"""
Energy measurement protocol (runs on the Raspberry Pi).

    python bench.py                      # real hardware, 30 interleaved trials
    python bench.py --sim --trials 5     # no hardware: tests the pipeline

For every trial the Pi asks the Pico to run a batch of inferences while the
Pico holds the marker pin high. A sampler thread reads the INA226 as fast as
I2C allows and keeps only the samples taken while the marker is high.

    P_run   = mean of V_bus * I over the window         [W]
    dt      = window length timed by the Pico's own microsecond timer
    E_total = P_run * dt / N                            [J per inference]
    E_dyn   = (P_run - P_idle) * dt / N                 [J per inference above idle]

Trials are interleaved in a random order (A, Z, L, P, I shuffled every round)
so drift in temperature or supply voltage hits every condition equally, and
every mode in a round sees the same images (paired design).
"""
import argparse
import collections
import csv
import os
import random
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_trapz = getattr(np, "trapezoid", None) or np.trapz     # NumPy 2 renamed it


class Sampler(threading.Thread):
    def __init__(self, sensor, link, history_s=12.0):
        super().__init__(daemon=True)
        self.sensor, self.link = sensor, link
        self.capture, self.capturing = [], False
        self.history = collections.deque(maxlen=4000)    # (t, P, marker) for the live plot
        self._stop = False
        self.history_s = history_s
        self.lock = threading.Lock()

    def run(self):
        last_hist = 0.0
        while not self._stop:
            t = time.perf_counter()
            mk = self.link.marker()
            v, i, p = self.sensor.read()
            if self.capturing and mk:
                self.capture.append((t, v, i, p))
            if t - last_hist > 0.01:
                with self.lock:
                    self.history.append((t, p, mk))
                last_hist = t
            if not hasattr(self.sensor, "bus"):
                time.sleep(0.0005)          # the fake sensor would otherwise spin a core

    def stop(self):
        self._stop = True

    def measure(self, fn):
        """Run fn() (a link.bench call) and return (its result, marker-high samples)."""
        self.capture = []
        self.capturing = True
        res = fn()
        time.sleep(0.02)                    # let the last samples land
        self.capturing = False
        return res, np.array(self.capture)


def trial(link, sampler, mode, start, count):
    res, s = sampler.measure(lambda: link.bench(mode, start, count))
    if len(s) < 5:
        raise RuntimeError(f"only {len(s)} samples in the {mode} window — check the marker wire")
    t, v, i, p = s.T
    dt = res["us"] * 1e-6
    e_trapz = _trapz(p, t)                    # trapezoid rule over the samples themselves
    res.update(P_mean=float(p.mean()), P_sd=float(p.std(ddof=1)), n_samples=len(p),
               V_mean=float(v.mean()), I_mean=float(i.mean()), E_trapz=float(e_trapz),
               E_window=float(p.mean() * dt))
    return res


def calibrate(link, modes, target_s):
    counts = {}
    for m in modes:
        r = link.bench(m, 0, 10)
        per = max(r["us"], 1) / 10
        counts[m] = max(10, int(round(target_s * 1e6 / per)))
        print(f"  {m}: {per:8.0f} us/inference -> {counts[m]} per trial")
    return counts


def make_link(args):
    from nnsim import Bundle
    bundle = Bundle(os.path.join(HERE, "model_bundle.npz"))
    if args.sim:
        from pico_link import SimLink
        from ina226 import FakePowerSensor
        link = SimLink(bundle)
        sensor = FakePowerSensor(link)
    else:
        from pico_link import PicoLink
        from ina226 import PowerSensor
        link = PicoLink(bundle, port=args.port)
        sensor = PowerSensor(chip=args.chip, r_shunt=args.shunt)
    return bundle, link, sensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--target-s", type=float, default=0.5, help="length of each measured window")
    ap.add_argument("--modes", default="AZLP")
    ap.add_argument("--port", default="/dev/serial0")
    ap.add_argument("--chip", default="ina226", choices=["ina226", "ina219"])
    ap.add_argument("--shunt", type=float, default=0.100, help="measured shunt resistance, ohms")
    ap.add_argument("--clock-khz", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "bench.csv"))
    args = ap.parse_args()

    bundle, link, sensor = make_link(args)
    print(link.info())
    if args.clock_khz:
        print(link.set_clock(args.clock_khz))
    modes = [m for m in args.modes if m != "P" or bundle.has_poisson]

    sampler = Sampler(sensor, link)
    sampler.start()
    print("calibrating batch sizes ...")
    counts = calibrate(link, modes, args.target_s)
    counts["I"] = int(args.target_s * 1e6)          # idle: count means microseconds

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ["round", "order", "mode", "count", "us", "correct", "spikes", "P_mean", "P_sd",
              "n_samples", "V_mean", "I_mean", "E_trapz", "E_window", "unix_time"]
    rng = random.Random(42)
    n_img = len(bundle.labels)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rnd in range(-1, args.trials):          # round -1 is a discarded warm-up
            order = modes + ["I"]
            rng.shuffle(order)
            for k, m in enumerate(order):
                start = (max(rnd, 0) * 97) % n_img
                r = trial(link, sampler, m, start, counts[m])
                if rnd < 0:
                    continue
                r.update(round=rnd, order=k, unix_time=time.time())
                w.writerow({k2: r[k2] for k2 in fields})
                f.flush()
                if m != "I":
                    print(f"round {rnd:2d} {m}  {r['P_mean'] * 1e3:7.2f} mW  "
                          f"{r['P_mean'] * r['us'] * 1e-6 / r['count'] * 1e6:8.1f} uJ/inf  "
                          f"acc {r['correct'] / r['count'] * 100:5.1f}%")
    sampler.stop()
    print("saved", args.out, "-> now run: python analyse.py", args.out)


if __name__ == "__main__":
    main()
