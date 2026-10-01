"""
Link to the Pico over the Pi's UART, plus a simulator with the same interface.

Wiring (Pi header -> Pico):
    GPIO14 TXD (pin 8)  -> GP1 UART0 RX (Pico pin 2)
    GPIO15 RXD (pin 10) <- GP0 UART0 TX (Pico pin 1)
    GPIO17     (pin 11) <- GP2 marker   (Pico pin 4)
    GND        (pin 6)  -- GND          (Pico pin 3)
"""
import threading
import time

import numpy as np

MODES = {"A": "ANN (dense)", "Z": "ANN (zero-skip)", "L": "SNN (latency)",
         "P": "SNN (Poisson)", "I": "idle"}


def _parse_trace(lines, bundle):
    T, H = bundle.T, bundle.H
    raster = np.zeros((T, H), bool)
    u = np.zeros((T, 3), np.int64)
    out = {}
    for ln in lines:
        tag, *rest = ln.split()
        if tag == "TI":
            out["in_time"] = np.frombuffer(bytes.fromhex(rest[0]), np.uint8).copy()
        elif tag == "TN":
            out["trace_idx"] = [int(v) for v in rest]
        elif tag == "TS":
            bits = np.unpackbits(np.frombuffer(bytes.fromhex(rest[1]), np.uint8), bitorder="little")
            raster[int(rest[0])] = bits[:H].astype(bool)
        elif tag == "TU":
            u[int(rest[0])] = [int(v) for v in rest[1:4]]
        elif tag == "TO":
            v = [int(s) for s in rest]
            out["c"] = np.array(v[:10])
            out["pred"], out["label"], out["spikes"], out["theta"] = v[10:14]
        elif tag == "TA":
            out["ann_pred"] = int(rest[0])
    out["raster"], out["u"] = raster, u
    out["c_hist"], _ = bundle.output_from_raster(raster, "latency")
    return out


class PicoLink:
    def __init__(self, bundle, port="/dev/serial0", baud=115200, marker_gpio=17):
        import serial
        self.bundle = bundle
        self.ser = serial.Serial(port, baud, timeout=0.2)
        self.lock = threading.Lock()
        try:
            from gpiozero import DigitalInputDevice
            self._marker = DigitalInputDevice(marker_gpio, pull_up=False)
        except Exception as e:                      # still usable without the marker
            print("marker GPIO unavailable:", e)
            self._marker = None
        self.ser.reset_input_buffer()

    def marker(self):
        return bool(self._marker.value) if self._marker else False

    def _cmd(self, text, until=lambda ln: True, timeout=30.0):
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write((text + "\n").encode())
            lines, t0 = [], time.time()
            while time.time() - t0 < timeout:
                ln = self.ser.readline().decode(errors="replace").strip()
                if not ln or ln.startswith("INFO boot"):
                    continue
                lines.append(ln)
                if until(ln):
                    return lines
            raise TimeoutError(f"no reply to {text!r}")

    def info(self):
        return self._cmd("?", lambda ln: ln.startswith("INFO"))[-1]

    def set_clock(self, khz):
        return self._cmd(f"C {khz}", lambda ln: ln.startswith(("OK", "ERR")))[-1]

    def bench(self, mode, start, count):
        ln = self._cmd(f"B {mode} {start} {count}", lambda l: l.startswith(("R ", "ERR")))[-1]
        if ln.startswith("ERR"):
            raise RuntimeError(ln)
        f = ln.split()                      # R m count us correct spikes [steps]
        return {"mode": f[1], "count": int(f[2]), "us": int(f[3]), "correct": int(f[4]),
                "spikes": int(f[5]), "steps": int(f[6]) if len(f) > 6 else 0}

    def trace(self, idx):
        lines = self._cmd(f"T {idx}", lambda l: l == "END")
        return _parse_trace(lines, self.bundle)


class SimLink:
    """Behaves like PicoLink using the NumPy integer model and a crude timing /
    power model (only for developing the display without hardware).
    ITS ENERGY NUMBERS ARE MADE UP — never quote them; only the Pico measures."""
    US_PER_OP = {"A": 0.045, "Z": 0.050, "L": 0.055, "P": 0.080, "E": 0.050}   # rough RP2040 guesses
    LOAD_W = {"A": 0.0120, "Z": 0.0115, "L": 0.0105, "P": 0.0110, "E": 0.0105, "I": 0.0}

    def __init__(self, bundle):
        self.bundle = bundle
        self._mk = False
        self._mode = "I"

    def marker(self):
        return self._mk

    def fake_load_w(self):
        return self.LOAD_W.get(self._mode, 0.0)

    def info(self):
        return f"INFO SIM H={self.bundle.H} T={self.bundle.T} NTEST={len(self.bundle.labels)}"

    def set_clock(self, khz):
        return f"OK {khz}"

    def bench(self, mode, start, count):
        b = self.bundle
        correct = spikes = ops = 0
        if mode == "I":
            dur = count * 1e-6
        else:
            for k in range(count):
                idx = (start + k) % len(b.labels)
                x = b.images[idx]
                if mode == "E":
                    r = b.snn_fast(x)
                    p, s = r["pred"], r["spikes"]
                    spikes += s
                    ops += r["in_spikes"] * b.H + s * 10 + r["steps"] * (b.H + 10) * 2
                elif mode in "AZ":
                    p, h = b.ann(x)
                    ops += 784 * b.H + b.H * 10 if mode == "A" else (x > 0).sum() * b.H + (h > 0).sum() * 10
                else:
                    r = b.snn(x, "latency" if mode == "L" else "poisson", idx)
                    p, s = r["pred"], r["spikes"]
                    spikes += s
                    ops += r["in_spikes"] * b.H + s * 10 + b.T * (b.H + 10) * 3
                    if mode == "P":
                        ops += (x > 0).sum() * b.T * 4          # one RNG draw per non-zero pixel per step
                correct += p == b.labels[idx]
            dur = ops * self.US_PER_OP[mode] * 1e-6
        self._mode, self._mk = mode, True
        time.sleep(dur)
        self._mk, self._mode = False, "I"
        return {"mode": mode, "count": count, "us": int(dur * 1e6), "correct": int(correct),
                "spikes": int(spikes)}

    def trace(self, idx):
        b = self.bundle
        idx %= len(b.labels)
        r = b.snn(b.images[idx], "latency", idx)
        r["label"] = int(b.labels[idx])
        r["ann_pred"] = b.ann(b.images[idx])[0]
        return r
