"""
Pepper's-ghost hologram for the Silicon Neuron demo (Raspberry Pi Zero 2 W).

    python3 hologram.py                 # real Pico + INA226, fullscreen
    python3 hologram.py --sim           # no hardware: windowed, energy numbers SIMULATED
    python3 hologram.py --sim --snapshot out.png --model E --view 0 --at 1.5

Five networks (zoo.py) and four visuals (views.py). Scroll the mouse wheel to
change network, click it (or hold the arcade button) to change visual, press
the arcade button for the next digit. Wiring and every key: controls.py.
Left alone for 90 s it tours by itself.

The screen lies face down above a 45-degree sheet; the viewer sees its mirror
image, so the frame is drawn normally then flipped left-right (--no-mirror for
a monitor). Black is invisible in the ghost: only light floats.

Energy numbers come from the Pico: a worker thread benchmarks every network in
turn (the current one most often) with the INA226, exactly as bench.py does.
The animation itself is computed on the Pi by nnsim.py, which is bit-exact
with the firmware, so the Pico is never interrupted while it is measured.
"""
import argparse
import collections
import json
import os
import threading
import time

import numpy as np

import views as V
import zoo
from controls import Controls

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1024, 600
TOUR_IDLE_S = 90


# ================================================================ energy worker
class EnergyWorker(threading.Thread):
    """Measures energy per inference of every network, current one first."""

    def __init__(self, link, sampler, codes, target_s=0.25):
        super().__init__(daemon=True)
        self.link, self.sampler, self.codes, self.target_s = link, sampler, codes, target_s
        self.uJ = {c: collections.deque(maxlen=7) for c in codes}
        self.counts = {}
        self.current = codes[0]
        self.status = "starting"
        self.lock = threading.Lock()

    def energy(self):
        with self.lock:
            return {c: float(np.median(v)) for c, v in self.uJ.items() if v}

    def run(self):
        from bench import trial
        k = 0
        while True:
            # the current network every other turn, the rest round-robin
            code = self.current if k % 2 == 0 else self.codes[(k // 2) % len(self.codes)]
            k += 1
            try:
                if code not in self.counts:
                    r = self.link.bench(code, 0, 3)
                    self.counts[code] = max(3, int(self.target_s * 1e6 / max(r["us"] / 3, 1)))
                r = trial(self.link, self.sampler, code, (k * 37) % 500, self.counts[code])
                with self.lock:
                    self.uJ[code].append(r["P_mean"] * r["us"] * 1e-6 / r["count"] * 1e6)
                self.status = "running"
            except Exception as e:                         # keep the show going
                self.status = f"Pico link: {e}"
                print(self.status, flush=True)
                time.sleep(1.0)


# ================================================================ app state
class App:
    def __init__(self, bundle, ink, models, sim, energy_worker, costs, energy_model):
        self.b, self.ink, self.models, self.sim = bundle, ink, models, sim
        self.ew, self.costs, self.em = energy_worker, costs, energy_model
        self.views = [cls(bundle, ink) for cls in V.VIEWS]
        self.mi, self.vi, self.idx = 0, 0, 0
        self.tr, self.t0 = None, 0.0
        self.tour = False
        self.digits_in_view = 0

    @property
    def model(self):
        return self.models[self.mi]

    def new_trace(self, now):
        self.tr = zoo.make_trace(self.b, self.model["code"], self.idx)
        self.t0 = now
        if self.ew:
            self.ew.current = self.model["code"]

    def handle(self, ev, now):
        if ev == "model+":
            self.mi = (self.mi + 1) % len(self.models)
        elif ev == "model-":
            self.mi = (self.mi - 1) % len(self.models)
        elif ev.startswith("pick"):
            self.mi = min(int(ev[4:]), len(self.models) - 1)
        elif ev == "view+":
            self.vi = (self.vi + 1) % len(self.views)
            return
        elif ev == "digit":
            self.idx = (self.idx + 1) % len(self.b.labels)
        elif ev == "tour":
            self.tour = not self.tour
            return
        self.new_trace(now)

    def info(self):
        p_active = 0.10
        pred = {}
        for m in self.models:
            c = self.costs.get(m["code"], {}).get("cycles")
            if c:
                pred[m["code"]] = self.em.pico_energy_uJ(c, p_active_w=p_active)
        return {"energy": self.ew.energy() if self.ew else {}, "predicted": pred, "sim": self.sim,
                "accuracy": {k: v.get("accuracy") for k, v in self.costs.items() if v.get("accuracy")},
                "models": self.models, "cycles_fn": self.em.pico_cycles}

    def draw(self, pg, s, now):
        tl = V.timeline(self.tr, now - self.t0)
        view = self.views[self.vi]
        view.draw(pg, s, now, self.tr, self.model, tl, self.info())
        self._chrome(pg, s, view)
        return tl

    def _chrome(self, pg, s, view):
        ink = self.ink
        ink.text(s, "SILICON NEURON", (24, 14), zoo.CYAN, "m")
        x = 250
        for k, m in enumerate(self.models):                 # the network selector
            sel = k == self.mi
            r = ink.text(s, m["short"], (x, 18), m["colour"] if sel else V.dim(m["colour"], 0.45), "sb" if sel else "s")
            if sel:
                pg.draw.line(s, m["colour"], (r.left, r.bottom + 3), (r.right, r.bottom + 3), 3)
            x = r.right + 18
        x = 720
        for k, v in enumerate(self.views):                   # the visual tabs
            sel = k == self.vi
            r = ink.text(s, v.name, (x, 18), V.WHITE if sel else V.dim(V.WHITE, 0.4), "sb" if sel else "s")
            if sel:
                pg.draw.line(s, V.WHITE, (r.left, r.bottom + 3), (r.right, r.bottom + 3), 2)
            x = r.right + 16
        ink.text(s, self.model["name"], (24, 46), self.model["colour"], "sb")
        hint = "wheel: network   click: view   button: next digit" + ("   (touring)" if self.tour else "")
        ink.text(s, hint, (W - 20, H - 22), V.dim(V.WHITE, 0.45), "xs", anchor="topright")
        if self.ew and self.ew.status not in ("running", "starting"):
            ink.text(s, self.ew.status[:80], (24, H - 22), V.RED, "xs")

    def tour_step(self, now):
        """Called when a digit's animation has finished while touring."""
        self.digits_in_view += 1
        if self.digits_in_view >= 2:
            self.digits_in_view = 0
            self.vi = (self.vi + 1) % len(self.views)
            if self.vi == 0:
                self.mi = (self.mi + 1) % len(self.models)
        self.idx = (self.idx + 1) % len(self.b.labels)
        self.new_trace(now)


def load_costs():
    p = os.path.join(HERE, "model_costs.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 180])
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--port", default="/dev/serial0")
    ap.add_argument("--chip", default="ina226")
    ap.add_argument("--shunt", type=float, default=0.100)
    ap.add_argument("--tour", action="store_true", help="start in the automatic tour")
    ap.add_argument("--model", default=None, help="start on this network (A Z L P E)")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--snapshot", help="render one frame to this PNG and exit (no display)")
    ap.add_argument("--at", type=float, default=1.5, help="snapshot: seconds into the animation")
    ap.add_argument("--idx", type=int, default=0, help="test image to start on")
    a = ap.parse_args()

    if a.snapshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame as pg
    import energy_model
    from bench import Sampler, make_link

    class _A:  # adapt to bench.make_link
        sim, port, chip, shunt = a.sim, a.port, a.chip, a.shunt
        calibration = os.path.join(HERE, "results", "calibration.json")
    bundle, link, sensor = make_link(_A)

    pg.init()
    fullscreen = not (a.windowed or a.sim or a.snapshot)
    screen = pg.display.set_mode((W, H), pg.FULLSCREEN if fullscreen else 0)
    pg.display.set_caption("Silicon Neuron hologram")
    pg.mouse.set_visible(False)
    ink = V.Ink(pg)
    models = zoo.available(bundle)
    canvas = pg.Surface((W, H))
    mirror = not a.no_mirror

    if a.snapshot:
        app = App(bundle, ink, models, a.sim, None, load_costs(), energy_model)
        if a.model:
            app.mi = [m["code"] for m in models].index(a.model)
        app.vi, app.idx = a.view, a.idx
        app.new_trace(0.0)
        canvas.fill((0, 0, 0))
        app.draw(pg, canvas, a.at)
        pg.image.save(pg.transform.flip(canvas, mirror, False), a.snapshot)
        print("saved", a.snapshot)
        return

    from sensors import StatusLEDs
    sampler = Sampler(sensor, link, leds=StatusLEDs(enabled=not a.sim))
    sampler.start()
    ew = EnergyWorker(link, sampler, [m["code"] for m in models])
    ew.start()
    controls = Controls(use_gpio=not a.sim)
    app = App(bundle, ink, models, a.sim, ew, load_costs(), energy_model)
    if a.model:
        app.mi = [m["code"] for m in models].index(a.model)
    app.vi, app.idx, app.tour = a.view, a.idx, a.tour
    app.new_trace(time.perf_counter())

    clock = pg.time.Clock()
    while True:
        now = time.perf_counter()
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                pg.quit()
                return
            cmd = controls.handle_pygame(pg, ev)
            if cmd == "quit":
                pg.quit()
                return
            if cmd == "mirror":
                mirror = not mirror
            if cmd == "fullscreen":
                pg.display.toggle_fullscreen()
        for ev in controls.events():
            app.handle(ev, now)
        if not app.tour and time.time() - controls.last_input > TOUR_IDLE_S:
            app.tour = True
        elif app.tour and time.time() - controls.last_input < 1:
            app.tour = False
        canvas.fill((0, 0, 0))
        tl = app.draw(pg, canvas, now)
        if tl.get("finished"):
            if app.tour:
                app.tour_step(now)
            else:                                    # replay the same digit until someone asks
                app.t0 = now
        out = pg.transform.flip(canvas, mirror, False)
        if a.rotate == 180:
            out = pg.transform.rotate(out, 180)
        screen.blit(out, (0, 0))
        pg.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
