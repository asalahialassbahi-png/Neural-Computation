"""
Pepper's-ghost hologram display for the Silicon Neuron demo (Raspberry Pi).

    python hologram.py              # real Pico + INA226, fullscreen
    python hologram.py --sim        # no hardware (laptop or Pi), windowed
    python hologram.py --sim --snapshot out.png --snapshot-step 9   # render one frame

The screen lies face-up under a 45-degree acrylic sheet. The viewer sees the
screen's REFLECTION, which is a mirror image standing upright behind the
sheet, so everything is drawn normally and then flipped left-right before it
reaches the screen (--no-mirror turns that off for testing on a monitor).
Black pixels emit no light and therefore vanish: only the bright lines and
dots appear to float in mid-air. Keep the palette saturated and the lines thick.

Keys: SPACE next digit · M toggle mirror · F toggle fullscreen · ESC quit
Button: Pi GPIO27 to GND = next digit
"""
import argparse
import collections
import json
import math
import os
import queue
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

W, H = 1024, 600
CYAN = (0, 225, 255)
MAGENTA = (255, 60, 210)
AMBER = (255, 185, 40)
GREEN = (60, 255, 140)
RED = (255, 80, 80)
WHITE = (235, 245, 255)
GREY = (70, 80, 95)
STEP_S = 0.18          # seconds of animation per SNN time step
HOLD_S = 3.0           # seconds the finished answer stays on screen


def dim(c, f):
    return tuple(int(v * f) for v in c)


# ================================================================ worker
class DemoWorker(threading.Thread):
    """Talks to the Pico (or simulator) off the render thread:
    fetch a trace, wait for the animation, run a short live benchmark."""

    def __init__(self, link, sampler, bundle, live_target_s=0.3):
        super().__init__(daemon=True)
        self.link, self.sampler, self.bundle = link, sampler, bundle
        self.traces = queue.Queue(maxsize=1)
        self.anim_done = threading.Event()
        self.skip = threading.Event()
        self.energy = {"A": collections.deque(maxlen=20), "L": collections.deque(maxlen=20)}
        self.power_idle = collections.deque(maxlen=20)
        self.idx = 0
        self.counts = None
        self.live_target_s = live_target_s
        self.status = "starting"

    def _calibrate(self):
        self.counts = {}
        for m in ("A", "L"):
            r = self.link.bench(m, 0, 5)
            self.counts[m] = max(5, int(self.live_target_s * 1e6 / max(r["us"] / 5, 1)))

    def _live_bench(self):
        from bench import trial
        for m in ("A", "L"):
            r = trial(self.link, self.sampler, m, self.idx, self.counts[m])
            self.energy[m].append(r["P_mean"] * r["us"] * 1e-6 / r["count"])
        r = trial(self.link, self.sampler, "I", 0, int(0.15e6))
        self.power_idle.append(r["P_mean"])

    def run(self):
        try:
            self._calibrate()
        except Exception as e:
            self.status = f"calibration failed: {e}"
            print(self.status, flush=True)
        while True:
            try:
                tr = self.link.trace(self.idx)
                tr["idx"] = self.idx
                self.anim_done.clear()
                self.traces.put(tr)
                self.status = "running"
                if self.counts and self.sampler:
                    self._live_bench()
                self.anim_done.wait()
            except Exception as e:                    # keep the show going
                self.status = f"link error: {e}"
                print(self.status, flush=True)
                time.sleep(1.0)
            self.idx = (self.idx + 1) % len(self.bundle.labels)


# ================================================================ scene
class Scene:
    def __init__(self, bundle, fonts):
        self.b, self.f = bundle, fonts
        g = np.linspace(-0.62, 0.62, 28)
        yy, zz = np.meshgrid(-g, g, indexing="ij")        # row 0 at the top
        self.p_in = np.stack([np.full(784, -1.7), yy.ravel(), zz.ravel()], 1)
        hy = np.linspace(-0.5, 0.5, 15)
        hz = np.linspace(-0.72, 0.72, 20)
        yy, zz = np.meshgrid(hy, hz, indexing="ij")
        self.p_hid = np.stack([np.zeros(300), yy.ravel(), zz.ravel()], 1)[:bundle.H]
        self.p_out = np.stack([np.full(10, 1.7), np.linspace(0.6, -0.6, 10), np.zeros(10)], 1)
        self.tr = None
        self.t_start = 0.0
        self.signalled = False       # finished-animation event sent for this trace

    def project(self, P, t, cx, cy, scale):
        yaw = -0.70 + 0.20 * math.sin(0.23 * t)      # layers seen obliquely, slowly swaying
        pitch = 0.22 + 0.05 * math.sin(0.17 * t)
        cyw, syw, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        x = P[:, 0] * cyw + P[:, 2] * syw
        z = -P[:, 0] * syw + P[:, 2] * cyw
        y = P[:, 1] * cp - z * sp
        z = P[:, 1] * sp + z * cp
        d = 6.0 + z                       # camera 6 units away: gentle perspective
        k = scale / d
        return np.stack([cx + x * k, cy - y * k], 1), k / scale * 6.0   # screen xy, depth factor

    # -------------------------------------------------------------- draw
    def draw(self, pg, surf, now, energy, idle, status, power_hist):
        tr = self.tr
        t_anim = now - self.t_start if tr else 0
        T = self.b.T
        step_f = min(t_anim / STEP_S, T - 1e-6) if tr else 0
        n = int(step_f)
        done = tr is not None and t_anim >= T * STEP_S

        self._header(pg, surf)
        self._network(pg, surf, now, tr, n, step_f, done)
        if tr is not None:
            self._membrane(pg, surf, tr, step_f)
            self._outputs(pg, surf, tr, n, done)
            self._verdict(pg, surf, tr, done)
        self._energy(pg, surf, energy, idle, power_hist)
        if status and status != "running":
            surf.blit(self.f["s"].render(status[:70], True, RED), (20, H - 24))
        return done and t_anim >= T * STEP_S + HOLD_S

    def _header(self, pg, s):
        s.blit(self.f["m"].render("SILICON NEURON", True, CYAN), (20, 12))
        s.blit(self.f["s"].render("spiking network vs standard network, live on a Raspberry Pi Pico",
                                  True, dim(WHITE, 0.7)), (260, 20))

    def _network(self, pg, s, now, tr, n, step_f, done):
        cx, cy, sc = 312, 290, 900
        pin, kin = self.project(self.p_in, now, cx, cy, sc)
        phid, khid = self.project(self.p_hid, now, cx, cy, sc)
        pout, kout = self.project(self.p_out, now, cx, cy, sc)
        # --- input layer: the digit glows faintly; pixels flash when they spike
        if tr is not None:
            it = tr["in_time"].astype(int)
            x = self.b.images[tr["idx"]]
            for i in np.nonzero(x >= 20)[0]:
                c = dim(CYAN, 0.18 + 0.25 * x[i] / 255)
                if it[i] <= n:
                    age = step_f - it[i]
                    c = CYAN if age < 1.2 else dim(CYAN, 0.45 + 0.2 * x[i] / 255)
                pg.draw.circle(s, c, pin[i], max(1, int(2.2 * kin[i])))
        else:
            for i in range(0, 784, 3):
                pg.draw.circle(s, dim(CYAN, 0.12), pin[i], 1)
        # --- hidden layer: every neuron is a dot; spikes flash magenta
        spk_now = tr["raster"][n] if tr is not None and not done else np.zeros(len(phid), bool)
        recent = tr["raster"][max(0, n - 2):n + 1].any(0) if tr is not None and not done else spk_now
        for j in range(len(phid)):
            r = max(2, int(3.0 * khid[j]))
            if spk_now[j]:
                pg.draw.circle(s, dim(MAGENTA, 0.35), phid[j], r + 5)
                pg.draw.circle(s, MAGENTA, phid[j], r + 1)
            elif recent[j]:
                pg.draw.circle(s, dim(MAGENTA, 0.55), phid[j], r)
            else:
                pg.draw.circle(s, dim(MAGENTA, 0.28), phid[j], r - 1)
        # --- spikes travelling to the output layer
        if tr is not None and not done:
            winner = int(np.argmax(tr["c_hist"][n]))
            for j in np.nonzero(spk_now)[0][:40]:
                pg.draw.line(s, dim(AMBER, 0.35), phid[j], pout[winner], 1)
        # --- output layer
        for k in range(10):
            on = tr is not None and ((done and k == tr["pred"]) or (not done and k == int(np.argmax(tr["c_hist"][n]))))
            col = AMBER if on else dim(AMBER, 0.3)
            rad = int((9 if on else 6) * kout[k])
            if on:
                pg.draw.circle(s, dim(AMBER, 0.3), pout[k], rad + 7)
            pg.draw.circle(s, col, pout[k], rad)
            s.blit(self.f["s"].render(str(k), True, col), (pout[k][0] + 12, pout[k][1] - 8))
        for label, xs, col in (("pixels -> spikes", pin, CYAN), ("300 spiking neurons", phid, MAGENTA),
                               ("digit", pout, AMBER)):
            img = self.f["s"].render(label, True, dim(col, 0.85))
            s.blit(img, (max(8, xs[:, 0].mean() - img.get_width() / 2), xs[:, 1].max() + 14))

    def _membrane(self, pg, s, tr, step_f):
        x0, y0, w, h = 590, 70, 410, 170
        pg.draw.rect(s, GREY, (x0, y0, w, h), 1)
        s.blit(self.f["s"].render("membrane voltage U(t) of three neurons", True, WHITE), (x0 + 6, y0 + 4))
        th = max(1, tr["theta"])
        T = self.b.T
        rel = tr["u"] / th
        lo, hi = min(-1.0, rel.min()) - 0.2, max(1.5, rel.max()) + 0.2     # auto-scale in units of theta
        ymap = lambda u: y0 + h - 10 - (u / th - lo) / (hi - lo) * (h - 50)
        ty = ymap(th)
        for xx in range(x0 + 4, x0 + w - 4, 14):
            pg.draw.line(s, dim(RED, 0.8), (xx, ty), (xx + 7, ty), 2)
        s.blit(self.f["s"].render("threshold", True, dim(RED, 0.9)), (x0 + w - 80, ty - 20))
        cols = (CYAN, MAGENTA, GREEN)
        tn = tr["trace_idx"]
        for q in range(3):
            pts = []
            for k in range(int(step_f) + 1):
                pts.append((x0 + 8 + k / (T - 1) * (w - 16), ymap(tr["u"][k][q])))
            if len(pts) > 1:
                pg.draw.lines(s, cols[q], False, pts, 3)
            for k in range(int(step_f) + 1):
                if tr["raster"][k][tn[q]]:
                    xx = x0 + 8 + k / (T - 1) * (w - 16)
                    pg.draw.line(s, cols[q], (xx, y0 + 26), (xx, y0 + 40), 3)

    def _outputs(self, pg, s, tr, n, done):
        x0, y0, w, h = 590, 255, 410, 125
        c = tr["c_hist"][-1 if done else n].astype(float)
        scale = max(1.0, np.abs(tr["c_hist"][-1]).max() / 6)
        e = np.exp((c - c.max()) / scale)
        p = e / e.sum()
        s.blit(self.f["s"].render("how sure the spiking network is", True, WHITE), (x0 + 6, y0))
        bw = w / 10
        for k in range(10):
            bh = p[k] * (h - 45)
            col = AMBER if k == int(np.argmax(c)) else dim(AMBER, 0.4)
            pg.draw.rect(s, col, (x0 + k * bw + 6, y0 + h - 20 - bh, bw - 12, bh))
            s.blit(self.f["s"].render(str(k), True, col), (x0 + k * bw + bw / 2 - 5, y0 + h - 18))

    def _verdict(self, pg, s, tr, done):
        if not done:
            return
        lab = tr["label"]
        for i, (name, pred, col) in enumerate((("spiking network", tr["pred"], MAGENTA),
                                               ("standard network", tr["ann_pred"], CYAN))):
            ok = pred == lab
            txt = f"{name}: {pred} {'correct' if ok else 'wrong'}"
            s.blit(self.f["m"].render(txt, True, col if ok else RED), (30, 520 + i * 34))

    def _energy(self, pg, s, energy, idle, power_hist):
        x0, y0, w, h = 590, 392, 410, 196
        pg.draw.rect(s, GREY, (x0, y0, w, h), 1)
        s.blit(self.f["s"].render("energy per answer, measured live on the Pico", True, WHITE), (x0 + 6, y0 + 4))
        ea = np.mean(energy["A"]) if energy["A"] else None
        el = np.mean(energy["L"]) if energy["L"] else None
        if ea and el:
            mx = max(ea, el)
            for i, (name, e, col) in enumerate((("standard", ea, CYAN), ("spiking", el, MAGENTA))):
                yy = y0 + 32 + i * 44
                pg.draw.rect(s, col, (x0 + 100, yy, (w - 200) * e / mx, 28))
                s.blit(self.f["s"].render(name, True, col), (x0 + 8, yy + 5))
                s.blit(self.f["s"].render(f"{e * 1e6:,.0f} uJ", True, col), (x0 + w - 92, yy + 5))
            ratio = ea / el
            msg = (f"spiking uses {ratio:.1f}x less energy" if ratio >= 1
                   else f"spiking uses {1 / ratio:.1f}x MORE energy")
            s.blit(self.f["m"].render(msg, True, GREEN if ratio >= 1 else RED), (x0 + 8, y0 + 110))
        else:
            s.blit(self.f["s"].render("measuring ...", True, dim(WHITE, 0.6)), (x0 + 8, y0 + 40))
        # live power strip (last ~6 s)
        if power_hist is not None and len(power_hist):
            ph = np.array(power_hist)
            ph = ph[ph[:, 0] > ph[-1, 0] - 6.0]
            if len(ph) > 2:
                lo, hi = ph[:, 1].min(), ph[:, 1].max() + 1e-4
                xs = x0 + 8 + (ph[:, 0] - ph[-1, 0] + 6.0) / 6.0 * (w - 16)
                ys = y0 + h - 8 - (ph[:, 1] - lo) / (hi - lo) * 26
                pg.draw.lines(s, dim(GREEN, 0.8), False, list(zip(xs, ys)), 2)
                s.blit(self.f["s"].render(f"Pico power now: {ph[-1, 1] * 1e3:.0f} mW", True, dim(GREEN, 0.8)),
                       (x0 + 8, y0 + h - 52))


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
    ap.add_argument("--snapshot", help="render one frame to this PNG and exit (no display)")
    ap.add_argument("--snapshot-step", type=float, default=9.5)
    ap.add_argument("--snapshot-idx", type=int, default=0)
    a = ap.parse_args()

    if a.snapshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame as pg
    from bench import Sampler, make_link

    class _A:  # adapt to bench.make_link
        sim, port, chip, shunt = a.sim, a.port, a.chip, a.shunt
    bundle, link, sensor = make_link(_A)

    pg.init()
    fullscreen = not (a.windowed or a.sim or a.snapshot)
    screen = pg.display.set_mode((W, H), pg.FULLSCREEN if fullscreen else 0)
    pg.display.set_caption("Silicon Neuron hologram")
    pg.mouse.set_visible(False)

    def font(sz, bold=False):
        try:
            return pg.font.SysFont("dejavusans", sz, bold=bold)
        except Exception:
            return pg.font.Font(None, sz + 6)
    fonts = {"s": font(16), "m": font(26, True), "l": font(48, True)}
    scene = Scene(bundle, fonts)
    canvas = pg.Surface((W, H))
    mirror = not a.no_mirror

    if a.snapshot:
        tr = link.trace(a.snapshot_idx)
        tr["idx"] = a.snapshot_idx
        scene.tr, scene.t_start = tr, 0.0
        energy = {"A": [], "L": []}          # a snapshot never shows invented energy numbers
        now = a.snapshot_step * STEP_S
        canvas.fill((0, 0, 0))
        scene.draw(pg, canvas, now, energy, [], "running", None)
        out = pg.transform.flip(canvas, mirror, False)
        pg.image.save(out, a.snapshot)
        print("saved", a.snapshot)
        return

    sampler = Sampler(sensor, link)
    sampler.start()
    worker = DemoWorker(link, sampler, bundle)
    worker.start()

    button = None
    if not a.sim:
        try:
            from gpiozero import Button
            button = Button(27, pull_up=True, bounce_time=0.05)
            button.when_pressed = lambda: worker.anim_done.set()
        except Exception as e:
            print("button unavailable:", e)

    clock = pg.time.Clock()
    while True:
        for ev in pg.event.get():
            if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE):
                pg.quit()
                return
            if ev.type == pg.KEYDOWN and ev.key == pg.K_SPACE:
                worker.anim_done.set()
            if ev.type == pg.KEYDOWN and ev.key == pg.K_m:
                mirror = not mirror
            if ev.type == pg.KEYDOWN and ev.key == pg.K_f:
                pg.display.toggle_fullscreen()
        now = time.perf_counter()
        try:
            tr = worker.traces.get_nowait()
            scene.tr, scene.t_start, scene.signalled = tr, now, False
        except queue.Empty:
            pass
        canvas.fill((0, 0, 0))
        with sampler.lock:
            ph = [(t, p, m) for t, p, m in sampler.history]
        finished = scene.draw(pg, canvas, now, worker.energy, list(worker.power_idle), worker.status, ph)
        if finished and not scene.signalled:
            scene.signalled = True
            worker.anim_done.set()
        out = pg.transform.flip(canvas, mirror, False)
        if a.rotate == 180:
            out = pg.transform.rotate(out, 180)
        screen.blit(out, (0, 0))
        pg.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
