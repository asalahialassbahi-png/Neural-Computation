"""
The four things the hologram can show, all driven by the same trace (zoo.py):

    NetworkView   the network computing, in 3D: pixels -> hidden layer -> digit.
                  ANN: a wave sweeps the input, every multiply flashes.
                  SNN: spikes travel as pulses, firing neurons ripple outwards.
                  A live counter shows how many operations the Pico is doing.
    RaceView      energy per digit for every model (measured on the Pico) and
                  accuracy against energy: which network wins, and by how much.
    EquationView  the model's governing equations, with this digit's numbers
                  substituted in, and the energy inequality that decides the race.
    NeuronView    inside the hidden layer: membrane voltages crossing threshold
                  and the spike raster (SNN), or the ReLU activations (ANN).

Pepper's ghost rules: black is invisible, so everything is a bright line or dot
on black; colours are saturated; nothing important sits near the screen edge.
"""
import io
import math

import numpy as np

from zoo import AMBER, CYAN, GREEN, MAGENTA, H, N_IN, N_OUT

W, HGT = 1024, 600
WHITE = (235, 245, 255)
GREY = (70, 80, 95)
RED = (255, 80, 80)
DENSE_OPS = N_IN * H + H * N_OUT                 # the dense ANN's multiply-adds: the yardstick

ANN_SWEEP_S, ANN_HID_S, ANN_OUT_S = 1.8, 0.8, 0.8
STEP_S = 0.34
SNN_MIN_S = 2.0          # an early exit after 1 step still animates for this long
HOLD_S = 3.0


def step_s(tr):
    return max(STEP_S, SNN_MIN_S / max(1, tr["steps"]))


def dim(c, f):
    f = max(0.0, min(1.0, f))
    return tuple(int(v * f) for v in c)


# ================================================================ timeline
def timeline(tr, t):
    """Where the animation of this trace is at time t (seconds since it started)."""
    if tr is None:
        return {"phase": "idle", "done": False, "ops": 0, "finished": False}
    if tr["family"] == "ann":
        t1, t2, t3 = ANN_SWEEP_S, ANN_SWEEP_S + ANN_HID_S, ANN_SWEEP_S + ANN_HID_S + ANN_OUT_S
        if t < t1:
            f = t / t1
            ops = int(f * len(tr["rows"])) * H
            return {"phase": "input", "f": f, "ops": ops, "done": False, "finished": False}
        rows_ops = len(tr["rows"]) * H
        if t < t2:
            return {"phase": "hidden", "f": (t - t1) / ANN_HID_S, "ops": rows_ops, "done": False, "finished": False}
        if t < t3:
            f = (t - t2) / ANN_OUT_S
            return {"phase": "output", "f": f, "ops": rows_ops + int(f * len(tr["hid_rows"])) * N_OUT,
                    "done": False, "finished": False}
        return {"phase": "done", "f": 1.0, "ops": tr["ops_total"], "done": True, "finished": t > t3 + HOLD_S}
    steps, ds = tr["steps"], step_s(tr)
    if t < steps * ds:
        sf = t / ds
        n = int(sf)
        prev = tr["ops"][n - 1] if n > 0 else 0
        ops = prev + int((sf - n) * (tr["ops"][n] - prev))
        return {"phase": "steps", "n": n, "f": sf - n, "sf": sf, "ops": ops, "done": False, "finished": False}
    return {"phase": "done", "n": steps - 1, "f": 1.0, "sf": steps, "ops": tr["ops_total"], "done": True,
            "finished": t > steps * ds + HOLD_S}


# ================================================================ helpers
class Ink:
    """Fonts, wrapped text and LaTeX equations (matplotlib mathtext) as surfaces."""

    def __init__(self, pg):
        self.pg = pg

        def font(sz, bold=False):
            try:
                return pg.font.SysFont("dejavusans", sz, bold=bold)
            except Exception:
                return pg.font.Font(None, sz + 6)
        self.f = {"xs": font(13), "s": font(16), "sb": font(16, True), "m": font(22, True),
                  "l": font(34, True), "xl": font(96, True)}
        self._tex = {}

    def text(self, surf, s, pos, col, size="s", anchor="topleft"):
        img = self.f[size].render(str(s), True, col)
        r = img.get_rect(**{anchor: pos})
        surf.blit(img, r)
        return r

    def wrap(self, surf, s, pos, col, width, size="s", gap=4):
        words, line, y = s.split(), "", pos[1]
        for w in words:
            trial = (line + " " + w).strip()
            if self.f[size].size(trial)[0] > width and line:
                self.text(surf, line, (pos[0], y), col, size)
                y += self.f[size].get_height() + gap
                line = w
            else:
                line = trial
        if line:
            self.text(surf, line, (pos[0], y), col, size)
            y += self.f[size].get_height() + gap
        return y

    def tex(self, s, px, col):
        """A LaTeX string rendered by matplotlib's mathtext, recoloured, cached."""
        key = (s, px, col)
        if key in self._tex:
            return self._tex[key]
        pg = self.pg
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import mathtext
            from matplotlib.font_manager import FontProperties
            buf = io.BytesIO()
            mathtext.math_to_image(s, buf, prop=FontProperties(size=px), dpi=100, format="png")
            buf.seek(0)
            img = pg.image.load(buf, "eq.png").convert_alpha()
            # mathtext draws dark ink: keep its coverage (darkness x alpha) as the alpha of a bright colour
            rgb = pg.surfarray.array3d(img).astype(np.float32)
            a = pg.surfarray.array_alpha(img).astype(np.float32)
            cov = (a / 255.0) * (1.0 - rgb.mean(2) / 255.0)
            out = pg.Surface(img.get_size(), pg.SRCALPHA)
            out.fill(col + (0,))
            pg.surfarray.pixels_alpha(out)[:] = np.clip(cov * 255, 0, 255).astype(np.uint8)
        except Exception:
            plain = s.replace("$", "").replace("\\", "")
            out = self.f["m"].render(plain, True, col)
        self._tex[key] = out
        return out


# layers turned towards the viewer, spaced so they do not overlap; the camera is far
# enough back that the near (input) layer stays on screen
YAW0, LAYER_DX, CAM_D = -1.05, 2.4, 9.0
NET_CX, NET_CY, NET_SCALE = 400, 300, 1300


def project(P, t, cx, cy, scale):
    yaw = YAW0 + 0.10 * math.sin(0.23 * t)           # layers seen obliquely, slowly swaying
    pitch = 0.22 + 0.05 * math.sin(0.17 * t)
    cyw, syw, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    x = P[:, 0] * cyw + P[:, 2] * syw
    z = -P[:, 0] * syw + P[:, 2] * cyw
    y = P[:, 1] * cp - z * sp
    z = P[:, 1] * sp + z * cp
    k = scale / (CAM_D + z)
    return np.stack([cx + x * k, cy - y * k], 1), k / scale * 6.0


# Network layout: each layer faces the viewer (so the digit stays readable in the
# ghost) with a slight slant and a slow sway that give it depth.
LAYERS = {"in": (175, 300, 7.6), "hid": (400, 300, 11.0), "out": (585, 300, 27.0)}   # centre x, y, px per unit


def place(local_yz, layer, t):
    cx, cy, px = LAYERS[layer]
    slant = 0.22 + 0.05 * math.sin(0.23 * t)
    squash = 0.82 + 0.04 * math.sin(0.17 * t)
    y, z = local_yz[:, 0], local_yz[:, 1]
    return np.stack([cx + z * px * squash, cy - y * px + z * px * slant], 1)


def softmax_scaled(c):
    c = np.asarray(c, float)
    s = max(1.0, np.abs(c).max() / 6)
    e = np.exp((c - c.max()) / s)
    return e / e.sum()


# ================================================================ network view
class NetworkView:
    name = "Network"

    def __init__(self, bundle, ink):
        self.b, self.ink = bundle, ink
        g = np.arange(28) - 13.5
        yy, zz = np.meshgrid(-g, g, indexing="ij")          # row 0 at the top
        self.l_in = np.stack([yy.ravel(), zz.ravel()], 1)
        hy, hz = np.arange(15) - 7, np.arange(20) - 9.5
        yy, zz = np.meshgrid(-hy, hz, indexing="ij")
        self.l_hid = np.stack([yy.ravel(), zz.ravel()], 1)
        self.l_out = np.stack([np.linspace(4.5, -4.5, 10), np.zeros(10)], 1)
        self.wiring = {}

    def _wires(self, code):
        """Each pixel's two strongest hidden targets and each hidden neuron's strongest
        output, from the model's own weights: the paths the pulses travel along."""
        if code not in self.wiring:
            z = self.b.z
            key = {"A": "ann", "Z": "ann", "L": "snn_latency", "P": "snn_poisson", "E": "snn_fast"}[code]
            W1, W2 = z[key + "_W1"].astype(np.int32), z[key + "_W2"].astype(np.int32)
            self.wiring[code] = (np.argsort(-W1, 1)[:, :2], np.argmax(W2, 1))
        return self.wiring[code]

    def draw(self, pg, s, now, tr, model, tl, info):
        pin, phid, pout = place(self.l_in, "in", now), place(self.l_hid, "hid", now), place(self.l_out, "out", now)
        kin, khid, kout = np.full(784, 1.1), np.full(300, 1.15), np.full(10, 1.2)
        col = model["colour"]
        if tr is None:
            for i in range(0, 784, 3):
                pg.draw.circle(s, dim(CYAN, 0.12), pin[i], 1)
            return
        w_in, w_out = self._wires(tr["code"])
        x = tr["x"]
        lit_in = np.zeros(784, float)          # brightness of each input dot this frame
        hid_b = np.full(300, 0.22)             # brightness of each hidden neuron
        pulses, rings = [], []                 # (p0, p1, f, colour), (centre, f, colour)

        if tr["family"] == "ann":
            rows = tr["rows"]
            if tl["phase"] == "input":
                k = int(tl["f"] * len(rows))
                done_rows = rows[:k]
                lit_in[done_rows] = 0.35 + 0.65 * x[done_rows] / 255
                head = rows[max(0, k - 18):k]                       # the sweep front
                for i in head:
                    bright = x[i] > 0
                    lit_in[i] = 1.0 if bright else 0.6
                    for j in w_in[i][:1 if tr["code"] == "Z" else 2]:
                        pulses.append((pin[i], phid[j], 0.5, col if bright else dim(RED, 0.8)))
                hid_b[:] = 0.22 + 0.3 * tl["f"]
            else:
                lit_in[rows] = 0.35 + 0.65 * x[rows] / 255
                hmax = max(1, tr["h"].max())
                fh = 1.0 if tl["phase"] != "hidden" else tl["f"]
                hid_b = 0.15 + 0.85 * fh * tr["h"] / hmax
                if tl["phase"] == "hidden":
                    for j in np.argsort(-tr["h"])[:24]:
                        rings.append((phid[j], tl["f"], col))
                if tl["phase"] in ("output", "done"):
                    f = tl["f"] if tl["phase"] == "output" else 1.0
                    for j in np.argsort(-tr["h"])[:30]:
                        pulses.append((phid[j], pout[w_out[j]], f, dim(AMBER, 0.8)))
            out_p = softmax_scaled(tr["logits"]) if tl["phase"] in ("output", "done") else np.zeros(10)
        else:
            n, f = (tl["n"], tl["f"]) if tl["phase"] in ("steps", "done") else (0, 0.0)
            frames = tr["frames"]
            # pixels that have already spiked stay faintly lit; this step's flash bright
            for m in range(min(n + 1, len(frames))):
                ii = frames[m]["in"]
                if m == n and not tl["done"]:
                    lit_in[ii] = 1.0
                    for i in ii[:70]:
                        pulses.append((pin[i], phid[w_in[i][0]], f, col))
                else:
                    lit_in[ii] = np.maximum(lit_in[ii], 0.35)
            fired_now = frames[n]["fired"] if not tl["done"] else np.array([], int)
            recent = np.zeros(300, bool)
            for m in range(max(0, n - 2), n + 1):
                recent[frames[m]["fired"]] = True
            hid_b = np.where(recent, 0.55, 0.2)
            hid_b[fired_now] = 1.0
            for j in fired_now[:40]:
                rings.append((phid[j], f, col))
                pulses.append((phid[j], pout[w_out[j]], f, dim(AMBER, 0.85)))
            out_p = softmax_scaled(frames[min(n, len(frames) - 1)]["c"])

        # ---- faint wiring of the digit's pixels, so the structure reads in the air
        for i in np.nonzero(x > 120)[0][::6]:
            pg.draw.line(s, dim(col, 0.10), pin[i], phid[w_in[i][0]], 1)
        # ---- input layer
        for i in np.nonzero(lit_in > 0)[0]:
            pg.draw.circle(s, dim(CYAN, lit_in[i]), pin[i], max(1, int(2.3 * kin[i])))
        if tr["code"] == "A" and tl["phase"] == "input":      # even blank pixels get multiplied
            for i in tr["rows"][max(0, int(tl["f"] * 784) - 18):int(tl["f"] * 784)]:
                if x[i] == 0:
                    pg.draw.circle(s, dim(RED, 0.7), pin[i], max(1, int(2.0 * kin[i])))
        # ---- pulses along the wires
        for p0, p1, fp, c in pulses:
            q = p0 + (p1 - p0) * fp
            pg.draw.line(s, dim(c, 0.35), p0, q, 1)
            pg.draw.circle(s, c, q, 2)
        # ---- hidden layer and ripples
        for j in range(300):
            r = max(2, int(3.0 * khid[j]))
            pg.draw.circle(s, dim(MAGENTA if tr["family"] == "snn" else col, hid_b[j]), phid[j], r)
        for c0, fr, c in rings:
            pg.draw.circle(s, dim(c, 0.9 * (1 - fr)), c0, int(4 + 22 * fr), 2)
        # ---- output layer
        best = int(np.argmax(out_p)) if out_p.any() else -1
        for k in range(10):
            on = k == best
            c = AMBER if on else dim(AMBER, 0.25 + 0.6 * out_p[k])
            rad = int((4 + 8 * out_p[k]) * kout[k])
            if on:
                pg.draw.circle(s, dim(AMBER, 0.3), pout[k], rad + 8)
            pg.draw.circle(s, c, pout[k], max(3, rad))
            self.ink.text(s, k, (pout[k][0] + 14, pout[k][1] - 9), c)
        self.ink.text(s, "pixels", (pin[:, 0].mean(), pin[:, 1].max() + 12), dim(CYAN, 0.8), anchor="midtop")
        self.ink.text(s, "300 hidden neurons", (phid[:, 0].mean(), phid[:, 1].max() + 12), dim(col, 0.9),
                      anchor="midtop")
        self.ink.text(s, "answer", (pout[:, 0].mean() + 10, pout[:, 1].max() + 14), dim(AMBER, 0.9), anchor="midtop")
        if tr["family"] == "snn":
            self._steps_bar(pg, s, tr, tl, col)
        self._panel(pg, s, tr, model, tl, info)

    def _steps_bar(self, pg, s, tr, tl, col):
        T, steps = tr["T"], tr["steps"]
        x0, y0, bw = 60, 548, min(34, 560 // T)
        n = tl.get("n", 0)
        for k in range(T):
            r = (x0 + k * bw, y0, bw - 4, 14)
            if k >= steps:
                pg.draw.rect(s, dim(GREY, 0.8), r, 1)               # never needed: skipped
            elif k < n or tl["done"]:
                pg.draw.rect(s, dim(col, 0.6), r)
            elif k == n:
                pg.draw.rect(s, col, r)
            else:
                pg.draw.rect(s, dim(col, 0.5), r, 1)
        lab = f"time step {min(n + 1, steps)} of {T}"
        if tr["exited"] and tl["done"]:
            lab = f"sure after {steps} of {T} steps: stopped early, {T - steps} steps skipped"
        self.ink.text(s, lab, (x0, y0 - 22), dim(WHITE, 0.85))

    def _panel(self, pg, s, tr, model, tl, info):
        x0 = 700
        col = model["colour"]
        y = self.ink.wrap(s, model["lesson"], (x0, 84), dim(WHITE, 0.9), 300)
        # the answer
        if tl["done"]:
            self.ink.text(s, tr["pred"], (x0 + 4, y + 2), AMBER, "xl")
            ok = tr["correct"]
            self.ink.text(s, "correct" if ok else f"wrong (it was {tr['label']})", (x0 + 90, y + 52),
                          GREEN if ok else RED, "m")
        else:
            self.ink.text(s, "thinking ...", (x0 + 4, y + 40), dim(WHITE, 0.6), "m")
        y += 120
        # the live work counter
        kind = "multiply-adds" if tr["mult"] else "spike additions"
        self.ink.text(s, f"{kind} for this digit", (x0, y), dim(WHITE, 0.8))
        self.ink.text(s, f"{tl['ops']:,}", (x0, y + 20), col, "l")
        frac = tl["ops"] / DENSE_OPS
        pg.draw.rect(s, dim(GREY, 0.9), (x0, y + 66, 300, 14), 1)
        pg.draw.rect(s, col, (x0 + 1, y + 67, max(1, int(298 * min(1, frac))), 12))
        self.ink.text(s, f"{frac * 100:.0f}% of the standard network's work", (x0, y + 84), dim(WHITE, 0.7), "xs")
        if tr["family"] == "snn":
            upd = (tl["n"] + 1 if tl["phase"] == "steps" else tr["steps"]) * (H + N_OUT)
            self.ink.text(s, f"+ {upd:,} neuron updates (leak, threshold)", (x0, y + 102), dim(WHITE, 0.7), "xs")
        # measured energy for this model
        y += 130
        e = info["energy"].get(tr["code"])
        if e is not None:
            tag = "SIMULATED" if info["sim"] else "measured on the Pico"
            self.ink.text(s, f"energy per digit ({tag})", (x0, y), dim(WHITE, 0.8))
            self.ink.text(s, f"{e:,.0f} uJ", (x0, y + 20), GREEN, "l")
        else:
            p = info["predicted"].get(tr["code"])
            self.ink.text(s, "energy: measuring ..." if p is None else "predicted from Pico cycles", (x0, y),
                          dim(WHITE, 0.7))
            if p is not None:
                self.ink.text(s, f"~{p:,.0f} uJ", (x0, y + 20), dim(GREEN, 0.8), "l")


# ================================================================ race view
class RaceView:
    name = "Race"

    def __init__(self, bundle, ink):
        self.b, self.ink = bundle, ink

    def draw(self, pg, s, now, tr, model, tl, info):
        models = info["models"]
        ink = self.ink
        ink.text(s, "Which network uses the least energy per digit?", (40, 70), WHITE, "m")
        # ---------------- energy bars
        x0, y0, bw = 40, 120, 380
        vals = {}
        for m in models:
            e = info["energy"].get(m["code"])
            vals[m["code"]] = (e, True) if e is not None else (info["predicted"].get(m["code"]), False)
        mx = max([v for v, _ in vals.values() if v] + [1e-9])
        for k, m in enumerate(models):
            y = y0 + k * 62
            v, meas = vals[m["code"]]
            sel = m["code"] == model["code"]
            c = m["colour"] if sel else dim(m["colour"], 0.6)
            ink.text(s, m["short"], (x0, y + 6), c, "sb")
            if v:
                L = int(bw * v / mx)
                if meas:
                    pg.draw.rect(s, c, (x0 + 80, y, L, 28))
                else:
                    pg.draw.rect(s, c, (x0 + 80, y, L, 28), 2)         # outline = predicted, not yet measured
                ink.text(s, f"{v:,.0f} uJ" + ("" if meas else " (pred.)"), (x0 + 88 + L, y + 6), c, "s")
            if sel:
                pg.draw.rect(s, c, (x0 - 8, y - 4, bw + 175, 36), 1)
            acc = info["accuracy"].get(m["code"])
            if acc is not None:
                ink.text(s, f"accuracy {acc * 100:.2f}%", (x0 + 80, y + 31), dim(WHITE, 0.6), "xs")
        tag = "SIMULATED numbers (no Pico connected)" if info["sim"] else "solid = measured live by the INA226; outline = predicted"
        ink.text(s, tag, (x0, y0 + len(models) * 62 + 4), dim(WHITE, 0.6), "xs")
        # ---------------- accuracy vs energy
        px0, py0, pw, ph = 620, 130, 360, 330
        pg.draw.line(s, dim(WHITE, 0.6), (px0, py0 + ph), (px0 + pw, py0 + ph), 2)
        pg.draw.line(s, dim(WHITE, 0.6), (px0, py0), (px0, py0 + ph), 2)
        ink.text(s, "energy per digit (log scale)  ->", (px0 + pw, py0 + ph + 10), dim(WHITE, 0.7), "xs",
                 anchor="topright")
        ink.text(s, "accuracy (up = better)", (px0 + 8, py0 - 26), dim(WHITE, 0.7), "xs")
        pts = [(m, vals[m["code"]][0], info["accuracy"].get(m["code"])) for m in models]
        pts = [p for p in pts if p[1] and p[2] is not None]
        if pts:
            es = np.log10([p[1] for p in pts])
            ac = np.array([p[2] for p in pts])
            elo, ehi = es.min() - 0.15, es.max() + 0.15
            alo, ahi = min(ac.min(), ac.max() - 0.01) - 0.002, ac.max() + 0.002
            ann_acc = info["accuracy"].get("A")
            if ann_acc is not None:                  # the "as accurate as the ANN" band (within 0.5 points)
                yb = py0 + ph - (ann_acc - 0.005 - alo) / (ahi - alo) * ph
                yt = py0 + ph - (ann_acc - alo) / (ahi - alo) * ph
                for yy in range(int(max(py0, yt)), int(min(py0 + ph, yb)), 6):
                    pg.draw.line(s, dim(GREEN, 0.18), (px0 + 2, yy), (px0 + pw, yy), 1)
                ink.text(s, "within 0.5 points of the ANN", (px0 + 6, max(py0, yt) + 2), dim(GREEN, 0.7), "xs")
            for (m, e, a), le in zip(pts, es):
                X = px0 + (le - elo) / (ehi - elo) * pw
                Y = py0 + ph - (a - alo) / (ahi - alo) * ph
                sel = m["code"] == model["code"]
                c = m["colour"]
                if sel:
                    pg.draw.circle(s, dim(c, 0.4), (X, Y), 18, 2)
                pg.draw.circle(s, c, (X, Y), 8 if sel else 6)
                ink.text(s, m["short"], (X + 12, Y - 20), c if sel else dim(c, 0.7), "sb" if sel else "s")
            ink.text(s, "<- better: less energy", (px0 + 10, py0 + ph - 22), dim(GREEN, 0.8), "xs")
        # ---------------- the headline ratio
        a = vals.get("A", (None,))[0]
        e = vals.get(model["code"], (None,))[0]
        if a and e and model["code"] != "A":
            ink.text(s, f"{model['short']} uses {a / e:.1f}x less energy than the standard ANN",
                     (40, 540), model["colour"], "m")
        z = vals.get("Z", (None,))[0]
        if z and e and model["family"] == "snn":
            r = z / e
            ink.text(s, (f"and {r:.2f}x less than the zero-skipping ANN" if r >= 1
                         else f"but {1 / r:.2f}x MORE than the zero-skipping ANN"),
                     (40, 570), GREEN if r >= 1 else RED, "s")


# ================================================================ equation view
class EquationView:
    name = "Equation"

    def __init__(self, bundle, ink):
        self.b, self.ink = bundle, ink

    def draw(self, pg, s, now, tr, model, tl, info):
        ink, col = self.ink, model["colour"]
        ink.text(s, f"The maths inside: {model['name']}", (40, 70), col, "m")
        y = 108
        for eq in model["eqs"]:
            img = ink.tex(eq, 24, WHITE)
            s.blit(img, (60, y))
            y += img.get_height() + 10
        y += 6
        pg.draw.line(s, dim(col, 0.5), (40, y), (984, y), 1)
        y += 14
        if tr is None:
            return
        ink.text(s, "with this digit's numbers:", (40, y), dim(WHITE, 0.7))
        y += 28
        for line in self._worked(tr, tl):
            ink.text(s, line.strip(), (60 if not line.startswith("  ") else 80, y),
                     col if not line.startswith("  ") else dim(WHITE, 0.85), "sb")
            y += 26
        # the energy inequality, live, anchored to the bottom of the screen
        lines = self._inequality(tr, info)
        y = HGT - 52 - 30 * len(lines)
        pg.draw.line(s, dim(GREEN, 0.5), (40, y), (984, y), 1)
        ink.text(s, "why it wins or loses on a Pico (estimated CPU cycles, from the compiled loops):", (40, y + 8),
                 dim(WHITE, 0.75), "xs")
        for k, line in enumerate(lines):
            ink.text(s, line, (60, y + 30 + 30 * k), GREEN if k == len(lines) - 1 else WHITE, "m")

    def _worked(self, tr, tl):
        if tr["family"] == "ann":
            j = int(np.argmax(tr["h"]))
            nz = int((tr["x"] > 0).sum())
            n_terms = 784 if tr["code"] == "A" else nz
            return [f"neuron {j}:  sum of {n_terms} products W x  =  {int(tr['acc'][j]):,}",
                    f"  ReLU keeps positives, rescale to 8 bit:  h = {int(tr['h'][j])}",
                    f"  {int((tr['h'] > 0).sum())} of 300 hidden neurons are active (non-zero)",
                    f"  answer = largest output:  {tr['pred']}"]
        n = min(tl.get("n", 0), tr["steps"] - 1)
        q = 0
        j = tr["trace_idx"][q]
        u = int(tr["u_all"][n][j])
        th = int(tr["theta"])
        fired = u >= th
        prev = int(tr["u_all"][n - 1][j]) if n > 0 else 0
        return [f"neuron {j}, step {n + 1}:  U = {u:,}   (it was {prev:,})",
                f"  threshold theta = {th}:  {'U >= theta, so it SPIKES and resets by theta' if fired else 'U < theta, so it stays silent'}",
                f"  spikes so far: {int(sum(len(f['fired']) for f in tr['frames'][:n + 1]))} in the whole hidden layer",
                f"  evidence leader: digit {int(np.argmax(tr['frames'][n]['c']))}"
                + (f"  (stops early after step {tr['steps']})" if tr["exited"] else "")]

    def _inequality(self, tr, info):
        cyc = info["cycles_fn"]
        nz = int((tr["x"] > 0).sum())
        nzh = int((self.b.ann(tr["x"])[1] > 0).sum())       # the ANN's active hidden neurons, same digit
        z = cyc("ann_skip", nz_in=nz, nz_hid=nzh)["total"]
        if tr["family"] == "ann":
            mine = cyc("ann_dense" if tr["code"] == "A" else "ann_skip", nz_in=nz, nz_hid=int((tr["h"] > 0).sum()))
            return [f"cycles = rows x (300 x 13 per multiply-add) + layer 2",
                    f"this digit: {mine['total'] / 1e3:,.0f}k cycles  = {mine['total'] / 125e3:.1f} ms at 125 MHz"]
        kind = {"L": "snn_latency", "P": "snn_latency", "E": "snn_fast"}[tr["code"]]
        ev = sum(len(f["in"]) for f in tr["frames"])
        mine = cyc(kind, events=ev, hid_spikes=tr["spikes"], steps=tr["steps"], T=tr["T"])["total"]
        upd = 40 if tr["code"] in "LP" else 17
        return [f"spikes x (300 x 12)  +  steps x 300 x {upd}   vs   pixels x (300 x 13)",
                f"{ev} x 3,600  +  {tr['steps']} x {300 * upd:,}   vs   {nz} x 3,900",
                f"{mine / 1e3:,.0f}k cycles  vs  zero-skip ANN {z / 1e3:,.0f}k:  "
                + (f"{z / mine:.2f}x less work" if mine < z else f"{mine / z:.2f}x MORE work")]


# ================================================================ neuron view
class NeuronView:
    name = "Neurons"

    def __init__(self, bundle, ink):
        self.b, self.ink = bundle, ink

    def draw(self, pg, s, now, tr, model, tl, info):
        ink, col = self.ink, model["colour"]
        if tr is None:
            return
        if tr["family"] == "ann":
            ink.text(s, "Inside the ANN: every hidden neuron is a number (ReLU of a weighted sum)", (40, 70), col, "m")
            h = tr["h"]
            order = np.argsort(-h)
            x0, y0, w, hh = 60, 130, 900, 300
            mx = max(1, h.max())
            f = 1.0 if tl["phase"] in ("output", "done") else (tl.get("f", 0) if tl["phase"] == "hidden" else 0)
            for k, j in enumerate(order):
                bh = hh * h[j] / mx * f
                pg.draw.line(s, col if h[j] > 0 else dim(GREY, 0.8), (x0 + k * 3, y0 + hh),
                             (x0 + k * 3, y0 + hh - max(1, bh)), 2)
            ink.text(s, f"{int((h > 0).sum())} of 300 neurons active (sorted)", (x0, y0 + hh + 12), dim(WHITE, 0.8))
            ink.text(s, "a zero still costs a multiply in the standard ANN; the zero-skip ANN and the",
                     (x0, y0 + hh + 50), dim(WHITE, 0.7))
            ink.text(s, "spiking networks only do work where something is happening", (x0, y0 + hh + 74), dim(WHITE, 0.7))
            return
        ink.text(s, "Inside the spiking network: membrane voltage, threshold, spike", (40, 70), col, "m")
        n = min(tl.get("n", 0), tr["steps"] - 1) if tl["phase"] != "done" else tr["steps"] - 1
        T = tr["T"]
        # membrane traces of three neurons
        x0, y0, w, hh = 60, 120, 560, 250
        th = max(1, tr["theta"])
        u = tr["u"]
        lo, hi = min(-0.5, (u / th).min()) - 0.2, max(1.5, (u / th).max()) + 0.2
        ymap = lambda v: y0 + hh - (v / th - lo) / (hi - lo) * hh
        for xx in range(x0, x0 + w, 14):
            pg.draw.line(s, dim(RED, 0.8), (xx, ymap(th)), (xx + 7, ymap(th)), 2)
        ink.text(s, "threshold", (x0 + w + 6, ymap(th) - 9), dim(RED, 0.9))
        for q, c in zip(range(3), (CYAN, col, GREEN)):
            pts = [(x0 + k / max(1, T - 1) * w, ymap(u[k][q])) for k in range(n + 1)]
            if len(pts) > 1:
                pg.draw.lines(s, c, False, pts, 3)
            for k in range(n + 1):
                if tr["raster"][k][tr["trace_idx"][q]]:
                    xx = x0 + k / max(1, T - 1) * w
                    pg.draw.line(s, c, (xx, y0 - 8), (xx, y0 + 8), 3)
        ink.text(s, "voltage U of three neurons over time; a tick on top = a spike", (x0, y0 + hh + 10), dim(WHITE, 0.75))
        # spike raster: the 60 busiest neurons x steps
        rx0, ry0, rw, rh = 680, 120, 300, 300
        busy = np.argsort(-tr["raster"].sum(0))[:60]
        cw, ch = rw / max(1, T), rh / 60
        for k in range(min(n + 1, tr["steps"])):
            for r, j in enumerate(busy):
                if tr["raster"][k][j]:
                    pg.draw.rect(s, col, (rx0 + k * cw + 1, ry0 + r * ch, max(2, cw - 2), max(2, ch - 1)))
        pg.draw.rect(s, dim(GREY, 0.9), (rx0, ry0, rw, rh), 1)
        if tr["steps"] < T:
            pg.draw.line(s, GREEN, (rx0 + tr["steps"] * cw, ry0 - 6), (rx0 + tr["steps"] * cw, ry0 + rh + 6), 2)
            ink.text(s, "stopped", (rx0 + tr["steps"] * cw + 4, ry0 + rh + 8), GREEN, "xs")
        ink.text(s, "spike raster: 60 busiest neurons x time", (rx0, ry0 + rh + 26), dim(WHITE, 0.75))
        sp = tr["spikes"]
        ink.text(s, f"{sp} hidden spikes in total = {sp / 300 / tr['steps']:.3f} spikes per neuron per step",
                 (60, 520), col, "m")


VIEWS = [NetworkView, RaceView, EquationView, NeuronView]
