"""
The five networks the demo switches between, and one common "trace" format
that every view can animate.

A trace is everything one inference did, step by step, computed on the Pi
with nnsim.py (bit-exact with the Pico firmware), so the animation shows
exactly what the Pico computes while the INA226 measures it.
"""
import numpy as np

CYAN = (0, 225, 255)
MAGENTA = (255, 60, 210)
AMBER = (255, 185, 40)
GREEN = (60, 255, 140)
VIOLET = (170, 120, 255)
ORANGE = (255, 130, 40)

H, N_IN, N_OUT = 300, 784, 10

# Each lesson is one short sentence the visitor can read in 5 seconds.
MODELS = [
    {"code": "A", "family": "ann", "name": "Standard network (ANN)", "short": "ANN", "colour": CYAN,
     "lesson": "Multiplies every pixel by every weight, even the blank ones.",
     "eqs": [r"$h_j=\mathrm{ReLU}\left(\sum_{i=1}^{784} W_{ij}\,x_i+b_j\right)$",
             r"$\hat{y}=\arg\max_k\ \sum_j V_{jk}\,h_j+c_k$"]},
    {"code": "Z", "family": "ann", "name": "ANN that skips blank pixels", "short": "ANN-Z", "colour": GREEN,
     "lesson": "Same answers, but skips the ~80% of pixels that are zero.",
     "eqs": [r"$h_j=\mathrm{ReLU}\left(\sum_{i:\,x_i\neq 0} W_{ij}\,x_i+b_j\right)$",
             r"$\mathrm{work}=N_{x\neq0}\times 300\ \mathrm{multiply\!-\!adds}$"]},
    {"code": "L", "family": "snn", "name": "Spiking network, latency code", "short": "SNN-L", "colour": MAGENTA,
     "lesson": "Bright pixels spike first, once. Adds instead of multiplies, but 16 steps of upkeep.",
     "eqs": [r"$I[n]=\alpha I[n-1]+\sum_i W_{ij}\,s_i[n]$",
             r"$U[n]=\beta U[n-1]+I[n]-\theta S[n-1],\quad S[n]=H(U[n]-\theta)$"]},
    {"code": "P", "family": "snn", "name": "Spiking network, rate code", "short": "SNN-P", "colour": ORANGE,
     "lesson": "Brightness as spike frequency: far more spikes, far more work.",
     "eqs": [r"$P(s_i[n]=1)=x_i/256$",
             r"$\mathrm{work}\propto\sum_n\sum_i s_i[n]\ \approx\ T\,\bar{x}\,N$"]},
    {"code": "E", "family": "snn", "name": "Spiking network, Pico-optimised", "short": "SNN-E", "colour": VIOLET,
     "lesson": "One-variable neurons, a few steps, and it stops the moment it is sure.",
     "eqs": [r"$U[n]=\beta U[n-1]+\sum_i W_{ij}\,s_i[n]-\theta S[n-1]$",
             r"$\mathrm{stop\ when}\ c_{1}-c_{2}\geq m\,(n+1)$"]},
]
BY_CODE = {m["code"]: m for m in MODELS}


def available(bundle):
    out = [m for m in MODELS if m["code"] in "AZL"]
    if bundle.has_poisson:
        out.append(BY_CODE["P"])
    if bundle.has_fast:
        out.append(BY_CODE["E"])
    return out


def make_trace(bundle, code, idx):
    """Run model `code` on test image idx and return a uniform trace dict:
        family, code, x, label, pred, correct
        ann : h (300), logits (10), acc (300 pre-activation), rows (input rows processed)
        snn : steps, T, frames[n] = {in: pixel indices spiking, fired: hidden indices,
              c: evidence (10)}, u (steps,3), trace_idx, theta, exited
        ops : cumulative synaptic operations after each frame (for the live counter),
              mult: whether those operations are multiplies (ANN) or adds (SNN)
    """
    x = bundle.images[idx]
    label = int(bundle.labels[idx])
    tr = {"code": code, "x": x, "label": label, "idx": idx}
    if code in "AZ":
        pred, h, logits, acc = bundle.ann(x, full=True)
        rows = np.arange(N_IN) if code == "A" else np.nonzero(x)[0]
        hid_rows = np.arange(H) if code == "A" else np.nonzero(h)[0]
        tr.update(family="ann", pred=pred, h=h, logits=logits, acc=acc, rows=rows, hid_rows=hid_rows,
                  ops_total=len(rows) * H + len(hid_rows) * N_OUT, mult=True)
    else:
        if code == "E":
            r = bundle.snn_fast(x)
        else:
            r = bundle.snn(x, "latency" if code == "L" else "poisson", idx)
        steps = r["steps"]
        frames, ops, cum = [], [], 0
        for n in range(steps):
            if code == "P":
                inp = np.nonzero(r["s_in"][n])[0]
            else:
                inp = np.nonzero(r["in_time"] == n)[0]
            fired = np.nonzero(r["raster"][n])[0]
            cum += len(inp) * H + len(fired) * N_OUT
            frames.append({"in": inp, "fired": fired, "c": r["c_hist"][n]})
            ops.append(cum)
        tr.update(family="snn", pred=r["pred"], steps=steps, T=r["T"], frames=frames, ops=ops,
                  u=r["u"], u_all=r["u_all"], trace_idx=r["trace_idx"], theta=r["theta"],
                  raster=r["raster"], exited=steps < r["T"], ops_total=cum, mult=False,
                  neuron_updates=steps * (H + N_OUT), in_spikes=r["in_spikes"], spikes=r["spikes"])
    tr["correct"] = tr["pred"] == label
    return tr
