"""
Design-space sweep for the Pico-optimised SNN (SNN-E).

Trains a small grid of first-order, per-step-readout latency SNNs, then scores
each one on the VALIDATION images (the last 5,000 training images, which
train.py never trains on; the test set is not touched here) with the integer
model, early exit and the Cortex-M0+ cycle model in energy_model.py.

    python sweep.py            # train the grid (skips configs already trained), then score
    python sweep.py --score    # score only

Writes weights/sweep/*.npz and results/sweep.json. The chosen design is the
cheapest one whose validation accuracy is within --tol of the ANN's.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import config as C
from energy_model import pico_cycles
from intsim import quantise_snn_fast, snn_fast_int, quantise_ann, ann_int
from mnist_idx import load_mnist

HERE = os.path.dirname(os.path.abspath(__file__))
SW = os.path.join(HERE, "weights", "sweep")
RES = os.path.join(os.path.dirname(HERE), "results")

GRID = {"T": [6, 8], "tin": [3, 4], "xmin": [96, 128, 160], "lam": [1.0]}
EPOCHS = 10
# exit margins in units of the threshold theta: "stop once the leading digit's mean
# evidence is k thresholds ahead of the runner-up"
MARGIN_K = [None, 0.5, 1, 2, 4, 8, 16, 32, 64]


def name(cfg):
    return f"T{cfg['T']}_tin{cfg['tin']}_x{cfg['xmin']}_lam{cfg['lam']:g}"


def train_one(cfg):
    out = os.path.join(SW, name(cfg) + ".npz")
    if os.path.exists(out):
        return out
    cmd = [sys.executable, os.path.join(HERE, "train.py"), "--model", "snn", "--enc", "latency",
           "--first-order", "--readout", "perstep", "--T", str(cfg["T"]), "--tin", str(cfg["tin"]),
           "--xmin", str(cfg["xmin"]), "--lam", str(cfg["lam"]), "--epochs", str(EPOCHS),
           "--val-only", "--out", out]
    log = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    with open(out.replace(".npz", ".log"), "w") as f:
        f.write(log.stdout + log.stderr)
    print("trained", name(cfg), log.stdout.strip().split("\n")[-2] if log.stdout else log.stderr[-300:],
          flush=True)
    return out


def score(path, xva, yva):
    q = quantise_snn_fast(dict(np.load(path)))
    rows = []
    for k in MARGIN_K:
        m = None if k is None else int(round(k * q["theta"]))
        pred, _, cnt = snn_fast_int(q, xva, exit_margin=m)
        cyc = pico_cycles("snn_fast", events=cnt["events"].mean(), hid_spikes=cnt["hid_spikes"].mean(),
                          steps=cnt["steps"].mean())
        rows.append({"margin": m, "margin_k": k, "acc": float((pred == yva).mean()), "steps": float(cnt["steps"].mean()),
                     "events": float(cnt["events"].mean()), "hid_spikes": float(cnt["hid_spikes"].mean()),
                     "cycles": cyc["total"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.004, help="max accuracy below the ANN (validation)")
    a = ap.parse_args()
    os.makedirs(SW, exist_ok=True)
    cfgs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    if not a.score:
        with ThreadPoolExecutor(a.jobs) as ex:
            list(ex.map(train_one, cfgs))

    xtr, ytr, _, _ = load_mnist()
    xva, yva = xtr[-5000:], ytr[-5000:]
    # the ANN reference on the same validation images, and its cycle count
    qa = quantise_ann(dict(np.load(os.path.join(HERE, "weights", "ann_float.npz"))), xtr[:5000])
    pa, h8, _ = ann_int(qa, xva)
    ann_acc = float((pa == yva).mean())
    ann_z = pico_cycles("ann_skip", nz_in=float((xva > 0).sum(1).mean()), nz_hid=float((h8 > 0).sum(1).mean()))
    print(f"ANN (int) validation accuracy {ann_acc * 100:.2f}%, zero-skip {ann_z['total'] / 1e3:.0f}k cycles")

    out = {"ann_val_acc": ann_acc, "ann_skip_cycles": ann_z["total"], "configs": {}}
    best = None
    for cfg in cfgs:
        path = os.path.join(SW, name(cfg) + ".npz")
        if not os.path.exists(path):
            continue
        rows = score(path, xva, yva)
        out["configs"][name(cfg)] = rows
        for r in rows:
            ok = r["acc"] >= ann_acc - a.tol
            print(f"{name(cfg):24s} margin {str(r['margin_k']):>4s} theta  acc {r['acc'] * 100:6.2f}%  steps {r['steps']:4.2f}"
                  f"  events {r['events']:6.1f}  cycles {r['cycles'] / 1e3:6.0f}k "
                  f"({r['cycles'] / ann_z['total']:.2f}x ANN-Z){'  ok' if ok else ''}")
            if ok and (best is None or r["cycles"] < best[2]["cycles"]):
                best = (name(cfg), cfg, r)
    out["chosen"] = {"name": best[0], "cfg": best[1], "score": best[2]} if best else None
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("chosen:", out["chosen"])


if __name__ == "__main__":
    main()
