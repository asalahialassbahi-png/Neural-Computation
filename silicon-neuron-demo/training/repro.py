"""
Repeatability: retrain the networks from scratch with several random seeds and
report every number as mean +/- standard deviation, so the result is a property
of the METHOD, not of one lucky run. (The same spirit as Eshraghian et al.,
"Training Spiking Neural Networks Using Lessons From Deep Learning", Proc.
IEEE 2023: fixed protocols, several seeds, everything recorded.)

    python repro.py                      # seeds 0-4: ANN and SNN-E (~15 min on 4 cores)
    python repro.py --seeds 0 1 2 3 4 --with-latency   # + the 16-step SNN-L (~+35 min)

For every seed, with exactly the deployed recipe:
    ANN    784-300-10 ReLU, Adam, 10 epochs                        (train.py --model ann)
    SNN-E  first-order LIF, T=8, 3 input steps, pixels >= 96,      (sweep winner)
           per-step readout, lambda = 1, 10 epochs
then quantise to int8, choose SNN-E's exit margin on the validation images with
the same rule as export.py, and evaluate ONCE on the 10,000 test images with the
integer (Pico-identical) models. Operation counts give the Pico cycle estimate.

Writes results/repro.json and results/repro.md, including a manifest: Python
and NumPy versions, the SHA-256 of the MNIST files and of every training
source file, so anyone can check they are re-running the same experiment.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys

import numpy as np

import config as C
from energy_model import pico_cycles
from intsim import (quantise_ann, ann_int, quantise_snn, snn_int, input_spikes_int,
                    quantise_snn_fast, snn_fast_int, choose_exit_margin)
from mnist_idx import load_mnist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "weights", "repro")
RES = os.path.join(ROOT, "results")

RECIPES = {
    "ann": ["--model", "ann", "--epochs", "10"],
    "snn_fast": ["--model", "snn", "--enc", "latency", "--first-order", "--readout", "perstep", "--T", "8",
                 "--tin", "3", "--xmin", "96", "--lam", "1.0", "--epochs", "10", "--val-only"],
    "snn_latency": ["--model", "snn", "--enc", "latency", "--epochs", "20", "--val-only"],
}
T_975 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest():
    data = os.path.join(HERE, "data")
    src = sorted(f for f in os.listdir(HERE) if f.endswith(".py"))
    return {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
            "mnist_sha256": {f: sha(os.path.join(data, f)) for f in sorted(os.listdir(data)) if f.endswith(".gz")},
            "source_sha256": {f: sha(os.path.join(HERE, f)) for f in src},
            "recipes": RECIPES}


def train(kind, seed):
    path = os.path.join(OUT, f"{kind}_seed{seed}.npz")
    if not os.path.exists(path):
        cmd = [sys.executable, os.path.join(HERE, "train.py"), *RECIPES[kind], "--seed", str(seed), "--out", path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        open(path.replace(".npz", ".log"), "w").write(r.stdout + r.stderr)
        if r.returncode:
            raise RuntimeError(f"{kind} seed {seed} failed:\n{r.stderr[-800:]}")
        print(f"trained {kind} seed {seed}: {r.stdout.strip().splitlines()[-2]}", flush=True)
    return dict(np.load(path))


def mean_ci(v):
    v = np.asarray(v, float)
    n = len(v)
    m, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
    half = T_975.get(n, 1.96) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"mean": m, "sd": sd, "ci95": [m - half, m + half], "values": v.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--with-latency", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    xtr, ytr, xte, yte = load_mnist()
    xva, yva = xtr[-5000:], ytr[-5000:]
    nz = (xte > 0).sum(1)

    rows = {"ann": [], "ann_skip_cycles": [], "snn_fast": [], "snn_fast_cycles": [], "snn_fast_steps": [],
            "snn_fast_margin": [], "gap_points": [], "cycle_ratio_vs_ann_skip": [], "snn_latency": [],
            "snn_latency_cycles": []}
    for seed in a.seeds:
        pa = train("ann", seed)
        qa = quantise_ann(pa, xtr[:5000])
        pred_a, h8, _ = ann_int(qa, xte)
        acc_a = float((pred_a == yte).mean())
        cyc_z = pico_cycles("ann_skip", nz_in=nz.mean(), nz_hid=(h8 > 0).sum(1).mean())["total"]

        pf = train("snn_fast", seed)
        qf = quantise_snn_fast(pf)
        m, _, _, _ = choose_exit_margin(qf, xva, yva, pico_cycles)
        pred_e, _, cnt = snn_fast_int(qf, xte, exit_margin=m if m >= 0 else None)
        acc_e = float((pred_e == yte).mean())
        cyc_e = pico_cycles("snn_fast", events=cnt["events"].mean(), hid_spikes=cnt["hid_spikes"].mean(),
                            steps=cnt["steps"].mean())["total"]
        rows["ann"].append(acc_a); rows["ann_skip_cycles"].append(cyc_z)
        rows["snn_fast"].append(acc_e); rows["snn_fast_cycles"].append(cyc_e)
        rows["snn_fast_steps"].append(float(cnt["steps"].mean())); rows["snn_fast_margin"].append(m / qf["theta"])
        rows["gap_points"].append((acc_a - acc_e) * 100)
        rows["cycle_ratio_vs_ann_skip"].append(cyc_e / cyc_z)
        line = (f"seed {seed}: ANN {acc_a * 100:.2f}%  SNN-E {acc_e * 100:.2f}% (gap {(acc_a - acc_e) * 100:+.2f} pts)"
                f"  SNN-E cycles {cyc_e / 1e3:.0f}k = {cyc_e / cyc_z:.2f}x ANN-Z, {cnt['steps'].mean():.2f} steps")
        if a.with_latency:
            pl = train("snn_latency", seed)
            ql = quantise_snn(pl)
            s_in = input_spikes_int(xte, "latency", np.arange(len(xte)))
            pred_l, _, S, _ = snn_int(ql, s_in)
            acc_l = float((pred_l == yte).mean())
            cyc_l = pico_cycles("snn_latency", events=s_in.sum((1, 2)).mean(), hid_spikes=S.sum((1, 2)).mean())["total"]
            rows["snn_latency"].append(acc_l); rows["snn_latency_cycles"].append(cyc_l)
            line += f"  SNN-L {acc_l * 100:.2f}% {cyc_l / cyc_z:.2f}x ANN-Z"
        print(line, flush=True)

    summary = {k: mean_ci(v) for k, v in rows.items() if v}
    out = {"seeds": a.seeds, "summary": summary, "manifest": manifest()}
    with open(os.path.join(RES, "repro.json"), "w") as f:
        json.dump(out, f, indent=1)

    def fmt(k, scale=100, unit="%", dp=2):
        s = summary[k]
        return (f"{s['mean'] * scale:.{dp}f}{unit} +/- {s['sd'] * scale:.{dp}f} "
                f"(95% CI {s['ci95'][0] * scale:.{dp}f} to {s['ci95'][1] * scale:.{dp}f})")
    md = [f"# Repeatability: {len(a.seeds)} independent training runs (seeds {', '.join(map(str, a.seeds))})", "",
          "Test accuracy is the int8 model the Pico runs, on all 10,000 MNIST test images, evaluated once per seed.",
          "", "| quantity | mean +/- sd (95% CI) |", "|---|---|",
          f"| ANN accuracy | {fmt('ann')} |",
          f"| SNN-E accuracy | {fmt('snn_fast')} |",
          f"| accuracy gap ANN - SNN-E (points) | {fmt('gap_points', 1, ' pts')} |",
          f"| SNN-E Pico cycles / ANN-Z Pico cycles | {fmt('cycle_ratio_vs_ann_skip', 1, 'x', 3)} |",
          f"| SNN-E mean time steps used (of 8) | {fmt('snn_fast_steps', 1, '', 2)} |"]
    if rows["snn_latency"]:
        md.append(f"| SNN-L accuracy | {fmt('snn_latency')} |")
    md += ["", "Cycle counts here use the hand model (energy_model.py, within ~4-9% of the emulator, always low);",
           "the ratio between models is what is compared. Manifest (versions, data and code hashes): repro.json."]
    open(os.path.join(RES, "repro.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
