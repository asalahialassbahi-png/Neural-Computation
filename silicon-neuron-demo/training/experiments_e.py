"""
Robustness experiments for the SNN-E recipe, judged on VALIDATION images only
(the test set is not touched). Each variant is trained with 3 seeds; reported:
validation accuracy without exit, with the deployed exit rule, and mean steps.

    python experiments_e.py
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from energy_model import pico_cycles
from intsim import quantise_snn_fast, snn_fast_int, choose_exit_margin
from mnist_idx import load_mnist

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weights", "exp_e")
BASE = ["--model", "snn", "--enc", "latency", "--first-order", "--readout", "perstep", "--T", "8",
        "--tin", "3", "--xmin", "96", "--val-only"]
VARIANTS = {"lam1_ep10": ["--lam", "1.0", "--epochs", "10"],
            "lam0.3_ep10": ["--lam", "0.3", "--epochs", "10"],
            "lam0.3_ep16": ["--lam", "0.3", "--epochs", "16"],
            "lam1_ep16_lr1e-3": ["--lam", "1.0", "--epochs", "16", "--lr", "1e-3"]}
SEEDS = [11, 12, 13]


def run(job):
    name, seed = job
    path = os.path.join(OUT, f"{name}_s{seed}.npz")
    if not os.path.exists(path):
        subprocess.run([sys.executable, os.path.join(HERE, "train.py"), *BASE, *VARIANTS[name], "--seed", str(seed),
                        "--out", path], capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(n, s) for n in VARIANTS for s in SEEDS]
    with ThreadPoolExecutor(2) as ex:
        paths = list(ex.map(run, jobs))
    xtr, ytr, _, _ = load_mnist()
    xva, yva = xtr[-5000:], ytr[-5000:]
    res = {}
    for (n, s), p in zip(jobs, paths):
        q = quantise_snn_fast(dict(np.load(p)))
        m, acc, base, _ = choose_exit_margin(q, xva, yva, pico_cycles)
        st = snn_fast_int(q, xva, exit_margin=m if m >= 0 else None)[2]["steps"].mean()
        res.setdefault(n, []).append((base, acc, st, m / q["theta"]))
    for n, r in res.items():
        r = np.array(r)
        print(f"{n:18s} val no-exit {r[:, 0].mean() * 100:.2f} +/- {r[:, 0].std(ddof=1) * 100:.2f}   "
              f"with exit {r[:, 1].mean() * 100:.2f} +/- {r[:, 1].std(ddof=1) * 100:.2f}   "
              f"steps {r[:, 2].mean():.2f}   margins {r[:, 3].tolist()}")


if __name__ == "__main__":
    main()
