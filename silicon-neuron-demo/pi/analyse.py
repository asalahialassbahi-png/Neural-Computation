"""
Statistics for the energy benchmark (formula sheet, "Hardware measurement").

    python analyse.py results/bench.csv            -> table + results/summary.json
    python analyse.py results/bench.csv --plots    -> also PNG figures

Per trial:   E_total = P_mean * dt / N          E_dyn = (P_mean - P_idle,round) * dt / N
where P_idle,round is the idle trial from the same round (paired, so slow drift cancels).

Reported per mode: mean +/- sd, standard error, accuracy, T*r (SNN).
Comparisons against the dense ANN:
    Welch t = (E1 - E2) / sqrt(s1^2/n1 + s2^2/n2),  Welch-Satterthwaite df
    Cohen's d = (E1 - E2) / s_pooled
    95% bootstrap CI of the ratio E_ANN / E_mode (10^4 resamples, percentiles 2.5 / 97.5)
Uncertainty of a single mean energy (quadrature):
    (sigma_E/E)^2 = (SE_P/P)^2 + (u_shunt)^2 + (u_gain)^2
The shunt tolerance (1%) and INA gain error (0.1%) are systematic: identical for
every mode, so they cancel exactly in the ratio E_ANN / E_SNN.
"""
import argparse
import csv
import json
import math
import os

import numpy as np

U_SHUNT = 0.01      # 1% resistor tolerance (use 0 if you measured R with a 4-wire meter)
U_GAIN = 0.001      # INA226 datasheet gain error 0.1%
NAMES = {"A": "ANN dense", "Z": "ANN zero-skip", "L": "SNN latency", "P": "SNN Poisson"}
N_HID = 300          # hidden neurons (training/config.py)


def welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df)
    except ImportError:
        p = float("nan")
    return t, df, p


def cohens_d(a, b):
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def bootstrap_ratio(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    ra = rng.choice(a, (n, len(a))).mean(1)
    rb = rng.choice(b, (n, len(b))).mean(1)
    r = ra / rb
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in r:
            if k != "mode":
                r[k] = float(r[k])
    return rows


def analyse(rows):
    idle = {int(r["round"]): r["P_mean"] for r in rows if r["mode"] == "I"}
    per = {}
    for r in rows:
        m = r["mode"]
        if m == "I":
            continue
        dt = r["us"] * 1e-6
        p_idle = idle.get(int(r["round"]), np.mean(list(idle.values())))
        per.setdefault(m, {"E_tot": [], "E_dyn": [], "acc": [], "Tr": [], "us_inf": [], "P": []})
        d = per[m]
        d["E_tot"].append(r["P_mean"] * dt / r["count"])
        d["E_dyn"].append((r["P_mean"] - p_idle) * dt / r["count"])
        d["acc"].append(r["correct"] / r["count"])
        d["Tr"].append(r["spikes"] / r["count"] / N_HID if m in "LP" else 0.0)   # T * mean rate
        d["us_inf"].append(r["us"] / r["count"])
        d["P"].append(r["P_mean"])
    out = {"P_idle_mW": float(np.mean(list(idle.values())) * 1e3), "modes": {}}
    for m, d in per.items():
        a = {k: np.array(v) for k, v in d.items()}
        n = len(a["E_tot"])
        se_rel = a["P"].std(ddof=1) / math.sqrt(n) / a["P"].mean()
        u_rel = math.sqrt(se_rel ** 2 + U_SHUNT ** 2 + U_GAIN ** 2)
        out["modes"][m] = {
            "name": NAMES[m], "n": n,
            "E_total_uJ": float(a["E_tot"].mean() * 1e6), "E_total_sd_uJ": float(a["E_tot"].std(ddof=1) * 1e6),
            "E_dyn_uJ": float(a["E_dyn"].mean() * 1e6), "E_dyn_sd_uJ": float(a["E_dyn"].std(ddof=1) * 1e6),
            "E_total_u_rel": u_rel,
            "us_per_inf": float(a["us_inf"].mean()), "P_mW": float(a["P"].mean() * 1e3),
            "accuracy": float(a["acc"].mean()), "T_r": float(a["Tr"].mean()),
        }
        per[m] = a
    if "A" in per:
        base = per["A"]
        for m in per:
            if m == "A":
                continue
            for kind in ("E_tot", "E_dyn"):
                t, df, p = welch(base[kind], per[m][kind])
                lo, hi = bootstrap_ratio(base[kind], per[m][kind])
                out["modes"][m][f"vs_ANN_{kind}"] = {
                    "ratio": float(base[kind].mean() / per[m][kind].mean()), "ci95": [lo, hi],
                    "welch_t": t, "df": df, "p": p, "cohens_d": cohens_d(base[kind], per[m][kind])}
    return out


def plots(summary, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = list(summary["modes"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, (key, lab) in enumerate((("E_total_uJ", "total energy / inference (uJ)"),
                                    ("E_dyn_uJ", "energy above idle / inference (uJ)"))):
        vals = [summary["modes"][m][key] for m in ms]
        sds = [summary["modes"][m][key.replace("_uJ", "_sd_uJ")] for m in ms]
        ax[k].bar([NAMES[m] for m in ms], vals, yerr=sds, capsize=4, color="#3b7dd8")
        ax[k].set_ylabel(lab)
        ax[k].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "energy_bars.png"), dpi=200)
    print("saved", os.path.join(folder, "energy_bars.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    s = analyse(load(a.csv))
    folder = os.path.dirname(os.path.abspath(a.csv))
    print(f"idle power {s['P_idle_mW']:.2f} mW")
    print(f"{'mode':15s} {'n':>3s} {'E_total uJ':>12s} {'E_dyn uJ':>12s} {'us/inf':>9s} {'acc %':>6s} {'T*r':>6s}")
    for m, d in s["modes"].items():
        print(f"{d['name']:15s} {d['n']:3d} {d['E_total_uJ']:8.1f}±{d['E_total_sd_uJ']:<5.1f}"
              f"{d['E_dyn_uJ']:8.1f}±{d['E_dyn_sd_uJ']:<5.1f}{d['us_per_inf']:8.0f} "
              f"{d['accuracy'] * 100:6.1f} {d['T_r']:6.3f}")
        for kind in ("E_tot", "E_dyn"):
            c = d.get(f"vs_ANN_{kind}")
            if c:
                print(f"    ANN/{m} {kind}: {c['ratio']:.2f}x  95% CI [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]"
                      f"  Welch t={c['welch_t']:.1f} (df {c['df']:.0f}, p={c['p']:.2g})  d={c['cohens_d']:.1f}")
    with open(os.path.join(folder, "summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    print("saved", os.path.join(folder, "summary.json"))
    if a.plots:
        plots(s, folder)


if __name__ == "__main__":
    main()
