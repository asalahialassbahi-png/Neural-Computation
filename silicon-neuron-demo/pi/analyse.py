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
every mode, so they cancel exactly in the ratio E_ANN / E_SNN. With
results/calibration.json present, u_shunt is the calibration's own uncertainty.

Power path (bench.py --rail): for "3v3" runs the chip-only energy is also reported,
    E_chip = V_rail * I_mean * dt / N      (the LDO passes its input current through)
Boards: a CSV holding Pico and Pico 2 trials is analysed per board.
Temperature: the range over the session and the correlation of idle power with
it are printed, so "temperature was controlled" is a measured claim.
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
            if k in ("mode", "board", "rail"):
                continue
            try:
                r[k] = float(r[k])
            except ValueError:
                r[k] = float("nan")
        r.setdefault("board", "pico")
        r.setdefault("rail", "vsys")
    return rows


def analyse(rows, u_shunt=U_SHUNT):
    idle = {int(r["round"]): r["P_mean"] for r in rows if r["mode"] == "I"}
    idle_i = {int(r["round"]): r["I_mean"] for r in rows if r["mode"] == "I"}
    chip = all(r.get("rail") == "3v3" and r.get("v_rail", float("nan")) == r.get("v_rail") for r in rows)
    per = {}
    for r in rows:
        m = r["mode"]
        if m == "I":
            continue
        dt = r["us"] * 1e-6
        p_idle = idle.get(int(r["round"]), np.mean(list(idle.values())))
        per.setdefault(m, {"E_tot": [], "E_dyn": [], "acc": [], "Tr": [], "us_inf": [], "P": [],
                           "E_chip": [], "E_chip_dyn": []})
        d = per[m]
        d["E_tot"].append(r["P_mean"] * dt / r["count"])
        d["E_dyn"].append((r["P_mean"] - p_idle) * dt / r["count"])
        if chip:
            i_idle = idle_i.get(int(r["round"]), np.mean(list(idle_i.values())))
            d["E_chip"].append(r["v_rail"] * r["I_mean"] * dt / r["count"])
            d["E_chip_dyn"].append(r["v_rail"] * (r["I_mean"] - i_idle) * dt / r["count"])
        d["acc"].append(r["correct"] / r["count"])
        d["Tr"].append(r["spikes"] / r["count"] / N_HID if m in "LP" else 0.0)   # T * mean rate
        d["us_inf"].append(r["us"] / r["count"])
        d["P"].append(r["P_mean"])
    out = {"P_idle_mW": float(np.mean(list(idle.values())) * 1e3), "modes": {},
           "rail": rows[0].get("rail", "vsys"), "board": rows[0].get("board", "pico")}
    temps = np.array([r.get("temp_c", float("nan")) for r in rows], float)
    if np.isfinite(temps).sum() > 2:
        it = [(r["temp_c"], r["P_mean"]) for r in rows if r["mode"] == "I" and np.isfinite(r["temp_c"])]
        tc = {"min": float(np.nanmin(temps)), "max": float(np.nanmax(temps))}
        if len(it) > 2 and np.std([t for t, _ in it]) > 0:
            tc["r_idle_power"] = float(np.corrcoef(np.array(it).T)[0, 1])
        out["temperature_c"] = tc
    for m, d in per.items():
        a = {k: np.array(v) for k, v in d.items()}
        n = len(a["E_tot"])
        se_rel = a["P"].std(ddof=1) / math.sqrt(n) / a["P"].mean()
        u_rel = math.sqrt(se_rel ** 2 + u_shunt ** 2 + U_GAIN ** 2)
        out["modes"][m] = {
            "name": NAMES[m], "n": n,
            "E_total_uJ": float(a["E_tot"].mean() * 1e6), "E_total_sd_uJ": float(a["E_tot"].std(ddof=1) * 1e6),
            "E_dyn_uJ": float(a["E_dyn"].mean() * 1e6), "E_dyn_sd_uJ": float(a["E_dyn"].std(ddof=1) * 1e6),
            "E_total_u_rel": u_rel,
            "us_per_inf": float(a["us_inf"].mean()), "P_mW": float(a["P"].mean() * 1e3),
            "accuracy": float(a["acc"].mean()), "T_r": float(a["Tr"].mean()),
        }
        if chip:
            out["modes"][m].update(E_chip_uJ=float(a["E_chip"].mean() * 1e6),
                                   E_chip_sd_uJ=float(a["E_chip"].std(ddof=1) * 1e6),
                                   E_chip_dyn_uJ=float(a["E_chip_dyn"].mean() * 1e6))
        per[m] = a
    if "A" in per:
        base = per["A"]
        for m in per:
            if m == "A":
                continue
            for kind in ("E_tot", "E_dyn") + (("E_chip",) if chip else ()):
                t, df, p = welch(base[kind], per[m][kind])
                lo, hi = bootstrap_ratio(base[kind], per[m][kind])
                out["modes"][m][f"vs_ANN_{kind}"] = {
                    "ratio": float(base[kind].mean() / per[m][kind].mean()), "ci95": [lo, hi],
                    "welch_t": t, "df": df, "p": p, "cohens_d": cohens_d(base[kind], per[m][kind])}
    return out


def plots(summary, folder, suffix=""):
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
    fig.savefig(os.path.join(folder, f"energy_bars{suffix}.png"), dpi=200)
    print("saved", os.path.join(folder, f"energy_bars{suffix}.png"))
    # the EPQ question in one picture: accuracy against energy per inference
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in ms:
        d = summary["modes"][m]
        ax.errorbar(d["E_total_uJ"], d["accuracy"] * 100, xerr=d["E_total_sd_uJ"], fmt="o", capsize=3)
        ax.annotate(NAMES[m], (d["E_total_uJ"], d["accuracy"] * 100), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("energy per inference (uJ)")
    ax.set_ylabel("accuracy on the Pico (%)")
    ax.set_title(f"accuracy vs energy ({summary.get('board', '')}, {summary.get('rail', '')})")
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f"accuracy_vs_energy{suffix}.png"), dpi=200)
    print("saved", os.path.join(folder, f"accuracy_vs_energy{suffix}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()
    folder = os.path.dirname(os.path.abspath(a.csv))
    u_shunt = U_SHUNT
    cal = os.path.join(folder, "calibration.json")
    if os.path.exists(cal):
        u_shunt = json.load(open(cal))["u_rel"]
        print(f"shunt uncertainty from calibration.json: {u_shunt * 100:.2f}%")
    rows = load(a.csv)
    boards = sorted({r["board"] for r in rows})
    summaries = {}
    for b in boards:
        s = analyse([r for r in rows if r["board"] == b], u_shunt)
        summaries[b] = s
        if len(boards) > 1:
            print(f"\n===== board: {b}")
        report(s)
    s = summaries[boards[0]] if len(boards) == 1 else {"boards": summaries}
    with open(os.path.join(folder, "summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    print("saved", os.path.join(folder, "summary.json"))
    if a.plots:
        for b, sb in summaries.items():
            plots(sb, folder, "" if len(boards) == 1 else f"_{b}")


def report(s):
    print(f"power path {s['rail']}, idle power {s['P_idle_mW']:.2f} mW")
    tc = s.get("temperature_c")
    if tc:
        print(f"temperature {tc['min']:.1f}-{tc['max']:.1f} C" +
              (f", correlation with idle power r = {tc['r_idle_power']:+.2f}" if "r_idle_power" in tc else ""))
    print(f"{'mode':15s} {'n':>3s} {'E_total uJ':>12s} {'E_dyn uJ':>12s} {'us/inf':>9s} {'acc %':>6s} {'T*r':>6s}")
    for m, d in s["modes"].items():
        print(f"{d['name']:15s} {d['n']:3d} {d['E_total_uJ']:8.1f}±{d['E_total_sd_uJ']:<5.1f}"
              f"{d['E_dyn_uJ']:8.1f}±{d['E_dyn_sd_uJ']:<5.1f}{d['us_per_inf']:8.0f} "
              f"{d['accuracy'] * 100:6.1f} {d['T_r']:6.3f}")
        if "E_chip_uJ" in d:
            print(f"    chip only (3V3 rail): {d['E_chip_uJ']:.1f} ± {d['E_chip_sd_uJ']:.1f} uJ, "
                  f"above idle {d['E_chip_dyn_uJ']:.1f} uJ")
        for kind in ("E_tot", "E_dyn", "E_chip"):
            c = d.get(f"vs_ANN_{kind}")
            if c:
                print(f"    ANN/{m} {kind}: {c['ratio']:.2f}x  95% CI [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]"
                      f"  Welch t={c['welch_t']:.1f} (df {c['df']:.0f}, p={c['p']:.2g})  d={c['cohens_d']:.1f}")


if __name__ == "__main__":
    main()
