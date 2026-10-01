"""
The evidence, in one run: is the Pico-optimised SNN as accurate as the ANN,
and does it need less energy on a Raspberry Pi Pico? And why?

    python prove.py          (after export.py and cycle_audit.py; repro.py optional)

Writes results/PROOF.md and four figures in results/:

 1. Accuracy, paired (both networks see the same 10,000 test digits):
      McNemar's exact test on the digits where exactly one network is right
      (H0: both networks are equally accurate), and a TOST equivalence test:
      is the accuracy difference inside +/- 0.5 percentage points?
 2. Work on the Pico, paired over the emulated digits (cycle_audit.py):
      ratio SNN-E / ANN-Z per digit, its bootstrap 95% CI, the fraction of
      digits where SNN-E is cheaper, and a sign test.
 3. The 45 nm "textbook" model next to the Pico model: the same operation
      counts priced on custom silicon and on a microcontroller.
 4. Why: the break-even line of MATHS.md eq. 8.4, with every test digit plotted.
"""
import json
import math
import os

import numpy as np

import config as C
from energy_model import CYC, energy_45nm, pico_cycles, N_HID
from intsim import (quantise_ann, ann_int, quantise_snn, snn_int, input_spikes_int,
                    quantise_snn_fast, snn_fast_int)
from mnist_idx import load_mnist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
W = os.path.join(HERE, "weights")


def binom_two_sided(k, n):
    """Exact two-sided binomial test, p = 0.5 (McNemar's exact test on the discordant pairs)."""
    from math import comb
    k = min(k, n - k)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def tost(diffs, margin):
    """Paired TOST on per-digit correctness differences d_i in {-1, 0, 1}:
    H0: |mean d| >= margin. Normal approximation (n = 10,000)."""
    d = np.asarray(diffs, float)
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
    p_lo = 1 - norm_cdf((m + margin) / se)     # H0: mean <= -margin
    p_hi = norm_cdf((m - margin) / se)         # H0: mean >= +margin
    return m, se, max(p_lo, p_hi)


def main():
    os.makedirs(RES, exist_ok=True)
    xtr, ytr, xte, yte = load_mnist()
    rep = json.load(open(os.path.join(W, "export_report.json")))
    margin = rep["snn_fast_params"]["exit_margin"]

    # ------------------------------------------------ predictions, all 10,000 digits
    qa = quantise_ann(dict(np.load(os.path.join(W, "ann_float.npz"))), xtr[:5000])
    pa, h8, _ = ann_int(qa, xte)
    qf = quantise_snn_fast(dict(np.load(os.path.join(W, "snn_fast_float.npz"))))
    pe, _, cnt = snn_fast_int(qf, xte, exit_margin=margin)
    ql = quantise_snn(dict(np.load(os.path.join(W, "snn_latency_float.npz"))))
    sl = input_spikes_int(xte, "latency", np.arange(len(xte)))
    pl, _, Sl, _ = snn_int(ql, sl)
    ca, ce, cl = pa == yte, pe == yte, pl == yte

    # ------------------------------------------------ 1. accuracy
    only_a, only_e = int((ca & ~ce).sum()), int((~ca & ce).sum())     # discordant pairs
    p_mc = binom_two_sided(only_e, only_a + only_e)
    d = ce.astype(int) - ca.astype(int)
    mdiff, se, p_tost05 = tost(d, 0.005)
    _, _, p_tost03 = tost(d, 0.003)
    acc = {"ANN": ca.mean(), "SNN-E": ce.mean(), "SNN-L": cl.mean()}

    # ------------------------------------------------ 2. work on the Pico (emulated)
    audit = json.load(open(os.path.join(RES, "cycle_audit.json")))
    M = audit["models"]
    z = np.array(M["Z"]["per_image"]); e = np.array(M["E"]["per_image"])
    ratio = e / z
    rng = np.random.default_rng(0)
    boot = [np.mean(e[i]) / np.mean(z[i]) for i in (rng.integers(0, len(z), len(z)) for _ in range(10000))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    wins = int((e < z).sum())
    p_sign = binom_two_sided(wins, len(z))

    # ------------------------------------------------ 3. operation counts, all digits, two cost models
    nz = (xte > 0).sum(1); nzh = (h8 > 0).sum(1)
    ops = {
        "ANN": {"MAC": C.N_IN * N_HID + N_HID * 10, "AC": 0, "upd": 0},
        "ANN-Z": {"MAC": float((nz * N_HID + nzh * 10).mean()), "AC": 0, "upd": 0},
        "SNN-L": {"MAC": 0, "AC": float((sl.sum((1, 2)) * N_HID + Sl.sum((1, 2)) * 10).mean()),
                  "upd": C.T_STEPS * (N_HID + 10)},
        "SNN-E": {"MAC": 0, "AC": float((cnt["events"] * N_HID + cnt["hid_spikes"] * 10).mean()),
                  "upd": float(cnt["steps"].mean() * (N_HID + 10))},
    }
    for k, o in ops.items():
        o["E45_nJ"] = energy_45nm(o["MAC"], o["AC"], o["upd"]) / 1e3
    emu = {"ANN": M["A"]["cycles_mean"], "ANN-Z": M["Z"]["cycles_mean"], "SNN-L": M["L"]["cycles_mean"],
           "SNN-E": M["E"]["cycles_mean"], "SNN-P": M["P"]["cycles_mean"]}

    # ------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    col = {"ANN": "#1f9bd1", "ANN-Z": "#2ca25f", "SNN-L": "#d6408f", "SNN-P": "#e6772e", "SNN-E": "#7a5fd6"}
    accs = {"ANN": rep["ann_int_acc"], "ANN-Z": rep["ann_int_acc"], "SNN-L": rep["snn_latency_int_acc"],
            "SNN-P": rep["snn_poisson_int_acc"], "SNN-E": rep["snn_fast_int_acc"]}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k in emu:
        ax.scatter(emu[k] / 1e3, accs[k] * 100, s=70, color=col[k], zorder=3)
        ax.annotate(k, (emu[k] / 1e3, accs[k] * 100), xytext=(7, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xticks([300, 600, 1000, 3000, 6000])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Pico CPU cycles per digit (thousands, emulated firmware, log scale)")
    ax.set_ylabel("test accuracy, int8 on 10,000 digits (%)")
    ax.axhspan(accs["ANN"] * 100 - 0.5, accs["ANN"] * 100, color="#2ca25f", alpha=0.08)
    ax.set_title("Accuracy against work on the Raspberry Pi Pico")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_pareto.png"), dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(z / 1e3, e / 1e3, s=12, color=col["SNN-E"], alpha=0.7)
    mx = max(z.max(), e.max()) / 1e3 * 1.05
    ax.plot([0, mx], [0, mx], color="grey", lw=1)
    ax.set_xlabel("ANN-Z cycles (thousands)"); ax.set_ylabel("SNN-E cycles (thousands)")
    ax.set_title(f"Same digit, both networks: SNN-E cheaper on {wins}/{len(z)}")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_per_digit.png"), dpi=180); plt.close(fig)

    # break-even (MATHS.md eq. 8.4): events SNN-E may process vs the ANN-Z work of the same digit
    cc = CYC
    be = (nz * (cc["ann_row"] + N_HID * cc["mac"]) - cnt["steps"] * N_HID * cc["fast_neuron_step"]) / \
         (cc["snn_row"] + N_HID * cc["ac"])
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(nz, cnt["events"], s=3, alpha=0.25, color=col["SNN-E"], label="SNN-E input spikes, each test digit")
    xs = np.linspace(nz.min(), nz.max(), 50)
    ax.plot(xs, (xs * (cc["ann_row"] + N_HID * cc["mac"]) - cnt["steps"].mean() * N_HID * cc["fast_neuron_step"])
            / (cc["snn_row"] + N_HID * cc["ac"]), color="k", lw=1.5, label="break-even: SNN-E = ANN-Z (eq. 8.4)")
    ax.set_xlabel("non-zero pixels (the ANN-Z's work)"); ax.set_ylabel("input spikes processed by SNN-E")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"Below the line SNN-E wins: {(cnt['events'] < be).mean() * 100:.1f}% of digits")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_break_even.png"), dpi=180); plt.close(fig)

    sweep = json.load(open(os.path.join(RES, "sweep.json")))
    rows = sweep["configs"]["T8_tin3_x96_lam1"]
    ks = [r["margin_k"] for r in rows if r["margin_k"] is not None]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(ks, [r["acc"] * 100 for r in rows if r["margin_k"] is not None], "o-", color=col["SNN-E"])
    ax2 = ax.twinx()
    ax2.plot(ks, [r["steps"] for r in rows if r["margin_k"] is not None], "s--", color="grey")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("exit margin m (thresholds)"); ax.set_ylabel("validation accuracy (%)")
    ax2.set_ylabel("mean time steps used (of 8)")
    ax.set_title("Early exit: confidence buys back time steps")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_early_exit.png"), dpi=180); plt.close(fig)

    # ------------------------------------------------ report
    repro = os.path.join(RES, "repro.md")
    L = ["# The evidence", "",
         "All numbers come from scripts in `training/`; rerun `python prove.py` to regenerate this file.", "",
         "## 1. Accuracy: is the spiking network as accurate as the ANN?", "",
         "Int8 models exactly as the Pico runs them, all 10,000 MNIST test digits, each seen by both networks.", "",
         "| network | test accuracy |", "|---|---|"]
    L += [f"| {k} | {v * 100:.2f}% |" for k, v in acc.items()]
    L += ["", f"Difference SNN-E minus ANN: **{mdiff * 100:+.2f} points** (standard error {se * 100:.2f}).",
          f"Digits only the ANN got right: {only_a}; only SNN-E got right: {only_e}. "
          f"McNemar exact test: p = {p_mc:.3f} " + ("(the difference is statistically significant at 5%)."
                                                     if p_mc < 0.05 else "(not significant at 5%)."),
          f"Equivalence (TOST): within +/-0.5 points, p = {p_tost05:.2g}"
          + (" -> equivalent." if p_tost05 < 0.05 else " -> not shown.")
          + f" Within +/-0.3 points, p = {p_tost03:.2g}" + (" -> equivalent." if p_tost03 < 0.05 else " -> not shown."),
          "", "## 2. Work on the Pico: the compiled firmware, emulated", "",
          f"Cycles per digit, emulated Cortex-M0+ (cycle_audit.py, {audit['n']} test digits; Poisson on fewer):", "",
          "| network | cycles per digit | ms at 125 MHz | compared with dense ANN | compared with ANN-Z |",
          "|---|---|---|---|---|"]

    def vs(a, b):
        if abs(a / b - 1) < 0.005:
            return "same"
        return f"{b / a:.2f}x less work" if a < b else f"{a / b:.2f}x MORE work"
    for k in ("ANN", "ANN-Z", "SNN-L", "SNN-P", "SNN-E"):
        L.append(f"| {k} | {emu[k] / 1e3:,.0f}k | {emu[k] / 125e3:.2f} | {vs(emu[k], emu['ANN'])} |"
                 f" {vs(emu[k], emu['ANN-Z'])} |")
    L += ["", f"Paired over the same digits: SNN-E / ANN-Z = **{np.mean(e) / np.mean(z):.3f}** "
              f"(bootstrap 95% CI {lo:.3f} to {hi:.3f}); SNN-E is cheaper on {wins} of {len(z)} digits "
              f"(sign test p = {p_sign:.2g}).",
          "The INA226 measures the real energy; first order, energy = active power x cycles / clock (MATHS.md 8.3).",
          "", "## 3. The same operations on two kinds of hardware", "",
          "| network | multiply-adds | spike additions | neuron updates | 45 nm arithmetic energy | Pico cycles |",
          "|---|---|---|---|---|---|"]
    for k, o in ops.items():
        L.append(f"| {k} | {o['MAC']:,.0f} | {o['AC']:,.0f} | {o['upd']:,.0f} | {o['E45_nJ']:.1f} nJ | {emu[k] / 1e3:,.0f}k |")
    L += ["", "On custom 45 nm silicon an addition costs about a seventh of a multiply-add, so every SNN looks "
              "far better; on the Pico the single-cycle multiplier makes them nearly equal (12 vs 13 cycles), "
              "and only doing FEWER operations helps. That is the design principle behind SNN-E.",
          "", "## 4. Why SNN-E wins", "",
          "![break-even](fig_break_even.png)", "", "![pareto](fig_pareto.png)", "",
          "![per digit](fig_per_digit.png)", "", "![early exit](fig_early_exit.png)", ""]
    if os.path.exists(repro):
        L += ["## 5. Repeatability", ""] + open(repro).read().splitlines()[2:]
    open(os.path.join(RES, "PROOF.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
