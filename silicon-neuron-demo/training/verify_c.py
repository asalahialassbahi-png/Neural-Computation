"""
Proves the C firmware core and the Python integer model are bit-identical.

Builds firmware/host_test, runs it on all N_TEST_PICO images, and compares
every prediction, spike count and output potential with training/intsim.py.

    python verify_c.py
"""
import os
import subprocess
import sys

import numpy as np

import config as C
from export import WDIR
from intsim import (quantise_ann, quantise_snn, ann_int, snn_int, input_spikes_int,
                    quantise_snn_fast, snn_fast_int)
from mnist_idx import load_mnist

HERE = os.path.dirname(os.path.abspath(__file__))
HT = os.path.join(os.path.dirname(HERE), "firmware", "host_test")


def main():
    subprocess.run(["make", "-s", "-C", HT], check=True)
    out = subprocess.run([os.path.join(HT, "host_test")], check=True,
                         capture_output=True, text=True).stdout.split("\n")
    rows = {"A": [], "L": [], "P": [], "F": [], "E": []}
    for ln in out:
        if ln:
            parts = ln.split()
            rows[parts[0]].append([int(v) for v in parts[1:]])
    rows = {k: np.array(v) for k, v in rows.items() if v}

    xtr, _, xte, yte = load_mnist()
    N = C.N_TEST_PICO
    x = xte[:N]
    bad = 0

    qa = quantise_ann(dict(np.load(os.path.join(WDIR, "ann_float.npz"))), xtr[:5000])
    pred, _, acc2 = ann_int(qa, x)
    A = rows["A"]
    bad += (A[:, 1] != pred).sum() + (A[:, 2] != pred).sum() + (A[:, 3:] != acc2).sum()
    print(f"ANN  : C vs Python mismatches = {(A[:, 1] != pred).sum()} preds, "
          f"{(A[:, 2] != pred).sum()} zero-skip preds, {(A[:, 3:] != acc2).sum()} logits;"
          f"  accuracy on Pico set {(pred == yte[:N]).mean() * 100:.1f}%")

    for code, enc in (("L", "latency"), ("P", "poisson")):
        if code not in rows:
            continue
        q = quantise_snn(dict(np.load(os.path.join(WDIR, f"snn_{enc}_float.npz"))))
        s_in = input_spikes_int(x, enc, np.arange(N))
        p, c, S, _ = snn_int(q, s_in)
        R = rows[code]
        m_pred = (R[:, 1] != p).sum()
        m_sp = (R[:, 2] != S.sum((1, 2))).sum()
        m_c = (R[:, 3:] != c).sum()
        bad += m_pred + m_sp + m_c
        print(f"SNN-{enc:7s}: mismatches = {m_pred} preds, {m_sp} spike counts, {m_c} potentials;"
              f"  accuracy on Pico set {(p == yte[:N]).mean() * 100:.1f}%,"
              f"  mean hidden spikes/inference {S.sum((1, 2)).mean():.1f}")

    # ---- SNN-E, without and with the deployed early exit
    if "F" in rows:
        z = dict(np.load(os.path.join(os.path.dirname(HERE), "pi", "model_bundle.npz")))
        margin = int(z["snn_fast_exit_margin"])
        qf = quantise_snn_fast(dict(np.load(os.path.join(WDIR, "snn_fast_float.npz"))))
        for code, m in (("F", None), ("E", margin if margin >= 0 else None)):
            p, c, cnt = snn_fast_int(qf, x, exit_margin=m)
            R = rows[code]
            m_pred = (R[:, 1] != p).sum()
            m_sp = (R[:, 2] != cnt["hid_spikes"]).sum()
            m_st = (R[:, 3] != cnt["steps"]).sum()
            m_c = (R[:, 4:] != c).sum()
            bad += m_pred + m_sp + m_st + m_c
            print(f"SNN-E {'no exit' if m is None else f'exit {m}':8s}: mismatches = {m_pred} preds, {m_sp} spike counts, "
                  f"{m_st} step counts, {m_c} potentials;  accuracy on Pico set {(p == yte[:N]).mean() * 100:.1f}%,"
                  f"  mean steps {cnt['steps'].mean():.2f}")

    # ---- the serial trace protocol, parsed exactly as the Pi parses it
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pi"))
    from nnsim import Bundle
    from pico_link import _parse_trace
    b = Bundle(os.path.join(os.path.dirname(HERE), "pi", "model_bundle.npz"))
    txt = subprocess.run([os.path.join(HT, "host_test"), "trace", "25"], check=True,
                         capture_output=True, text=True).stdout.split("\n")
    chunks, cur = [], []
    for ln in txt:
        if not ln:
            continue
        cur.append(ln)
        if ln == "END":
            chunks.append(cur); cur = []
    m_tr = 0
    for idx, ch in enumerate(chunks):
        got = _parse_trace(ch, b)
        ref = b.snn(b.images[idx], "latency", idx)
        m_tr += int(got["trace_idx"] != ref["trace_idx"]) + int((got["u"] != ref["u"]).any()) \
            + int((got["raster"] != ref["raster"]).any()) + int((got["in_time"] != ref["in_time"]).any()) \
            + int((got["c"] != ref["c"]).any()) + int(got["pred"] != ref["pred"]) \
            + int(got["ann_pred"] != b.ann(b.images[idx])[0])
    print(f"trace protocol: {len(chunks)} traces, {m_tr} mismatches vs the Pi's NumPy model")
    bad += m_tr

    print("BIT-EXACT: PASS" if bad == 0 else f"BIT-EXACT: FAIL ({bad} differences)")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
