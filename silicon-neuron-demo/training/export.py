"""
Quantise the trained float models, verify integer accuracy, and generate:

  firmware/generated/model_data.h / model_data.c   (C arrays for the Pico)
  pi/model_bundle.npz                              (same data for the Pi display
                                                    + its offline simulator)
  weights/export_report.json                       (numbers for the dissertation)

    python export.py            (after train.py has produced the three float models)
"""
import json
import os

import numpy as np

import config as C
from energy_model import pico_cycles
from intsim import (quantise_ann, quantise_snn, ann_int, snn_int, input_spikes_int, op_counts,
                    quantise_snn_fast, snn_fast_int)
from mnist_idx import load_mnist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WDIR = os.path.join(HERE, "weights")
GEN = os.path.join(ROOT, "firmware", "generated")


def c_array(ctype, name, arr, per_line=24, const=True):
    flat = np.asarray(arr).ravel()
    body = ",\n".join("  " + ",".join(str(int(v)) for v in flat[i:i + per_line])
                      for i in range(0, len(flat), per_line))
    return f"{'const ' if const else ''}{ctype} {name}[{len(flat)}] = {{\n{body}\n}};\n"


def main():
    xtr, ytr, xte, yte = load_mnist()
    rep = {}

    # ------------------------------------------------------------ ANN
    pa = dict(np.load(os.path.join(WDIR, "ann_float.npz")))
    qa = quantise_ann(pa, xtr[:5000])
    pred, _, acc2 = ann_int(qa, xte)
    rep["ann_float_acc"] = float(pa["test_acc"])
    rep["ann_int_acc"] = float((pred == yte).mean())
    assert np.abs(acc2).max() < 2 ** 31

    # ------------------------------------------------------------ SNNs
    qs = {}
    for enc in ("latency", "poisson"):
        path = os.path.join(WDIR, f"snn_{enc}_float.npz")
        if not os.path.exists(path):
            print("skip", enc, "(not trained)")
            continue
        ps = dict(np.load(path))
        q = quantise_snn(ps)
        n_eval = 10000 if enc == "latency" else 2000     # poisson sim is slower in NumPy
        s_in = input_spikes_int(xte[:n_eval], enc, np.arange(n_eval))
        p, c, S, _ = snn_int(q, s_in)
        assert np.abs(c).max() < 2 ** 31
        rep[f"snn_{enc}_float_acc"] = float(ps["test_acc"])
        rep[f"snn_{enc}_int_acc"] = float((p == yte[:n_eval]).mean())
        rep[f"snn_{enc}_ops"] = op_counts(xte[:n_eval], s_in, S)
        rep[f"snn_{enc}_theta_q"] = q["theta"]
        qs[enc] = q

    # ------------------------------------------------------------ SNN-E (Pico-optimised)
    qf = None
    pf_path = os.path.join(WDIR, "snn_fast_float.npz")
    if os.path.exists(pf_path):
        pf = dict(np.load(pf_path))
        qf = quantise_snn_fast(pf)
        # exit margin chosen on VALIDATION images (the last 5,000 training images):
        # the cheapest margin whose accuracy is within 0.1 percentage points of no exit
        xva, yva = xtr[-5000:], ytr[-5000:]
        base = (snn_fast_int(qf, xva)[0] == yva).mean()
        best = (None, base, None)
        for k in (0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):     # margin in units of theta
            m = int(round(k * qf["theta"]))
            p_, _, cnt = snn_fast_int(qf, xva, exit_margin=m)
            acc = (p_ == yva).mean()
            cyc = pico_cycles("snn_fast", events=cnt["events"].mean(), hid_spikes=cnt["hid_spikes"].mean(),
                              steps=cnt["steps"].mean())["total"]
            if acc >= base - 0.001 and (best[2] is None or cyc < best[2]):
                best = (m, acc, cyc)
        qf["exit_margin"] = int(best[0]) if best[0] is not None else -1
        qf["n_min"] = 1
        p_, c_, cnt = snn_fast_int(qf, xte, exit_margin=qf["exit_margin"] if qf["exit_margin"] >= 0 else None)
        assert np.abs(c_).max() < 2 ** 31
        rep["snn_fast_float_acc"] = float(pf["test_acc"])
        rep["snn_fast_int_acc_no_exit"] = float((snn_fast_int(qf, xte)[0] == yte).mean())
        rep["snn_fast_int_acc"] = float((p_ == yte).mean())
        rep["snn_fast_params"] = {k: qf[k] for k in ("T", "t_in", "x_min", "k_beta", "theta", "exit_margin", "n_min")}
        rep["snn_fast_ops"] = {"mean_steps": float(cnt["steps"].mean()), "in_AC": float(cnt["events"].mean() * C.N_HID),
                               "in_events": float(cnt["events"].mean()), "hid_spikes": float(cnt["hid_spikes"].mean()),
                               "hid_AC": float(cnt["hid_spikes"].mean() * C.N_OUT),
                               "neuron_updates": float(cnt["steps"].mean() * (C.N_HID + C.N_OUT))}
        print(f"SNN-E: exit margin {qf['exit_margin']} (validation acc {best[1] * 100:.2f}% vs {base * 100:.2f}% "
              f"without exit), test acc {rep['snn_fast_int_acc'] * 100:.2f}%, mean steps {cnt['steps'].mean():.2f}")

    rep["ann_ops"] = op_counts(xte, np.zeros((1, 1, 1)), np.zeros((1, 1, 1)))
    for k in ("snn_in_AC", "snn_hid_AC", "snn_neuron_updates", "snn_Tr"):
        rep["ann_ops"].pop(k)

    # ------------------------------------------------------------ emit C
    os.makedirs(GEN, exist_ok=True)
    N = C.N_TEST_PICO
    h = f"""// GENERATED by training/export.py — do not edit by hand.
#pragma once
#include <stdint.h>

#define N_IN      {C.N_IN}
#define N_HID     {C.N_HID}
#define N_OUT     {C.N_OUT}
#define T_STEPS   {C.T_STEPS}
#define T_IN      {C.T_IN}
#define X_MIN     {C.X_MIN}
#define K_ALPHA   {C.K_ALPHA}
#define K_BETA    {C.K_BETA}
#define N_TEST    {N}
#define POISSON_SEED 0x{C.POISSON_SEED:08X}u
#define ANN_SHIFT {C.ANN_REQUANT_SHIFT}
#define ANN_M     {qa['M']}
#define HAS_POISSON {1 if 'poisson' in qs else 0}

#define HAS_FAST  {1 if qf else 0}
#define FAST_T    {qf['T'] if qf else 1}
#define FAST_T_IN {qf['t_in'] if qf else 1}
#define FAST_X_MIN {qf['x_min'] if qf else 0}
#define FAST_K_BETA {qf['k_beta'] if qf else 1}
#define FAST_THETA {qf['theta'] if qf else 0}
#define FAST_EXIT_MARGIN {qf['exit_margin'] if qf else -1}
#define FAST_N_MIN {qf['n_min'] if qf else 1}

#define SNN_LAT_THETA {qs['latency']['theta']}
#define SNN_POI_THETA {qs['poisson']['theta'] if 'poisson' in qs else 0}

// weight layout: W[i * n_out + j] = weight from input i to output j
extern const int8_t  ann_W1[N_IN * N_HID];
extern const int32_t ann_b1[N_HID];
extern const int8_t  ann_W2[N_HID * N_OUT];
extern const int32_t ann_b2[N_OUT];
extern const int8_t  snn_lat_W1[N_IN * N_HID];
extern const int8_t  snn_lat_W2[N_HID * N_OUT];
extern const int8_t  fast_W1[];
extern const int8_t  fast_W2[];
extern const int8_t  snn_poi_W1[];
extern const int8_t  snn_poi_W2[];
extern const uint8_t test_images[N_TEST * N_IN];
extern const uint8_t test_labels[N_TEST];
"""
    src = ['// GENERATED by training/export.py — do not edit by hand.\n#include "model_data.h"\n\n',
           c_array("int8_t", "ann_W1", qa["W1"]), c_array("int32_t", "ann_b1", qa["b1"]),
           c_array("int8_t", "ann_W2", qa["W2"]), c_array("int32_t", "ann_b2", qa["b2"]),
           c_array("int8_t", "snn_lat_W1", qs["latency"]["W1"]),
           c_array("int8_t", "snn_lat_W2", qs["latency"]["W2"])]
    if "poisson" in qs:
        src += [c_array("int8_t", "snn_poi_W1", qs["poisson"]["W1"]),
                c_array("int8_t", "snn_poi_W2", qs["poisson"]["W2"])]
    else:
        src += ["const int8_t snn_poi_W1[1] = {0};\n", "const int8_t snn_poi_W2[1] = {0};\n"]
    if qf:
        src += [c_array("int8_t", "fast_W1", qf["W1"]), c_array("int8_t", "fast_W2", qf["W2"])]
    else:
        src += ["const int8_t fast_W1[1] = {0};\n", "const int8_t fast_W2[1] = {0};\n"]
    src += [c_array("uint8_t", "test_images", xte[:N], per_line=28),
            c_array("uint8_t", "test_labels", yte[:N])]
    with open(os.path.join(GEN, "model_data.h"), "w") as f:
        f.write(h)
    with open(os.path.join(GEN, "model_data.c"), "w") as f:
        f.write("".join(src))

    # ------------------------------------------------------------ emit Pi bundle
    bundle = {"ann_" + k: v for k, v in qa.items()}
    for enc, q in qs.items():
        bundle.update({f"snn_{enc}_" + k: v for k, v in q.items()})
    if qf:
        bundle.update({"snn_fast_" + k: v for k, v in qf.items()})
    cfg = {k: getattr(C, k) for k in dir(C) if k.isupper()}
    cfg.pop("TRACE_NEURONS", None)
    bundle.update(test_images=xte[:N], test_labels=yte[:N], config_json=json.dumps(cfg))
    np.savez_compressed(os.path.join(ROOT, "pi", "model_bundle.npz"), **bundle)

    # the Pi folder is copied to the Pi on its own: give it the cost model too
    import shutil
    shutil.copy(os.path.join(HERE, "energy_model.py"), os.path.join(ROOT, "pi", "energy_model.py"))

    with open(os.path.join(WDIR, "export_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
