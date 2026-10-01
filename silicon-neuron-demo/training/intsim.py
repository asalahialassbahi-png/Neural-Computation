"""
Integer (fixed-point) reference implementation — bit-exact with the Pico
firmware in firmware/nn_core.c. If this file and the C disagree on a single
prediction, spike count or output potential, the host test fails.

Quantisation
------------
ANN   weights int8 per layer:  W_q = round(W / s),  s = max|W| / 127
      input  uint8 x (scale 1/255)
      acc1 = x . W1_q + b1_q                      (int32, unit s1/255)
      h8   = clamp((acc1 * M + 2^23) >> 24, 0, 255),  M = round(2^24 (s1/255)/s_h)
      acc2 = h8 . W2_q + b2_q  -> argmax
SNN   weights int8, state int32 in units of the layer weight scale
      I <- I - (I >> K_ALPHA) + sum of rows of W1_q for input spikes
      U <- U - (U >> K_BETA) + I - theta_q * S_prev,   theta_q = round(theta / s1)
      S  = [U >= theta_q]
      output: I2, U2 the same without threshold, c += U2, prediction argmax c
Arithmetic right shift of negative numbers rounds towards -infinity in both
NumPy and GCC-for-ARM, so the two agree exactly.
"""
import numpy as np

import config as C
from encoding import latency_times, NO_SPIKE


# ============================================================ quantisation
def q8(W):
    s = np.abs(W).max() / 127.0
    return np.clip(np.round(W / s), -127, 127).astype(np.int8), float(s)


def quantise_ann(p, x_calib):
    W1q, s1 = q8(p["W1"])
    W2q, s2 = q8(p["W2"])
    b1q = np.round(p["b1"] / (s1 / 255.0)).astype(np.int32)
    h = np.maximum(x_calib / 255.0 @ p["W1"] + p["b1"], 0)
    s_h = np.percentile(h[h > 0], C.ACT_PERCENTILE) / 255.0
    M = int(round((s1 / 255.0) / s_h * 2 ** C.ANN_REQUANT_SHIFT))
    assert 0 < M < 2 ** 31, M
    b2q = np.round(p["b2"] / (s_h * s2)).astype(np.int32)
    return {"W1": W1q, "b1": b1q, "M": M, "W2": W2q, "b2": b2q,
            "s1": s1, "s2": s2, "s_h": s_h}


def quantise_snn(p):
    W1q, s1 = q8(p["W1"])
    W2q, s2 = q8(p["W2"])
    theta_q = int(round(C.THETA / s1))
    return {"W1": W1q, "W2": W2q, "theta": theta_q, "s1": s1, "s2": s2}


# ================================================================ ANN (int)
def ann_int(q, x_u8, skip_zeros=False):
    """Batch integer ANN. Returns predictions, hidden uint8 activations.
    skip_zeros changes only the operation count, never the result."""
    x = x_u8.astype(np.int64)
    acc1 = x @ q["W1"].astype(np.int64) + q["b1"]
    h8 = np.clip((acc1 * q["M"] + (1 << (C.ANN_REQUANT_SHIFT - 1))) >> C.ANN_REQUANT_SHIFT, 0, 255)
    h8 = np.where(acc1 > 0, h8, 0)
    acc2 = h8 @ q["W2"].astype(np.int64) + q["b2"]
    return acc2.argmax(1), h8, acc2


# ================================================================ SNN (int)
def xorshift_vec(s):
    s = s ^ ((s << np.uint32(13)) & np.uint32(0xFFFFFFFF))
    s = s ^ (s >> np.uint32(17))
    s = s ^ ((s << np.uint32(5)) & np.uint32(0xFFFFFFFF))
    return s


def input_spikes_int(x_u8, enc, img_index):
    """(B, T, 784) bool spikes exactly as the firmware generates them.

    latency : single spike at t_i = ((255 - x) * T_IN) >> 8 for x >= X_MIN
    poisson : per image the xorshift32 state is seeded with
              POISSON_SEED ^ (index * 0x9E3779B9); each step, pixels are
              visited in order and ONLY non-zero pixels draw a number;
              spike iff (r >> 24) < x.
    """
    B = x_u8.shape[0]
    T = C.T_STEPS
    if enc == "latency":
        t = latency_times(x_u8)
        return t[:, None, :] == np.arange(T, dtype=np.uint8)[None, :, None]
    state = (np.uint32(C.POISSON_SEED) ^
             (np.asarray(img_index, np.uint64) * 0x9E3779B9 & 0xFFFFFFFF).astype(np.uint32))
    state = state.astype(np.uint32)
    state[state == 0] = 1                      # xorshift must never be seeded with 0
    x = x_u8.astype(np.uint32)
    out = np.zeros((B, T, C.N_IN), bool)
    for n in range(T):
        for i in range(C.N_IN):
            m = x[:, i] > 0
            if not m.any():
                continue
            ns = xorshift_vec(state[m])
            state[m] = ns
            out[m, n, i] = (ns >> np.uint32(24)) < x[m, i]
    return out


def snn_int(q, s_in, trace_neurons=None):
    """Batch integer SNN on precomputed bool spikes (B, T, 784).
    Returns preds, c (B,10), hidden spikes (B,T,H) bool, traces (B,T,k)."""
    B, T, _ = s_in.shape
    H = q["W1"].shape[1]
    W1 = q["W1"].astype(np.int32); W2 = q["W2"].astype(np.int32)
    th = np.int32(q["theta"])
    I1 = np.zeros((B, H), np.int32); U1 = np.zeros((B, H), np.int32)
    S1 = np.zeros((B, H), bool)
    I2 = np.zeros((B, C.N_OUT), np.int32); U2 = np.zeros((B, C.N_OUT), np.int32)
    c = np.zeros((B, C.N_OUT), np.int32)
    S_hist = np.zeros((B, T, H), bool)
    tr = [] if trace_neurons is not None else None
    for n in range(T):
        I1 = I1 - (I1 >> C.K_ALPHA) + s_in[:, n].astype(np.int32) @ W1
        U1 = U1 - (U1 >> C.K_BETA) + I1 - th * S1
        S1 = U1 >= th
        S_hist[:, n] = S1
        I2 = I2 - (I2 >> C.K_ALPHA) + S1.astype(np.int32) @ W2
        U2 = U2 - (U2 >> C.K_BETA) + I2
        c = c + U2
        if tr is not None:
            tr.append(U1[:, list(trace_neurons)].copy())
    traces = np.stack(tr, 1) if tr is not None else None
    return c.argmax(1), c, S_hist, traces


# ======================================================== operation counts
def op_counts(x_u8, S_in, S_hid, H=C.N_HID):
    """Synaptic operations actually executed per inference (mean over batch)."""
    B = x_u8.shape[0]
    return {
        "ann_dense_MAC": C.N_IN * H + H * C.N_OUT,
        "ann_skip_MAC": float((x_u8 > 0).sum(1).mean() * H + H * C.N_OUT),
        "snn_in_AC": float(S_in.sum() / B * H),
        "snn_hid_AC": float(S_hid.sum() / B * C.N_OUT),
        "snn_neuron_updates": C.T_STEPS * (H + C.N_OUT),
        "snn_Tr": float(S_hid.mean() * C.T_STEPS),
    }


# ============================================ SNN-E: the Pico-optimised SNN (int)
def quantise_snn_fast(p):
    """SNN-E: first-order LIF (no synaptic current), its own T / latency code,
    trained with the per-step readout. Same int8 weight quantisation."""
    q = quantise_snn(p)
    q.update(T=int(p["T"]), t_in=int(p["t_in"]), x_min=int(p["x_min"]), k_beta=int(p["k_beta"]))
    return q


def snn_fast_int(q, x_u8, exit_margin=None, n_min=1):
    """Integer SNN-E with early exit, bit-exact with snn_fast_infer() in nn_core.c.

    Per step n (all integer, shifts only):
        U <- U - (U >> k_beta) - theta * S_prev       (leak, then reset by subtraction)
        U <- U + sum of W1 rows of the pixels spiking at n
        S  = [U >= theta]
        U2 <- U2 - (U2 >> k_beta) + sum of W2 rows of the hidden spikes
        c  <- c + U2                                   (cumulative evidence)
    Early exit after step n (n + 1 >= n_min):  stop when
        c_best - c_second >= exit_margin * (n + 1)
    i.e. when the mean output margin so far reaches exit_margin.

    Returns pred, c, and per-inference counts: steps used, input events
    processed, hidden spikes, so the operation count is exact.
    """
    B = x_u8.shape[0]
    T, kb, th = q["T"], q["k_beta"], np.int32(q["theta"])
    t = latency_times(x_u8, q["t_in"], q["x_min"])
    W1 = q["W1"].astype(np.int32); W2 = q["W2"].astype(np.int32)
    H = W1.shape[1]
    U1 = np.zeros((B, H), np.int32); S1 = np.zeros((B, H), bool)
    U2 = np.zeros((B, C.N_OUT), np.int32); c = np.zeros((B, C.N_OUT), np.int32)
    active = np.ones(B, bool)
    steps = np.full(B, T, np.int32)
    events = np.zeros(B, np.int64); hid = np.zeros(B, np.int64)
    for n in range(T):
        a = np.nonzero(active)[0]
        if len(a) == 0:
            break
        inp = t[a] == n
        events[a] += inp.sum(1)
        U = U1[a]
        U = U - (U >> kb) - th * S1[a]
        U = U + inp.astype(np.int32) @ W1
        S = U >= th
        U1[a], S1[a] = U, S
        hid[a] += S.sum(1)
        V = U2[a]
        V = V - (V >> kb) + S.astype(np.int32) @ W2
        U2[a] = V
        c[a] += V
        if exit_margin is not None and n + 1 >= n_min and n < T - 1:
            top = np.sort(c[a], 1).astype(np.int64)        # int64: c_best - c_second cannot wrap
            done = (top[:, -1] - top[:, -2]) >= exit_margin * (n + 1)
            steps[a[done]] = n + 1
            active[a[done]] = False
    return c.argmax(1), c, {"steps": steps, "events": events, "hid_spikes": hid}


EXIT_MARGIN_K = (0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)   # candidate margins, in thresholds


def choose_exit_margin(q, x_val, y_val, cycles_fn, max_changed=0.001):
    """The deployed early-exit rule, chosen on VALIDATION images only: the margin
    (a whole number of thresholds) with the fewest predicted Pico cycles for which
    stopping early CHANGES THE ANSWER on at most max_changed of the images,
    compared with always running every step. Agreement between two predictions
    of the same images is a much less noisy statistic than the difference of two
    accuracies (5,000 images give accuracy +/- 0.2 points by sampling alone).
    Returns (margin, acc, base_acc, cycles)."""
    p_full = snn_fast_int(q, x_val)[0]
    base = float((p_full == y_val).mean())
    best = (-1, base, base, None)
    for k in EXIT_MARGIN_K:
        m = int(round(k * q["theta"]))
        p, _, cnt = snn_fast_int(q, x_val, exit_margin=m)
        changed = float((p != p_full).mean())
        cyc = cycles_fn("snn_fast", events=cnt["events"].mean(), hid_spikes=cnt["hid_spikes"].mean(),
                        steps=cnt["steps"].mean())["total"]
        if changed <= max_changed and (best[3] is None or cyc < best[3]):
            best = (m, float((p == y_val).mean()), base, cyc)
    return best
