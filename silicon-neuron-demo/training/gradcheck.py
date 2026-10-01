"""
Gradient-check gate for the hand-written backward passes.

ANN: central differences, float64, relative error < 1e-6.
SNN: swap the Heaviside for a steep sigmoid (k = 20) in the FORWARD pass and
     use its exact derivative in the backward pass, so the analytic gradient
     is the true gradient of that smooth network.

The script prints the SNN error for shrinking eps: it falls ~100x for every
10x smaller eps, i.e. O(eps^2), the signature of a correct gradient whose
only error is central-difference truncation (eps^2 * L_ttt / 6). At the very
smallest eps floating-point round-off (~1e-16 * L / eps) takes over and the
error rises again: the usual V-shape. A wrong gradient shows a floor that
does not move with eps at all. Gate: the best eps of each block < 1e-6.

Run:  python gradcheck.py
"""
import numpy as np

from models import ANN, SNN, cross_entropy_logits
from encoding import latency_spikes


def rel_err(a, b):
    return np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b))


def check(model, loss_fn, grads, n_probe=5, eps=1e-5, rng=None):
    """Directional derivative test: for random directions d compare
    (L(W + eps d) - L(W - eps d)) / 2eps   with   <dL/dW, d>.
    Robust where single-element probes hit tiny gradients (round-off)."""
    worst = 0.0
    for name, W in model.p.items():
        for _ in range(n_probe):
            d = rng.normal(size=W.shape)
            W += eps * d; lp = loss_fn()
            W -= 2 * eps * d; lm = loss_fn()
            W += eps * d
            num = (lp - lm) / (2 * eps)
            ana = float((grads[name] * d).sum())
            worst = max(worst, rel_err(num, ana))
    return worst


def main():
    rng = np.random.default_rng(0)
    B = 4
    x = rng.integers(0, 256, (B, 784)).astype(np.uint8)
    y = rng.integers(0, 10, B)

    # ------------------------------------------------------------- ANN
    ann = ANN(rng, n_hid=16, dtype=np.float64)
    xf = x / 255.0
    z2, _ = ann.forward(xf)
    g = ann.backward(z2, y)
    loss = lambda: cross_entropy_logits(ann.forward(xf)[0], y)
    print(f"ANN worst relative error: {check(ann, loss, g, rng=rng):.2e}  (gate < 1e-6)")

    # ------------------------------------------------------------- SNN
    snn = SNN(rng, n_hid=16, dtype=np.float64, spike_mode="sigmoid", sig_k=20.0)
    snn.p["W1"] *= 3                           # make sure units are active
    s_in = latency_spikes(x, T=8).astype(np.float64)
    lam = 0.5
    logits, _ = snn.forward(s_in)
    g = snn.backward(logits, y, lam=lam)

    def loss():
        lg, S = snn.forward(s_in)
        return cross_entropy_logits(lg, y) + lam * S.mean()

    best = []
    errs = []
    for eps in (1e-4, 1e-5, 1e-6, 1e-7):
        errs.append(check(snn, loss, g, eps=eps, rng=rng))
        print(f"SNN eps={eps:.0e}  worst relative error: {errs[-1]:.2e}")
    best.append(min(errs))

    # ------------------------------------- SNN-E: per-step readout, first-order LIF
    # L = (1/T) sum_n CE(c[n]/(n+1)) + lam*mean(S): the early-exit training loss
    # (alpha = 0: the first-order LIF used by the Pico-optimised SNN-E)
    snn2 = SNN(rng, n_hid=16, dtype=np.float64, spike_mode="sigmoid", sig_k=20.0, readout="perstep",
               alpha=0.0)
    snn2.p["W1"] *= 3
    logits, _ = snn2.forward(s_in)
    g2 = snn2.backward(logits, y, lam=lam)

    def loss2():
        lg, S = snn2.forward(s_in)
        return snn2.loss(lg, y) + lam * S.mean()

    errs = []
    for eps in (1e-4, 1e-5, 1e-6, 1e-7):
        errs.append(check(snn2, loss2, g2, eps=eps, rng=rng))
        print(f"SNN-E eps={eps:.0e}  worst relative error: {errs[-1]:.2e}")
    best.append(min(errs))
    ok = all(b < 1e-6 for b in best)
    print(f"GRADIENT CHECK: {'PASS' if ok else 'FAIL'}  (best errors {', '.join(f'{b:.1e}' for b in best)}; gate < 1e-6)")


if __name__ == "__main__":
    main()
