"""
Train the float models.

    python train.py --model ann                      -> weights/ann_float.npz
    python train.py --model snn --enc latency        -> weights/snn_latency_float.npz
    python train.py --model snn --enc poisson        -> weights/snn_poisson_float.npz
    python train.py --model snn --enc latency --lam 2.0 --tag lam2   (sparsity sweep)
    python train.py --model snn --enc latency --T 8 --tin 4 --xmin 128 --readout perstep --tag fast
                                                     -> weights/snn_latency_fast_float.npz
                                                        (the Pico-optimised early-exit SNN)

Plain NumPy, CPU only. On a 4-core laptop: ANN ~1 min, SNN ~2-4 min per epoch.
"""
import argparse
import os
import time

import numpy as np

import config as C
from encoding import latency_spikes, poisson_spikes
from mnist_idx import load_mnist
from models import ANN, SNN, Adam, cross_entropy_logits

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(HERE, "weights")


def batches(n, bs, rng):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def encode(xb, enc, rng, T, tin, xmin):
    if enc == "latency":
        return latency_spikes(xb, T=T, t_in=tin, x_min=xmin)
    return poisson_spikes(xb, rng, T=T)


def evaluate(model, x, y, kind, enc, rng, bs=500, T=C.T_STEPS, tin=C.T_IN, xmin=C.X_MIN):
    correct, spikes = 0, 0.0
    for i in range(0, len(x), bs):
        xb, yb = x[i:i + bs], y[i:i + bs]
        if kind == "ann":
            logits, _ = model.forward(xb / 255.0)
        else:
            logits, S = model.forward(encode(xb, enc, rng, T, tin, xmin))
            spikes += S.sum()
        correct += (logits.argmax(1) == yb).sum()
    rate = spikes / (len(x) * T * C.N_HID) if kind == "snn" else 0.0
    return correct / len(x), rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ann", "snn"], required=True)
    ap.add_argument("--enc", choices=["latency", "poisson"], default="latency")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lam", type=float, default=0.5,
                    help="firing-rate regulariser weight (the sparsity dial)")
    ap.add_argument("--T", type=int, default=C.T_STEPS, help="SNN time steps")
    ap.add_argument("--tin", type=int, default=C.T_IN, help="latency code: input spikes land in steps 0..tin-1")
    ap.add_argument("--xmin", type=int, default=C.X_MIN, help="latency code: dimmer pixels never spike")
    ap.add_argument("--readout", choices=["mean", "perstep"], default="mean",
                    help="perstep trains a good answer at every step (for early exit)")
    ap.add_argument("--first-order", action="store_true",
                    help="no synaptic current (alpha = 0): the Pico-optimised neuron")
    ap.add_argument("--kbeta", type=int, default=C.K_BETA, help="membrane leak beta = 1 - 2^-kbeta")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="explicit output path (repro.py)")
    ap.add_argument("--val-only", action="store_true", help="skip the final test-set evaluation (sweeps)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=None, help="train on first N images (quick tests)")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    xtr, ytr, xte, yte = load_mnist()
    if a.limit:
        xtr, ytr = xtr[:a.limit], ytr[:a.limit]

    # hold out 5,000 training images for validation; the test set is touched once
    xva, yva = xtr[-5000:], ytr[-5000:]
    xtr, ytr = xtr[:-5000], ytr[:-5000]

    if a.model == "ann":
        model = ANN(rng)
        epochs, lr = a.epochs or 10, a.lr or 1e-3
    else:
        model = SNN(rng, readout=a.readout, alpha=0.0 if a.first_order else C.ALPHA,
                    beta=1.0 - 2.0 ** -a.kbeta)
        epochs, lr = a.epochs or 12, a.lr or 2e-3
    opt = Adam(model.p, lr=lr)

    log = []
    for ep in range(epochs):
        if ep == int(epochs * 0.7):
            opt.lr *= 0.3                          # simple step decay
        t0, tot = time.time(), 0.0
        for bi in batches(len(xtr), a.bs, rng):
            xb, yb = xtr[bi], ytr[bi]
            if a.model == "ann":
                logits, _ = model.forward(xb / 255.0)
                grads = model.backward(logits, yb)
            else:
                logits, S = model.forward(encode(xb, a.enc, rng, a.T, a.tin, a.xmin))
                grads = model.backward(logits, yb, lam=a.lam)
            tot += (cross_entropy_logits(logits, yb) if a.model == "ann" else model.loss(logits, yb)) * len(yb)
            opt.step(grads)
        acc, rate = evaluate(model, xva, yva, a.model, a.enc, rng, T=a.T, tin=a.tin, xmin=a.xmin)
        log.append((ep, tot / len(xtr), acc, rate))
        print(f"epoch {ep + 1:2d}/{epochs}  loss {tot / len(xtr):.4f}  val acc {acc * 100:.2f}%"
              + (f"  mean rate r = {rate:.4f} spikes/neuron/step  T*r = {rate * a.T:.3f}"
                 if a.model == "snn" else "") + f"  ({time.time() - t0:.0f}s)")

    val_acc = acc
    if a.val_only:
        acc = float("nan")
    else:
        acc, rate = evaluate(model, xte, yte, a.model, a.enc, rng, T=a.T, tin=a.tin, xmin=a.xmin)
        print(f"TEST accuracy (float): {acc * 100:.2f}%" +
              (f"   T*r = {rate * a.T:.3f}" if a.model == "snn" else ""))

    os.makedirs(WDIR, exist_ok=True)
    name = "ann" if a.model == "ann" else f"snn_{a.enc}"
    if a.tag:
        name += "_" + a.tag
    path = a.out or os.path.join(WDIR, name + "_float.npz")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **model.p, test_acc=acc, val_acc=val_acc, rate=rate, lam=a.lam, log=np.array(log),
             T=a.T, t_in=a.tin, x_min=a.xmin, readout=a.readout, seed=a.seed, epochs=epochs,
             first_order=a.first_order, k_beta=a.kbeta)
    print("saved", path)


if __name__ == "__main__":
    main()
