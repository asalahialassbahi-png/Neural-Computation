"""
Train the float models.

    python train.py --model ann                      -> weights/ann_float.npz
    python train.py --model snn --enc latency        -> weights/snn_latency_float.npz
    python train.py --model snn --enc poisson        -> weights/snn_poisson_float.npz
    python train.py --model snn --enc latency --lam 2.0 --tag lam2   (sparsity sweep)

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


def evaluate(model, x, y, kind, enc, rng, bs=500):
    correct, spikes = 0, 0.0
    for i in range(0, len(x), bs):
        xb, yb = x[i:i + bs], y[i:i + bs]
        if kind == "ann":
            logits, _ = model.forward(xb / 255.0)
        else:
            s_in = latency_spikes(xb) if enc == "latency" else poisson_spikes(xb, rng)
            logits, S = model.forward(s_in)
            spikes += S.sum()
        correct += (logits.argmax(1) == yb).sum()
    rate = spikes / (len(x) * C.T_STEPS * C.N_HID) if kind == "snn" else 0.0
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
    ap.add_argument("--seed", type=int, default=0)
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
        model = SNN(rng)
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
                s_in = latency_spikes(xb) if a.enc == "latency" else poisson_spikes(xb, rng)
                logits, S = model.forward(s_in)
                grads = model.backward(logits, yb, lam=a.lam)
            tot += cross_entropy_logits(logits, yb) * len(yb)
            opt.step(grads)
        acc, rate = evaluate(model, xva, yva, a.model, a.enc, rng)
        log.append((ep, tot / len(xtr), acc, rate))
        print(f"epoch {ep + 1:2d}/{epochs}  loss {tot / len(xtr):.4f}  val acc {acc * 100:.2f}%"
              + (f"  mean rate r = {rate:.4f} spikes/neuron/step  T*r = {rate * C.T_STEPS:.3f}"
                 if a.model == "snn" else "") + f"  ({time.time() - t0:.0f}s)")

    acc, rate = evaluate(model, xte, yte, a.model, a.enc, rng)
    print(f"TEST accuracy (float): {acc * 100:.2f}%" +
          (f"   T*r = {rate * C.T_STEPS:.3f}" if a.model == "snn" else ""))

    os.makedirs(WDIR, exist_ok=True)
    name = "ann" if a.model == "ann" else f"snn_{a.enc}"
    if a.tag:
        name += "_" + a.tag
    path = os.path.join(WDIR, name + "_float.npz")
    np.savez(path, **model.p, test_acc=acc, rate=rate, lam=a.lam, log=np.array(log))
    print("saved", path)


if __name__ == "__main__":
    main()
