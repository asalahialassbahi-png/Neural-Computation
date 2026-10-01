"""
ANN and SNN in plain NumPy with hand-written backward passes.

The SNN backward pass is exactly the BPTT recursion on the formula sheet:

    eps[n] = sigma'(U[n] - theta) * (g_S[n] - theta * eps[n+1]) + beta * eps[n+1]
    psi[n] = eps[n] + alpha * psi[n+1]
    dL/dW  = sum_n  S_in[n]^T psi[n]
    g_S(prev layer)[n] = psi[n] W^T

with eps[T] = psi[T] = 0 and n running T-1 ... 0.

Weight layout is (n_in, n_out) so that row i holds every weight leaving
input i. That is the layout the Pico uses: one input spike = add one row.
"""
import numpy as np

from config import ALPHA, BETA, THETA, SURROGATE_K, N_IN, N_HID, N_OUT


# ================================================================ utilities
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)       # stable softmax: subtract max
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_logits(z, y):
    """L = -z_y + log sum_j e^{z_j}, computed stably (no epsilon fudge)."""
    m = z.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))[:, 0]
    return (lse - z[np.arange(len(y)), y]).mean()


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ====================================================================== ANN
class ANN:
    """784-300-10 ReLU MLP. Input x in [0, 1] (= pixel / 255)."""

    def __init__(self, rng, n_hid=N_HID, dtype=np.float32):
        he = lambda n_in, n_out: rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
        self.p = {"W1": he(N_IN, n_hid).astype(dtype), "b1": np.zeros(n_hid, dtype),
                  "W2": he(n_hid, N_OUT).astype(dtype), "b2": np.zeros(N_OUT, dtype)}

    def forward(self, x):
        p = self.p
        z1 = x @ p["W1"] + p["b1"]
        h = np.maximum(z1, 0)
        z2 = h @ p["W2"] + p["b2"]
        self.cache = (x, z1, h)
        return z2, h

    def backward(self, z2, y):
        x, z1, h = self.cache
        B = len(y)
        d2 = softmax(z2)
        d2[np.arange(B), y] -= 1                   # dL/dz2 = p - y
        d2 /= B
        d1 = (d2 @ self.p["W2"].T) * (z1 > 0)      # ReLU' = Heaviside(z)
        return {"W2": h.T @ d2, "b2": d2.sum(0),
                "W1": x.T @ d1, "b1": d1.sum(0)}


# ====================================================================== SNN
def surrogate_grad(u, k=SURROGATE_K):
    """SuperSpike: sigma'(u) = 1 / (k|u| + 1)^2 (used in the backward pass only)."""
    return 1.0 / (k * np.abs(u) + 1.0) ** 2


class SNN:
    """Two-layer LIF network with current-based synapses.

    Hidden layer (spiking):
        I[n] = alpha I[n-1] + S_in[n] W1
        U[n] = beta  U[n-1] + I[n] - theta S[n-1]      (reset by subtraction)
        S[n] = Heaviside(U[n] - theta)
    Output layer (non-spiking leaky integrator):
        I2[n] = alpha I2[n-1] + S[n] W2
        U2[n] = beta  U2[n-1] + I2[n]
        logits = (1/T) sum_n U2[n]
    """

    def __init__(self, rng, n_hid=N_HID, init_rate=0.02, dtype=np.float32,
                 spike_mode="hard", sig_k=100.0):
        # sigma = theta / sqrt(n_in * p) gives std(U) ~ theta   (formula sheet)
        s1 = THETA / np.sqrt(N_IN * init_rate)
        s2 = THETA / np.sqrt(n_hid * 0.1)
        self.p = {"W1": rng.normal(0, s1 * 0.5, (N_IN, n_hid)).astype(dtype),
                  "W2": rng.normal(0, s2 * 0.5, (n_hid, N_OUT)).astype(dtype)}
        self.spike_mode, self.sig_k = spike_mode, sig_k

    def _spike(self, u):
        if self.spike_mode == "hard":
            return (u >= 0).astype(u.dtype)
        return 1.0 / (1.0 + np.exp(-self.sig_k * u))      # gradient-check mode

    def _dspike(self, u, s):
        if self.spike_mode == "hard":
            return surrogate_grad(u)
        return self.sig_k * s * (1 - s)                     # exact derivative

    def forward(self, s_in):
        """s_in: (B, T, 784) spikes. Returns logits (B, 10) and hidden spikes (B, T, H)."""
        W1, W2 = self.p["W1"], self.p["W2"]
        B, T, _ = s_in.shape
        H = W1.shape[1]
        dt = W1.dtype
        I1 = np.zeros((B, H), dt); U1 = np.zeros((B, H), dt); S1 = np.zeros((B, H), dt)
        I2 = np.zeros((B, N_OUT), dt); U2 = np.zeros((B, N_OUT), dt)
        csum = np.zeros((B, N_OUT), dt)
        U_hist = np.empty((B, T, H), dt); S_hist = np.empty((B, T, H), dt)
        drive = (s_in.reshape(B * T, -1) @ W1).reshape(B, T, H)   # one big matmul
        for n in range(T):
            I1 = ALPHA * I1 + drive[:, n]
            U1 = BETA * U1 + I1 - THETA * S1
            S1 = self._spike(U1 - THETA)
            U_hist[:, n], S_hist[:, n] = U1, S1
            I2 = ALPHA * I2 + S1 @ W2
            U2 = BETA * U2 + I2
            csum += U2
        self.cache = (s_in, U_hist, S_hist)
        return csum / T, S_hist

    def backward(self, logits, y, lam=0.0):
        """BPTT. lam weights the firing-rate regulariser  lam * mean(S)."""
        s_in, U_hist, S_hist = self.cache
        W1, W2 = self.p["W1"], self.p["W2"]
        B, T, H = S_hist.shape

        g = softmax(logits)
        g[np.arange(B), y] -= 1
        g /= B                                   # dL/dlogits

        # ---- output layer: dL/dU2[n] = g/T  (directly, every step)
        eps2 = np.zeros_like(g); psi2 = np.zeros_like(g)
        psi2_hist = np.empty((B, T, N_OUT), g.dtype)
        for n in range(T - 1, -1, -1):
            eps2 = g / T + BETA * eps2
            psi2 = eps2 + ALPHA * psi2
            psi2_hist[:, n] = psi2
        dW2 = S_hist.reshape(B * T, H).T @ psi2_hist.reshape(B * T, N_OUT)
        gS = (psi2_hist.reshape(B * T, N_OUT) @ W2.T).reshape(B, T, H)
        gS += lam / (B * T * H)                  # d(lam*mean S)/dS

        # ---- hidden layer BPTT (formula sheet recursion)
        eps1 = np.zeros((B, H), g.dtype); psi1 = np.zeros((B, H), g.dtype)
        psi1_hist = np.empty((B, T, H), g.dtype)
        for n in range(T - 1, -1, -1):
            u = U_hist[:, n] - THETA
            eps1 = self._dspike(u, S_hist[:, n]) * (gS[:, n] - THETA * eps1) + BETA * eps1
            psi1 = eps1 + ALPHA * psi1
            psi1_hist[:, n] = psi1
        dW1 = s_in.reshape(B * T, -1).T @ psi1_hist.reshape(B * T, H)
        return {"W1": dW1, "W2": dW2}
