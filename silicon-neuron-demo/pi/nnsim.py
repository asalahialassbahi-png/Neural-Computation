"""
Stand-alone integer model for the Pi (NumPy only), identical to the Pico
firmware. Used for (a) the --sim mode when no Pico is connected, and
(b) rebuilding the output-layer potentials from a streamed spike raster.
"""
import json

import numpy as np


class Bundle:
    def __init__(self, path="model_bundle.npz"):
        z = np.load(path)
        self.z = z
        self.cfg = json.loads(str(z["config_json"]))
        self.images = z["test_images"]
        self.labels = z["test_labels"]
        c = self.cfg
        self.H, self.T = c["N_HID"], c["T_STEPS"]
        self.ka, self.kb = c["K_ALPHA"], c["K_BETA"]
        self.has_poisson = "snn_poisson_W1" in z.files
        self.has_fast = "snn_fast_W1" in z.files

    # ------------------------------------------------------------ ANN
    def ann(self, x, full=False):
        z, c = self.z, self.cfg
        acc = x.astype(np.int64) @ z["ann_W1"].astype(np.int64) + z["ann_b1"]
        sh = c["ANN_REQUANT_SHIFT"]
        h = np.clip((acc * int(z["ann_M"]) + (1 << (sh - 1))) >> sh, 0, 255)
        h = np.where(acc > 0, h, 0)
        out = h @ z["ann_W2"].astype(np.int64) + z["ann_b2"]
        if full:
            return int(out.argmax()), h, out, acc
        return int(out.argmax()), h

    # ------------------------------------------------------------ SNN-E
    def fast_params(self):
        z = self.z
        return {k: int(z["snn_fast_" + k]) for k in ("T", "t_in", "x_min", "k_beta", "theta", "exit_margin", "n_min")}

    def snn_fast(self, x, exit=True):
        """SNN-E exactly as snn_fast_infer() runs it, with everything the display needs:
        raster (T,H), membrane (T,H) after input, evidence c per step, the step it exited."""
        z = self.z
        P = self.fast_params()
        W1 = z["snn_fast_W1"].astype(np.int32); W2 = z["snn_fast_W2"].astype(np.int32)
        H, T, kb, th = W1.shape[1], P["T"], P["k_beta"], P["theta"]
        xi = x.astype(np.int32)
        t_in = np.where(xi >= P["x_min"], ((255 - xi) * P["t_in"]) >> 8, 255).astype(np.uint8)
        U = np.zeros(H, np.int32); S = np.zeros(H, bool)
        U2 = np.zeros(10, np.int32); c = np.zeros(10, np.int64)
        raster = np.zeros((T, H), bool); u_all = np.zeros((T, H), np.int64)
        c_hist = np.zeros((T, 10), np.int64)
        steps, events = T, 0
        for n in range(T):
            U = U - (U >> kb) - th * S
            inp = t_in == n
            events += int(inp.sum())
            U = U + inp.astype(np.int32) @ W1
            S = U >= th
            raster[n], u_all[n] = S, U
            U2 = U2 - (U2 >> kb) + S.astype(np.int32) @ W2
            c += U2
            c_hist[n] = c
            if exit and P["exit_margin"] >= 0 and n + 1 >= P["n_min"] and n < T - 1:
                top = np.sort(c)
                if top[-1] - top[-2] >= P["exit_margin"] * (n + 1):
                    steps = n + 1
                    break
        pick = self.pick_neurons(raster[:steps], u_all[:steps])
        return {"pred": int(c.argmax()), "c": c, "raster": raster[:steps], "u": u_all[:steps, pick],
                "u_all": u_all[:steps], "in_time": t_in, "theta": th, "c_hist": c_hist[:steps],
                "spikes": int(raster[:steps].sum()), "trace_idx": pick, "in_spikes": events,
                "steps": steps, "T": T}

    # ------------------------------------------------------------ SNN
    def latency_times(self, x):
        c = self.cfg
        x = x.astype(np.int32)
        t = ((255 - x) * c["T_IN"]) >> 8
        return np.where(x >= c["X_MIN"], t, 255).astype(np.uint8)

    def input_spikes(self, x, enc, idx):
        T = self.T
        if enc == "latency":
            t = self.latency_times(x)
            return t[None, :] == np.arange(T)[:, None]
        s = (self.cfg["POISSON_SEED"] ^ ((idx * 0x9E3779B9) & 0xFFFFFFFF)) & 0xFFFFFFFF
        s = s or 1
        out = np.zeros((T, x.size), bool)
        for n in range(T):
            for i in np.nonzero(x)[0]:
                s ^= (s << 13) & 0xFFFFFFFF
                s ^= s >> 17
                s ^= (s << 5) & 0xFFFFFFFF
                out[n, i] = (s >> 24) < x[i]
        return out

    def snn(self, x, enc="latency", idx=0):
        """Returns dict: pred, c, raster (T,H) bool, u (T,3), in_time, theta."""
        z = self.z
        W1 = z[f"snn_{enc}_W1"].astype(np.int32)
        W2 = z[f"snn_{enc}_W2"].astype(np.int32)
        th = int(z[f"snn_{enc}_theta"])
        s_in = self.input_spikes(x, enc, idx)
        H, T = W1.shape[1], self.T
        I1 = np.zeros(H, np.int32); U1 = np.zeros(H, np.int32); S1 = np.zeros(H, bool)
        raster = np.zeros((T, H), bool); u_all = np.zeros((T, H), np.int64)
        for n in range(T):
            I1 = I1 - (I1 >> self.ka) + s_in[n].astype(np.int32) @ W1
            U1 = U1 - (U1 >> self.kb) + I1 - th * S1
            S1 = U1 >= th
            raster[n] = S1
            u_all[n] = U1
        pick = self.pick_neurons(raster, u_all)
        u = u_all[:, pick]
        c_hist, c = self.output_from_raster(raster, enc)
        return {"pred": int(c.argmax()), "c": c, "raster": raster, "u": u, "u_all": u_all,
                "in_time": self.latency_times(x) if enc == "latency" else np.full(784, 255, np.uint8),
                "theta": th, "c_hist": c_hist, "spikes": int(raster.sum()),
                "trace_idx": pick, "in_spikes": int(s_in.sum()), "s_in": s_in, "steps": T, "T": T}

    @staticmethod
    def pick_neurons(raster, u_all):
        """Same rule as the firmware: 3 most-firing neurons on this image
        (ties -> lower index), gaps filled by highest peak membrane."""
        cnt = raster.sum(0)
        peak = u_all.max(0)
        score = np.where(cnt > 0, 1_000_000 * cnt - np.arange(len(cnt)),
                         np.where(peak > -1_000_000, peak - 1_000_000, -1_999_999))
        pick = []
        for _ in range(3):
            s = score.copy()
            s[pick] = np.iinfo(np.int64).min
            pick.append(int(np.argmax(s)))      # argmax returns the first (lowest) index on ties
        return pick

    def output_from_raster(self, raster, enc="latency"):
        """Output integrator driven by hidden spikes -> (T,10) potentials, final c."""
        W2 = self.z[f"snn_{enc}_W2"].astype(np.int32)
        I2 = np.zeros(10, np.int32); U2 = np.zeros(10, np.int32); c = np.zeros(10, np.int64)
        hist = np.zeros((len(raster), 10), np.int64)
        for n, s in enumerate(raster):
            I2 = I2 - (I2 >> self.ka) + s.astype(np.int32) @ W2
            U2 = U2 - (U2 >> self.kb) + I2
            c += U2
            hist[n] = c
        return hist, c
