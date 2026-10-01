"""
Input encoders. Both are used identically in training (float) and in the
integer reference simulator / Pico firmware.
"""
import numpy as np

from config import N_IN, T_STEPS, T_IN, X_MIN, POISSON_SEED

NO_SPIKE = 255  # sentinel spike time meaning "this pixel never fires"


def latency_times(x_u8, t_in=T_IN, x_min=X_MIN):
    """x_u8: (..., 784) uint8 -> spike time per pixel (uint8), NO_SPIKE if silent.

    t_i = ((255 - x_i) * t_in) >> 8  for x_i >= x_min.
    Brightest pixels (255) fire at step 0; x_min-level pixels fire last.
    """
    x = x_u8.astype(np.int32)
    t = ((255 - x) * t_in) >> 8
    t = np.where(x >= x_min, t, NO_SPIKE)
    return t.astype(np.uint8)


def latency_spikes(x_u8, T=T_STEPS, t_in=T_IN, x_min=X_MIN):
    """Dense spike tensor (B, T, 784) float32 for training."""
    t = latency_times(x_u8, t_in, x_min)          # (B, 784)
    steps = np.arange(T, dtype=np.uint8)[None, :, None]
    return (t[:, None, :] == steps).astype(np.float32)


def poisson_spikes(x_u8, rng, T=T_STEPS):
    """Bernoulli/Poisson rate code with P(spike) = x/256 each step (training)."""
    p = x_u8.astype(np.float32) / 256.0
    u = rng.random((x_u8.shape[0], T, N_IN), dtype=np.float32)
    return (u < p[:, None, :]).astype(np.float32)


class XorShift32:
    """Marsaglia xorshift32, bit-identical to the C version in nn_core.c."""

    def __init__(self, seed=POISSON_SEED):
        self.s = seed & 0xFFFFFFFF

    def next(self):
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        self.s = s
        return s
