"""
Single source of truth for every constant shared by training, the integer
reference simulator, the Pico firmware (via the generated header) and the
Raspberry Pi hologram display.

Notation follows the project formula sheet:
    U  membrane potential      I  synaptic current      S  spike (0/1)
    theta  threshold           alpha = exp(-dt/tau_syn) beta = exp(-dt/tau_mem)
    T  time steps per inference
"""

# ---------------------------------------------------------------- network --
N_IN = 784          # 28 x 28 MNIST pixels
N_HID = 300         # hidden layer width (matches the 784-300-10 dissertation MLP)
N_OUT = 10          # digits 0-9

# ------------------------------------------------------------ SNN dynamics --
T_STEPS = 16        # time steps per SNN inference
THETA = 1.0         # firing threshold (float units, before quantisation)

# alpha and beta are restricted to 1 - 2^-k so that the Pico can apply the
# leak with one shift and one subtract:  x <- x - (x >> k)   (no multiply).
K_ALPHA = 1         # alpha = 1 - 2^-1 = 0.5    (tau_syn ~ 1.44 steps)
K_BETA = 3          # beta  = 1 - 2^-3 = 0.875  (tau_mem ~ 7.5 steps)
ALPHA = 1.0 - 2.0 ** -K_ALPHA
BETA = 1.0 - 2.0 ** -K_BETA

SURROGATE_K = 10.0  # SuperSpike slope: sigma'(u) = 1 / (k|u| + 1)^2

# ----------------------------------------------------------- input coding --
# Latency (time-to-first-spike) code: brighter pixel -> earlier single spike.
#   t_i = ((255 - x_i) * T_IN) >> 8     for x_i >= X_MIN, otherwise no spike
T_IN = 10           # input spikes land in steps 0 .. T_IN-1
X_MIN = 64          # pixels dimmer than this never spike (~25% grey)

# Poisson (rate) code, the dissertation's reference encoding:
#   spike_i[n] = 1  if  (xorshift32() >> 24) < x_i     so  P = x_i / 256
POISSON_SEED = 0x2545F491

# ------------------------------------------------------------- deployment --
N_TEST_PICO = 500   # test images copied into Pico flash (500 x 784 B = 392 kB)
ANN_REQUANT_SHIFT = 24   # hidden-layer requantisation: h8 = (acc*M + 2^23) >> 24
ACT_PERCENTILE = 99.9    # hidden activation clipping percentile for the ANN
