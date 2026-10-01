"""
Energy model: how much work each network does per inference, and what that
work costs on two very different pieces of hardware.

Two cost models, used side by side on purpose (docs/MATHS.md, section 8):

1. 45 nm CMOS operation energies (Horowitz, ISSCC 2014) — the model used in
   most SNN papers. A multiply-accumulate costs a multiply plus an add; a
   spike-driven accumulate costs only the add. On custom silicon this is where
   the famous "SNNs are cheaper" argument comes from.

       E = N_MAC * (E_mult + E_add) + N_AC * E_add      (+ memory, see below)

2. Raspberry Pi Pico (RP2040, Cortex-M0+) CPU cycles. The RP2040 has a
   SINGLE-CYCLE 32x32 multiplier, so a MAC is barely dearer than an add:
   the compiled inner loops are 13 cycles per MAC and 12 per accumulate
   (cycle_audit.py prints the disassembly and counts them). On the Pico the
   energy of an inference is, to first order,

       E = P_active * cycles / f_clk

   so the SNN can only win by doing FEWER operations, not cheaper ones.
   That is why SNN-E (first-order LIF, short latency code, early exit) exists.

The per-component cycle costs below were read off the -O2 disassembly of
firmware/nn_core.c for cortex-m0plus and then checked against an instruction-
level emulation of the same compiled code on real MNIST images
(cycle_audit.py; it refits them and reports the error).
"""

# ---------------------------------------------------- 45 nm energies (pJ)
# Horowitz, "Computing's energy problem (and what we can do about it)", ISSCC 2014
E45 = {"add8": 0.03, "mult8": 0.2, "add32": 0.1, "mult32": 3.1, "sram_read_32b_8kB": 5.0}


def energy_45nm(n_mac, n_ac, n_state_updates=0, bits=8):
    """Arithmetic-only energy in pJ (memory excluded: see MATHS.md 8.1 for why
    that flatters every model equally little on a microcontroller but a lot on
    custom silicon)."""
    add = E45["add8"] if bits == 8 else E45["add32"]
    mul = E45["mult8"] if bits == 8 else E45["mult32"]
    # a neuron state update = one shift-subtract leak + compare: ~2 adds of 32 bit
    return n_mac * (mul + add) + n_ac * add + n_state_updates * 2 * E45["add32"]


# ----------------------------------------------- Pico (Cortex-M0+) cycles
# cycles per component, from the compiled firmware (see the docstring)
CYC = {
    # ANN
    "ann_pixel_scan": 10,      # zero-skip test per input pixel
    "ann_row": 14,             # set-up per processed input row
    "mac": 13,                 # ldrsb, ldr, muls, adds, stmia, adds, cmp, bne
    "ann_requant": 20,         # ReLU + requantise per hidden neuron
    "ann_row2": 12,            # set-up per processed hidden row (layer 2)
    # SNN
    "ac": 12,                  # ldr, ldrsb, adds, adds, stmia, cmp, bne
    "lat_encode_pixel": 50,    # SNN-L: latency time + two-pass counting sort, per pixel
    "fast_encode_pixel": 16,   # SNN-E: one-pass bucket encoder, per pixel
    "snn_row": 15,             # set-up per input-spike row
    "l_neuron_step": 40,       # SNN-L: current leak (9) + membrane update (31) per neuron per step
    "fast_neuron_step": 17,    # SNN-E: fused threshold / leak / reset per neuron per step
    "out_step": 160,           # output layer leak + evidence sum, per step (10 neurons)
    "hid_row": 15,             # set-up per hidden spike row into the output layer
    "exit_check": 80,          # SNN-E: best and second-best of 10, per step
}
N_IN, N_HID, N_OUT = 784, 300, 10
F_CLK = 125e6                  # RP2040 default system clock (Hz)


def pico_cycles(kind, nz_in=0.0, nz_hid=0.0, events=0.0, hid_spikes=0.0, steps=0.0, T=16):
    """Estimated CPU cycles for one inference.

    kind: ann_dense | ann_skip | snn_latency | snn_fast
    nz_in / nz_hid : non-zero input pixels / hidden activations (ANN)
    events         : input spikes processed (SNN)
    hid_spikes     : hidden spikes (SNN)
    steps          : time steps actually run (SNN-E, with early exit)
    """
    c = CYC
    p = {}
    if kind in ("ann_dense", "ann_skip"):
        rows_in = N_IN if kind == "ann_dense" else nz_in
        rows_hid = N_HID if kind == "ann_dense" else nz_hid
        p["scan"] = N_IN * c["ann_pixel_scan"] if kind == "ann_skip" else 0
        p["layer1"] = rows_in * (c["ann_row"] + N_HID * c["mac"])
        p["activation"] = N_HID * c["ann_requant"]
        p["layer2"] = rows_hid * (c["ann_row2"] + N_OUT * c["mac"])
    elif kind == "snn_latency":
        p["encode"] = N_IN * c["lat_encode_pixel"]
        p["layer1"] = events * (c["snn_row"] + N_HID * c["ac"])
        p["neurons"] = T * N_HID * c["l_neuron_step"]
        p["layer2"] = hid_spikes * (c["hid_row"] + N_OUT * c["ac"]) + T * c["out_step"]
    elif kind == "snn_fast":
        p["encode"] = N_IN * c["fast_encode_pixel"]
        p["layer1"] = events * (c["snn_row"] + N_HID * c["ac"])
        p["neurons"] = steps * N_HID * c["fast_neuron_step"]
        p["layer2"] = hid_spikes * (c["hid_row"] + N_OUT * c["ac"]) + steps * (c["out_step"] + c["exit_check"])
    else:
        raise ValueError(kind)
    p["total"] = float(sum(p.values()))
    return p


def pico_energy_uJ(cycles, p_active_w=0.10, f_clk=F_CLK):
    """First-order energy: active power x time. Only the INA226 measures the real value."""
    return p_active_w * cycles / f_clk * 1e6


def break_even_events(nz_in, steps, c=CYC):
    """How many input spikes SNN-E may process and still beat the zero-skip ANN
    (layer 1 + per-step overhead only; MATHS.md eq. 8.4):
        e * (row + H ac) + steps * H * c_n  <  nz * (row + H mac)
    """
    ann = nz_in * (c["ann_row"] + N_HID * c["mac"])
    over = steps * N_HID * c["fast_neuron_step"]
    return (ann - over) / (c["snn_row"] + N_HID * c["ac"])
