// Portable integer ANN / SNN inference core.
// Compiles for the Pico (RP2040 / RP2350) and for a desktop PC (host_test/),
// and is bit-exact with training/intsim.py.
#pragma once
#include <stdint.h>
#include "model_data.h"

enum { ENC_LATENCY = 0, ENC_POISSON = 1 };

typedef struct {
    uint8_t in_time[N_IN];                    // latency spike time per pixel (255 = none)
    uint8_t raster[T_STEPS][(N_HID + 7) / 8]; // hidden spikes, 1 bit per neuron
    int32_t u[T_STEPS][N_HID];                // membrane of every hidden neuron (19 kB)
    int32_t theta;                            // threshold in the same units
} snn_trace_t;

// Integer ANN. skip_zeros = 1 skips zero inputs and zero hidden activations
// (identical result, fewer operations). acc_out (10 values) may be NULL.
int ann_infer(const uint8_t *x, int skip_zeros, int32_t *acc_out);

// Integer SNN. enc = ENC_LATENCY or ENC_POISSON. img_index seeds the Poisson RNG.
// spikes_out (may be NULL) receives the total hidden spike count, c_out
// (may be NULL) the 10 summed output potentials, tr (may be NULL) the trace.
int snn_infer(const uint8_t *x, int enc, uint32_t img_index,
              uint32_t *spikes_out, int32_t *c_out, snn_trace_t *tr);

#if HAS_FAST
// SNN-E, the Pico-optimised spiking network: first-order LIF (no synaptic
// current), short latency code, evidence accumulated every step and an early
// exit once c_best - c_second >= exit_margin * (steps so far). exit_margin < 0
// disables the exit (always FAST_T steps). Bit-exact with intsim.snn_fast_int.
int snn_fast_infer(const uint8_t *x, int32_t exit_margin, uint32_t *spikes_out,
                   uint32_t *steps_out, int32_t *c_out);
#endif
