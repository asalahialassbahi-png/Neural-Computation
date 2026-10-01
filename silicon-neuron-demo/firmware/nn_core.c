// Integer ANN / SNN inference — see nn_core.h and training/intsim.py.
//
// Every right shift below is applied to signed int32. GCC for ARM (and for
// x86) implements that as an arithmetic shift (rounds towards -infinity),
// which is exactly what NumPy's >> does, so results are bit-identical.
#include <string.h>
#include "nn_core.h"

#define NO_SPIKE 255u

// ======================================================================= ANN
int ann_infer(const uint8_t *x, int skip_zeros, int32_t *acc_out)
{
    static int32_t acc[N_HID];
    static uint8_t h8[N_HID];
    int32_t out[N_OUT];

    // layer 1: acc = b1 + sum_i x_i * W1[i, :]   (row-wise: one pixel at a time)
    for (int j = 0; j < N_HID; j++) acc[j] = ann_b1[j];
    for (int i = 0; i < N_IN; i++) {
        int32_t xi = x[i];
        if (skip_zeros && xi == 0) continue;
        const int8_t *row = &ann_W1[i * N_HID];
        for (int j = 0; j < N_HID; j++) acc[j] += xi * row[j];
    }
    // ReLU + requantise to uint8:  h8 = clamp((acc*M + 2^(S-1)) >> S, 0, 255)
    for (int j = 0; j < N_HID; j++) {
        int32_t a = acc[j];
        if (a <= 0) { h8[j] = 0; continue; }
        int64_t v = ((int64_t)a * ANN_M + (1LL << (ANN_SHIFT - 1))) >> ANN_SHIFT;
        h8[j] = (uint8_t)(v > 255 ? 255 : v);
    }
    // layer 2
    for (int k = 0; k < N_OUT; k++) out[k] = ann_b2[k];
    for (int j = 0; j < N_HID; j++) {
        int32_t hj = h8[j];
        if (skip_zeros && hj == 0) continue;
        const int8_t *row = &ann_W2[j * N_OUT];
        for (int k = 0; k < N_OUT; k++) out[k] += hj * row[k];
    }
    int best = 0;
    for (int k = 1; k < N_OUT; k++) if (out[k] > out[best]) best = k;
    if (acc_out) memcpy(acc_out, out, sizeof out);
    return best;
}

// ======================================================================= SNN
static inline uint32_t xorshift32(uint32_t *s)
{
    uint32_t v = *s;
    v ^= v << 13;
    v ^= v >> 17;
    v ^= v << 5;
    return *s = v;
}

int snn_infer(const uint8_t *x, int enc, uint32_t img_index,
              uint32_t *spikes_out, int32_t *c_out, snn_trace_t *tr)
{
    static int32_t I1[N_HID], U1[N_HID];
    static uint8_t S1[N_HID];
    static uint16_t fired[N_HID];                // indices of hidden spikes this step
    static uint16_t order[N_IN];                 // latency: pixels sorted by spike time
    uint16_t start[T_STEPS + 1];
    int32_t I2[N_OUT] = {0}, U2[N_OUT] = {0}, c[N_OUT] = {0};
    uint32_t total_spikes = 0, rng = 0;

    const int8_t *W1 = enc == ENC_LATENCY ? snn_lat_W1 : snn_poi_W1;
    const int8_t *W2 = enc == ENC_LATENCY ? snn_lat_W2 : snn_poi_W2;
    const int32_t theta = enc == ENC_LATENCY ? SNN_LAT_THETA : SNN_POI_THETA;

    memset(I1, 0, sizeof I1);
    memset(U1, 0, sizeof U1);
    memset(S1, 0, sizeof S1);

    if (enc == ENC_LATENCY) {
        // t_i = ((255 - x_i) * T_IN) >> 8 for x_i >= X_MIN, then a counting
        // sort so each step touches only the pixels that fire in it.
        uint16_t count[T_STEPS] = {0};
        for (int i = 0; i < N_IN; i++) {
            uint8_t t = x[i] >= X_MIN ? (uint8_t)(((255 - x[i]) * T_IN) >> 8) : NO_SPIKE;
            if (tr) tr->in_time[i] = t;
            if (t < T_STEPS) count[t]++;
        }
        start[0] = 0;
        for (int n = 0; n < T_STEPS; n++) start[n + 1] = start[n] + count[n];
        uint16_t fill[T_STEPS];
        memcpy(fill, start, sizeof fill);
        for (int i = 0; i < N_IN; i++) {
            if (x[i] < X_MIN) continue;
            uint8_t t = (uint8_t)(((255 - x[i]) * T_IN) >> 8);
            order[fill[t]++] = (uint16_t)i;
        }
    } else {
        rng = POISSON_SEED ^ (img_index * 0x9E3779B9u);
        if (rng == 0) rng = 1;
        if (tr) memset(tr->in_time, NO_SPIKE, N_IN);
    }
    if (tr) { memset(tr->raster, 0, sizeof tr->raster); tr->theta = theta; }

    for (int n = 0; n < T_STEPS; n++) {
        // ---- synaptic current: I <- alpha*I + sum of rows for input spikes
        for (int j = 0; j < N_HID; j++) I1[j] -= I1[j] >> K_ALPHA;
        if (enc == ENC_LATENCY) {
            for (int k = start[n]; k < start[n + 1]; k++) {
                const int8_t *row = &W1[order[k] * N_HID];
                for (int j = 0; j < N_HID; j++) I1[j] += row[j];      // accumulate only
            }
        } else {
            for (int i = 0; i < N_IN; i++) {
                uint32_t xi = x[i];
                if (xi == 0) continue;
                if ((xorshift32(&rng) >> 24) < xi) {
                    const int8_t *row = &W1[i * N_HID];
                    for (int j = 0; j < N_HID; j++) I1[j] += row[j];
                }
            }
        }
        // ---- membrane: U <- beta*U + I - theta*S_prev ; S = [U >= theta]
        int nf = 0;
        for (int j = 0; j < N_HID; j++) {
            int32_t u = U1[j] - (U1[j] >> K_BETA) + I1[j] - (S1[j] ? theta : 0);
            U1[j] = u;
            S1[j] = u >= theta;
            if (S1[j]) fired[nf++] = (uint16_t)j;
        }
        total_spikes += nf;
        // ---- output leaky integrator, event driven on hidden spikes
        for (int k = 0; k < N_OUT; k++) I2[k] -= I2[k] >> K_ALPHA;
        for (int f = 0; f < nf; f++) {
            const int8_t *row = &W2[fired[f] * N_OUT];
            for (int k = 0; k < N_OUT; k++) I2[k] += row[k];
        }
        for (int k = 0; k < N_OUT; k++) {
            U2[k] = U2[k] - (U2[k] >> K_BETA) + I2[k];
            c[k] += U2[k];
        }
        if (tr) {
            for (int f = 0; f < nf; f++) tr->raster[n][fired[f] >> 3] |= (uint8_t)(1u << (fired[f] & 7));
            memcpy(tr->u[n], U1, sizeof U1);
        }
    }
    int best = 0;
    for (int k = 1; k < N_OUT; k++) if (c[k] > c[best]) best = k;
    if (spikes_out) *spikes_out = total_spikes;
    if (c_out) memcpy(c_out, c, sizeof c);
    return best;
}

// ===================================================================== SNN-E
#if HAS_FAST
int snn_fast_infer(const uint8_t *x, int32_t exit_margin, uint32_t *spikes_out,
                   uint32_t *steps_out, int32_t *c_out)
{
    static int32_t U[N_HID];                     // membrane, stored ALREADY leaked and reset
    static uint16_t fired[N_HID];
    static uint16_t bucket[FAST_T_IN][N_IN];     // pixels spiking at each step (one-pass encoder)
    uint16_t nb[FAST_T_IN];
    int32_t U2[N_OUT] = {0}, c[N_OUT] = {0};
    uint32_t total = 0;
    int steps = FAST_T;

    memset(U, 0, sizeof U);
    memset(nb, 0, sizeof nb);
    // ---- latency code in ONE pass: t = ((255 - x) * T_IN) >> 8 for x >= X_MIN
    for (int i = 0; i < N_IN; i++) {
        uint32_t xi = x[i];
        if (xi < FAST_X_MIN) continue;
        uint32_t t = ((255u - xi) * FAST_T_IN) >> 8;
        bucket[t][nb[t]++] = (uint16_t)i;
    }
    for (int n = 0; n < FAST_T; n++) {
        // ---- input spikes add their weight rows straight into the membrane
        int ne = n < FAST_T_IN ? nb[n] : 0;     // t < T_IN always, so later steps have no input
        for (int k = 0; k < ne; k++) {
            const int8_t *row = &fast_W1[bucket[n][k] * N_HID];
            for (int j = 0; j < N_HID; j++) U[j] += row[j];
        }
        // ---- one fused pass: spike test, then leak and reset ready for step n+1
        //      S = [U >= theta];  U <- U - (U >> K) - theta*S
        int nf = 0;
        for (int j = 0; j < N_HID; j++) {
            int32_t u = U[j];
            int32_t v = u - (u >> FAST_K_BETA);
            if (u >= FAST_THETA) { fired[nf++] = (uint16_t)j; v -= FAST_THETA; }
            U[j] = v;
        }
        total += nf;
        // ---- output: U2 <- U2 - (U2 >> K) + rows of fired neurons; c += U2
        for (int k = 0; k < N_OUT; k++) U2[k] -= U2[k] >> FAST_K_BETA;
        for (int f = 0; f < nf; f++) {
            const int8_t *row = &fast_W2[fired[f] * N_OUT];
            for (int k = 0; k < N_OUT; k++) U2[k] += row[k];
        }
        for (int k = 0; k < N_OUT; k++) c[k] += U2[k];
        // ---- early exit: is the leading digit far enough ahead of the runner-up?
        if (exit_margin >= 0 && n + 1 >= FAST_N_MIN && n < FAST_T - 1) {
            int32_t m1 = c[0], m2 = INT32_MIN;
            for (int k = 1; k < N_OUT; k++) {
                if (c[k] > m1) { m2 = m1; m1 = c[k]; }
                else if (c[k] > m2) m2 = c[k];
            }
            if ((int64_t)m1 - m2 >= (int64_t)exit_margin * (n + 1)) { steps = n + 1; break; }
        }
    }
    int best = 0;
    for (int k = 1; k < N_OUT; k++) if (c[k] > c[best]) best = k;
    if (spikes_out) *spikes_out = total;
    if (steps_out) *steps_out = (uint32_t)steps;
    if (c_out) memcpy(c_out, c, sizeof c);
    return best;
}
#endif
