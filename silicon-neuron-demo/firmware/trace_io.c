#include <stdio.h>
#include "nn_core.h"
#include "trace_io.h"

static snn_trace_t trace;          // 24 kB, static so it is not on the stack

static void send_hex(const uint8_t *p, int n)
{
    static const char hx[] = "0123456789abcdef";
    for (int i = 0; i < n; i++) {
        putchar(hx[p[i] >> 4]);
        putchar(hx[p[i] & 15]);
    }
}

int emit_trace(uint32_t idx)
{
    idx %= N_TEST;
    const uint8_t *x = &test_images[idx * N_IN];
    int32_t c[N_OUT];
    uint32_t sp;
    int pred = snn_infer(x, ENC_LATENCY, idx, &sp, c, &trace);
    int ann_pred = ann_infer(x, 0, NULL);

    // pick the three neurons that fired most on THIS image (ties -> lower index;
    // if fewer than three fired, the highest peak membrane fills the gaps)
    int pick[3] = {-1, -1, -1};
    for (int q = 0; q < 3; q++) {
        long best_score = -2147483647L;
        for (int j = 0; j < N_HID; j++) {
            if (j == pick[0] || j == pick[1]) continue;
            int cnt = 0;
            int32_t peak = -2147483647;
            for (int n = 0; n < T_STEPS; n++) {
                cnt += (trace.raster[n][j >> 3] >> (j & 7)) & 1;
                if (trace.u[n][j] > peak) peak = trace.u[n][j];
            }
            long score = cnt > 0 ? 1000000L * cnt - j
                                 : (peak > -1000000 ? (long)peak - 1000000L : -1999999L);
            if (score > best_score) { best_score = score; pick[q] = j; }
        }
    }
    printf("TN %d %d %d\n", pick[0], pick[1], pick[2]);
    printf("TI ");
    send_hex(trace.in_time, N_IN);
    printf("\n");
    for (int n = 0; n < T_STEPS; n++) {
        printf("TS %d ", n);
        send_hex(trace.raster[n], (N_HID + 7) / 8);
        printf("\n");
        printf("TU %d %ld %ld %ld\n", n, (long)trace.u[n][pick[0]], (long)trace.u[n][pick[1]],
               (long)trace.u[n][pick[2]]);
    }
    printf("TO");
    for (int k = 0; k < N_OUT; k++) printf(" %ld", (long)c[k]);
    printf(" %d %d %lu %ld\n", pred, test_labels[idx], (unsigned long)sp, (long)trace.theta);
    printf("TA %d\n", ann_pred);
    printf("END\n");
    return pred == test_labels[idx];
}
