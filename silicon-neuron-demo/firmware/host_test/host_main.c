// Host (PC) test harness: runs the exact firmware inference core on every
// test image stored for the Pico and prints one line per inference.
// training/verify_c.py compares these lines with the Python integer model.
//
//   make -C firmware/host_test && python training/verify_c.py
#include <stdio.h>
#include <string.h>
#include "nn_core.h"
#include "trace_io.h"

int main(int argc, char **argv)
{
    if (argc > 2 && strcmp(argv[1], "trace") == 0) {      // host_test trace N
        int n = 0;
        sscanf(argv[2], "%d", &n);
        for (int i = 0; i < n; i++) emit_trace((uint32_t)i);
        return 0;
    }
    for (int idx = 0; idx < N_TEST; idx++) {
        const uint8_t *x = &test_images[idx * N_IN];
        int32_t acc[N_OUT], c[N_OUT];
        uint32_t sp;

        int pa = ann_infer(x, 0, acc);
        int pz = ann_infer(x, 1, NULL);
        printf("A %d %d %d", idx, pa, pz);
        for (int k = 0; k < N_OUT; k++) printf(" %ld", (long)acc[k]);
        printf("\n");

        int pl = snn_infer(x, ENC_LATENCY, idx, &sp, c, NULL);
        printf("L %d %d %lu", idx, pl, (unsigned long)sp);
        for (int k = 0; k < N_OUT; k++) printf(" %ld", (long)c[k]);
        printf("\n");

#if HAS_POISSON
        int pp = snn_infer(x, ENC_POISSON, idx, &sp, c, NULL);
        printf("P %d %d %lu", idx, pp, (unsigned long)sp);
        for (int k = 0; k < N_OUT; k++) printf(" %ld", (long)c[k]);
        printf("\n");
#endif
    }
    return 0;
}
