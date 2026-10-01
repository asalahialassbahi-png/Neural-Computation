// Silicon Neuron demo — Pico firmware (RP2040 Pico or RP2350 Pico 2).
//
// The Pico is the device under test. It stores the trained ANN and SNN in
// flash and runs them on command from the Raspberry Pi. While it computes it
// holds MARKER_PIN high, so the Pi knows exactly which INA226 power samples
// belong to the benchmark. Nothing is transmitted during a measured window.
//
// Serial protocol (UART0: GP0 = TX -> Pi RXD, GP1 = RX <- Pi TXD, 115200 8N1):
//   ?                     -> INFO ...
//   B <m> <start> <count> -> R <m> <count> <us> <correct> <spikes>
//        m = A  ANN dense         Z  ANN zero-skip
//            L  SNN latency code  P  SNN Poisson code
//            I  idle busy-wait for <count> microseconds (baseline)
//   T <idx>               -> TI/TS/TU/TO/TA lines then END   (not measured)
//   C <khz>               -> OK <khz>   change system clock (e.g. 48000..200000)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/uart.h"
#include "nn_core.h"
#include "trace_io.h"

#define MARKER_PIN 2
#ifdef PICO_DEFAULT_LED_PIN
#define LED_PIN PICO_DEFAULT_LED_PIN
#endif

static void do_bench(char mode, uint32_t start, uint32_t count)
{
    uint32_t correct = 0, spikes_total = 0;
#ifdef LED_PIN
    gpio_put(LED_PIN, 0);                    // LED must not add current to the window
#endif
    stdio_flush();
    uart_tx_wait_blocking(uart0);            // UART idle before the window opens
    sleep_ms(2);

    gpio_put(MARKER_PIN, 1);
    uint64_t t0 = time_us_64();
    if (mode == 'I') {
        // idle baseline: CPU spins for 'count' microseconds doing nothing useful
        while (time_us_64() - t0 < count) tight_loop_contents();
    } else {
        for (uint32_t k = 0; k < count; k++) {
            uint32_t idx = (start + k) % N_TEST;
            const uint8_t *x = &test_images[idx * N_IN];
            int pred;
            uint32_t sp = 0;
            switch (mode) {
            case 'A': pred = ann_infer(x, 0, NULL); break;
            case 'Z': pred = ann_infer(x, 1, NULL); break;
            case 'L': pred = snn_infer(x, ENC_LATENCY, idx, &sp, NULL, NULL); break;
            default:  pred = snn_infer(x, ENC_POISSON, idx, &sp, NULL, NULL); break;
            }
            correct += (pred == test_labels[idx]);
            spikes_total += sp;
        }
    }
    uint64_t dt = time_us_64() - t0;
    gpio_put(MARKER_PIN, 0);

    printf("R %c %lu %llu %lu %lu\n", mode, (unsigned long)count,
           (unsigned long long)dt, (unsigned long)correct, (unsigned long)spikes_total);
}

static void do_trace(uint32_t idx)
{
    int correct = emit_trace(idx);           // prints the TN/TI/TS/TU/TO/TA/END lines
#ifdef LED_PIN
    gpio_put(LED_PIN, correct);              // LED on = the SNN answered correctly
#else
    (void)correct;
#endif
}

int main(void)
{
    stdio_init_all();                        // UART only (see CMakeLists.txt)
    gpio_init(MARKER_PIN);
    gpio_set_dir(MARKER_PIN, GPIO_OUT);
    gpio_put(MARKER_PIN, 0);
#ifdef LED_PIN
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
#endif

    char line[64];
    int len = 0;
    printf("INFO boot\n");
    while (true) {
        int ch = getchar();
        if (ch == '\r') continue;
        if (ch != '\n') {
            if (len < (int)sizeof line - 1) line[len++] = (char)ch;
            continue;
        }
        line[len] = 0;
        len = 0;
        char m = 0;
        unsigned long a = 0, b = 0;
        if (line[0] == '?') {
            printf("INFO H=%d T=%d NTEST=%d POISSON=%d CLK=%lu\n", N_HID, T_STEPS, N_TEST,
                   HAS_POISSON, (unsigned long)clock_get_hz(clk_sys));
        } else if (sscanf(line, "B %c %lu %lu", &m, &a, &b) == 3) {
            if (m == 'P' && !HAS_POISSON) printf("ERR no poisson model\n");
            else do_bench(m, a, b);
        } else if (sscanf(line, "T %lu", &a) == 1) {
            do_trace(a);
        } else if (sscanf(line, "C %lu", &a) == 1) {
            bool ok = set_sys_clock_khz(a, false);
            stdio_init_all();                    // re-derive UART baud from new clock
            printf(ok ? "OK %lu\n" : "ERR clock %lu\n", a);
        } else if (line[0]) {
            printf("ERR ?\n");
        }
    }
}
