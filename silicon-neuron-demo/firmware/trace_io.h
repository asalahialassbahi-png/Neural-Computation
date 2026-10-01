// Serial "trace" dump of one SNN inference for the hologram (portable C, stdio).
#pragma once
#include <stdint.h>

// Runs the latency SNN (with full trace) and the ANN on test image idx and
// prints the TN / TI / TS / TU / TO / TA / END lines. Returns 1 if the SNN
// prediction is correct.
int emit_trace(uint32_t idx);
