"""
Extra instruments around the measurement, all driven by the Pi (never the Pico,
so none of their current flows through the INA226):

    DS18B20 temperature probe   1-Wire on GPIO4 (pin 7), 4.7 kOhm pull-up to 3.3 V,
                                body taped beside the Pico in the tin
    "measuring" LED (green)     GPIO22 (pin 15) -> 330 Ohm -> LED -> GND
    "ready" LED (red)           GPIO23 (pin 16) -> 330 Ohm -> LED -> GND

The LEDs mirror the marker from the Pi side. Driving them from the Pico's marker
pin directly would add their ~5 mA to the very current being measured.

Temperature matters because the RP2040's leakage (idle) power rises with
temperature: bench.py logs it for every trial so it can be checked as a
controlled variable (analyse.py reports the range and its correlation with idle power).
"""
import glob
import os
import random
import threading
import time

W1_GLOB = "/sys/bus/w1/devices/28-*"


class Thermometer(threading.Thread):
    """Reads the DS18B20 every ~2 s in the background (a conversion takes 750 ms),
    so the latest value can be logged instantly with each trial."""

    def __init__(self, sim=False, period_s=2.0):
        super().__init__(daemon=True)
        self.sim, self.period_s = sim, period_s
        self.value = float("nan")
        self._dev = None
        if not sim:
            devs = sorted(glob.glob(W1_GLOB))
            if devs:
                self._dev = devs[0]
            else:
                print("DS18B20 not found (check 1-Wire is enabled and the 4.7k pull-up); logging NaN")
        self._rng = random.Random(3)
        self._t0 = time.time()

    def read_once(self):
        if self.sim:                              # slow warm-up drift, for testing only
            return 22.0 + 1.5 * (1 - 2.718 ** (-(time.time() - self._t0) / 600)) + self._rng.gauss(0, 0.03)
        if not self._dev:
            return float("nan")
        path = os.path.join(self._dev, "temperature")
        if os.path.exists(path):                  # newer kernels: millidegrees in one line
            return int(open(path).read().strip()) / 1000.0
        lines = open(os.path.join(self._dev, "w1_slave")).read().splitlines()
        if not lines or not lines[0].endswith("YES"):
            return float("nan")                   # CRC failed: keep the old value
        return int(lines[1].split("t=")[1]) / 1000.0

    def run(self):
        while True:
            try:
                v = self.read_once()
                if v == v:                        # not NaN
                    self.value = v
            except (OSError, ValueError, IndexError):
                pass
            time.sleep(self.period_s)


class StatusLEDs:
    """Green while a measured window is open, red otherwise. Silent no-op without GPIO."""

    def __init__(self, green_gpio=22, red_gpio=23, enabled=True):
        self.g = self.r = None
        if enabled:
            try:
                from gpiozero import LED
                self.g, self.r = LED(green_gpio), LED(red_gpio)
                self.r.on()
            except Exception as e:
                print("status LEDs unavailable:", e)

    def measuring(self, on):
        if self.g:
            (self.g.on if on else self.g.off)()
            (self.r.off if on else self.r.on)()


def board_from_info(info):
    """'pico' (RP2040, boots at 125 MHz) or 'pico2' (RP2350, 150 MHz) from the INFO line."""
    for tok in info.split():
        if tok.startswith("CLK="):
            khz = int(tok[4:])
            khz = khz // 1000 if khz > 10_000_000 else khz       # firmware may print Hz
            return "pico2" if khz >= 145_000 else "pico"
    return "unknown"
