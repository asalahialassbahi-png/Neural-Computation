"""
Calibrate the INA226 against known loads before a measuring session (~5 min).

    python3 calibrate.py              # interactive, real hardware
    python3 calibrate.py --sim        # made-up numbers, to try the procedure

Hardware (the calibration board in the cap, lid off):
    OUT row   = the meter's VIN- (the Pico's supply).  GND row = Pi GND.
    R_A = 470 Ohm, 1%, metal film                  -> about 10.6 mA at 5 V
    R_B = 220 Ohm, 1%, metal film, in series with the 1 kOhm pot -> 4 to 23 mA
The Pico draws roughly 20 mA, so the six points (4-23 mA) bracket it.

Method (no circuit has to be broken, so the multimeter only ever reads volts and ohms):
  1. With everything unpowered, measure R_A and R_B with the multimeter (ohms range).
  2. Unplug the tin's red jumper from OUT, so the Pico is unpowered.
  3. For each load the script averages the INA226 for 2 s while you read the
     voltage ACROSS the reference resistor with the multimeter (20 V range):
         I_true = V_dmm / R_dmm
  4. Least squares  V_shunt = R_eff * I_true + offset
     R_eff becomes the shunt value bench.py uses; the residuals show linearity.
Optional cross-check with the USB inline meter on the Pi's supply lead: the
change in its current reading between "no load" and "R_B, pot at zero" should
match the INA226's change to within the USB meter's resolution.

Writes results/calibration.json, read automatically by bench.py and hologram.py.
Every number in it traces back to two multimeter readings per point.
"""
import argparse
import datetime
import json
import math
import os
import random
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NOMINAL = 0.100
DMM_U_REL = 0.01          # cheap DMM: ~0.5% on DC volts and ~0.8% on ohms -> ~1% on I = V / R
POT_STEPS = ["fully anticlockwise (0 ohm)", "a quarter turn", "half way", "three quarters",
             "fully clockwise (1 kohm)"]


def ask_float(prompt, sim_value=None):
    if sim_value is not None:
        print(f"{prompt} {sim_value:.4g}  (sim)")
        return sim_value
    while True:
        s = input(prompt + " ").strip().replace(",", ".")
        try:
            return float(s)
        except ValueError:
            print("  type a number, e.g. 2.451")


def average_shunt(sensor, seconds=2.0):
    """Mean shunt voltage (V) and bus voltage (V) over a few seconds."""
    vs, vb = [], []
    t0 = time.time()
    while time.time() - t0 < seconds:
        v_bus, i, _ = sensor.read()
        vs.append(i * sensor.r)                   # back to raw shunt volts
        vb.append(v_bus)
        time.sleep(0.002)
    return float(np.mean(vs)), float(np.mean(vb))


class SimSensor:
    """Pretends to be an INA226 with a 0.1032 ohm shunt and 3 uV offset."""

    def __init__(self):
        self.r, self.i, self.rng = NOMINAL, 0.0, random.Random(7)

    def read(self):
        v = 5.02 - 0.3 * self.i
        vs = self.i * 0.1032 + 3e-6 + self.rng.gauss(0, 2e-6)
        return v, vs / self.r, v * vs / self.r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--chip", default="ina226")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "calibration.json"))
    a = ap.parse_args()

    if a.sim:
        sensor = SimSensor()
        r_a_true, r_b_true = 468.9, 219.3
    else:
        from ina226 import PowerSensor
        sensor = PowerSensor(chip=a.chip, r_shunt=NOMINAL, avg=16)
    from sensors import Thermometer
    thermo = Thermometer(sim=a.sim)
    temp_c = thermo.read_once()

    print(__doc__.split("Method")[0].strip(), "\n")
    print("Step 1 - everything unpowered. Multimeter on the ohms range:")
    r_a = ask_float("  R_A (470 ohm) measures, ohm:", r_a_true if a.sim else None)
    r_b = ask_float("  R_B (220 ohm) measures, ohm:", r_b_true if a.sim else None)

    print("\nStep 2 - power on, then unplug the tin's RED jumper from the OUT row.")
    a.sim or input("  press Enter when the Pico is disconnected ")
    if a.sim:
        sensor.i = 0.0
    v0, vb0 = average_shunt(sensor)
    print(f"  no load: shunt {v0 * 1e6:7.1f} uV, bus {vb0:.3f} V")

    usb0 = None
    if not a.sim:
        s = input("  USB meter current now, mA (Enter to skip the cross-check): ").strip()
        usb0 = float(s) if s else None

    points = []                                    # (I_true, V_shunt, label)
    loads = [("R_A alone: one leg in OUT, the other in GND", r_a, None)]
    loads += [(f"R_B + pot, pot {pos}", r_b, pos) for pos in POT_STEPS]
    print("\nStep 3 - known loads. Multimeter on DC volts, probes on the legs of the named resistor.")
    for k, (what, r_ref, pot) in enumerate(loads):
        print(f"\n  load {k + 1}/{len(loads)}: {what}")
        if pot is not None and k == 1:
            print("    (R_B from OUT to the pot's middle pin; pot's outer pin to GND)")
        if a.sim:
            r_pot = 1000.0 * (k - 1) / (len(POT_STEPS) - 1) if pot else 0.0
            sensor.i = 5.0 / (r_ref + r_pot + 0.4)
        else:
            input("    press Enter once it is plugged in ")
        vs, vb = average_shunt(sensor)
        v_dmm = ask_float(f"    voltage across the {'470' if r_ref is r_a else '220'} ohm resistor, V:",
                          sensor.i * r_ref * (1 + random.gauss(0, 0.002)) if a.sim else None)
        i_true = v_dmm / r_ref
        points.append((i_true, vs, what))
        print(f"    I_true {i_true * 1e3:6.2f} mA   INA226 (nominal shunt) {vs / NOMINAL * 1e3:6.2f} mA")
        if k == 1 and usb0 is not None:
            s = input("    USB meter current now, mA (Enter to skip): ").strip()
            if s:
                usb_delta = (float(s) - usb0) * 1e-3
                print(f"    USB meter change {usb_delta * 1e3:.1f} mA vs I_true {i_true * 1e3:.2f} mA "
                      f"({(usb_delta / i_true - 1) * 100:+.1f}%)")
                points[-1] = points[-1] + (usb_delta,)

    I = np.array([p[0] for p in points])
    V = np.array([p[1] for p in points])
    A = np.vstack([I, np.ones_like(I)]).T
    (r_eff, off), res, *_ = np.linalg.lstsq(A, V, rcond=None)
    fit = A @ np.array([r_eff, off])
    resid_rel = (V - fit) / V.max()
    n = len(I)
    s2 = float(((V - fit) ** 2).sum() / max(n - 2, 1))
    se_slope = math.sqrt(s2 / ((I - I.mean()) ** 2).sum())
    u_rel = math.sqrt((se_slope / r_eff) ** 2 + DMM_U_REL ** 2)

    print("\nResult")
    print(f"  effective shunt R_eff = {r_eff:.5f} ohm  ({(r_eff / NOMINAL - 1) * 100:+.2f}% vs the 0.1 ohm label)")
    print(f"  offset {off * 1e6:+.1f} uV  (= {off / r_eff * 1e6:+.0f} uA)")
    print(f"  linearity: worst point {np.abs(resid_rel).max() * 100:.2f}% of full scale over "
          f"{I.min() * 1e3:.1f}-{I.max() * 1e3:.1f} mA")
    print(f"  relative uncertainty of R_eff {u_rel * 100:.2f}% (fit {se_slope / r_eff * 100:.2f}%, multimeter "
          f"{DMM_U_REL * 100:.0f}%)")
    if abs(r_eff / NOMINAL - 1) > 0.05:
        print("  WARNING: more than 5% from 0.1 ohm - check the module really has an R100 shunt and the wiring")

    out = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "sim": a.sim,
           "r_shunt_ohm": float(r_eff), "offset_v": float(off), "u_rel": u_rel,
           "linearity_max_dev_fs": float(np.abs(resid_rel).max()), "temp_c": temp_c,
           "r_a_ohm": r_a, "r_b_ohm": r_b,
           "points": [{"what": p[2], "i_true_a": p[0], "v_shunt_v": p[1],
                       **({"usb_delta_a": p[3]} if len(p) > 3 else {})} for p in points]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", a.out, "- plug the tin's red jumper back into OUT before running bench.py")


if __name__ == "__main__":
    main()
