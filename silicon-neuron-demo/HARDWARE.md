# Silicon Neuron — hardware, v2 (final prototype)

The v1 box measured the Pico's energy with one INA226. v2 keeps that and adds what
makes the number defensible in an EPQ: a **calibration** of the meter against
known loads, a **chip-only** power path, a **second processor** (Pico 2), a
**temperature** log, an **independent cross-check** (the USB inline meter), and
status LEDs that do not disturb the measurement. Every addition answers a question
an examiner can ask about the energy-vs-accuracy result.

| Question an examiner can ask | What answers it |
|---|---|
| How do you know the meter reads correctly? | `calibrate.py`: 470 Ω, 220 Ω and a 1 kΩ pot give 6 known currents (4–23 mA); multimeter V/R gives the truth; least squares gives the real shunt value, offset and linearity |
| Is the difference the network, or the Pico's regulator? | chip-only mode: MCP1700 LDO into the Pico's 3V3 pin with 3V3_EN grounded; a linear regulator passes its input current through, so chip power = V_rail × I |
| Is the result just a quirk of one chip? | the same firmware on a Pico 2 (RP2350, Cortex-M33) through the service hatch |
| Did temperature drift bias the comparison? | DS18B20 beside the Pico, logged every trial; `analyse.py` reports the range and its correlation with idle power |
| Does a second instrument agree? | the USB inline meter on the supply: its change between no load and a known load matches the INA226 |

## Upgrade parts (on top of the v1 list)

Prices are approximate (UK, Oct 2026) — check stock. Items marked *owned?* are
usually in a starter kit; buy them only if you don't have them.

| Item | Qty | ≈ £ | Where it goes | Why |
|---|---|---|---|---|
| Raspberry Pi Pico 2 **with headers** (Pico 2 H) | 1 | 6.00 | swapped into the tin (step 25) | second architecture; `firmware/uf2/silicon_neuron_pico2_rp2350.uf2` is ready |
| MCP1700-3302E/TO, 3.3 V LDO, TO-92 | 1 | 0.50 | tin b24-b26 | chip-only power path (1.6 µA quiescent, so it adds nothing measurable) |
| 1 µF ceramic capacitor | 2 | 0.30 *owned?* | tin d24-d25, e24-e26 | the MCP1700 needs them on its input and output to be stable |
| DS18B20 temperature sensor, TO-92 | 1 | 2.50 | tin i22-i24 | controlled variable |
| 4.7 kΩ resistor | 1 | *owned?* | calibration board g1-g2 | DS18B20 pull-up |
| 330 Ω resistor | 2 | *owned?* | calibration board g6-g9, g11-g14 | LED current limit (~4 mA) |
| 470 Ω and 220 Ω, 1% metal film | 1 + 1 | *owned?* | parked on the calibration board | reference loads (0.11 W max: 0.25 W parts are fine) |
| 1 kΩ potentiometer (breadboard pins) | 1 | *owned?* | calibration board b15-b17 | sweeps the load 4–23 mA for the linearity check (≤ 30 mW in the pot) |
| 5 mm LEDs, green + red | 2 | *owned?* | cap front, top left | measuring / ready, driven by the **Pi** |
| 170-point mini breadboard | 1 | 1.50 | roof, in front of the INA226 | the calibration board |
| Jumper wires, 20 cm, M-M and M-F | 1 pack each | 3.00 | everywhere | 7 tin jumpers (was 5) plus 11 short cap wires |
| Digital multimeter (UNI-T UT33 class) | 1 | 13.00 | calibration only | the reference for every calibration point |
| micro-USB to USB-A cable, second one | 1 | 3.00 | Pi power, through the USB meter | the Pi 12.5 W supply has a fixed lead, so the inline meter needs a USB-A charger + cable |
| 5 V USB-A charger, ≥ 2.4 A | 1 | *owned?* (6.00) | on the table | powers everything through the USB meter |
| USB inline power meter | 1 | **owned** | between the charger and the cable | whole-demo power and the cross-check |
| Beam-splitter film (optional) | 1 | 9.00 | on the acrylic | 30–50 % reflection in a lit room |

**New spend ≈ £30** (Pico 2, LDO, DS18B20, mini board, jumpers, multimeter, cable),
**≈ £45** if you also buy the resistor/capacitor/LED bits, a charger and the film —
well inside the £100.

Your existing **micro-USB to USB-A cable** has two jobs: flashing the Pico from the
laptop (step 4, then unplugged), and in the box it runs from the OTG adapter to the
screen's USB (step 20). That is why a second one is listed for the Pi's power.

## Wiring (v2, complete)

Pi header pins are physical pin numbers (pin 1 = 3.3 V, next to the SD card end).

**Pi ↔ INA226 (cap)**

| From | To | Colour |
|---|---|---|
| Pi pin 2 (5 V) | INA226 VIN+ | red |
| Pi pin 1 (3.3 V) | INA226 VCC | orange |
| Pi pin 3 (SDA) | INA226 SDA | white |
| Pi pin 5 (SCL) | INA226 SCL | grey |
| Pi pin 6 (GND) | INA226 GND | black |

**Calibration board (cap)** — front half: row 1 = OUT, row 3 = GND; back half: row 1 = 3V3, row 2 = DQ, row 3 = GND

| From | To | Colour |
|---|---|---|
| INA226 VIN- | a1 (OUT) | red, short |
| Pi pin 20 (GND) | a3 | black |
| e3 | f3 (bridge the two GND rows) | black link |
| Pi pin 17 (3.3 V) | j1 | orange |
| Pi pin 7 (GPIO4, 1-Wire) | j2 | grey |
| Pi pin 15 (GPIO22) | j6 → 330 Ω g6-g9 → j9 → green LED + | green |
| Pi pin 16 (GPIO23) | j11 → 330 Ω g11-g14 → j14 → red LED + | brown |
| h3 / i3 (GND) | green LED − / red LED − | black |
| 4.7 kΩ | g1 - g2 (pull-up, DQ to 3.3 V) | — |

**Tin ↔ cap (7 jumpers through the roof hole)**

| Tin end | Cap end | Colour |
|---|---|---|
| back + rail, row 10 (→ 1N5819 → VSYS) | calibration board c1 (OUT = meter VIN-) | red |
| a3 = Pico pin 3 (GND) | Pi pin 9 (GND) | black |
| a2 = Pico pin 2 (GP1, UART RX) | Pi pin 8 (TXD) | yellow |
| a1 = Pico pin 1 (GP0, UART TX) | Pi pin 10 (RXD) | green |
| a4 = Pico pin 4 (GP2, marker) | Pi pin 11 (GPIO17) | blue |
| j23 (DS18B20 DQ) | calibration board i2 (DQ) | grey |
| j24 (DS18B20 VDD) | calibration board i1 (3.3 V) | orange |

**Inside the tin**

| Part / link | Holes |
|---|---|
| 1N5819 | anode back + rail row 6, cathode (band) j2 (Pico pin 39, VSYS) |
| MCP1700 | b24 GND, b25 VIN, b26 VOUT, flat face to the front |
| 1 µF, 1 µF | d24-d25 (input), e24-e26 (output) |
| black link | a24 - a18 (regulator GND to Pico pin 18 GND) |
| DS18B20 | i22 GND, i23 DQ, i24 VDD, flat face to the front |
| black link | j22 - j18 (probe GND to Pico pin 23 GND) |

**Button and screen** — as v1: Pi pin 13 (GPIO27) and pin 14 (GND) to the button
legs (purple / black); mini-HDMI to the screen; OTG adapter + micro-USB to USB-A
cable to the screen's USB.

**Power** — USB charger → USB inline meter → micro-USB to USB-A cable → back
slot → Pi PWR port. The Pico is never powered from the Pi's 5 V directly: its
supply always passes the INA226 shunt.

## Chip-only power mode (through the service hatch)

1. Pull the hatch and slide the tin out (all jumpers stay plugged in).
2. Move the **red** jumper's tin end from the + rail (row 10) to **a25** (regulator VIN).
3. Add a white link **a26 → j5** (regulator VOUT → Pico pin 36, 3V3).
4. Add a grey link **j4 → j3** (Pico pin 37 3V3_EN → pin 38 GND: switches off the Pico's own regulator).
5. Measure the 3V3 pin with the multimeter (e.g. 3.297 V), slide the tin back, then
   `python3 bench.py --rail 3v3 --v-rail 3.297`.

Undo all three to return to the normal (whole-board) path. **Never fit the two
links while the red jumper is still on the + rail.**

## Session routine

```
python3 calibrate.py                     # ~5 min, lid off, writes results/calibration.json
python3 bench.py                         # whole board (VSYS), Pico or Pico 2 auto-detected
python3 bench.py --rail 3v3 --v-rail 3.297 --out results/bench_chip.csv
python3 analyse.py results/bench.csv --plots      # per board, temperature check, accuracy-vs-energy
```

The board is detected from the clock the firmware reports (RP2040 125 MHz, RP2350
150 MHz); check the printed `board:` line, and pass `--board pico2` if it is wrong.
