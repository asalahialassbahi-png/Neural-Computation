# Silicon Neuron — Pepper's-ghost demo

A spiking neural network (SNN) and a conventional ANN both run on a Raspberry Pi
Pico; an INA226 measures the Pico's power; a Raspberry Pi Zero 2 W turns the
spikes, membrane voltages and measured energy into a floating "hologram" in a
45-degree Pepper's-ghost box.

The full build guide (parts, optics, wiring, maths, demo script) is the Claude
Doc that accompanies this folder. **`HARDWARE.md` is the v2 hardware: the upgrade
parts list, the complete wiring tables, calibration and the chip-only power mode.**
This README is the command reference.

## Folder map

| Folder | What is in it |
|---|---|
| `training/` | NumPy-only training (hand-written BPTT), gradient check, quantisation, export, C-vs-Python bit-exact test |
| `firmware/` | Pico C firmware (Pico SDK). `uf2/` holds ready-built files for Pico (RP2040) and Pico 2 (RP2350) |
| `pi/` | Raspberry Pi software: INA226 driver, Pico link, benchmark protocol, statistics, hologram display |
| `blender/` | `build_demo.py` builds the enclosure in Blender, writes `cut_sheet.svg`, renders the four views |

## The five networks

| code | network | what it shows |
|---|---|---|
| A | ANN, dense | the yardstick: 238,200 multiply-adds per digit |
| Z | ANN, skips zero pixels | a fair, sparsity-aware ANN |
| L | SNN, latency code, 16 steps | the textbook SNN: on a Pico it narrowly LOSES to Z |
| P | SNN, rate (Poisson) code | why rate coding is the wrong code for a CPU |
| E | SNN-E, Pico-optimised | first-order LIF, short latency code, early exit: 1.86x less work than Z, 9.8x less than A |

The maths, derived from first principles: `docs/MATHS.md`. The evidence
(statistics, emulated cycles, figures): `results/PROOF.md`. Repeatability over
five seeds: `results/repro.md`.

## Order of operations

On your PC (Python 3.10+, `pip install numpy`; `matplotlib` for figures;
`unicorn capstone pyelftools` + `arm-none-eabi-gcc` for the cycle audit):

```
cd training
python gradcheck.py                          # gate: GRADIENT CHECK: PASS (ANN, SNN, SNN-E)
python train.py --model ann                  # ~1 min
python train.py --model snn --enc latency --epochs 20
python train.py --model snn --enc poisson --epochs 20
python sweep.py                              # SNN-E design sweep on validation data (~25 min)
cp weights/sweep/T8_tin3_x96_lam1.npz weights/snn_fast_float.npz   # the chosen SNN-E
python export.py                             # int8 weights -> firmware/generated + pi/model_bundle.npz
python verify_c.py                           # needs gcc; must print BIT-EXACT: PASS
python cycle_audit.py --n 200                # emulated Pico cycles per network -> results/, pi/model_costs.json
python prove.py                              # statistics + figures -> results/PROOF.md
python repro.py                              # 5 seeds from scratch -> results/repro.md (~20 min)
```

Firmware (skip if you use the prebuilt `firmware/uf2/*.uf2`): install the
Raspberry Pi Pico VS Code extension, open `firmware/`, pick board `pico` or
`pico2`, build. Flash: hold BOOTSEL, plug in USB, drag the `.uf2` onto RPI-RP2.
**Unplug USB afterwards** — during measurements the Pico is powered only
through the INA226.

On the Pi Zero 2 W (Raspberry Pi OS Lite, Bookworm), copy the `pi/` folder over, then:

```
cd pi && chmod +x setup_pi.sh && ./setup_pi.sh && sudo reboot
i2cdetect -y 1                  # INA226 at 0x40
python3 bench.py                # 30 interleaved rounds (~3 min) -> results/bench.csv
python3 analyse.py results/bench.csv --plots
python3 hologram.py             # starts automatically at boot after setup
                                # scroll wheel = network, click = view, button = next digit
```

Before each measuring session, calibrate the meter (lid off, ~5 min; see HARDWARE.md):
`python3 calibrate.py`. For the chip-only power path add `--rail 3v3 --v-rail <volts>` to
`bench.py`. Temperature, board (Pico / Pico 2) and power path are logged with every trial.

No hardware yet? Everything on the Pi side runs with `--sim` (made-up energy
numbers — never quote them):

```
python3 hologram.py --sim --windowed --no-mirror      # arrows: network/view, SPACE: digit, T: tour
python3 bench.py --sim --trials 3
```

Blender (4.2 or newer): Scripting tab → open `blender/build_demo.py` → Run
Script. Terminal: `blender -b -P blender/build_demo.py -- --render`.

Step-by-step assembly animation (LEGO-manual style, 28 steps: 0 cover, 1 kit, 2-24 the build,
25-26 the optional Pico 2 swap and chip-only mode through the service hatch, 27 power on):

- `blender/renders/assembly.mp4` — 1280x720, 24 fps, 112 s; one chapter per step, so VLC or
  QuickTime can jump straight to a step; a keyframe every second for smooth scrubbing.
- `blender/renders/assembly_booklet.pdf` — one page per step (the settled frame at its end).
- `blender/silicon_neuron_assembly.blend` — open it, press Numpad 0 to look
  through the camera, Space to play, and drag the timeline to go back. Every
  step is a named marker on the timeline (`Step 5: Schottky diode 1N5819`).
- Check framing first: `python blender/assembly_animation.py --preview` renders one
  captioned frame per step to `renders/preview/` and `renders/preview_sheet.jpg`
  (`--preview=40,89` adds a mid-flight frame per step).
- Rebuild or re-render: `python blender/assembly_animation.py --render`
  (`--every=2` for a 12 fps draft; frames already rendered are skipped, so it resumes).
  Captions, MP4 and booklet come from `blender/make_video.py` (Pillow + ffmpeg);
  run it alone to redo the captions. Workbench takes about 1 s per 1280x720 frame on
  a 4-core CPU with software OpenGL; a laptop GPU is faster.
- Never run two renders into the same folder; stop one by its PID, not `pkill -f`.

## Measured so far (from code, before hardware)

| Model | Test accuracy (int8, 10k images) | Synaptic ops per inference | T·r̄ |
|---|---|---|---|
| ANN 784-300-10, dense | 98.26% | 238,200 MAC | — |
| ANN, zero inputs skipped | 98.26% | ≈48,300 MAC | — |
| SNN, latency code, T=16 | 98.08% | ≈37,800 AC + 4,960 neuron updates | 0.20 |
| SNN, Poisson code, T=16 | 97.45% | ≈456,000 AC + 4,960 neuron updates | 1.05 |

Energy on the Pico is the one number the hardware must supply.
