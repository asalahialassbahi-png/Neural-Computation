# Silicon Neuron — Pepper's-ghost demo

A spiking neural network (SNN) and a conventional ANN both run on a Raspberry Pi
Pico; an INA226 measures the Pico's power; a Raspberry Pi Zero 2 W turns the
spikes, membrane voltages and measured energy into a floating "hologram" in a
45-degree Pepper's-ghost box.

The full build guide (parts, optics, wiring, maths, demo script) is the Claude
Doc that accompanies this folder. This README is the command reference.

## Folder map

| Folder | What is in it |
|---|---|
| `training/` | NumPy-only training (hand-written BPTT), gradient check, quantisation, export, C-vs-Python bit-exact test |
| `firmware/` | Pico C firmware (Pico SDK). `uf2/` holds ready-built files for Pico (RP2040) and Pico 2 (RP2350) |
| `pi/` | Raspberry Pi software: INA226 driver, Pico link, benchmark protocol, statistics, hologram display |
| `blender/` | `build_demo.py` builds the enclosure in Blender, writes `cut_sheet.svg`, renders the four views |

## Order of operations

On your PC (Python 3.10+, `pip install numpy`):

```
cd training
python gradcheck.py                          # gate: errors shrink ~100x per row, last < 1e-6
python train.py --model ann                  # ~1 min
python train.py --model snn --enc latency --epochs 20
python train.py --model snn --enc poisson --epochs 20
python export.py                             # int8 weights -> firmware/generated + pi/model_bundle.npz
python verify_c.py                           # needs gcc; must print BIT-EXACT: PASS
```

Sparsity dial (dissertation Pareto curve): retrain the latency SNN with
`--lam 0.0 / 0.5 / 2 / 5 --tag lamX` and re-run export on each.

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
```

No hardware yet? Everything on the Pi side runs with `--sim` (made-up energy
numbers — never quote them):

```
python3 hologram.py --sim --windowed --no-mirror
python3 bench.py --sim --trials 3
```

Blender (4.2 or newer): Scripting tab → open `blender/build_demo.py` → Run
Script. Terminal: `blender -b -P blender/build_demo.py -- --render`.

Step-by-step assembly animation (LEGO-manual style, 22 steps: 0 cover, 1 kit, 2-21 the build):

- `blender/renders/assembly.mp4` — 1280x720, 24 fps, 88 s; one chapter per step, so VLC or
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
