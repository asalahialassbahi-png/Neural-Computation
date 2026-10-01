# Repeatability: 5 independent training runs (seeds 0, 1, 2, 3, 4)

Test accuracy is the int8 model the Pico runs, on all 10,000 MNIST test images, evaluated once per seed.

| quantity | mean +/- sd (95% CI) |
|---|---|
| ANN accuracy | 98.19% +/- 0.05 (95% CI 98.13 to 98.26) |
| SNN-E accuracy | 97.84% +/- 0.12 (95% CI 97.69 to 97.99) |
| accuracy gap ANN - SNN-E (points) | 0.35 pts +/- 0.12 (95% CI 0.21 to 0.50) |
| SNN-E Pico cycles / ANN-Z Pico cycles | 0.579x +/- 0.003 (95% CI 0.574 to 0.583) |
| SNN-E mean time steps used (of 8) | 1.36 +/- 0.08 (95% CI 1.25 to 1.46) |

Cycle counts here use the hand model (energy_model.py, within ~4-9% of the emulator, always low);
the ratio between models is what is compared. Manifest (versions, data and code hashes): repro.json.
