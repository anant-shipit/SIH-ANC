# GTCRN Defense Noise Suppression — Ablation Comparison

| Experiment | Mode | Best Ep | PESQ | STOI | SI-SNR (dB) | ΔSI-SNR (dB) | ΔPESQ |
|---|---|---|---|---|---|---|---|
| `experiment_001_random_baseline` | random | 1 | **2.402** | 0.908 | 15.28 | **+5.89** | -0.259 |

## Per-SNR SI-SDR Improvement (ΔSI-SDR dB)

| Experiment | -10 dB | -5 dB | 0 dB | 5 dB | 10 dB | 15 dB |
|---|---|---|---|---|---|---|
| `experiment_001_random_baseline` | +12.48 | +8.72 | +8.74 | +6.22 | +5.02 | -1.53 |

## Per-Noise Type Performance (PESQ / SI-SDR dB)

| Experiment | background | environmental | gunfire | unseen_gunfire |
|---|---|---|---|---|
| `experiment_001_random_baseline` | 1.54 / 9.1 | 1.50 / 8.9 | 1.99 / 13.3 | 1.28 / 6.0 |