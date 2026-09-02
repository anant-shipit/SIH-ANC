#!/usr/bin/env python3
"""
eval_identity.py — Verify identity test: clean vs clean yields PESQ ≈ 4.5, SI-SNR > 80 dB.
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sih26052.eval.metrics import compute_all_metrics


def main():
    print("Running identity evaluation (clean -> clean test)...")
    sr = 16000
    duration_s = 3.0
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr

    # Synthetic multi-tone signal with envelope
    signal = (
        0.3 * np.sin(2 * np.pi * 200 * t)
        + 0.2 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.sin(2 * np.pi * 1500 * t)
    ).astype(np.float32)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    signal = (signal * envelope).astype(np.float32)

    res = compute_all_metrics(signal, signal, sr=sr)
    print(f"SI-SNR:  {res.si_snr:.2f} dB (Expected > 80 dB)")
    print(f"STOI:    {res.stoi if res.stoi is not None else 'N/A'}")
    print(f"PESQ:    {res.pesq if res.pesq is not None else 'N/A'}")

    passed = res.si_snr > 80.0 and (res.pesq is None or res.pesq > 4.0)
    print(f"Identity Test: {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
