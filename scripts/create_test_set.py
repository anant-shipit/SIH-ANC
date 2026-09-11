#!/usr/bin/env python3
"""
create_test_set.py — Create a fixed test set using MAD military test noise and clean speech.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf
from sih26052.data.manifest import ManifestWriter
from sih26052.data.mixer import mix_at_snr


def main():
    repo_root = Path(__file__).resolve().parent.parent
    clean_dir = repo_root / "models" / "datasets" / "processed" / "clean"
    noise_dir = repo_root / "models" / "datasets" / "processed" / "noise" / "broadband" / "military" / "MAD_dataset" / "test"
    out_dir = repo_root / "data" / "test_audio"
    manifest_path = repo_root / "data" / "test_manifest.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_files = sorted([p for p in clean_dir.rglob("*.wav") if not p.name.startswith(".")])
    noise_files = sorted([p for p in noise_dir.rglob("*.wav") if not p.name.startswith(".")])

    print(f"Found {len(clean_files)} clean files and {len(noise_files)} test noise files.")

    # Select 100 fixed pairs across various SNRs
    rng = np.random.default_rng(42)
    n_pairs = 100
    target_len = 32000  # 2 seconds at 16kHz
    snrs = np.linspace(-5.0, 15.0, n_pairs)
    rng.shuffle(snrs)

    with ManifestWriter(manifest_path) as mw:
        for idx in range(n_pairs):
            c_file = clean_files[idx % len(clean_files)]
            n_file = noise_files[idx % len(noise_files)]

            clean, sr_c = sf.read(str(c_file), dtype="float32")
            noise, sr_n = sf.read(str(n_file), dtype="float32")

            if clean.ndim > 1:
                clean = clean.mean(axis=1)
            if noise.ndim > 1:
                noise = noise.mean(axis=1)

            # Crop or pad clean
            if len(clean) > target_len:
                start = rng.integers(0, len(clean) - target_len + 1)
                clean = clean[start : start + target_len]
            else:
                padded = np.zeros(target_len, dtype=np.float32)
                padded[: len(clean)] = clean
                clean = padded

            # Crop or repeat noise
            if len(noise) < target_len:
                repeats = int(np.ceil(target_len / len(noise)))
                noise = np.tile(noise, repeats)[:target_len]
            else:
                start = rng.integers(0, len(noise) - target_len + 1)
                noise = noise[start : start + target_len]

            snr_db = float(snrs[idx])
            noisy, clean_aligned = mix_at_snr(clean, noise, snr_db, impulsive=False, rng=rng)

            noisy_path = out_dir / f"test_{idx:04d}_noisy.wav"
            clean_path = out_dir / f"test_{idx:04d}_clean.wav"

            sf.write(str(noisy_path), noisy, 16000)
            sf.write(str(clean_path), clean_aligned, 16000)

            mw.write_entry(
                noisy_path=str(noisy_path),
                clean_path=str(clean_path),
                snr_db=snr_db,
                noise_class="military_broadband",
                subset="military_test",
                duration_s=target_len / 16000.0,
            )

    print(f"Generated {n_pairs} test pairs in {out_dir} and saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
