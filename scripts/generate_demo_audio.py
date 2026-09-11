#!/usr/bin/env python3
"""
generate_demo_audio.py — Generate clean, noisy, and enhanced audio demonstration files.

Processes speech files mixed with diverse defense/environmental noise types
(gunshot, siren, engine/helicopter, industrial drilling, environmental background)
across multiple SNR levels (-10, -5, 0, 5, 10 dB).

Produces structured WAV files in:
    results/demo_audio/<noise_category>/
        <sample_id>_clean.wav
        <sample_id>_noisy_<snr>db.wav
        <sample_id>_enhanced_<snr>db.wav

Also outputs an index table with PESQ, STOI, and SI-SNR scores.

Usage:
    python scripts/generate_demo_audio.py
    python scripts/generate_demo_audio.py --checkpoint models/checkpoints/checkpoint_best.pth
    python scripts/generate_demo_audio.py --snr-levels -10 -5 0 5 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models" / "gtcrn"))

# Auto-reexec with virtual environment if needed
if sys.prefix == sys.base_prefix:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = REPO_ROOT.parent / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import numpy as np
import soundfile as sf
import torch

from gtcrn import GTCRN
from sih26052.data.mixer import mix_at_snr
from sih26052.eval.full_metrics import compute_full_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("demo_generator")

NFFT = 512
HOP = 256
WIN_LEN = 512


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate clean/noisy/enhanced demo audio samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to GTCRN checkpoint. Defaults to best available.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(REPO_ROOT / "results" / "demo_audio"),
        help="Directory to save generated audio files",
    )
    parser.add_argument(
        "--snr-levels", type=float, nargs="+", default=[-10.0, -5.0, 0.0, 5.0, 10.0],
        help="SNR levels to generate (in dB)",
    )
    parser.add_argument(
        "--samples-per-category", type=int, default=2,
        help="Number of distinct speech/noise samples per category",
    )
    parser.add_argument(
        "--crop-samples", type=int, default=48000,  # 3.0 seconds at 16kHz
        help="Number of audio samples per clip (16kHz)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    return parser.parse_args()


def find_checkpoint(user_path: Optional[str]) -> Path:
    if user_path and Path(user_path).exists():
        return Path(user_path)

    candidates = [
        REPO_ROOT / "models" / "checkpoints" / "checkpoint_best.pth",
        REPO_ROOT / "models" / "gtcrn" / "checkpoints" / "model_trained_on_dns3.tar",
    ]
    # Also check experiments
    exp_dir = REPO_ROOT / "experiments"
    if exp_dir.exists():
        for sub in exp_dir.glob("*/checkpoint_best.pth"):
            candidates.insert(0, sub)

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError("No GTCRN checkpoint found. Please provide --checkpoint.")


def load_gtcrn_model(ckpt_path: Path, device: torch.device) -> GTCRN:
    model = GTCRN().to(device)
    logger.info("Loading model checkpoint from: %s", ckpt_path)
    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = state_dict.get("model", state_dict.get("model_state_dict", state_dict))
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def load_clean_and_noise_files() -> tuple[List[Path], Dict[str, List[Path]]]:
    clean_manifest = REPO_ROOT / "data" / "manifests" / "clean_train.json"
    noise_manifest = REPO_ROOT / "data" / "manifests" / "noise_pools.json"

    clean_files: List[Path] = []
    if clean_manifest.exists():
        with open(clean_manifest) as f:
            clean_files = [Path(p) for p in json.load(f)]
    else:
        vb_dir = REPO_ROOT / "Datasets" / "VoiceBank" / "clean_testset_wav"
        if vb_dir.exists():
            clean_files = sorted(list(vb_dir.glob("*.wav")))

    noise_pools: Dict[str, List[Path]] = {}
    if noise_manifest.exists():
        with open(noise_manifest) as f:
            pools = json.load(f)
            raw_pools = pools.get("test", pools.get("train", {}))
            for cat, paths in raw_pools.items():
                noise_pools[cat] = [Path(p) for p in paths]
    else:
        # Fallback: find any noise wavs in Datasets or data/processed
        for cat in ["gunfire", "environmental", "background"]:
            d = REPO_ROOT / "data" / "processed" / "defense_noise" / "train" / cat
            if d.exists():
                noise_pools[cat] = sorted(list(d.glob("*.wav")))

    return clean_files, noise_pools


def enhance_audio(model: GTCRN, noisy_wav: np.ndarray, window: torch.Tensor, device: torch.device) -> np.ndarray:
    orig_len = len(noisy_wav)
    with torch.no_grad():
        t = torch.from_numpy(noisy_wav.astype(np.float32)).unsqueeze(0).to(device)
        stft_c = torch.stft(t, n_fft=NFFT, hop_length=HOP, win_length=WIN_LEN, window=window, return_complex=True)
        stft_real = torch.view_as_real(stft_c)
        enh_stft_real = model(stft_real)
        enh_stft_c = torch.complex(enh_stft_real[..., 0], enh_stft_real[..., 1])
        enh_t = torch.istft(enh_stft_c, n_fft=NFFT, hop_length=HOP, win_length=WIN_LEN, window=window, length=orig_len)
        return enh_t.squeeze(0).cpu().numpy()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = find_checkpoint(args.checkpoint)
    model = load_gtcrn_model(ckpt_path, device)
    window = torch.hann_window(WIN_LEN).pow(0.5).to(device)

    clean_files, noise_pools = load_clean_and_noise_files()
    assert clean_files, "No clean speech files found for demonstration!"
    assert noise_pools, "No noise files found for demonstration!"

    logger.info("Found %d clean speech files and %d noise categories.", len(clean_files), len(noise_pools))

    rng = np.random.default_rng(42)
    demo_catalog = []

    for cat_name, n_files in noise_pools.items():
        if not n_files:
            continue
        cat_dir = out_dir / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)

        for s_idx in range(args.samples_per_category):
            clean_p = clean_files[rng.integers(0, len(clean_files))]
            noise_p = n_files[rng.integers(0, len(n_files))]

            try:
                raw_clean, sr_c = sf.read(str(clean_p), dtype="float32")
                raw_noise, sr_n = sf.read(str(noise_p), dtype="float32")
            except Exception as e:
                logger.warning("Failed to load audio: %s", e)
                continue

            if raw_clean.ndim > 1:
                raw_clean = raw_clean.mean(axis=1)
            if raw_noise.ndim > 1:
                raw_noise = raw_noise.mean(axis=1)

            # Crop or pad to crop_samples
            target_len = args.crop_samples
            if len(raw_clean) > target_len:
                st = rng.integers(0, len(raw_clean) - target_len + 1)
                clean = raw_clean[st:st + target_len]
            else:
                clean = np.pad(raw_clean, (0, target_len - len(raw_clean)))

            # Save clean reference once
            sample_id = f"sample_{s_idx + 1:02d}"
            clean_wav_name = f"{sample_id}_clean.wav"
            clean_wav_path = cat_dir / clean_wav_name
            sf.write(str(clean_wav_path), clean, 16000)

            for snr_db in args.snr_levels:
                noisy, clean_target = mix_at_snr(clean, raw_noise, snr_db, rng=rng)
                enhanced = enhance_audio(model, noisy, window, device)

                noisy_wav_name = f"{sample_id}_noisy_snr{int(snr_db):+03d}db.wav"
                enh_wav_name = f"{sample_id}_enhanced_snr{int(snr_db):+03d}db.wav"

                noisy_wav_path = cat_dir / noisy_wav_name
                enh_wav_path = cat_dir / enh_wav_name

                sf.write(str(noisy_wav_path), noisy, 16000)
                sf.write(str(enh_wav_path), enhanced, 16000)

                # Compute metrics
                m = compute_full_metrics(clean, noisy, enhanced, sr=16000)

                record = {
                    "category": cat_name,
                    "sample_id": sample_id,
                    "snr_db": snr_db,
                    "clean_path": str(clean_wav_path.relative_to(REPO_ROOT)),
                    "noisy_path": str(noisy_wav_path.relative_to(REPO_ROOT)),
                    "enhanced_path": str(enh_wav_path.relative_to(REPO_ROOT)),
                    "input_metrics": {
                        "pesq": round(m.pesq_input, 3) if m.pesq_input is not None else None,
                        "stoi": round(m.stoi_input, 3) if m.stoi_input is not None else None,
                        "si_sdr": round(m.si_sdr_input, 2),
                    },
                    "enhanced_metrics": {
                        "pesq": round(m.pesq_output, 3) if m.pesq_output is not None else None,
                        "stoi": round(m.stoi_output, 3) if m.stoi_output is not None else None,
                        "si_sdr": round(m.si_sdr_output, 2),
                    },
                    "improvement": {
                        "si_sdr_gain": round(m.si_sdr_improvement, 2),
                        "pesq_gain": round(m.pesq_output - m.pesq_input, 3) if (m.pesq_output is not None and m.pesq_input is not None) else None,
                    },
                }
                demo_catalog.append(record)
                logger.info(
                    "[%s | %s | %+03d dB] SI-SDR: %5.2f -> %5.2f dB (%+5.2f dB) | PESQ: %s -> %s",
                    cat_name, sample_id, int(snr_db),
                    m.si_sdr_input, m.si_sdr_output, m.si_sdr_improvement,
                    f"{m.pesq_input:.2f}" if m.pesq_input is not None else "N/A",
                    f"{m.pesq_output:.2f}" if m.pesq_output is not None else "N/A",
                )

    # Save demo catalog JSON
    catalog_path = out_dir / "index.json"
    with open(catalog_path, "w") as f:
        json.dump(demo_catalog, f, indent=2)
    logger.info("Saved demo audio index to %s", catalog_path)

    # Generate Markdown Summary
    readme_path = out_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("# GTCRN Speech Enhancement Audio Demos\n\n")
        f.write(f"Generated from checkpoint: `{ckpt_path.name}`\n\n")
        f.write("| Noise Type | Sample | SNR (dB) | Input SI-SDR | Enhanced SI-SDR | Gain (dB) | Input PESQ | Enhanced PESQ |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in demo_catalog:
            cat = r["category"]
            sid = r["sample_id"]
            snr = int(r["snr_db"])
            in_sdr = r["input_metrics"]["si_sdr"]
            out_sdr = r["enhanced_metrics"]["si_sdr"]
            gain = r["improvement"]["si_sdr_gain"]
            in_p = r["input_metrics"]["pesq"] or "-"
            out_p = r["enhanced_metrics"]["pesq"] or "-"
            f.write(f"| {cat} | {sid} | {snr:+d} | {in_sdr:.1f} | **{out_sdr:.1f}** | **{gain:+.1f}** | {in_p} | **{out_p}** |\n")
    logger.info("Saved demo audio README table to %s", readme_path)


if __name__ == "__main__":
    main()
