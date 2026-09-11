#!/usr/bin/env python3
"""
create_defense_test_manifests.py — Build the 10 defense evaluation groups and unseen gunfire test set.

Groups:
    1. Background only
    2. Gunshot only
    3. Siren
    4. Engine
    5. Helicopter
    6. Machinery (drilling / jackhammer / chainsaw)
    7. Background + gunshot
    8. Background + siren
    9. Background + engine
    10. Background + multiple gunshots
    + Dedicated UNSEEN GUNFIRE test set (never seen during training)
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# Auto-reexec with virtual environment if needed
if sys.prefix == sys.base_prefix:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = repo_root.parent / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)


import numpy as np
import soundfile as sf
import yaml
from sih26052.data.manifest import ManifestWriter
from sih26052.data.mixer import mix_at_snr, mix_defense_sample, synthesize_gunfire_track

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Create defense test manifests and audio pairs")
    parser.add_argument("--config", type=Path, default=repo_root / "configs" / "gtcrn_defense.yaml")
    parser.add_argument("--out-audio", type=Path, default=repo_root / "data" / "defense_test_audio")
    parser.add_argument("--out-manifests", type=Path, default=repo_root / "data" / "defense_test_manifests")
    parser.add_argument("--pairs-per-group", type=int, default=30, help="Number of test pairs per group")
    args = parser.parse_args()

    args.out_audio.mkdir(parents=True, exist_ok=True)
    args.out_manifests.mkdir(parents=True, exist_ok=True)

    with open(repo_root / "data" / "manifests" / "clean_test.json", "r") as f:
        clean_test_files = [Path(p) for p in json.load(f)]

    with open(repo_root / "data" / "manifests" / "noise_pools.json", "r") as f:
        noise_pools = json.load(f)

    test_bg_files = [Path(p) for p in noise_pools["test"]["background"]]
    test_env_files = [Path(p) for p in noise_pools["test"]["environmental"]]
    test_gf_files = [Path(p) for p in noise_pools["test"]["gunfire"]]
    test_unseen_gf_files = [Path(p) for p in noise_pools["test"]["unseen_gunfire"]]

    logger.info("Loaded test files: clean=%d, bg=%d, env=%d, gf=%d, unseen_gf=%d",
                len(clean_test_files), len(test_bg_files), len(test_env_files),
                len(test_gf_files), len(test_unseen_gf_files))

    rng = np.random.default_rng(42)
    snr_levels = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
    target_samples = 32000  # 2.0 seconds at 16kHz
    n_pairs = args.pairs_per_group

    # Helper function to crop or pad
    def get_clean_crop(idx: int) -> tuple[np.ndarray, Path]:
        c_path = clean_test_files[idx % len(clean_test_files)]
        data, sr = sf.read(str(c_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if len(data) > target_samples:
            start = rng.integers(0, len(data) - target_samples + 1)
            return data[start : start + target_samples], c_path
        else:
            out = np.zeros(target_samples, dtype=np.float32)
            out[: len(data)] = data
            return out, c_path

    # Helper to find specific class files
    def filter_by_keywords(files: list[Path], keywords: list[str]) -> list[Path]:
        matched = [f for f in files if any(kw in f.name.lower() for kw in keywords)]
        return matched if matched else files

    siren_files = filter_by_keywords(test_env_files, ["siren"])
    engine_files = filter_by_keywords(test_bg_files + test_env_files, ["engine"])
    heli_files = filter_by_keywords(test_env_files, ["helicopter"])
    machinery_files = filter_by_keywords(test_env_files, ["drilling", "jackhammer", "chainsaw", "hand_saw"])

    groups = [
        ("01_background_only", "background_only", test_bg_files, "single", False),
        ("02_gunshot_only", "gunshot_only", test_gf_files, "single", False),
        ("03_siren", "siren", siren_files, "single", False),
        ("04_engine", "engine", engine_files, "single", False),
        ("05_helicopter", "helicopter", heli_files, "single", False),
        ("06_machinery", "machinery", machinery_files, "single", False),
        ("07_background_gunshot", "background_gunshot", test_bg_files, "composite_bg_gf", False),
        ("08_background_siren", "background_siren", test_bg_files, "composite_bg_siren", False),
        ("09_background_engine", "background_engine", test_bg_files, "composite_bg_engine", False),
        ("10_background_multiple_gunshots", "background_multiple_gunshots", test_bg_files, "composite_bg_multi_gf", False),
        ("unseen_gunfire", "unseen_gunfire", test_unseen_gf_files, "unseen_gunfire", True),
    ]

    master_manifest_path = args.out_manifests / "defense_all_test.jsonl"
    with ManifestWriter(master_manifest_path) as master_mw:
        for group_id, group_name, source_files, mode, is_unseen in groups:
            group_manifest_path = args.out_manifests / f"{group_name}.jsonl"
            logger.info("Generating group: %s (%d pairs)...", group_name, n_pairs)

            with ManifestWriter(group_manifest_path) as mw:
                for i in range(n_pairs):
                    clean_crop, c_path = get_clean_crop(i)
                    snr_db = float(snr_levels[i % len(snr_levels)])
                    prefix = f"{group_name}_{i:03d}"

                    noisy_wav = np.zeros_like(clean_crop)
                    clean_wav = clean_crop.copy()

                    if mode == "single" or is_unseen:
                        n_path = source_files[i % len(source_files)]
                        noise_data, _ = sf.read(str(n_path), dtype="float32")
                        if noise_data.ndim > 1:
                            noise_data = noise_data.mean(axis=1)
                        is_imp = ("gun" in group_name or is_unseen)
                        noisy_wav, clean_wav = mix_at_snr(clean_crop, noise_data, snr_db=snr_db, impulsive=is_imp, rng=rng)

                    elif mode == "composite_bg_gf":
                        bg_p = test_bg_files[i % len(test_bg_files)]
                        gf_p = test_gf_files[i % len(test_gf_files)]
                        bg_data, _ = sf.read(str(bg_p), dtype="float32")
                        gf_data, _ = sf.read(str(gf_p), dtype="float32")
                        if bg_data.ndim > 1: bg_data = bg_data.mean(axis=1)
                        if gf_data.ndim > 1: gf_data = gf_data.mean(axis=1)
                        noisy_wav, clean_wav, _ = mix_defense_sample(
                            clean_crop, background=bg_data, gunfire=gf_data, mixture_type="D", snr_db=snr_db, rng=rng
                        )

                    elif mode == "composite_bg_siren":
                        bg_p = test_bg_files[i % len(test_bg_files)]
                        sr_p = siren_files[i % len(siren_files)]
                        bg_data, _ = sf.read(str(bg_p), dtype="float32")
                        sr_data, _ = sf.read(str(sr_p), dtype="float32")
                        if bg_data.ndim > 1: bg_data = bg_data.mean(axis=1)
                        if sr_data.ndim > 1: sr_data = sr_data.mean(axis=1)
                        noisy_wav, clean_wav, _ = mix_defense_sample(
                            clean_crop, background=bg_data, event=sr_data, mixture_type="E", snr_db=snr_db, rng=rng
                        )

                    elif mode == "composite_bg_engine":
                        bg_p = test_bg_files[i % len(test_bg_files)]
                        eng_p = engine_files[i % len(engine_files)]
                        bg_data, _ = sf.read(str(bg_p), dtype="float32")
                        eng_data, _ = sf.read(str(eng_p), dtype="float32")
                        if bg_data.ndim > 1: bg_data = bg_data.mean(axis=1)
                        if eng_data.ndim > 1: eng_data = eng_data.mean(axis=1)
                        noisy_wav, clean_wav, _ = mix_defense_sample(
                            clean_crop, background=bg_data, event=eng_data, mixture_type="E", snr_db=snr_db, rng=rng
                        )

                    elif mode == "composite_bg_multi_gf":
                        bg_p = test_bg_files[i % len(test_bg_files)]
                        bg_data, _ = sf.read(str(bg_p), dtype="float32")
                        if bg_data.ndim > 1: bg_data = bg_data.mean(axis=1)

                        gf_samples = []
                        for k in range(3):
                            p_gf = test_gf_files[(i + k) % len(test_gf_files)]
                            w_gf, _ = sf.read(str(p_gf), dtype="float32")
                            if w_gf.ndim > 1: w_gf = w_gf.mean(axis=1)
                            gf_samples.append(w_gf)

                        gf_track, _ = synthesize_gunfire_track(gf_samples, target_len=target_samples, rng=rng, mode="multiple", min_shots=3, max_shots=4)
                        noisy_wav, clean_wav, _ = mix_defense_sample(
                            clean_crop, background=bg_data, gunfire=gf_track, mixture_type="D", snr_db=snr_db, rng=rng
                        )

                    noisy_path = args.out_audio / f"{prefix}_noisy.wav"
                    clean_path = args.out_audio / f"{prefix}_clean.wav"
                    sf.write(str(noisy_path), noisy_wav, 16000)
                    sf.write(str(clean_path), clean_wav, 16000)

                    entry_args = {
                        "noisy_path": str(noisy_path),
                        "clean_path": str(clean_path),
                        "snr_db": snr_db,
                        "noise_class": group_name,
                        "subset": "unseen_test" if is_unseen else group_name,
                        "duration_s": target_samples / 16000.0,
                    }
                    mw.write_entry(**entry_args)
                    master_mw.write_entry(**entry_args)

    logger.info("=" * 70)
    logger.info("Defense test sets successfully created:")
    logger.info("  Audio Directory: %s", args.out_audio)
    logger.info("  Manifests Directory: %s", args.out_manifests)
    logger.info("  Total groups: 10 + 1 unseen gunfire = 11 manifests")
    logger.info("  Master manifest: %s", master_manifest_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
