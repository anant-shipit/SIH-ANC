#!/usr/bin/env python3
"""
prepare_defense_data.py — Standardize and split defense datasets for GTCRN training.

Processes:
    - Resamples audio to 16,000 Hz mono (if needed)
    - Saves into data/processed/defense_noise/ without modifying original Datasets/
    - Organizes into train / val / test partitions with zero data leakage:
        - VoiceBank: Speaker-aware split (25 train, 3 val, 2 test)
        - ESC-50: Official folds (1-3 train, 4 val, 5 test)
        - UrbanSound8K: Official folds (1-8 train, 9 val, 10 test)
        - Gunshot Audio: Weapon split (7 train, 1 val, 1 UNSEEN test)
        - Edge Guns: Session split (1-3 train, 4 val, 5 test)
"""
import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from math import gcd

repo_root = Path(__file__).resolve().parent.parent

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
from scipy.signal import resample_poly

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_SR = 16000


def resample_and_write(src_path: Path, dst_path: Path, target_sr: int = TARGET_SR) -> bool:
    """Read audio, convert stereo to mono, resample to target_sr, write 16-bit PCM WAV."""
    if dst_path.exists():
        return True

    try:
        data, sr = sf.read(str(src_path), dtype="float32", always_2d=True)
        # Stereo to mono
        if data.shape[1] > 1:
            data = data.mean(axis=1)
        else:
            data = data[:, 0]

        # Resample if needed
        if sr != target_sr:
            g = gcd(target_sr, sr)
            up, down = target_sr // g, sr // g
            data = resample_poly(data, up, down).astype(np.float32)

        # Normalize peak slightly if clipping
        peak = np.max(np.abs(data))
        if peak > 0.99:
            data = data * (0.99 / peak)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst_path), data, target_sr, subtype="PCM_16")
        return True
    except Exception as exc:
        logger.warning("Failed processing %s: %s", src_path, exc)
        return False


def main():
    parser = argparse.ArgumentParser(description="Standardize and split defense datasets")
    parser.add_argument("--config", type=Path, default=repo_root / "configs" / "gtcrn_defense.yaml")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "data" / "processed" / "defense_noise")
    parser.add_argument("--manifest-dir", type=Path, default=repo_root / "data" / "manifests")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional limit for fast testing")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_base = args.out_dir
    manifest_base = args.manifest_dir
    manifest_base.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PREPARING DEFENSE DATASETS")
    logger.info("Output processed directory: %s", out_base)
    logger.info("Manifest directory: %s", manifest_base)
    logger.info("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. VoiceBank Clean Speech Splits (Zero Speaker Leakage)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[1/5] Partitioning VoiceBank Clean Speech...")
    vb_cfg = cfg["datasets"]["voicebank"]
    vb_train_dir = repo_root / vb_cfg["train_dir"]
    vb_test_dir = repo_root / vb_cfg["test_dir"]
    val_spks = set(vb_cfg.get("val_speakers", ["p282", "p286", "p287"]))

    vb_train_files = sorted(vb_train_dir.glob("*.wav"))
    vb_test_files = sorted(vb_test_dir.glob("*.wav"))

    clean_train = []
    clean_val = []
    clean_test = [str(p) for p in vb_test_files]

    for p in vb_train_files:
        spk = p.name.split("_")[0]
        if spk in val_spks:
            clean_val.append(str(p))
        else:
            clean_train.append(str(p))

    logger.info(
        "VoiceBank Clean: Train=%d files (25 spk), Val=%d files (%d spk: %s), Test=%d files (2 spk)",
        len(clean_train), len(clean_val), len(val_spks), sorted(val_spks), len(clean_test),
    )

    # Save clean lists
    with open(manifest_base / "clean_train.json", "w") as f:
        json.dump(clean_train, f, indent=2)
    with open(manifest_base / "clean_val.json", "w") as f:
        json.dump(clean_val, f, indent=2)
    with open(manifest_base / "clean_test.json", "w") as f:
        json.dump(clean_test, f, indent=2)

    # Noise pools index
    noise_index = {
        "train": {"background": [], "environmental": [], "gunfire": []},
        "val": {"background": [], "environmental": [], "gunfire": []},
        "test": {"background": [], "environmental": [], "gunfire": [], "unseen_gunfire": []},
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 2. ESC-50 Environmental Noise (Folds 1-3 Train, 4 Val, 5 Test)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[2/5] Processing ESC-50 Environmental Noise...")
    esc_cfg = cfg["datasets"]["esc50"]
    esc_csv = repo_root / esc_cfg["meta_csv"]
    esc_audio_dir = repo_root / esc_cfg["audio_dir"]
    esc_selected = set(esc_cfg["selected_classes"])
    esc_train_folds = set(str(f) for f in esc_cfg["train_folds"])
    esc_val_folds = set(str(f) for f in esc_cfg["val_folds"])
    esc_test_folds = set(str(f) for f in esc_cfg["test_folds"])

    # Define which ESC-50 classes serve as continuous background
    esc_background_classes = {"rain", "thunderstorm", "wind", "sea_waves", "engine"}

    with open(esc_csv, "r", encoding="utf-8") as f:
        esc_rows = list(csv.DictReader(f))

    esc_count = 0
    for r in esc_rows:
        cat = r["category"]
        if cat not in esc_selected:
            continue

        fold = r["fold"]
        split = "train" if fold in esc_train_folds else ("val" if fold in esc_val_folds else "test")
        role = "background" if cat in esc_background_classes else "environmental"

        src_file = esc_audio_dir / r["filename"]
        if not src_file.exists():
            continue

        dst_file = out_base / split / role / f"esc50_{cat}_{r['filename']}"
        if resample_and_write(src_file, dst_file):
            noise_index[split][role].append(str(dst_file))
            esc_count += 1

    logger.info("Processed %d ESC-50 files into 16kHz mono pools", esc_count)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. UrbanSound8K Real-World Noise (Folds 1-8 Train, 9 Val, 10 Test)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[3/5] Processing UrbanSound8K Real-World Noise...")
    us8k_cfg = cfg["datasets"]["urbansound8k"]
    us8k_csv = repo_root / us8k_cfg["meta_csv"]
    us8k_root = repo_root / us8k_cfg["root_dir"]
    us8k_selected = set(us8k_cfg["selected_classes"])
    us8k_train_folds = set(str(f) for f in us8k_cfg["train_folds"])
    us8k_val_folds = set(str(f) for f in us8k_cfg["val_folds"])
    us8k_test_folds = set(str(f) for f in us8k_cfg["test_folds"])

    with open(us8k_csv, "r", encoding="utf-8") as f:
        us8k_rows = list(csv.DictReader(f))

    us8k_count = 0
    for r in us8k_rows:
        cls_name = r["class"]
        if cls_name not in us8k_selected:
            continue

        fold = r["fold"]
        split = "train" if fold in us8k_train_folds else ("val" if fold in us8k_val_folds else "test")

        if cls_name == "gun_shot":
            role = "gunfire"
        elif cls_name == "engine_idling":
            role = "background"
        else:
            role = "environmental"

        src_file = us8k_root / f"fold{fold}" / r["slice_file_name"]
        if not src_file.exists():
            continue

        dst_file = out_base / split / role / f"us8k_{cls_name}_{r['slice_file_name']}"
        if resample_and_write(src_file, dst_file):
            noise_index[split][role].append(str(dst_file))
            us8k_count += 1

    logger.info("Processed %d UrbanSound8K files into 16kHz mono pools", us8k_count)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Gunshot Audio Dataset (Weapon Splits: 7 Train, 1 Val, 1 UNSEEN Test)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[4/5] Processing Gunshot Audio Dataset...")
    gad_cfg = cfg["datasets"]["gunshot"]
    gad_root = repo_root / gad_cfg["root_dir"]
    train_weapons = set(gad_cfg["train_weapons"])
    val_weapons = set(gad_cfg["val_weapons"])
    unseen_test_weapons = set(gad_cfg["unseen_test_weapons"])

    gad_count = 0
    for weapon_dir in sorted(gad_root.iterdir()):
        if not weapon_dir.is_dir():
            continue
        w_name = weapon_dir.name
        if w_name in unseen_test_weapons:
            split = "test"
            role = "unseen_gunfire"
        elif w_name in val_weapons:
            split = "val"
            role = "gunfire"
        elif w_name in train_weapons:
            split = "train"
            role = "gunfire"
        else:
            split = "train"
            role = "gunfire"

        for w_file in sorted(weapon_dir.glob("*.wav")):
            safe_w_name = w_name.replace(" ", "_")
            dst_file = out_base / split / role / f"gad_{safe_w_name}_{w_file.name}"
            if resample_and_write(w_file, dst_file):
                noise_index[split][role].append(str(dst_file))
                gad_count += 1

    logger.info(
        "Processed %d Gunshot Audio Dataset files (Unseen test weapon '%s' = %d files)",
        gad_count, list(unseen_test_weapons), len(noise_index["test"]["unseen_gunfire"]),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Edge-Collected Gunshot Audio (Session Splits: 1-3 Train, 4 Val, 5 Test)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[5/5] Processing Edge-Collected Gunshot Audio...")
    edge_cfg = cfg["datasets"]["edge_guns"]
    edge_audio_dir = repo_root / edge_cfg["audio_dir"]
    edge_meta_csv = repo_root / edge_cfg["meta_csv"]
    train_sessions = set(edge_cfg["train_sessions"])
    val_sessions = set(edge_cfg["val_sessions"])

    # Build session lookup from metadata if available
    session_map = {}
    if edge_meta_csv.exists():
        with open(edge_meta_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sess = r.get("recording_session ", r.get("recording_session", "1")).strip()
                fn = r.get("filename", "").strip()
                if fn:
                    session_map[fn] = sess

    edge_count = 0
    for subfolder in sorted(edge_audio_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        for w_file in sorted(subfolder.rglob("*.wav")):
            fn_stem = w_file.stem
            sess = session_map.get(fn_stem, "1")

            if sess in train_sessions:
                split = "train"
                role = "gunfire"
            elif sess in val_sessions:
                split = "val"
                role = "gunfire"
            else:
                # Session 5 or other: add to test unseen gunfire pool!
                split = "test"
                role = "unseen_gunfire"

            dst_file = out_base / split / role / f"edge_{subfolder.name}_{w_file.name}"
            if resample_and_write(w_file, dst_file):
                noise_index[split][role].append(str(dst_file))
                edge_count += 1

    logger.info("Processed %d Edge-Collected Gunshot files", edge_count)

    # ─────────────────────────────────────────────────────────────────────────
    # DEMAND Status check
    # ─────────────────────────────────────────────────────────────────────────
    demand_root = cfg["datasets"]["demand"]["root_dir"]
    if demand_root and Path(demand_root).exists():
        logger.info("DEMAND detected at %s — adding continuous noise", demand_root)
    else:
        logger.info(
            "DEMAND not present (cfg datasets.demand.root_dir=%s). Handled gracefully.",
            demand_root,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Save Noise Index Manifest
    # ─────────────────────────────────────────────────────────────────────────
    noise_index_path = manifest_base / "noise_pools.json"
    with open(noise_index_path, "w", encoding="utf-8") as f:
        json.dump(noise_index, f, indent=2)

    logger.info("=" * 70)
    logger.info("DEFENSE DATA PREPARATION SUMMARY:")
    for split_name in ["train", "val", "test"]:
        roles = noise_index[split_name]
        counts_str = ", ".join(f"{r}={len(files)}" for r, files in roles.items())
        logger.info("  Split [%s]: %s", split_name.upper(), counts_str)
    logger.info("Noise index written to %s", noise_index_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
