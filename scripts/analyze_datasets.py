#!/usr/bin/env python3
"""
analyze_datasets.py — Comprehensive inspection and audit script for SIH-ANC datasets.

Reports:
    - Dataset name and path
    - Total files, usable files, corrupt files
    - Total duration, average duration
    - Sample rate and channel counts
    - All classes and selected defense classes
    - Gunfire-specific totals and weapon breakdowns
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from collections import Counter
from typing import Any

repo_root = Path(__file__).resolve().parent.parent

# Auto-reexec with virtual environment if needed
if sys.prefix == sys.base_prefix:
    venv_python = repo_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = repo_root.parent / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

import soundfile as sf


def audit_audio_files(paths: list[Path]) -> dict:
    """Scan list of audio files and return summary statistics."""
    total = len(paths)
    usable = 0
    corrupt = 0
    total_dur = 0.0
    srs = Counter()
    channels = Counter()

    for p in paths:
        try:
            info = sf.info(str(p))
            usable += 1
            total_dur += info.duration
            srs[info.samplerate] += 1
            channels[info.channels] += 1
        except Exception:
            corrupt += 1

    avg_dur = total_dur / usable if usable > 0 else 0.0
    return {
        "total_files": total,
        "usable_files": usable,
        "corrupt_files": corrupt,
        "total_duration_s": round(total_dur, 2),
        "total_duration_h": round(total_dur / 3600.0, 2),
        "avg_duration_s": round(avg_dur, 2),
        "sample_rates": dict(srs),
        "channels": dict(channels),
    }


def analyze_voicebank(vb_dir: Path) -> dict:
    clean_train_dir = vb_dir / "clean_trainset_28spk_wav"
    clean_test_dir = vb_dir / "clean_testset_wav"

    train_files = sorted(clean_train_dir.glob("*.wav")) if clean_train_dir.exists() else []
    test_files = sorted(clean_test_dir.glob("*.wav")) if clean_test_dir.exists() else []

    train_stats = audit_audio_files(train_files)
    test_stats = audit_audio_files(test_files)

    train_spks = sorted(set(p.name.split("_")[0] for p in train_files))
    test_spks = sorted(set(p.name.split("_")[0] for p in test_files))

    return {
        "dataset": "VoiceBank",
        "path": str(vb_dir),
        "train_set": {
            "path": str(clean_train_dir),
            "speakers_count": len(train_spks),
            "speakers": train_spks,
            **train_stats,
        },
        "test_set": {
            "path": str(clean_test_dir),
            "speakers_count": len(test_spks),
            "speakers": test_spks,
            **test_stats,
        },
        "speaker_leakage": len(set(train_spks) & set(test_spks)) > 0,
    }


def analyze_esc50(esc_dir: Path, selected_classes: list[str]) -> dict:
    meta_csv = esc_dir / "meta" / "esc50.csv"
    audio_dir = esc_dir / "audio"

    if not meta_csv.exists() or not audio_dir.exists():
        return {"dataset": "ESC50", "path": str(esc_dir), "status": "missing"}

    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    all_categories = Counter(r["category"] for r in rows)
    folds = Counter(r["fold"] for r in rows)

    selected_set = set(selected_classes)
    present_selected = {c: all_categories[c] for c in selected_classes if c in all_categories}
    missing_selected = [c for c in selected_classes if c not in all_categories]

    # Scan selected class audio files
    selected_files = [
        audio_dir / r["filename"] for r in rows if r["category"] in selected_set
    ]
    audio_stats = audit_audio_files(selected_files)

    return {
        "dataset": "ESC-50",
        "path": str(esc_dir),
        "total_categories": len(all_categories),
        "all_categories": sorted(all_categories.keys()),
        "folds": dict(folds),
        "selected_classes_present": present_selected,
        "selected_classes_missing": missing_selected,
        "selected_audio_stats": audio_stats,
    }


def analyze_urbansound8k(us8k_dir: Path, selected_classes: list[str]) -> dict:
    meta_csv = us8k_dir / "UrbanSound8K.csv"
    if not meta_csv.exists():
        return {"dataset": "UrbanSound8K", "path": str(us8k_dir), "status": "missing"}

    with open(meta_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    all_classes = Counter(r["class"] for r in rows)
    folds = Counter(r["fold"] for r in rows)

    selected_set = set(selected_classes)
    selected_counts = {c: all_classes[c] for c in selected_classes if c in all_classes}

    # Selected audio files
    selected_files = []
    for r in rows:
        if r["class"] in selected_set:
            file_path = us8k_dir / f"fold{r['fold']}" / r["slice_file_name"]
            if file_path.exists():
                selected_files.append(file_path)

    audio_stats = audit_audio_files(selected_files)

    return {
        "dataset": "UrbanSound8K",
        "path": str(us8k_dir),
        "all_classes": dict(all_classes),
        "folds": dict(folds),
        "selected_classes_counts": selected_counts,
        "selected_audio_stats": audio_stats,
    }


def analyze_gunshot_audio_dataset(gad_dir: Path) -> dict:
    if not gad_dir.exists():
        return {"dataset": "Gunshot_Audio_Dataset", "path": str(gad_dir), "status": "missing"}

    weapon_stats = {}
    all_files = []
    for p in sorted(gad_dir.iterdir()):
        if p.is_dir():
            w_files = sorted(p.glob("*.wav"))
            weapon_stats[p.name] = len(w_files)
            all_files.extend(w_files)

    audio_stats = audit_audio_files(all_files)
    return {
        "dataset": "Gunshot_Audio_Dataset",
        "path": str(gad_dir),
        "weapons": weapon_stats,
        "total_weapons": len(weapon_stats),
        "audio_stats": audio_stats,
    }


def analyze_edge_collected_guns(edge_dir: Path) -> dict:
    if not edge_dir.exists():
        return {"dataset": "edge-collected-guns", "path": str(edge_dir), "status": "missing"}

    meta_csv = edge_dir / "gunshot-audio-all-metadata.csv"
    meta_stats: dict[str, Any] = {}
    if meta_csv.exists():
        with open(meta_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        meta_stats["total_metadata_rows"] = len(rows)
        meta_stats["labels"] = dict(Counter(r.get("label", "unknown") for r in rows))
        meta_stats["firearms"] = dict(Counter(r.get("firearm", "unknown") for r in rows))
        meta_stats["calibers"] = dict(Counter(r.get("caliber", "unknown") for r in rows))
        meta_stats["sessions"] = dict(Counter(r.get("recording_session ", r.get("recording_session", "unknown")) for r in rows))

    audio_subdir = edge_dir / "edge-collected-gunshot-audio"
    folder_counts = {}
    all_files = []
    if audio_subdir.exists():
        for p in sorted(audio_subdir.iterdir()):
            if p.is_dir():
                w_files = sorted(p.rglob("*.wav"))
                folder_counts[p.name] = len(w_files)
                all_files.extend(w_files)

    audio_stats = audit_audio_files(all_files)
    return {
        "dataset": "edge-collected-gunshot-audio",
        "path": str(edge_dir),
        "subfolders": folder_counts,
        "metadata_summary": meta_stats,
        "audio_stats": audio_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze datasets for GTCRN training")
    parser.add_argument("--datasets-dir", type=Path, default=repo_root / "Datasets")
    parser.add_argument("--json-out", type=Path, default=repo_root / "reports" / "dataset_analysis.json")
    args = parser.parse_args()

    ds_dir = args.datasets_dir
    print("=" * 80)
    print(f"ANALYZE DATASETS — Auditing {ds_dir}")
    print("=" * 80)

    # 1. VoiceBank
    vb_res = analyze_voicebank(ds_dir / "VoiceBank")
    print("\n[1] VoiceBank (Clean Target Speech)")
    print(f"    Path: {vb_res['path']}")
    print(f"    Train: {vb_res['train_set']['usable_files']} files, {vb_res['train_set']['total_duration_h']}h, {vb_res['train_set']['speakers_count']} speakers, Corrupt: {vb_res['train_set']['corrupt_files']}")
    print(f"    Test:  {vb_res['test_set']['usable_files']} files, {vb_res['test_set']['total_duration_h']}h, {vb_res['test_set']['speakers_count']} speakers, Corrupt: {vb_res['test_set']['corrupt_files']}")
    print(f"    Speaker Leakage: {'YES (FAIL)' if vb_res['speaker_leakage'] else 'NO (PASS)'}")

    # 2. ESC-50
    requested_esc_classes = [
        "gun_shot", "helicopter", "airplane", "siren", "car_horn", "engine", "train",
        "chainsaw", "hand_saw", "fireworks", "crackling_fire", "rain", "thunderstorm",
        "wind", "sea_waves", "water_drops", "footsteps", "breathing", "coughing",
        "clapping", "laughing", "sneezing", "snoring", "crying_baby"
    ]
    esc_res = analyze_esc50(ds_dir / "ESC50" / "ESC-50-master", requested_esc_classes)
    print("\n[2] ESC-50 (Environmental / Event Noise)")
    print(f"    Path: {esc_res['path']}")
    print(f"    Selected Classes Present ({len(esc_res['selected_classes_present'])}): {list(esc_res['selected_classes_present'].keys())}")
    print(f"    Missing Classes: {esc_res['selected_classes_missing']} (Note: gun_shot is not in ESC-50 metadata)")
    s_stats = esc_res["selected_audio_stats"]
    print(f"    Selected Files: {s_stats['usable_files']}, {s_stats['total_duration_h']}h, SR: {list(s_stats['sample_rates'].keys())}, Channels: {list(s_stats['channels'].keys())}, Corrupt: {s_stats['corrupt_files']}")

    # 3. UrbanSound8K
    requested_us8k_classes = ["gun_shot", "siren", "engine_idling", "jackhammer", "drilling", "car_horn"]
    us8k_res = analyze_urbansound8k(ds_dir / "urbansound8k", requested_us8k_classes)
    print("\n[3] UrbanSound8K (Real-World Event Noise)")
    print(f"    Path: {us8k_res['path']}")
    print(f"    Selected Classes Counts: {us8k_res['selected_classes_counts']}")
    u_stats = us8k_res["selected_audio_stats"]
    print(f"    Selected Files: {u_stats['usable_files']}, {u_stats['total_duration_h']}h, SR: {list(u_stats['sample_rates'].keys())}, Channels: {list(u_stats['channels'].keys())}, Corrupt: {u_stats['corrupt_files']}")

    # 4. Gunshot_Audio_Dataset
    gad_res = analyze_gunshot_audio_dataset(ds_dir / "Gunshot_Audio_Dataset")
    print("\n[4] Gunshot_Audio_Dataset (Gunfire Noise)")
    print(f"    Path: {gad_res['path']}")
    print(f"    Weapons ({gad_res['total_weapons']}): {gad_res['weapons']}")
    g_stats = gad_res["audio_stats"]
    print(f"    Total Recordings: {g_stats['usable_files']}, {g_stats['total_duration_h']}h, SR: {list(g_stats['sample_rates'].keys())}, Channels: {list(g_stats['channels'].keys())}, Corrupt: {g_stats['corrupt_files']}")

    # 5. edge-collected-gunshot-audio
    edge_res = analyze_edge_collected_guns(ds_dir / "edge-collected-gunshot-audio (1)")
    print("\n[5] Edge-Collected-Gunshot-Audio (Gunfire Noise)")
    print(f"    Path: {edge_res['path']}")
    print(f"    Subfolders: {edge_res['subfolders']}")
    e_stats = edge_res["audio_stats"]
    print(f"    Total Recordings: {e_stats['usable_files']}, {e_stats['total_duration_h']}h, SR: {list(e_stats['sample_rates'].keys())}, Channels: {list(e_stats['channels'].keys())}, Corrupt: {e_stats['corrupt_files']}")

    # 6. DEMAND
    demand_dir = ds_dir / "DEMAND"
    demand_exists = demand_dir.exists()
    print("\n[6] DEMAND (Background Continuous Noise)")
    print(f"    Path: {demand_dir} — Status: {'PRESENT' if demand_exists else 'NOT INSTALLED (Handled gracefully as optional future source)'}")

    # Summary of usable gunfire recordings
    total_gunfire_files = gad_res["audio_stats"]["usable_files"] + edge_res["audio_stats"]["usable_files"] + us8k_res["selected_classes_counts"].get("gun_shot", 0)
    print(f"\n[TOTAL USABLE GUNFIRE RECORDINGS POOL]: {total_gunfire_files} files")
    print(f"   - Gunshot_Audio_Dataset: {gad_res['audio_stats']['usable_files']}")
    print(f"   - Edge-Collected Guns:   {edge_res['audio_stats']['usable_files']}")
    print(f"   - UrbanSound8K Gunshot:  {us8k_res['selected_classes_counts'].get('gun_shot', 0)}")
    print("=" * 80)

    # Save report
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "voicebank": vb_res,
        "esc50": esc_res,
        "urbansound8k": us8k_res,
        "gunshot_audio_dataset": gad_res,
        "edge_collected_guns": edge_res,
        "demand": {"status": "present" if demand_exists else "not_installed"},
        "summary": {
            "total_gunfire_recordings": total_gunfire_files,
            "total_clean_train": vb_res["train_set"]["usable_files"],
            "total_clean_test": vb_res["test_set"]["usable_files"],
            "total_selected_esc50": esc_res["selected_audio_stats"]["usable_files"],
            "total_selected_us8k": us8k_res["selected_audio_stats"]["usable_files"],
        }
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
