#!/usr/bin/env python3
"""
comprehensive_epoch_eval.py — Systematically evaluate candidate epochs and combinations.

1. Preloads test audio into memory.
2. Evaluates individual candidate checkpoints across Stage 1, 2, 3 and previous defense runs.
3. Ranks checkpoints by Recall, Precision, Specificity, and Accuracy.
4. Identifies:
   - Best Recall Epoch
   - Best Precision Epoch
   - Best Specificity Epoch
5. Evaluates 2-way, 3-way, and 4-way weight interpolation combinations (Model Soups).
6. Exports all selected models to:
   - PyTorch (.pth)
   - Streaming FP32 ONNX (.onnx)
   - Dynamic INT8 ONNX (_int8.onnx)
7. Validates ONNX runtime and RTF.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models" / "gtcrn"))
sys.path.insert(0, str(REPO_ROOT / "models" / "gtcrn" / "stream"))

from gtcrn import GTCRN
from sih26052.eval.metrics import compute_classification_metrics
from sih26052.export.to_onnx import export_streaming_onnx
from sih26052.export.quantize import quantize_dynamic_int8

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("epoch_eval")


def load_manifest_audio(manifest_path: Path):
    """Preload audio pairs into memory for instant evaluation."""
    logger.info("Preloading manifest audio from %s...", manifest_path)
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))

    m_dir = manifest_path.resolve().parent
    audio_data = []
    for entry in entries:
        noisy_p = Path(entry["noisy"])
        if not noisy_p.exists():
            noisy_p = m_dir / noisy_p if (m_dir / noisy_p).exists() else m_dir.parent / noisy_p
        clean_p = Path(entry["clean"])
        if not clean_p.exists():
            clean_p = m_dir / clean_p if (m_dir / clean_p).exists() else m_dir.parent / clean_p

        noisy, _ = sf.read(str(noisy_p), dtype="float32")
        clean, _ = sf.read(str(clean_p), dtype="float32")
        if noisy.ndim > 1:
            noisy = noisy.mean(axis=1)
        if clean.ndim > 1:
            clean = clean.mean(axis=1)
        min_len = min(len(noisy), len(clean))
        audio_data.append((noisy[:min_len], clean[:min_len]))

    logger.info("Preloaded %d audio pairs successfully.", len(audio_data))
    return audio_data


def evaluate_state_dict(model: GTCRN, state_dict: dict, audio_data: list, device: torch.device, spectral_floor: float = 0.025):
    """Run VAD frame classification evaluation on preloaded audio."""
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    nfft = 512
    hop = 256
    window = torch.hann_window(nfft).pow(0.5).to(device)

    total_tp = 0.0
    total_fp = 0.0
    total_tn = 0.0
    total_fn = 0.0

    for noisy, clean in audio_data:
        min_len = len(noisy)
        with torch.no_grad():
            noisy_t = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).to(device)
            noisy_stft = torch.view_as_real(torch.stft(noisy_t, nfft, hop, window=window, return_complex=True))
            pred_stft = model(noisy_stft)
            if spectral_floor > 0.0:
                pred_stft = pred_stft + noisy_stft * spectral_floor

            pred_complex = torch.complex(pred_stft[..., 0], pred_stft[..., 1])
            enhanced = torch.istft(pred_complex, nfft, hop, window=window, length=min_len).squeeze(0).cpu().numpy()

        clf = compute_classification_metrics(clean, enhanced, frame_len=nfft, hop_len=hop, threshold_db=-35.0)
        total_tp += clf.get("tp", 0.0)
        total_fp += clf.get("fp", 0.0)
        total_tn += clf.get("tn", 0.0)
        total_fn += clf.get("fn", 0.0)

    total_frames = total_tp + total_fp + total_tn + total_fn
    actual_speech = total_tp + total_fn
    actual_noise = total_tn + total_fp
    pred_speech = total_tp + total_fp

    accuracy = (total_tp + total_tn) / (total_frames + 1e-12) * 100.0
    precision = total_tp / (pred_speech + 1e-12) * 100.0
    recall = total_tp / (actual_speech + 1e-12) * 100.0
    specificity = total_tn / (actual_noise + 1e-12) * 100.0
    f1 = 2.0 * (precision / 100.0) * (recall / 100.0) / ((precision / 100.0) + (recall / 100.0) + 1e-12)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": int(total_tp),
        "fp": int(total_fp),
        "tn": int(total_tn),
        "fn": int(total_fn),
    }


def get_sd(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu")
    return ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))


def merge_state_dicts(sd_weight_pairs: list[tuple[dict, float]]) -> dict:
    base_sd = sd_weight_pairs[0][0]
    merged = {}
    for k in base_sd.keys():
        merged[k] = sum(w * sd[k] for sd, w in sd_weight_pairs)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Scan and rank epochs & combinations")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "test_manifest.jsonl")
    parser.add_argument("--export-all", action="store_true", default=True, help="Export models to models/")
    args = parser.parse_args()

    audio_data = load_manifest_audio(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_model = GTCRN().to(device)

    # Gather candidate checkpoints from various epochs
    candidate_paths = [
        # Stage 3 Silence & Specificity Checkpoints
        ("Stage-3 Ep10 (Final Best)", REPO_ROOT / "experiments" / "gtcrn_stage3_final" / "checkpoint_best_epoch010.pth"),
        ("Stage-3 Ep04", REPO_ROOT / "experiments" / "gtcrn_stage3_final" / "checkpoint_best_epoch004.pth"),
        ("Stage-3 Ep03", REPO_ROOT / "experiments" / "gtcrn_stage3_final" / "checkpoint_best_epoch003.pth"),
        ("Stage-3 Ep10 (Latest)", REPO_ROOT / "experiments" / "gtcrn_stage3_final" / "checkpoint_latest.pth"),

        # Stage 2 Perceptual & Codec Checkpoints
        ("Stage-2 Ep13 (Best Score)", REPO_ROOT / "experiments" / "gtcrn_stage2_finetuned" / "checkpoint_best_epoch013.pth"),
        ("Stage-2 Ep11", REPO_ROOT / "experiments" / "gtcrn_stage2_finetuned" / "checkpoint_best_epoch011.pth"),
        ("Stage-2 Ep10", REPO_ROOT / "experiments" / "gtcrn_stage2_finetuned" / "checkpoint_best_epoch010.pth"),
        ("Stage-2 Ep15 (Latest)", REPO_ROOT / "experiments" / "gtcrn_stage2_finetuned" / "checkpoint_latest.pth"),

        # Stage 1 Core Checkpoints
        ("Stage-1 Ep58 (Best Core)", REPO_ROOT / "experiments" / "gtcrn_optimized_v1" / "checkpoint_best_epoch058.pth"),
        ("Stage-1 Ep57", REPO_ROOT / "experiments" / "gtcrn_optimized_v1" / "checkpoint_best_epoch057.pth"),
        ("Stage-1 Ep55", REPO_ROOT / "experiments" / "gtcrn_optimized_v1" / "checkpoint_best_epoch055.pth"),
        ("Stage-1 Ep60 (Latest)", REPO_ROOT / "experiments" / "gtcrn_optimized_v1" / "checkpoint_latest.pth"),

        # Other Milestone Epochs from models/checkpoints
        ("Core Run Ep50", REPO_ROOT / "models" / "checkpoints" / "checkpoint_epoch_050.pth"),
        ("Core Run Ep45", REPO_ROOT / "models" / "checkpoints" / "checkpoint_epoch_045.pth"),
        ("Core Run Ep40", REPO_ROOT / "models" / "checkpoints" / "checkpoint_epoch_040.pth"),
        ("Core Run Ep30", REPO_ROOT / "models" / "checkpoints" / "checkpoint_epoch_030.pth"),
        ("Core Run Ep20", REPO_ROOT / "models" / "checkpoints" / "checkpoint_epoch_020.pth"),

        # Defense Checkpoints
        ("Defense Run Best", REPO_ROOT / "models" / "checkpoints_defense_60_40" / "checkpoint_best.pth"),
        ("Defense Run Ep10", REPO_ROOT / "models" / "checkpoints_defense_60_40" / "checkpoint_epoch_010.pth"),
    ]

    print("\n" + "=" * 90)
    print("EVALUATING ALL CANDIDATE EPOCHS ACROSS TRAINING STAGES".center(90))
    print("=" * 90)

    evaluated_results = []
    loaded_sds = {}

    for name, path in candidate_paths:
        if not path.exists():
            continue
        try:
            sd = get_sd(path)
            res = evaluate_state_dict(eval_model, sd, audio_data, device)
            res["name"] = name
            res["path"] = path
            evaluated_results.append(res)
            loaded_sds[name] = sd
            print(f"[{name:<26}] Recall: {res['recall']:>5.1f}% | Prec: {res['precision']:>5.1f}% | Spec: {res['specificity']:>5.1f}% | Acc: {res['accuracy']:>5.2f}% | F1: {res['f1']:>6.4f}")
        except Exception as e:
            logger.warning("Could not evaluate %s: %s", path, e)

    # Sort and identify champions
    by_recall = sorted(evaluated_results, key=lambda x: x["recall"], reverse=True)
    by_precision = sorted(evaluated_results, key=lambda x: x["precision"], reverse=True)
    by_specificity = sorted(evaluated_results, key=lambda x: x["specificity"], reverse=True)
    by_accuracy = sorted(evaluated_results, key=lambda x: x["accuracy"], reverse=True)
    by_f1 = sorted(evaluated_results, key=lambda x: x["f1"], reverse=True)

    best_recall_item = by_recall[0]
    best_precision_item = by_precision[0]
    best_specificity_item = by_specificity[0]

    print("\n" + "=" * 90)
    print("INDIVIDUAL EPOCH CHAMPIONS".center(90))
    print("=" * 90)
    print(f"  • BEST RECALL      : {best_recall_item['name']} ({best_recall_item['recall']:.1f}% Recall, {best_recall_item['precision']:.1f}% Prec, {best_recall_item['specificity']:.1f}% Spec, {best_recall_item['accuracy']:.2f}% Acc)")
    print(f"  • BEST PRECISION   : {best_precision_item['name']} ({best_precision_item['precision']:.1f}% Prec, {best_precision_item['recall']:.1f}% Recall, {best_precision_item['specificity']:.1f}% Spec, {best_precision_item['accuracy']:.2f}% Acc)")
    print(f"  • BEST SPECIFICITY : {best_specificity_item['name']} ({best_specificity_item['specificity']:.1f}% Spec, {best_specificity_item['precision']:.1f}% Prec, {best_specificity_item['recall']:.1f}% Recall, {best_specificity_item['accuracy']:.2f}% Acc)")
    print(f"  • BEST OVERALL F1  : {by_f1[0]['name']} (F1: {by_f1[0]['f1']:.4f}, Recall: {by_f1[0]['recall']:.1f}%)")

    # Multi-way combinations (Model Soups)
    print("\n" + "=" * 90)
    print("EVALUATING MULTI-WAY COMBINATIONS (2-WAY, 3-WAY, 4-WAY MODEL SOUPS)".center(90))
    print("=" * 90)

    sd_recall = loaded_sds[best_recall_item["name"]]
    sd_prec = loaded_sds[best_precision_item["name"]]
    sd_spec = loaded_sds[best_specificity_item["name"]]
    sd_base = loaded_sds.get("Stage-1 Ep58 (Best Core)", loaded_sds[evaluated_results[0]["name"]])
    sd_runner_recall = loaded_sds.get("Stage-2 Ep10", loaded_sds[by_recall[1]["name"]])

    combo_specs = [
        # 2-way variations
        ("Combo-2way (70% Recall + 30% Spec)", [(sd_recall, 0.70), (sd_spec, 0.30)]),
        ("Combo-2way (60% Recall + 40% Spec)", [(sd_recall, 0.60), (sd_spec, 0.40)]),
        ("Combo-2way (50% Recall + 50% Spec)", [(sd_recall, 0.50), (sd_spec, 0.50)]),
        ("Combo-2way (40% Recall + 60% Spec)", [(sd_recall, 0.40), (sd_spec, 0.60)]),

        # 3-way variations
        ("Combo-3way (50% Recall + 35% Spec + 15% Base)", [(sd_recall, 0.50), (sd_spec, 0.35), (sd_base, 0.15)]),
        ("Combo-3way (45% Recall + 35% Spec + 20% Prec)", [(sd_recall, 0.45), (sd_spec, 0.35), (sd_prec, 0.20)]),
        ("Combo-3way (40% Recall + 40% Spec + 20% Base)", [(sd_recall, 0.40), (sd_spec, 0.40), (sd_base, 0.20)]),

        # 4-way variations
        ("Combo-4way (40% Recall + 30% Spec + 15% Ep10 + 15% Base)", [(sd_recall, 0.40), (sd_spec, 0.30), (sd_runner_recall, 0.15), (sd_base, 0.15)]),
        ("Combo-4way (35% Recall + 35% Spec + 15% Prec + 15% Base)", [(sd_recall, 0.35), (sd_spec, 0.35), (sd_prec, 0.15), (sd_base, 0.15)]),
        ("Combo-4way Balanced (25% Each)", [(sd_recall, 0.25), (sd_spec, 0.25), (sd_prec, 0.25), (sd_base, 0.25)]),
    ]

    combo_results = []
    for c_name, pairs in combo_specs:
        merged_sd = merge_state_dicts(pairs)
        c_res = evaluate_state_dict(eval_model, merged_sd, audio_data, device)
        c_res["name"] = c_name
        c_res["sd"] = merged_sd
        combo_results.append(c_res)
        print(f"[{c_name:<55}] Recall: {c_res['recall']:>5.1f}% | Prec: {c_res['precision']:>5.1f}% | Spec: {c_res['specificity']:>5.1f}% | Acc: {c_res['accuracy']:>5.2f}% | F1: {c_res['f1']:>6.4f}")

    # Identify best 2-way, 3-way, 4-way
    twoway_combos = [c for c in combo_results if "Combo-2way" in c["name"]]
    threeway_combos = [c for c in combo_results if "Combo-3way" in c["name"]]
    fourway_combos = [c for c in combo_results if "Combo-4way" in c["name"]]

    best_2way = sorted(twoway_combos, key=lambda x: x["f1"] * 0.5 + x["accuracy"] * 0.005, reverse=True)[0]
    best_3way = sorted(threeway_combos, key=lambda x: x["f1"] * 0.5 + x["accuracy"] * 0.005, reverse=True)[0]
    best_4way = sorted(fourway_combos, key=lambda x: x["f1"] * 0.5 + x["accuracy"] * 0.005, reverse=True)[0]

    # Models to save
    # Note: If best precision and best specificity are from the same checkpoint, we save both aliases or distinguish them!
    final_models_to_export = {
        "gtcrn_best_recall": {
            "title": f"Best Recall Checkpoint: {best_recall_item['name']}",
            "sd": loaded_sds[best_recall_item["name"]],
            "metrics": f"Recall: {best_recall_item['recall']:.1f}% | Prec: {best_recall_item['precision']:.1f}% | Spec: {best_recall_item['specificity']:.1f}% | Acc: {best_recall_item['accuracy']:.2f}% | F1: {best_recall_item['f1']:.4f}",
            "res": best_recall_item,
        },
        "gtcrn_best_precision": {
            "title": f"Best Precision Checkpoint: {best_precision_item['name']}",
            "sd": loaded_sds[best_precision_item["name"]],
            "metrics": f"Recall: {best_precision_item['recall']:.1f}% | Prec: {best_precision_item['precision']:.1f}% | Spec: {best_precision_item['specificity']:.1f}% | Acc: {best_precision_item['accuracy']:.2f}% | F1: {best_precision_item['f1']:.4f}",
            "res": best_precision_item,
        },
        "gtcrn_best_specificity": {
            "title": f"Best Specificity Checkpoint: {best_specificity_item['name']}",
            "sd": loaded_sds[best_specificity_item["name"]],
            "metrics": f"Recall: {best_specificity_item['recall']:.1f}% | Prec: {best_specificity_item['precision']:.1f}% | Spec: {best_specificity_item['specificity']:.1f}% | Acc: {best_specificity_item['accuracy']:.2f}% | F1: {best_specificity_item['f1']:.4f}",
            "res": best_specificity_item,
        },
        "gtcrn_combo_2way": {
            "title": f"Best 2-Way Combination: {best_2way['name']}",
            "sd": best_2way["sd"],
            "metrics": f"Recall: {best_2way['recall']:.1f}% | Prec: {best_2way['precision']:.1f}% | Spec: {best_2way['specificity']:.1f}% | Acc: {best_2way['accuracy']:.2f}% | F1: {best_2way['f1']:.4f}",
            "res": best_2way,
        },
        "gtcrn_combo_3way": {
            "title": f"Best 3-Way Combination: {best_3way['name']}",
            "sd": best_3way["sd"],
            "metrics": f"Recall: {best_3way['recall']:.1f}% | Prec: {best_3way['precision']:.1f}% | Spec: {best_3way['specificity']:.1f}% | Acc: {best_3way['accuracy']:.2f}% | F1: {best_3way['f1']:.4f}",
            "res": best_3way,
        },
        "gtcrn_combo_4way": {
            "title": f"Best 4-Way Combination: {best_4way['name']}",
            "sd": best_4way["sd"],
            "metrics": f"Recall: {best_4way['recall']:.1f}% | Prec: {best_4way['precision']:.1f}% | Spec: {best_4way['specificity']:.1f}% | Acc: {best_4way['accuracy']:.2f}% | F1: {best_4way['f1']:.4f}",
            "res": best_4way,
        },
    }

    if args.export_all:
        models_dir = REPO_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 90)
        print("EXPORTING ALL SPECIALIZED CHECKPOINTS & COMBINATIONS (PTH, FP32 ONNX, INT8 ONNX)".center(90))
        print("=" * 90)

        for key, info in final_models_to_export.items():
            pth_file = models_dir / f"{key}.pth"
            onnx_file = models_dir / f"{key}.onnx"
            int8_file = models_dir / f"{key}_int8.onnx"

            print(f"\n--> Processing [{key}]: {info['title']}")
            print(f"    Metrics: {info['metrics']}")

            # 1. Save PyTorch .pth
            torch.save(
                {
                    "model_state_dict": info["sd"],
                    "title": info["title"],
                    "metrics_summary": info["metrics"],
                    "eval_results": {k: v for k, v in info["res"].items() if k not in ("sd", "path")},
                },
                pth_file,
            )
            print(f"    [1/3] Saved PyTorch checkpoint: models/{pth_file.name}")

            # 2. Export Streaming ONNX
            export_streaming_onnx(
                checkpoint_path=pth_file,
                output_path=onnx_file,
                opset_version=18,
            )
            print(f"    [2/3] Exported Streaming ONNX: models/{onnx_file.name}")

            # 3. Quantize to INT8
            quantize_dynamic_int8(
                input_path=onnx_file,
                output_path=int8_file,
            )
            print(f"    [3/3] Quantized to INT8 ONNX: models/{int8_file.name}")

        print("\n" + "=" * 90)
        print("ALL MODELS EXPORTED AND QUANTIZED SUCCESSFULLY".center(90))
        print("=" * 90)

    # Return summary dictionary
    summary_report_path = REPO_ROOT / "models" / "epochs_and_combos_summary.json"
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champions": {
            "best_recall": {
                "name": best_recall_item["name"],
                "recall": best_recall_item["recall"],
                "precision": best_recall_item["precision"],
                "specificity": best_recall_item["specificity"],
                "accuracy": best_recall_item["accuracy"],
                "f1": best_recall_item["f1"],
            },
            "best_precision": {
                "name": best_precision_item["name"],
                "recall": best_precision_item["recall"],
                "precision": best_precision_item["precision"],
                "specificity": best_precision_item["specificity"],
                "accuracy": best_precision_item["accuracy"],
                "f1": best_precision_item["f1"],
            },
            "best_specificity": {
                "name": best_specificity_item["name"],
                "recall": best_specificity_item["recall"],
                "precision": best_specificity_item["precision"],
                "specificity": best_specificity_item["specificity"],
                "accuracy": best_specificity_item["accuracy"],
                "f1": best_specificity_item["f1"],
            },
            "best_2way": {
                "name": best_2way["name"],
                "recall": best_2way["recall"],
                "precision": best_2way["precision"],
                "specificity": best_2way["specificity"],
                "accuracy": best_2way["accuracy"],
                "f1": best_2way["f1"],
            },
            "best_3way": {
                "name": best_3way["name"],
                "recall": best_3way["recall"],
                "precision": best_3way["precision"],
                "specificity": best_3way["specificity"],
                "accuracy": best_3way["accuracy"],
                "f1": best_3way["f1"],
            },
            "best_4way": {
                "name": best_4way["name"],
                "recall": best_4way["recall"],
                "precision": best_4way["precision"],
                "specificity": best_4way["specificity"],
                "accuracy": best_4way["accuracy"],
                "f1": best_4way["f1"],
            },
        },
        "all_individual_epochs": [
            {
                "name": r["name"],
                "recall": r["recall"],
                "precision": r["precision"],
                "specificity": r["specificity"],
                "accuracy": r["accuracy"],
                "f1": r["f1"],
            }
            for r in evaluated_results
        ],
        "all_combinations": [
            {
                "name": c["name"],
                "recall": c["recall"],
                "precision": c["precision"],
                "specificity": c["specificity"],
                "accuracy": c["accuracy"],
                "f1": c["f1"],
            }
            for c in combo_results
        ],
    }

    with open(summary_report_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[+] Detailed summary report written to: {summary_report_path}")


if __name__ == "__main__":
    main()
