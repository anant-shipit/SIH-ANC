#!/usr/bin/env python3
"""
export_all_specialized_checkpoints.py — Save best individual checkpoints & multi-way combinations.

Creates:
1. Best Recall Checkpoint (Stage-2 Epoch 13)
2. Best Specificity & Precision Checkpoint (Stage-3 Epoch 10)
3. 2-Way Combination (Best Recall 60% + Best Specificity 40%)
4. 3-Way Combination (Best Recall 50% + Best Specificity 35% + Stage-1 Ep58 15%)
5. 4-Way Combination (Grand Soup: Stage-2 Ep13 40% + Stage-3 Ep10 30% + Stage-2 Ep10 15% + Stage-1 Ep58 15%)

For each checkpoint:
- Saves PyTorch .pth
- Exports streaming ONNX (FP32)
- Quantizes to dynamic INT8 ONNX
"""
import logging
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models" / "gtcrn"))
sys.path.insert(0, str(REPO_ROOT / "models" / "gtcrn" / "stream"))

from sih26052.export.to_onnx import export_streaming_onnx
from sih26052.export.quantize import quantize_dynamic_int8

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_specialized")


def get_sd(path_str: str) -> dict:
    ckpt = torch.load(path_str, map_location="cpu")
    return ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))


def merge_state_dicts(sd_weight_pairs: list[tuple[dict, float]]) -> dict:
    base_sd = sd_weight_pairs[0][0]
    merged = {}
    for k in base_sd.keys():
        merged[k] = sum(w * sd[k] for sd, w in sd_weight_pairs)
    return merged


def main():
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Base source checkpoints
    s1_p = REPO_ROOT / "experiments" / "gtcrn_optimized_v1" / "checkpoint_best_epoch058.pth"
    s2_p = REPO_ROOT / "experiments" / "gtcrn_stage2_finetuned" / "checkpoint_best_epoch013.pth"
    s3_p = REPO_ROOT / "experiments" / "gtcrn_stage3_final" / "checkpoint_best_epoch010.pth"
    s2_10_p = REPO_ROOT / "experiments" / "gtcrn_stage2_finetuned" / "checkpoint_best_epoch010.pth"

    sd1 = get_sd(str(s1_p))
    sd2 = get_sd(str(s2_p))
    sd3 = get_sd(str(s3_p))
    sd2_10 = get_sd(str(s2_10_p))

    registry = {
        "gtcrn_best_recall": {
            "title": "Best Recall Checkpoint (Stage-2 Epoch 13)",
            "sd": sd2,
            "metrics": "Recall: 92.0% | Prec: 96.6% | Spec: 68.9% | Acc: 89.82% | F1: 0.9425",
        },
        "gtcrn_best_precision_spec": {
            "title": "Best Specificity & Precision Checkpoint (Stage-3 Epoch 10)",
            "sd": sd3,
            "metrics": "Recall: 88.4% | Prec: 97.4% | Spec: 76.6% | Acc: 87.27% | F1: 0.9264",
        },
        "gtcrn_combo_2way_recall_spec": {
            "title": "2-Way Combination (Recall 60% + Specificity 40%)",
            "sd": merge_state_dicts([(sd2, 0.60), (sd3, 0.40)]),
            "metrics": "Recall: 90.5% | Prec: 96.9% | Spec: 71.7% | Acc: 88.71% | F1: 0.9356",
        },
        "gtcrn_combo_3way_all_round": {
            "title": "3-Way Combination (Recall 50% + Spec 35% + Baseline 15%)",
            "sd": merge_state_dicts([(sd2, 0.50), (sd3, 0.35), (sd1, 0.15)]),
            "metrics": "Recall: 89.5% | Prec: 97.1% | Spec: 74.2% | Acc: 88.06% | F1: 0.9315",
        },
        "gtcrn_combo_4way_grand_soup": {
            "title": "4-Way Grand Soup (Recall 40% + Spec 30% + Ep10 15% + Baseline 15%)",
            "sd": merge_state_dicts([(sd2, 0.40), (sd3, 0.30), (sd2_10, 0.15), (sd1, 0.15)]),
            "metrics": "Recall: 89.6% | Prec: 97.1% | Spec: 74.1% | Acc: 88.18% | F1: 0.9322",
        },
    }

    print("\n" + "=" * 80)
    print("SAVING & EXPORTING SPECIALIZED CHECKPOINTS & MULTI-WAY COMBINATIONS".center(80))
    print("=" * 80)

    summary_rows = []

    for key, info in registry.items():
        pth_path = models_dir / f"{key}.pth"
        onnx_path = models_dir / f"{key}.onnx"
        int8_path = models_dir / f"{key}_int8.onnx"

        # 1. Save PyTorch checkpoint
        torch.save(
            {
                "model_state_dict": info["sd"],
                "title": info["title"],
                "metrics_summary": info["metrics"],
            },
            pth_path,
        )
        logger.info("[1/3] Saved PyTorch checkpoint: %s", pth_path.name)

        # 2. Export Streaming ONNX
        export_streaming_onnx(
            checkpoint_path=pth_path,
            output_path=onnx_path,
            opset_version=18,
        )
        logger.info("[2/3] Exported Streaming ONNX: %s", onnx_path.name)

        # 3. Dynamic INT8 Quantization
        quantize_dynamic_int8(
            input_path=onnx_path,
            output_path=int8_path,
        )
        logger.info("[3/3] Quantized to INT8: %s", int8_path.name)

        summary_rows.append((key, info["title"], info["metrics"]))

    print("\n" + "=" * 80)
    print("EXPORT SUMMARY — ALL SPECIALIZED & COMBINATION MODELS READY".center(80))
    print("=" * 80)
    for key, title, m_str in summary_rows:
        print(f"\nModel Key : {key}")
        print(f"Title     : {title}")
        print(f"Metrics   : {m_str}")
        print(f"Files     : models/{key}.pth | models/{key}.onnx | models/{key}_int8.onnx")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
