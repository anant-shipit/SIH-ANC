#!/usr/bin/env python3
"""
train_and_export_unified.py — End-to-end training, evaluation, ONNX export & benchmark.

1. Runs 15 epochs of multi-objective fine-tuning starting from models/gtcrn_combo_2way.pth.
2. Evaluates the best checkpoint on data/test_manifest.jsonl for Recall, Precision, Specificity, and Accuracy.
3. Automatically exports to PyTorch (.pth), Streaming FP32 ONNX (.onnx), and INT8 ONNX (_int8.onnx).
4. Measures RTF and latency via sih26052.export.benchmark.
"""
import json
import logging
import os
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
from sih26052.export.benchmark import benchmark_rtf
from sih26052.train.train import main as train_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_unified")


def evaluate_checkpoint_confusion(ckpt_path: Path, manifest_path: Path, spectral_floor: float = 0.025) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GTCRN().to(device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    model.load_state_dict(sd, strict=False)
    model.eval()

    nfft = 512
    hop = 256
    window = torch.hann_window(nfft).pow(0.5).to(device)

    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))

    m_dir = manifest_path.resolve().parent
    total_tp = total_fp = total_tn = total_fn = 0.0

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

        with torch.no_grad():
            noisy_t = torch.from_numpy(noisy[:min_len].astype(np.float32)).unsqueeze(0).to(device)
            noisy_stft = torch.view_as_real(torch.stft(noisy_t, nfft, hop, window=window, return_complex=True))
            pred_stft = model(noisy_stft)
            if spectral_floor > 0.0:
                pred_stft = pred_stft + noisy_stft * spectral_floor
            pred_complex = torch.complex(pred_stft[..., 0], pred_stft[..., 1])
            enh = torch.istft(pred_complex, nfft, hop, window=window, length=min_len).squeeze(0).cpu().numpy()

        clf = compute_classification_metrics(clean[:min_len], enh, frame_len=nfft, hop_len=hop, threshold_db=-35.0)
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
        "total_frames": int(total_frames),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "tn": int(total_tn),
        "fn": int(total_fn),
    }


def main():
    exp_name = "gtcrn_multi_objective_run"
    exp_dir = REPO_ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt = REPO_ROOT / "models" / "gtcrn_combo_2way.pth"
    if not base_ckpt.exists():
        base_ckpt = REPO_ROOT / "experiments" / "gtcrn_stage2_finetuned" / "checkpoint_best.pth"

    logger.info("=" * 80)
    logger.info("STARTING GTCRN MULTI-OBJECTIVE TRAINING RUN (15 EPOCHS)")
    logger.info("  Base Checkpoint : %s", base_ckpt)
    logger.info("  Experiment Dir  : %s", exp_dir)
    logger.info("=" * 80)

    # 1. Run Training
    cli_args = [
        "train.py",
        "--finetune-from", str(base_ckpt),
        "--epochs", "15",
        "--batch-size", "16",
        "--lr", "2e-5",
        "--scheduler", "cosine",
        "--warmup-epochs", "1",
        "--weight-decay", "0.01",
        "--si-snr-weight", "0.2",
        "--asym-penalty", "2.5",
        "--silence-penalty", "2.0",
        "--spectral-floor", "0.025",
        "--multi-res-loss",
        "--multi-res-weight", "0.4",
        "--loss-warmup-epochs", "0",
        "--snr-dist", "gaussian",
        "--gain-jitter",
        "--simulate-reverb",
        "--simulate-codec",
        "--experiment-name", exp_name,
        "--n-pairs", "10000",
        "--val-samples", "50",
    ]
    sys.argv = cli_args
    train_main()

    # 2. Identify Best Checkpoint
    best_pth = exp_dir / "checkpoint_best.pth"
    if not best_pth.exists():
        candidates = sorted(exp_dir.glob("checkpoint_best_epoch*.pth"))
        best_pth = candidates[-1] if candidates else exp_dir / "checkpoint_latest.pth"

    logger.info("Training completed! Best checkpoint: %s", best_pth)

    # 3. Evaluate Confusion Matrix on test manifest
    manifest_path = REPO_ROOT / "data" / "test_manifest.jsonl"
    logger.info("Evaluating on test manifest: %s", manifest_path)
    metrics = evaluate_checkpoint_confusion(best_pth, manifest_path)

    print("\n" + "=" * 80)
    print("                    TRAINED MODEL EVALUATION RESULTS                     ")
    print("=" * 80)
    print(f"  • Accuracy (Overall Correctness) : {metrics['accuracy']:.2f}%")
    print(f"  • Recall (Speech Preservation)   : {metrics['recall']:.1f}%  (Target: >90%)")
    print(f"  • Precision (Speech Purity)      : {metrics['precision']:.1f}%")
    print(f"  • Specificity (Noise Suppression): {metrics['specificity']:.1f}%")
    print(f"  • F1-Score                       : {metrics['f1']:.4f}")
    print(f"  • Confusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")
    print("=" * 80)

    # 4. Save & Export Models to models/
    models_dir = REPO_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    dest_pth = models_dir / "gtcrn_trained_best.pth"
    dest_onnx = models_dir / "gtcrn_trained_best.onnx"
    dest_int8 = models_dir / "gtcrn_trained_best_int8.onnx"

    # Save PTH
    sd = torch.load(str(best_pth), map_location="cpu", weights_only=False)
    state_dict = sd.get("model_state_dict", sd.get("state_dict", sd.get("model", sd)))
    torch.save(
        {
            "model_state_dict": state_dict,
            "title": "GTCRN Trained Multi-Objective Model (15 Epochs)",
            "metrics": metrics,
        },
        dest_pth,
    )
    logger.info("[1/3] Saved model checkpoint to %s", dest_pth.name)

    # Export Streaming ONNX
    export_streaming_onnx(
        checkpoint_path=dest_pth,
        output_path=dest_onnx,
        opset_version=18,
    )
    logger.info("[2/3] Exported Streaming ONNX to %s", dest_onnx.name)

    # Quantize to INT8
    quantize_dynamic_int8(
        input_path=dest_onnx,
        output_path=dest_int8,
    )
    logger.info("[3/3] Quantized to Dynamic INT8 ONNX: %s", dest_int8.name)

    # 5. Benchmark RTF
    logger.info("Benchmarking RTF on CPU (60s streaming)...")
    bench = benchmark_rtf(dest_int8, duration_s=60.0, n_runs=3)
    logger.info("Benchmark complete: Median RTF = %.4f (%.3f ms/frame)", bench["rtf_median"], bench["inference_ms_per_frame"])

    print("\n" + "=" * 80)
    print("GTCRN TRAINING, EXPORT, AND BENCHMARK COMPLETE".center(80))
    print(f"  PyTorch Checkpoint : models/{dest_pth.name}")
    print(f"  Streaming ONNX     : models/{dest_onnx.name}")
    print(f"  INT8 Quantized ONNX: models/{dest_int8.name}")
    print(f"  Accuracy           : {metrics['accuracy']:.2f}%")
    print(f"  Recall             : {metrics['recall']:.1f}%")
    print(f"  Precision          : {metrics['precision']:.1f}%")
    print(f"  Specificity        : {metrics['specificity']:.1f}%")
    print(f"  F1-Score           : {metrics['f1']:.4f}")
    print(f"  RTF                : {bench['rtf_median']:.4f} ({bench['fps']:.0f} FPS, {bench['inference_ms_per_frame']:.3f} ms/frame)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
