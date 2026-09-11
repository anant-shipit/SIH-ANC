"""
confusion_matrix.py — Voice Activity Detection (VAD) & Signal Confusion Matrix computation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np


def compute_vad_frames(
    wav: np.ndarray,
    frame_len: int = 512,
    hop_len: int = 256,
    threshold_db: float = -35.0,
) -> np.ndarray:
    """Compute frame-level energy and threshold to determine binary voice activity."""
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)

    n_frames = 1 + (len(wav) - frame_len) // hop_len
    if n_frames <= 0:
        return np.array([False])

    frames = np.lib.stride_tricks.as_strided(
        wav,
        shape=(n_frames, frame_len),
        strides=(wav.strides[0] * hop_len, wav.strides[0]),
    )
    frame_energy = np.mean(frames ** 2, axis=1) + 1e-12
    peak_energy = np.max(frame_energy)
    energy_db = 10.0 * np.log10(frame_energy / (peak_energy + 1e-12))
    return energy_db > threshold_db


def compute_frame_confusion_matrix(
    clean: np.ndarray,
    enh: np.ndarray,
    noisy: Optional[np.ndarray] = None,
    frame_len: int = 512,
    hop_len: int = 256,
    threshold_db: float = -35.0,
) -> Dict[str, int]:
    """Compute frame-level TP, FP, TN, FN between clean target and enhanced output."""
    vad_clean = compute_vad_frames(clean, frame_len, hop_len, threshold_db)
    vad_enh = compute_vad_frames(enh, frame_len, hop_len, threshold_db)

    min_len = min(len(vad_clean), len(vad_enh))
    vad_clean = vad_clean[:min_len]
    vad_enh = vad_enh[:min_len]

    tp = int(np.sum((vad_clean == True) & (vad_enh == True)))
    fp = int(np.sum((vad_clean == False) & (vad_enh == True)))
    tn = int(np.sum((vad_clean == False) & (vad_enh == False)))
    fn = int(np.sum((vad_clean == True) & (vad_enh == False)))

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Total": tp + fp + tn + fn,
    }


def aggregate_confusion_matrices(cm_list: List[Dict[str, int]]) -> Dict[str, Any]:
    """Aggregate a list of confusion matrices and compute classification metrics."""
    tp = sum(cm.get("TP", 0) for cm in cm_list)
    fp = sum(cm.get("FP", 0) for cm in cm_list)
    tn = sum(cm.get("TN", 0) for cm in cm_list)
    fn = sum(cm.get("FN", 0) for cm in cm_list)
    total = tp + fp + tn + fn

    speech_preservation = float(tp / (tp + fn + 1e-12) * 100.0)
    noise_suppression = float(tn / (tn + fp + 1e-12) * 100.0)
    speech_distortion = float(fn / (tp + fn + 1e-12) * 100.0)
    noise_leakage = float(fp / (tn + fp + 1e-12) * 100.0)
    accuracy = float((tp + tn) / (total + 1e-12) * 100.0)
    precision = float(tp / (tp + fp + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))
    f1 = float(2 * precision * recall / (precision + recall + 1e-12))

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Total": total,
        "speech_preservation_recall": speech_preservation,
        "noise_suppression_specificity": noise_suppression,
        "speech_distortion_fn_rate": speech_distortion,
        "noise_leakage_fp_rate": noise_leakage,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def format_confusion_matrix_ascii(
    cm: Optional[Dict[str, Any]],
    pesq: Optional[float] = None,
    stoi: Optional[float] = None,
    si_snr: Optional[float] = None,
    delta_pesq: Optional[float] = None,
    model_name: str = "SPEECH ENHANCEMENT",
) -> str:
    """Format confusion matrix and acoustic metrics as a clean ASCII/Unicode table."""
    if cm is None:
        return "No confusion matrix data available."

    tp = cm.get("TP", 0)
    fp = cm.get("FP", 0)
    tn = cm.get("TN", 0)
    fn = cm.get("FN", 0)
    tot_speech = tp + fn
    tot_noise = tn + fp

    pct_tp = (tp / tot_speech * 100) if tot_speech > 0 else 0
    pct_fn = (fn / tot_speech * 100) if tot_speech > 0 else 0
    pct_fp = (fp / tot_noise * 100) if tot_noise > 0 else 0
    pct_tn = (tn / tot_noise * 100) if tot_noise > 0 else 0

    sp_recall = cm.get("speech_preservation_recall", pct_tp)
    ns_spec = cm.get("noise_suppression_specificity", pct_tn)
    sd_fn = cm.get("speech_distortion_fn_rate", pct_fn)
    nl_fp = cm.get("noise_leakage_fp_rate", pct_fp)
    acc = cm.get("accuracy", ((tp + tn) / max(tp + tn + fp + fn, 1) * 100))
    f1 = cm.get("f1_score", 0.0)

    lines = [
        "=" * 68,
        f"   {model_name.upper()} SPEECH ENHANCEMENT EVALUATION & CONFUSION MATRIX",
        "=" * 68,
        "                              PREDICTED / ENHANCED",
        "                         Speech Active      Suppressed/Silence",
        "  GROUND TRUTH       ┌────────────────────┬────────────────────┐",
        f"   Speech Frame      │ TP: {tp:7d} ({pct_tp:5.1f}%) │ FN: {fn:7d} ({pct_fn:5.1f}%) │",
        "                     ├────────────────────┼────────────────────┤",
        f"   Noise / Silence   │ FP: {fp:7d} ({pct_fp:5.1f}%) │ TN: {tn:7d} ({pct_tn:5.1f}%) │",
        "                     └────────────────────┴────────────────────┘",
        "-" * 68,
        "  VAD Performance Metrics:",
        f"    • Speech Preservation Rate (Recall) :  {sp_recall:5.2f}%  [Target: >90%]",
        f"    • Noise Suppression Rate (Spec.)    :  {ns_spec:5.2f}%  [Target: >85%]",
        f"    • Speech Distortion / Clipping (FN) :  {sd_fn:5.2f}%  [Target: <10%]",
        f"    • Noise Leakage Rate (FP)           :  {nl_fp:5.2f}%  [Target: <15%]",
        f"    • Overall Classification Accuracy   :  {acc:5.2f}%",
        f"    • Harmonic F1-Score                 :  {f1:.4f}",
    ]

    if pesq is not None or stoi is not None or si_snr is not None:
        lines.append("-" * 68)
        lines.append("  Acoustic Quality Metrics:")
        if pesq is not None:
            delta_str = f" (Δ: {delta_pesq:+.3f})" if delta_pesq is not None else ""
            lines.append(f"    • PESQ (Wideband MOS)               :  {pesq:.4f}{delta_str}")
        if stoi is not None:
            lines.append(f"    • STOI (Intelligibility)            :  {stoi:.4f}")
        if si_snr is not None:
            lines.append(f"    • SI-SNR                            :  {si_snr:.2f} dB")

    lines.append("=" * 68)
    return "\n".join(lines)
