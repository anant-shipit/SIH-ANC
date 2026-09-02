"""
harness.py — Run metrics over a manifest, split by 3 subsets.

This is the core evaluation engine.  It reads a manifest, loads each
(noisy, clean) pair, optionally passes through a model, computes metrics,
and produces the headline table:

    | Subset         | PESQ in | PESQ out | Δ    | STOI in | STOI out | SI-SNR out |
    |----------------|---------|----------|------|---------|----------|------------|
    | Stationary     |         |          |      |         |          |            |
    | Impulsive      |         |          |      |         |          |            |
    | Real recordings|         |          |      |         |          |            |
    | **Overall**    |         |          |      |         |          |            |

Usage:
    from sih26052.eval.harness import run_eval_harness

    results = run_eval_harness(
        manifest_path="data/manifest.jsonl",
        enhance_fn=my_model_enhance,  # or None for "before" table
    )
    print(results.format_table())
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from sih26052.data.manifest import ManifestEntry, read_manifest
from sih26052.eval.alignment import align_signals
from sih26052.eval.metrics import MetricResult, compute_all_metrics

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────

@dataclasses.dataclass
class SubsetMetrics:
    """Aggregated metrics for one subset (e.g. "stationary")."""
    subset: str
    pesq_in_scores: list[float]     = dataclasses.field(default_factory=list)
    pesq_out_scores: list[float]    = dataclasses.field(default_factory=list)
    stoi_in_scores: list[float]     = dataclasses.field(default_factory=list)
    stoi_out_scores: list[float]    = dataclasses.field(default_factory=list)
    si_snr_out_scores: list[float]  = dataclasses.field(default_factory=list)
    pesq_skips: int = 0
    stoi_skips: int = 0
    count: int = 0

    @property
    def pesq_in_mean(self) -> float:
        return float(np.mean(self.pesq_in_scores)) if self.pesq_in_scores else 0.0

    @property
    def pesq_out_mean(self) -> float:
        return float(np.mean(self.pesq_out_scores)) if self.pesq_out_scores else 0.0

    @property
    def pesq_delta(self) -> float:
        return self.pesq_out_mean - self.pesq_in_mean

    @property
    def stoi_in_mean(self) -> float:
        return float(np.mean(self.stoi_in_scores)) if self.stoi_in_scores else 0.0

    @property
    def stoi_out_mean(self) -> float:
        return float(np.mean(self.stoi_out_scores)) if self.stoi_out_scores else 0.0

    @property
    def si_snr_out_mean(self) -> float:
        return float(np.mean(self.si_snr_out_scores)) if self.si_snr_out_scores else 0.0


@dataclasses.dataclass
class EvalResults:
    """Full evaluation results across all subsets."""
    subsets: dict[str, SubsetMetrics] = dataclasses.field(default_factory=dict)
    overall: SubsetMetrics = dataclasses.field(default_factory=lambda: SubsetMetrics("Overall"))

    def format_table(self) -> str:
        """Format the headline table as a string."""
        lines = []
        header = (
            f"{'Subset':<18} | {'PESQ in':>8} | {'PESQ out':>8} | "
            f"{'Δ':>6} | {'STOI in':>8} | {'STOI out':>8} | {'SI-SNR out':>10}"
        )
        sep = "-" * len(header)

        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        # Subsets in consistent order
        for subset_name in ["stationary", "impulsive", "real"]:
            if subset_name in self.subsets:
                sm = self.subsets[subset_name]
                lines.append(_format_row(sm))

        lines.append(sep)
        lines.append(_format_row(self.overall))
        lines.append(sep)

        return "\n".join(lines)


def _format_row(sm: SubsetMetrics) -> str:
    """Format one row of the results table."""
    return (
        f"{sm.subset:<18} | {sm.pesq_in_mean:>8.3f} | {sm.pesq_out_mean:>8.3f} | "
        f"{sm.pesq_delta:>+6.3f} | {sm.stoi_in_mean:>8.3f} | {sm.stoi_out_mean:>8.3f} | "
        f"{sm.si_snr_out_mean:>10.2f}"
    )


# ── Evaluation engine ─────────────────────────────────────────────────────

def run_eval_harness(
    manifest_path: str | Path,
    enhance_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    sr: int = 16000,
    max_pairs: int | None = None,
    align: bool = True,
) -> EvalResults:
    """Run the full evaluation harness.

    Parameters
    ----------
    manifest_path : path to the JSONL manifest
    enhance_fn    : function that takes noisy waveform → enhanced waveform.
                    If None, we compute "input" metrics only (noisy vs clean).
    sr            : sample rate (must match manifest files)
    max_pairs     : limit evaluation to this many pairs (for quick testing)
    align         : if True, align enhanced signal to reference via cross-correlation

    Returns
    -------
    EvalResults with per-subset and overall metrics.
    """
    entries = read_manifest(manifest_path)
    if max_pairs is not None:
        entries = entries[:max_pairs]

    results = EvalResults()

    for i, entry in enumerate(entries):
        try:
            _process_entry(entry, enhance_fn, sr, align, results)
        except Exception as exc:
            logger.warning("Failed on entry %d (%s): %s", i, entry.noisy, exc)

        if (i + 1) % 100 == 0:
            logger.info("Evaluated %d / %d pairs", i + 1, len(entries))

    logger.info(
        "Evaluation complete: %d pairs, PESQ skips=%d, STOI skips=%d",
        results.overall.count,
        results.overall.pesq_skips,
        results.overall.stoi_skips,
    )
    return results


def _process_entry(
    entry: ManifestEntry,
    enhance_fn: Callable | None,
    sr: int,
    align: bool,
    results: EvalResults,
) -> None:
    """Process a single manifest entry and accumulate metrics."""
    # Load audio
    noisy, sr_n = sf.read(entry.noisy, dtype="float32")
    clean, sr_c = sf.read(entry.clean, dtype="float32")

    # Mono safety
    if noisy.ndim > 1:
        noisy = noisy.mean(axis=1)
    if clean.ndim > 1:
        clean = clean.mean(axis=1)

    # ── Input metrics (noisy vs clean, before enhancement) ──
    input_metrics = compute_all_metrics(clean, noisy, sr)

    # ── Output metrics ──
    if enhance_fn is not None:
        enhanced = enhance_fn(noisy)
        if align:
            clean_aligned, enhanced_aligned, delay = align_signals(clean, enhanced)
            output_metrics = compute_all_metrics(clean_aligned, enhanced_aligned, sr)
        else:
            output_metrics = compute_all_metrics(clean, enhanced, sr)
    else:
        # No model — output = input (for "before" table)
        output_metrics = input_metrics

    # ── Accumulate ──
    subset = entry.subset
    if subset not in results.subsets:
        results.subsets[subset] = SubsetMetrics(subset)

    for sm in [results.subsets[subset], results.overall]:
        sm.count += 1

        if input_metrics.pesq is not None:
            sm.pesq_in_scores.append(input_metrics.pesq)
        else:
            sm.pesq_skips += 1

        if output_metrics.pesq is not None:
            sm.pesq_out_scores.append(output_metrics.pesq)

        if input_metrics.stoi is not None:
            sm.stoi_in_scores.append(input_metrics.stoi)
        else:
            sm.stoi_skips += 1

        if output_metrics.stoi is not None:
            sm.stoi_out_scores.append(output_metrics.stoi)

        sm.si_snr_out_scores.append(output_metrics.si_snr)
