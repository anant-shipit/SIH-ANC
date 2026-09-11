#!/usr/bin/env python3
"""
compare_experiments.py — Compare GTCRN ablation experiments.

Scans the `experiments/` directory for `training_report.json` files,
tabulates key speech enhancement metrics (PESQ, STOI, SI-SNR, SI-SNRi),
and generates markdown and JSON comparison reports.

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --experiments-dir experiments
    python scripts/compare_experiments.py --output reports/ablation_comparison.md
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("compare_experiments")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare GTCRN ablation experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiments-dir", type=str, default=str(REPO_ROOT / "experiments"),
        help="Directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--output-md", type=str, default=str(REPO_ROOT / "experiments" / "comparison_table.md"),
        help="Path to write markdown comparison table",
    )
    parser.add_argument(
        "--output-json", type=str, default=str(REPO_ROOT / "experiments" / "comparison_summary.json"),
        help="Path to write JSON summary",
    )
    return parser.parse_args()


def load_experiments(exp_root: Path) -> List[Dict[str, Any]]:
    runs = []
    if not exp_root.exists():
        logger.warning("Experiments directory %s does not exist.", exp_root)
        return runs

    for d in sorted(exp_root.iterdir()):
        if not d.is_dir():
            continue
        report_path = d / "training_report.json"
        if not report_path.exists():
            continue

        try:
            with open(report_path) as f:
                data = json.load(f)
            data["exp_name"] = d.name
            data["exp_dir"] = str(d)
            runs.append(data)
        except Exception as e:
            logger.warning("Could not read %s: %s", report_path, e)

    return runs


def format_markdown_table(runs: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("# GTCRN Defense Noise Suppression — Ablation Comparison\n")
    lines.append("| Experiment | Mode | Best Ep | PESQ | STOI | SI-SNR (dB) | ΔSI-SNR (dB) | ΔPESQ |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for r in runs:
        name = r.get("exp_name", "Unknown")
        mode = r.get("mode", "unknown")
        best_ep = r.get("best_epoch", 0)
        final = r.get("final_metrics", {})
        init = r.get("initial_metrics", {})
        imp = r.get("improvement", {})

        pesq = final.get("pesq", 0.0)
        stoi = final.get("stoi", 0.0)
        sisnr = final.get("si_snr", 0.0)
        sisnri = final.get("si_snr_improvement", 0.0)
        d_pesq = imp.get("pesq", pesq - init.get("pesq", 0.0))

        lines.append(
            f"| `{name}` | {mode} | {best_ep} | **{pesq:.3f}** | {stoi:.3f} | {sisnr:.2f} | **{sisnri:+.2f}** | {d_pesq:+.3f} |"
        )

    # Per-SNR breakdown table if available
    has_snr = any(r.get("per_snr") for r in runs)
    if has_snr:
        lines.append("\n## Per-SNR SI-SDR Improvement (ΔSI-SDR dB)\n")
        # Collect all SNR levels
        all_snrs = set()
        for r in runs:
            if r.get("per_snr"):
                all_snrs.update(r["per_snr"].keys())
        snr_cols = sorted(list(all_snrs), key=lambda x: float(x))

        header = "| Experiment | " + " | ".join(f"{s} dB" for s in snr_cols) + " |"
        sep = "|---|" + "|".join(["---"] * len(snr_cols)) + "|"
        lines.append(header)
        lines.append(sep)

        for r in runs:
            row = [f"`{r.get('exp_name')}`"]
            ps = r.get("per_snr") or {}
            for s in snr_cols:
                val = ps.get(s, {}).get("si_sdr_improvement", None)
                if val is not None:
                    row.append(f"{val:+.2f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

    # Per-Noise breakdown table if available
    has_noise = any(r.get("per_noise") for r in runs)
    if has_noise:
        lines.append("\n## Per-Noise Type Performance (PESQ / SI-SDR dB)\n")
        all_cats = set()
        for r in runs:
            if r.get("per_noise"):
                all_cats.update(r["per_noise"].keys())
        noise_cols = sorted(list(all_cats))

        header = "| Experiment | " + " | ".join(noise_cols) + " |"
        sep = "|---|" + "|".join(["---"] * len(noise_cols)) + "|"
        lines.append(header)
        lines.append(sep)

        for r in runs:
            row = [f"`{r.get('exp_name')}`"]
            pn = r.get("per_noise") or {}
            for c in noise_cols:
                met = pn.get(c, {})
                pesq_val = met.get("pesq_output")
                sisdr_val = met.get("si_sdr_output")
                if pesq_val is not None and sisdr_val is not None:
                    row.append(f"{pesq_val:.2f} / {sisdr_val:.1f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    args = parse_args()
    exp_dir = Path(args.experiments_dir)
    runs = load_experiments(exp_dir)

    if not runs:
        logger.warning("No completed experiment reports found in %s.", exp_dir)
        return

    logger.info("Found %d completed experiment runs.", len(runs))
    md_content = format_markdown_table(runs)
    print("\n" + md_content + "\n")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md_content)
    logger.info("Saved comparison markdown to %s", out_md)

    out_json = Path(args.output_json)
    summary_data = {
        "num_experiments": len(runs),
        "experiments": [
            {
                "name": r.get("exp_name"),
                "mode": r.get("mode"),
                "best_epoch": r.get("best_epoch"),
                "final_metrics": r.get("final_metrics"),
                "improvement": r.get("improvement"),
            }
            for r in runs
        ],
    }
    with open(out_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info("Saved comparison summary JSON to %s", out_json)


if __name__ == "__main__":
    main()
