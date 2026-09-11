#!/usr/bin/env python3
"""
run_ablation.py — Orchestrate GTCRN ablation experiments.

Ablation Suite:
    1. experiment_001_random_baseline      — Uniform random SNR & static mixing
    2. experiment_002_adaptive_snr         — Adaptive SNR sampling
    3. experiment_003_adaptive_noise       — Adaptive noise category sampling
    4. experiment_004_adaptive_snr_noise   — Joint adaptive SNR & noise category sampling
    5. experiment_005_full_adaptive        — Full adaptive policy (SNR, category, gain, overlap)

Usage:
    python scripts/run_ablation.py --quick                  # 2 epochs each for testing
    python scripts/run_ablation.py --experiments 1 2       # Run specific experiments
    python scripts/run_ablation.py --dry-run               # Inspect commands only
    python scripts/run_ablation.py --epochs 30             # Full 30-epoch training suite
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Auto-reexec with virtual environment if needed
if sys.prefix == sys.base_prefix:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = REPO_ROOT.parent / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ablation_runner")

EXPERIMENTS: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "experiment_001_random_baseline",
        "title": "Random Baseline (Uniform SNR & Fixed Category Weights)",
        "mode": "random",
        "extra_args": [
            "--snr-range", "-10.0", "15.0",
            "--clean-ratio", "0.40",
            "--noise-ratio", "0.60",
        ],
    },
    2: {
        "name": "experiment_002_adaptive_snr",
        "title": "Adaptive SNR (Curriculum / Hard-SNR Mining)",
        "mode": "adaptive",
        "extra_args": [
            "--clean-ratio", "0.40",
            "--noise-ratio", "0.60",
        ],
    },
    3: {
        "name": "experiment_003_adaptive_noise",
        "title": "Adaptive Noise Category (Adversarial Category Weighting)",
        "mode": "adaptive",
        "extra_args": [
            "--clean-ratio", "0.40",
            "--noise-ratio", "0.60",
        ],
    },
    4: {
        "name": "experiment_004_adaptive_snr_noise",
        "title": "Joint Adaptive SNR & Noise Category",
        "mode": "adaptive",
        "extra_args": [
            "--clean-ratio", "0.40",
            "--noise-ratio", "0.60",
        ],
    },
    5: {
        "name": "experiment_005_full_adaptive",
        "title": "Full Adaptive Policy (SNR, Noise Category, Gain, Overlap)",
        "mode": "adaptive",
        "extra_args": [
            "--clean-ratio", "0.40",
            "--noise-ratio", "0.60",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GTCRN ablation experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiments", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="Experiment IDs to run (1-5)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per experiment")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n-pairs", type=int, default=10000, help="Pairs per epoch")
    parser.add_argument("--quick", action="store_true", help="Quick test run (2 epochs, 100 pairs)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip if experiment training_report.json already exists")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    return parser.parse_args()


def run_experiment(exp_id: int, exp_info: Dict[str, Any], args: argparse.Namespace) -> bool:
    exp_dir = REPO_ROOT / "experiments" / exp_info["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    report_file = exp_dir / "training_report.json"
    log_file = exp_dir / "training.log"

    if args.skip_existing and report_file.exists():
        logger.info("⏩ Skipping %s (already completed, report exists at %s)",
                    exp_info["name"], report_file)
        return True

    python_bin = sys.executable
    train_script = REPO_ROOT / "sih26052" / "train" / "train.py"

    epochs = 2 if args.quick else args.epochs
    n_pairs = 100 if args.quick else args.n_pairs
    max_batches = 10 if args.quick else 0

    cmd = [
        python_bin, str(train_script),
        "--mode", exp_info["mode"],
        "--experiment-name", exp_info["name"],
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--n-pairs", str(n_pairs),
        "--device", args.device,
    ]
    if max_batches > 0:
        cmd.extend(["--max-batches-per-epoch", str(max_batches)])

    cmd.extend(exp_info["extra_args"])

    logger.info("=" * 80)
    logger.info("▶ Launching Experiment %d: %s", exp_id, exp_info["title"])
    logger.info("Directory: %s", exp_dir)
    logger.info("Command: %s", " ".join(cmd))
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("[DRY-RUN] Command not executed.")
        return True

    t0 = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
        proc.wait()

    elapsed = time.time() - t0
    if proc.returncode == 0:
        logger.info("✅ Experiment %d finished successfully in %.1f minutes.", exp_id, elapsed / 60)
        return True
    else:
        logger.error("❌ Experiment %d failed with return code %d.", exp_id, proc.returncode)
        return False


def main():
    args = parse_args()
    experiments_dir = REPO_ROOT / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Ablation Suite for experiments: %s", args.experiments)
    results = {}

    for exp_id in args.experiments:
        if exp_id not in EXPERIMENTS:
            logger.warning("Unknown experiment ID %d. Skipping.", exp_id)
            continue
        success = run_experiment(exp_id, EXPERIMENTS[exp_id], args)
        results[exp_id] = success
        if not success and not args.dry_run:
            logger.warning("Experiment %d did not complete cleanly.", exp_id)

    logger.info("=" * 80)
    logger.info("ABLATION SUITE SUMMARY:")
    for exp_id, ok in results.items():
        name = EXPERIMENTS[exp_id]["name"]
        status = "PASSED" if ok else "FAILED"
        logger.info("  Exp %d (%s): %s", exp_id, name, status)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
