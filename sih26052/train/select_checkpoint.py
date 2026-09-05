"""
select_checkpoint.py — Best-by-PESQ checkpoint selector.

Scans a directory of checkpoints, runs the eval harness on each,
and picks the one with the highest overall PESQ.

Usage:
    python -m sih26052.train.select_checkpoint \\
        --checkpoint-dir models/checkpoints \\
        --val-manifest data/manifest_val.jsonl \\
        --output models/best_model.pth
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def select_best_checkpoint(
    checkpoint_dir: str | Path,
    val_manifest: str | Path,
    model_factory,
    device: torch.device | None = None,
    max_pairs: int | None = 100,
) -> tuple[Path, dict]:
    """Evaluate all checkpoints and return the best one.

    Parameters
    ----------
    checkpoint_dir : directory containing checkpoint_epoch_NNN.pth files
    val_manifest   : path to validation manifest
    model_factory  : callable that returns an untrained model instance
    device         : torch device
    max_pairs      : limit eval pairs for speed

    Returns
    -------
    best_path   : path to the best checkpoint
    best_metrics : dict of its metrics
    """
    from sih26052.train.validate import validate_epoch

    checkpoint_dir = Path(checkpoint_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    logger.info("Found %d checkpoints in %s", len(checkpoints), checkpoint_dir)

    best_path = None
    best_pesq = -float("inf")
    best_metrics = {}
    all_results = []

    for ckpt_path in checkpoints:
        logger.info("Evaluating %s...", ckpt_path.name)

        model = model_factory().to(device)
        state = torch.load(str(ckpt_path), map_location=device)

        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
            epoch = state.get("epoch", 0)
        else:
            model.load_state_dict(state, strict=False)
            epoch = 0

        metrics = validate_epoch(
            model, val_manifest, device, epoch=epoch, max_pairs=max_pairs,
        )

        pesq_out = metrics.get("overall_pesq_out", 0.0)
        all_results.append({"checkpoint": ckpt_path.name, "pesq": pesq_out, **metrics})

        if pesq_out > best_pesq:
            best_pesq = pesq_out
            best_path = ckpt_path
            best_metrics = metrics

        logger.info("  PESQ=%.3f (best so far: %.3f from %s)",
                    pesq_out, best_pesq, best_path.name if best_path else "—")

    logger.info("=" * 60)
    logger.info("Best checkpoint: %s (PESQ=%.3f)", best_path.name, best_pesq)

    return best_path, best_metrics


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Select best checkpoint by PESQ.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Copy best checkpoint to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import sys
    import shutil
    
    logger.info("Select best checkpoint from %s", args.checkpoint_dir)
    
    checkpoints = list(args.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        logger.error("No checkpoints found in %s", args.checkpoint_dir)
        sys.exit(1)
        
    gtcrn_root = checkpoints[0].parent.parent
    
    stream_dir = gtcrn_root / "stream"
    if stream_dir.exists():
        sys.path.insert(0, str(stream_dir))
    sys.path.insert(0, str(gtcrn_root))
    
    try:
        from gtcrn import GTCRN  # type: ignore
    except ImportError:
        logger.error("Could not import GTCRN from %s. Is the checkpoint path correct?", gtcrn_root)
        sys.exit(1)
        
    best_path, best_metrics = select_best_checkpoint(
        args.checkpoint_dir, args.val_manifest, model_factory=GTCRN
    )
    
    if args.output and best_path:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, args.output)
        logger.info("Copied best checkpoint to %s", args.output)


if __name__ == "__main__":
    main()
