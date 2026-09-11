#!/usr/bin/env python3
"""
train.py — Entry point for GTCRN training.

Thin wrapper that delegates to sih26052.train.train:main().

Usage:
    python scripts/train.py                      # defaults
    python scripts/train.py --epochs 100         # long run
    python scripts/train.py --config configs/gtcrn_defense.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
GTCRN_DIR = REPO_ROOT / "models" / "gtcrn"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if GTCRN_DIR.exists() and str(GTCRN_DIR) not in sys.path:
    sys.path.insert(0, str(GTCRN_DIR))

from sih26052.train.train import main

if __name__ == "__main__":
    main()
