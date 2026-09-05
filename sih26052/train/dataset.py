"""
dataset.py — On-the-fly mixing DataLoader for training.

Adapted from ~/Downloads/SEtrain/ patterns.

Key design:
    - Training: on-the-fly mixing in __getitem__ for infinite variety.
    - Validation: pre-generated fixed pairs for consistent epoch-to-epoch comparison.
    - Crops to 2 seconds (32,000 samples at 16kHz).
    - Batch composition: 40% broadband, 40% impulsive, 20% stationary defense.

Why on-the-fly mixing for training?
    Pre-generating all pairs would require terabytes of disk space.
    On-the-fly mixing with random SNR draws creates effectively infinite
    training data from a finite set of clean+noise files.

Why fixed validation?
    Because if validation mixes are different each epoch, we can't tell
    whether metric changes are from model improvement or noise variation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class SpeechEnhancementDataset:
    """On-the-fly mixing dataset for GTCRN training.

    Usage (with PyTorch DataLoader):
        dataset = SpeechEnhancementDataset(
            clean_dir="data/processed/clean",
            noise_dirs={"broadband": "data/processed/DEMAND", ...},
            crop_samples=32000,
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
    """

    def __init__(
        self,
        clean_dir: str | Path,
        noise_dirs: dict[str, str | Path],
        crop_samples: int = 32000,
        snr_range: tuple[float, float] = (-5.0, 20.0),
        composition: dict[str, float] | None = None,
        seed: int | None = None,
        n_pairs: int = 10000,
        augment: bool = False,
    ):
        """
        Parameters
        ----------
        clean_dir     : directory of clean 16kHz mono WAVs
        noise_dirs    : mapping from noise category to directory
                        e.g. {"broadband": "...", "impulsive": "...", "stationary": "..."}
        crop_samples  : crop length (32000 = 2s at 16kHz)
        snr_range     : uniform SNR range in dB
        composition   : category mixing ratios (default: 40/40/20)
        seed          : RNG seed
        n_pairs       : virtual dataset size per epoch
        """
        self.crop_samples = crop_samples
        self.snr_range = snr_range
        self.n_pairs = n_pairs
        self.rng = np.random.default_rng(seed)

        # Default composition: 40% broadband, 40% impulsive, 20% stationary
        self.composition = composition or {
            "broadband": 0.4,
            "impulsive": 0.4,
            "stationary": 0.2,
        }

        if augment:
            from sih26052.data.augment import AugmentPipeline
            self.augment_pipeline = AugmentPipeline()
        else:
            self.augment_pipeline = None

        # ── Discover files ──
        self.clean_files = sorted(Path(clean_dir).rglob("*.wav"))
        assert len(self.clean_files) > 0, f"No clean WAVs found in {clean_dir}"

        self.noise_files: dict[str, list[Path]] = {}
        for category, noise_dir in noise_dirs.items():
            files = sorted(Path(noise_dir).rglob("*.wav"))
            if files:
                self.noise_files[category] = files
                logger.info("Noise category '%s': %d files", category, len(files))
            else:
                logger.warning("No WAVs found for noise category '%s' in %s", category, noise_dir)

        logger.info(
            "Dataset: %d clean files, %d noise categories, %d virtual pairs/epoch",
            len(self.clean_files), len(self.noise_files), n_pairs,
        )

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, idx: int) -> dict:
        """Generate one (noisy, clean) training pair on-the-fly.

        Returns
        -------
        dict with keys:
            'noisy'       : float32 tensor, shape (crop_samples,)
            'clean'       : float32 tensor, shape (crop_samples,)
            'snr_db'      : float
            'noise_class' : str
        """
        import torch
        from sih26052.data.mixer import mix_at_snr

        # ── Select noise category by composition ──
        category = self._sample_category()

        # ── Random clean file ──
        clean_path = self.rng.choice(self.clean_files)
        clean, sr = sf.read(str(clean_path), dtype="float32")
        if clean.ndim > 1:
            clean = clean.mean(axis=1)

        if self.augment_pipeline is not None:
            clean = self.augment_pipeline(clean, sr, rng=self.rng)

        # ── Random noise file ──
        if category in self.noise_files and self.noise_files[category]:
            noise_path = self.rng.choice(self.noise_files[category])
            noise, sr_n = sf.read(str(noise_path), dtype="float32")
            if noise.ndim > 1:
                noise = noise.mean(axis=1)
            is_impulsive = category == "impulsive"
        else:
            # Fallback: white noise
            noise = self.rng.standard_normal(len(clean)).astype(np.float32) * 0.3
            is_impulsive = False

        # ── Crop clean to target length ──
        clean = self._crop_or_pad(clean, self.crop_samples)

        # ── Random SNR ──
        snr_db = self.rng.uniform(self.snr_range[0], self.snr_range[1])

        # ── Mix ──
        noisy, clean_out = mix_at_snr(
            clean, noise, snr_db,
            impulsive=is_impulsive, rng=self.rng,
        )

        return {
            "noisy": torch.from_numpy(noisy),
            "clean": torch.from_numpy(clean_out),
            "snr_db": float(snr_db),
            "noise_class": category,
        }

    def _sample_category(self) -> str:
        """Sample a noise category according to composition ratios."""
        categories = list(self.composition.keys())
        probs = np.array([self.composition[c] for c in categories])
        probs /= probs.sum()  # normalize
        return self.rng.choice(categories, p=probs)

    def _crop_or_pad(self, audio: np.ndarray, target_len: int) -> np.ndarray:
        """Crop to target length (random start) or zero-pad if too short."""
        if len(audio) > target_len:
            start = self.rng.integers(0, len(audio) - target_len + 1)
            return audio[start:start + target_len]
        elif len(audio) < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[:len(audio)] = audio
            return padded
        return audio


class FixedValidationDataset:
    """Pre-generated fixed validation pairs.

    Unlike the training dataset, these pairs are generated once and
    reused every epoch.  This ensures consistent evaluation.
    """

    def __init__(self, manifest_path: str | Path):
        from sih26052.data.manifest import read_manifest
        self.entries = read_manifest(manifest_path)
        logger.info("Validation dataset: %d pairs from %s", len(self.entries), manifest_path)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        import torch
        entry = self.entries[idx]
        noisy, _ = sf.read(entry.noisy, dtype="float32")
        clean, _ = sf.read(entry.clean, dtype="float32")
        if noisy.ndim > 1:
            noisy = noisy.mean(axis=1)
        if clean.ndim > 1:
            clean = clean.mean(axis=1)

        return {
            "noisy": torch.from_numpy(noisy),
            "clean": torch.from_numpy(clean),
            "snr_db": entry.snr_db,
            "noise_class": entry.noise_class,
            "subset": entry.subset,
        }
