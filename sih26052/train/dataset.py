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
from typing import Any, TYPE_CHECKING, Sequence

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch.utils.data import Dataset
else:
    try:
        from torch.utils.data import Dataset
    except ImportError:
        class Dataset:
            pass


class SpeechEnhancementDataset(Dataset):
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
        clean_dir: str | Path | Sequence[str | Path],
        noise_dirs: dict[str, Any],
        crop_samples: int = 32000,
        snr_range: tuple[float, float] = (-10.0, 15.0),
        composition: dict[str, float] | None = None,
        seed: int | None = None,
        n_pairs: int = 10000,
        augment: bool = False,
        composite_mixing: bool = False,
        mixing_probabilities: dict[str, float] | None = None,
        clean_ratio: float | None = 0.40,
        noise_ratio: float | None = 0.60,
        snr_dist: str = "uniform",
        gain_jitter: bool = False,
        simulate_reverb: bool = False,
        simulate_codec: bool = False,
    ):
        """
        Parameters
        ----------
        clean_dir            : directory, json manifest path, or list of clean 16kHz mono WAV paths
        noise_dirs           : mapping from noise category to directory or list of file paths
                               e.g. {"background": "...", "environmental": "...", "gunfire": "..."}
        crop_samples         : crop length (32000 = 2s at 16kHz)
        snr_range            : uniform SNR range in dB
        composition          : category mixing ratios (for single category mixing)
        seed                 : RNG seed
        n_pairs              : virtual dataset size per epoch
        augment              : whether to apply audio augmentations to clean speech
        composite_mixing     : whether to use multi-layer defense mixing (Types A-F)
        mixing_probabilities : probabilities for Types A-F if composite_mixing is True
        clean_ratio          : clean speech proportion (default 0.40 = 40%)
        noise_ratio          : noise/background proportion (default 0.60 = 60%)
        snr_dist             : 'uniform' or 'gaussian'
        gain_jitter          : randomize clean speech RMS level (-32 to -16 dBFS)
        simulate_reverb      : simulate cabin/room acoustic reflections
        simulate_codec       : simulate streaming codec companding & quantization
        """
        self.crop_samples = crop_samples
        self.snr_range = snr_range
        self.n_pairs = n_pairs
        self.rng = np.random.default_rng(seed)
        self.composite_mixing = composite_mixing
        self.mixing_probabilities = mixing_probabilities
        self.clean_ratio = clean_ratio
        self.noise_ratio = noise_ratio
        self.snr_dist = snr_dist
        self.gain_jitter = gain_jitter
        self.simulate_reverb = simulate_reverb
        self.simulate_codec = simulate_codec



        # Default composition: match available noise categories or fallback to 40/40/20
        if composition is not None:
            self.composition = composition
        elif noise_dirs:
            n_cats = len(noise_dirs)
            self.composition = {cat: 1.0 / n_cats for cat in noise_dirs}
        else:
            self.composition = {
                "broadband": 0.4,
                "impulsive": 0.4,
                "stationary": 0.2,
            }

        if augment:
            from sih26052.data.augment import AugmentPipeline
            self.augment_pipeline = AugmentPipeline()
        else:
            self.augment_pipeline = None

        # ── Discover clean files ──
        if isinstance(clean_dir, (str, Path)):
            clean_p = Path(clean_dir)
            if clean_p.is_file() and clean_p.suffix == ".json":
                with open(clean_p, "r", encoding="utf-8") as f:
                    file_list = json.load(f)
                self.clean_files = [Path(p) for p in file_list if Path(p).exists()]
            else:
                self.clean_files = sorted([p for p in clean_p.rglob("*.wav") if not p.name.startswith(".")])
        else:
            self.clean_files = [Path(p) for p in clean_dir if Path(p).exists()]

        assert len(self.clean_files) > 0, f"No clean WAVs found in {clean_dir}"

        # ── Discover noise files ──
        self.noise_files: dict[str, list[Path]] = {}
        for category, noise_source in noise_dirs.items():
            if isinstance(noise_source, (list, tuple)):
                files = [Path(p) for p in noise_source if Path(p).exists()]
            else:
                ns_p = Path(noise_source)
                if ns_p.is_file() and ns_p.suffix == ".json":
                    with open(ns_p, "r", encoding="utf-8") as f:
                        file_list = json.load(f)
                    files = [Path(p) for p in file_list if Path(p).exists()]
                elif ns_p.is_dir():
                    files = sorted([p for p in ns_p.rglob("*.wav") if not p.name.startswith(".")])
                else:
                    files = []

            if files:
                self.noise_files[category] = files
                logger.info("Noise category '%s': %d files", category, len(files))
            else:
                logger.warning("No WAVs found for noise category '%s' in %s", category, noise_source)

        logger.info(
            "Dataset: %d clean files, %d noise categories, %d virtual pairs/epoch (composite=%s)",
            len(self.clean_files), len(self.noise_files), n_pairs, composite_mixing,
        )

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, index: int) -> dict:
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
        from sih26052.data.mixer import (
            mix_at_snr,
            mix_defense_sample,
            sample_snr,
            apply_speech_gain_jitter,
            apply_synthetic_reverb,
            apply_simulated_codec,
        )

        # ── Random clean file ──
        clean_idx = int(self.rng.integers(0, len(self.clean_files)))
        clean_path = self.clean_files[clean_idx]
        clean, sr = sf.read(str(clean_path), dtype="float32")
        if clean.ndim > 1:
            clean = clean.mean(axis=1)

        if self.augment_pipeline is not None:
            clean = self.augment_pipeline(clean, sr, rng=self.rng)

        # ── Crop clean to target length ──
        clean = self._crop_or_pad(clean, self.crop_samples)

        # ── Optional Reverberation Simulation (cabin / vehicle reflections) ──
        if self.simulate_reverb and (self.rng.random() < 0.35):
            clean = apply_synthetic_reverb(clean, sr=16000, rng=self.rng)

        # ── Optional Codec Companding & Quantization Simulation ──
        if self.simulate_codec and (self.rng.random() < 0.40):
            clean = apply_simulated_codec(clean, sr=16000, rng=self.rng)

        # ── Optional Speech Gain Jitter ──
        if self.gain_jitter:
            clean = apply_speech_gain_jitter(clean, rng=self.rng)

        # ── Sample SNR according to distribution ──
        snr_db = sample_snr(self.snr_range, dist=self.snr_dist, rng=self.rng)

        # ── Composite defense mixing (Types A-F) ──
        if self.composite_mixing:
            bg_audio = None
            if "background" in self.noise_files and self.noise_files["background"]:
                bg_files = self.noise_files["background"]
                bg_path = bg_files[int(self.rng.integers(0, len(bg_files)))]
                bg_audio, _ = sf.read(str(bg_path), dtype="float32")
                if bg_audio.ndim > 1:
                    bg_audio = bg_audio.mean(axis=1)

            ev_audio = None
            if "environmental" in self.noise_files and self.noise_files["environmental"]:
                ev_files = self.noise_files["environmental"]
                ev_path = ev_files[int(self.rng.integers(0, len(ev_files)))]
                ev_audio, _ = sf.read(str(ev_path), dtype="float32")
                if ev_audio.ndim > 1:
                    ev_audio = ev_audio.mean(axis=1)

            gf_audio = None
            if "gunfire" in self.noise_files and self.noise_files["gunfire"]:
                gf_files = self.noise_files["gunfire"]
                gf_path = gf_files[int(self.rng.integers(0, len(gf_files)))]
                gf_audio, _ = sf.read(str(gf_path), dtype="float32")
                if gf_audio.ndim > 1:
                    gf_audio = gf_audio.mean(axis=1)

            noisy, clean_out, meta = mix_defense_sample(
                clean,
                background=bg_audio,
                event=ev_audio,
                gunfire=gf_audio,
                mixture_type="auto",
                snr_db=snr_db,
                rng=self.rng,
                probabilities=self.mixing_probabilities,
                clean_ratio=self.clean_ratio,
                noise_ratio=self.noise_ratio,
            )

            return {
                "noisy": torch.from_numpy(noisy),
                "clean": torch.from_numpy(clean_out),
                "snr_db": float(meta.get("snr_db", snr_db)),
                "noise_class": str(meta.get("mixture_type", "composite")),
            }

        # ── Standard single category mixing ──
        category = self._sample_category()

        if category in self.noise_files and self.noise_files[category]:
            n_files = self.noise_files[category]
            noise_idx = int(self.rng.integers(0, len(n_files)))
            noise_path = n_files[noise_idx]
            noise, sr_n = sf.read(str(noise_path), dtype="float32")
            if noise.ndim > 1:
                noise = noise.mean(axis=1)
            is_impulsive = (category in ["impulsive", "gunfire"])
        else:
            # Fallback: white noise
            noise = self.rng.standard_normal(len(clean)).astype(np.float32) * 0.3
            is_impulsive = False

        if self.clean_ratio is not None and self.noise_ratio is not None:
            from sih26052.data.mixer import mix_at_ratio
            noisy, clean_out = mix_at_ratio(
                clean, noise,
                clean_ratio=self.clean_ratio,
                noise_ratio=self.noise_ratio,
                impulsive=is_impulsive, rng=self.rng,
            )
            out_snr = float(20.0 * np.log10(self.clean_ratio / self.noise_ratio))
        else:
            noisy, clean_out = mix_at_snr(
                clean, noise, snr_db,
                impulsive=is_impulsive, rng=self.rng,
            )
            out_snr = float(snr_db)

        return {
            "noisy": torch.from_numpy(noisy),
            "clean": torch.from_numpy(clean_out),
            "snr_db": out_snr,
            "noise_class": category,
        }

    def _sample_category(self) -> str:
        """Sample a noise category according to composition ratios."""
        categories = list(self.composition.keys())
        probs = np.array([self.composition[c] for c in categories])
        probs /= probs.sum()  # normalize
        return str(self.rng.choice(categories, p=probs))

    def _crop_or_pad(self, audio: np.ndarray, target_len: int) -> np.ndarray:
        """Crop to target length (random start) or zero-pad if too short."""
        if len(audio) > target_len:
            start = int(self.rng.integers(0, len(audio) - target_len + 1))
            return audio[start:start + target_len]
        elif len(audio) < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[:len(audio)] = audio
            return padded
        return audio


class FixedValidationDataset(Dataset):
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

    def __getitem__(self, index: int) -> dict:
        import torch
        entry = self.entries[index]
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
