"""
dccrn_dataset.py — Paired Speech Dataset for audio enhancement model training and evaluation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class PairedSpeechDataset(Dataset):
    """Loads paired (noisy, clean) speech WAV files for training or evaluation."""

    def __init__(
        self,
        clean_dir: str | Path,
        noisy_dir: str | Path,
        crop_samples: int = 32000,
        is_train: bool = True,
        max_samples: Optional[int] = None,
        target_sr: int = 16000,
    ):
        self.crop_samples = crop_samples
        self.is_train = is_train
        self.target_sr = target_sr
        self.pairs: List[Tuple[Path, Path, str]] = []

        clean_path = Path(clean_dir)
        noisy_path = Path(noisy_dir)

        # 1. If manifests were provided (JSON or JSONL)
        if clean_path.is_file() and clean_path.suffix == ".jsonl":
            with open(clean_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        c_p = Path(item["clean"])
                        n_p = Path(item["noisy"])
                        self.pairs.append((n_p, c_p, c_p.stem))
        elif clean_path.exists() and noisy_path.exists() and clean_path.is_dir() and noisy_path.is_dir():
            clean_files = {f.name: f for f in clean_path.glob("*.wav")}
            noisy_files = {f.name: f for f in noisy_path.glob("*.wav")}
            common = sorted(set(clean_files.keys()) & set(noisy_files.keys()))

            for name in common:
                self.pairs.append((noisy_files[name], clean_files[name], Path(name).stem))
        elif clean_path.exists() and clean_path.is_dir():
            # Fallback scan
            for c_f in sorted(clean_path.glob("*.wav")):
                n_f = noisy_path / c_f.name if noisy_path.exists() else c_f
                self.pairs.append((n_f, c_f, c_f.stem))

        if max_samples is not None and max_samples > 0:
            self.pairs = self.pairs[:max_samples]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        noisy_p, clean_p, name = self.pairs[index]

        try:
            clean_wav, sr_c = sf.read(str(clean_p), dtype="float32")
            noisy_wav, sr_n = sf.read(str(noisy_p), dtype="float32")
        except Exception:
            clean_wav = np.zeros(self.crop_samples, dtype=np.float32)
            noisy_wav = np.zeros(self.crop_samples, dtype=np.float32)

        if clean_wav.ndim > 1:
            clean_wav = clean_wav.mean(axis=1)
        if noisy_wav.ndim > 1:
            noisy_wav = noisy_wav.mean(axis=1)

        min_len = min(len(clean_wav), len(noisy_wav))
        if min_len < self.crop_samples:
            pad_len = self.crop_samples - min_len
            clean_wav = np.pad(clean_wav[:min_len], (0, pad_len))
            noisy_wav = np.pad(noisy_wav[:min_len], (0, pad_len))
        elif self.is_train and min_len > self.crop_samples:
            offset = np.random.randint(0, min_len - self.crop_samples + 1)
            clean_wav = clean_wav[offset : offset + self.crop_samples]
            noisy_wav = noisy_wav[offset : offset + self.crop_samples]
        else:
            clean_wav = clean_wav[: self.crop_samples]
            noisy_wav = noisy_wav[: self.crop_samples]

        return (
            torch.from_numpy(noisy_wav.astype(np.float32)),
            torch.from_numpy(clean_wav.astype(np.float32)),
            name,
        )
