"""
manifest.py — Write and read JSONL manifests for the dataset.

A manifest is a line-delimited JSON file where each line describes one
(noisy, clean) training pair.  Fields:

    {
        "noisy":       "data/mixed/train/pair_00042_noisy.wav",
        "clean":       "data/mixed/train/pair_00042_clean.wav",
        "snr_db":      7.3,
        "noise_class": "gunshot",
        "subset":      "impulsive",
        "duration_s":  3.21
    }

Why JSONL instead of CSV?
    - Robust to commas in filenames.
    - Easy to append without rewriting the whole file.
    - Human-readable and grep-friendly.

Why store durations?
    - Lets the DataLoader estimate total hours for logging.
    - Lets the harness report per-subset stats without re-loading audio.

Usage:
    from sih26052.data.manifest import ManifestWriter, read_manifest

    with ManifestWriter("data/manifest.jsonl") as mw:
        mw.write_entry(noisy_path, clean_path, 7.3, "gunshot", "impulsive", 3.21)

    entries = read_manifest("data/manifest.jsonl")
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────────

@dataclasses.dataclass
class ManifestEntry:
    """One row in the manifest."""
    noisy: str               # relative path to noisy WAV
    clean: str               # relative path to clean WAV
    snr_db: float
    noise_class: str         # e.g. "gunshot", "engine", "white_noise"
    subset: str              # "stationary" | "impulsive" | "real"
    duration_s: float = 0.0  # seconds — computed from audio length

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ManifestEntry:
        return cls(**d)


# ── Writer ─────────────────────────────────────────────────────────────────

class ManifestWriter:
    """Append-mode JSONL writer.

    Use as a context manager:
        with ManifestWriter("manifest.jsonl") as mw:
            mw.write_entry(...)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fp = None
        self._count = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        if self._fp:
            self._fp.close()
        logger.info("Manifest: wrote %d entries to %s", self._count, self.path)

    def write_entry(
        self,
        noisy_path: str | Path,
        clean_path: str | Path,
        snr_db: float,
        noise_class: str,
        subset: str,
        duration_s: float = 0.0,
    ) -> None:
        """Write a single entry to the manifest."""
        entry = ManifestEntry(
            noisy=str(noisy_path),
            clean=str(clean_path),
            snr_db=round(snr_db, 2),
            noise_class=noise_class,
            subset=subset,
            duration_s=round(duration_s, 3),
        )
        line = json.dumps(entry.to_dict(), ensure_ascii=False)
        self._fp.write(line + "\n")
        self._count += 1

    def write_from_mix_result(
        self,
        noisy_path: str | Path,
        clean_path: str | Path,
        snr_db: float,
        noise_class: str,
        subset: str,
        sr: int = 16000,
    ) -> None:
        """Write an entry, computing duration from the noisy file on disk."""
        info = sf.info(str(noisy_path))
        duration_s = info.frames / info.samplerate
        self.write_entry(noisy_path, clean_path, snr_db, noise_class, subset, duration_s)


# ── Reader ─────────────────────────────────────────────────────────────────

def read_manifest(path: str | Path) -> list[ManifestEntry]:
    """Read all entries from a JSONL manifest file."""
    entries = []
    with open(path, "r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                entries.append(ManifestEntry.from_dict(d))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Manifest line %d malformed: %s", line_no, exc)
    return entries


def iter_manifest(path: str | Path) -> Iterator[ManifestEntry]:
    """Lazily iterate entries — useful for very large manifests."""
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield ManifestEntry.from_dict(json.loads(line))


def manifest_stats(entries: Sequence[ManifestEntry]) -> dict:
    """Compute summary statistics from a list of manifest entries.

    Returns
    -------
    dict with keys:
        total_pairs    : int
        total_hours    : float
        subset_counts  : dict[str, int]
        noise_class_counts : dict[str, int]
        snr_mean       : float
        snr_std        : float
    """
    if not entries:
        return {"total_pairs": 0}

    snrs = np.array([e.snr_db for e in entries])
    subset_counts: dict[str, int] = {}
    noise_counts: dict[str, int] = {}
    total_dur = 0.0

    for e in entries:
        subset_counts[e.subset] = subset_counts.get(e.subset, 0) + 1
        noise_counts[e.noise_class] = noise_counts.get(e.noise_class, 0) + 1
        total_dur += e.duration_s

    return {
        "total_pairs": len(entries),
        "total_hours": round(total_dur / 3600, 2),
        "subset_counts": subset_counts,
        "noise_class_counts": noise_counts,
        "snr_mean": round(float(snrs.mean()), 2),
        "snr_std": round(float(snrs.std()), 2),
    }
