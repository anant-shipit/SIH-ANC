"""
preprocess.py — Walk dataset directories, resample everything to 16 kHz mono, cache to disk.

Why 16 kHz?
    GTCRN's STFT uses a 512-point FFT at 16 kHz ⇒ Nyquist at 8 kHz, which
    covers the full speech band.  Higher rates waste compute on the Pi 4B.

Why mono?
    GTCRN processes single-channel spectrograms.  Stereo files get averaged
    to mono (not discarded) so we don't lose any data.

Usage:
    python -m sih26052.data.preprocess \\
        --input-dirs ~/Downloads/VoiceBank ~/Downloads/DEMAND ~/Downloads/ESC-50 \\
        --output-dir data/processed/ \\
        --sr 16000
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

# ── Constants ──────────────────────────────────────────────────────────────
TARGET_SR = 16000        # Hz  — GTCRN's native rate
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}

logger = logging.getLogger(__name__)


# ── Core functions ─────────────────────────────────────────────────────────

def discover_audio_files(root: Path) -> list[Path]:
    """Recursively find all audio files under *root*.

    We match by extension (case-insensitive) rather than trying to open
    every file, because failed opens are slow on spinning disks.
    """
    files = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if Path(fn).suffix.lower() in AUDIO_EXTS:
                files.append(Path(dirpath) / fn)
    return files


def load_and_resample(filepath: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load an audio file, convert to mono float32, and resample if needed.

    Returns
    -------
    audio : np.ndarray, shape (samples,), dtype float32
    sr    : int — always *target_sr*

    Notes
    -----
    - We use soundfile for loading because it's fast and dependency-light.
    - Resampling is done with scipy.signal.resample_poly for quality
      (it's a polyphase FIR, not the FFT-based resample that rings on
      non-periodic signals).
    - Stereo → mono: average channels, don't just pick one.
    """
    data, sr = sf.read(filepath, dtype="float32", always_2d=True)
    # data shape: (samples, channels)

    # ── Mono mixdown ──
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    data = data.squeeze()  # (samples,)

    # ── Resample if native rate differs ──
    if sr != target_sr:
        from scipy.signal import resample_poly
        # resample_poly wants integer up/down factors.
        # gcd reduction keeps the filter small.
        from math import gcd
        g = gcd(target_sr, sr)
        up, down = target_sr // g, sr // g
        data = resample_poly(data, up, down).astype(np.float32)

    return data, target_sr


def save_wav(audio: np.ndarray, path: Path, sr: int = TARGET_SR) -> None:
    """Write a mono float32 waveform to disk as 16-bit PCM WAV.

    Why 16-bit PCM instead of float32?
        Smaller files (half the size), and every tool on the Pi reads them.
        We lose ~0.001 dB of dynamic range which is inaudible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def process_directory(
    input_dir: Path,
    output_dir: Path,
    target_sr: int = TARGET_SR,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Resample and cache every audio file under *input_dir* into *output_dir*.

    The directory structure is preserved.  e.g.:
        input_dir/clean/p226_001.wav → output_dir/clean/p226_001.wav

    Returns a dict of stats: {"processed": N, "skipped": M, "errors": E}
    """
    audio_files = discover_audio_files(input_dir)
    stats = {"processed": 0, "skipped": 0, "errors": 0}

    for src in audio_files:
        # Build the mirror path under output_dir
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst = dst.with_suffix(".wav")  # normalise extension

        # Skip if already processed (idempotent re-runs)
        if dst.exists():
            stats["skipped"] += 1
            continue

        if dry_run:
            logger.info("Would process: %s → %s", src, dst)
            stats["processed"] += 1
            continue

        try:
            audio, sr = load_and_resample(src, target_sr)
            save_wav(audio, dst, sr)
            stats["processed"] += 1
            if stats["processed"] % 500 == 0:
                logger.info("Processed %d files...", stats["processed"])
        except Exception as exc:
            logger.warning("Failed to process %s: %s", src, exc)
            stats["errors"] += 1

    return stats


def process_multiple_directories(
    input_dirs: Sequence[Path],
    output_dir: Path,
    target_sr: int = TARGET_SR,
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Process several dataset roots into a single output tree.

    Each input directory gets its own sub-folder under output_dir to
    avoid filename collisions.  e.g.:
        ~/Downloads/VoiceBank/  → data/processed/VoiceBank/
        ~/Downloads/DEMAND/     → data/processed/DEMAND/
    """
    totals = {"processed": 0, "skipped": 0, "errors": 0}

    for inp in input_dirs:
        inp = Path(inp).expanduser().resolve()
        if not inp.is_dir():
            logger.warning("Input directory does not exist: %s — skipping", inp)
            continue

        sub_out = output_dir / inp.name
        logger.info("Processing %s → %s", inp, sub_out)
        stats = process_directory(inp, sub_out, target_sr, dry_run=dry_run)

        for k in totals:
            totals[k] += stats[k]

        logger.info(
            "  %s: processed=%d, skipped=%d, errors=%d",
            inp.name, stats["processed"], stats["skipped"], stats["errors"],
        )

    logger.info(
        "Total: processed=%d, skipped=%d, errors=%d",
        totals["processed"], totals["skipped"], totals["errors"],
    )
    return totals


# ── CLI entry point ────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Resample audio datasets to 16 kHz mono WAV.",
    )
    parser.add_argument(
        "--input-dirs", nargs="+", type=Path, required=True,
        help="One or more dataset root directories to process.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed"),
        help="Where to write the resampled files (default: data/processed/).",
    )
    parser.add_argument(
        "--sr", type=int, default=TARGET_SR,
        help=f"Target sample rate (default: {TARGET_SR}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without writing files.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    process_multiple_directories(
        args.input_dirs,
        args.output_dir.expanduser().resolve(),
        args.sr,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
