"""
test_preprocess.py — Verify that preprocess.py produces 16 kHz mono WAVs.

Tests:
    1. A multi-channel file gets mixed down to mono.
    2. A non-16 kHz file gets resampled correctly.
    3. Output is float32 and finite.
    4. discover_audio_files finds .wav, .flac, ignores .txt.
    5. process_directory is idempotent (re-run skips existing).
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from sih26052.data.preprocess import (
    TARGET_SR,
    discover_audio_files,
    load_and_resample,
    process_directory,
    save_wav,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_audio_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with various audio files for testing."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    rng = np.random.default_rng(42)

    # 1. Stereo WAV at 44100 Hz
    stereo = rng.standard_normal((44100, 2)).astype(np.float32) * 0.5
    sf.write(str(audio_dir / "stereo_44k.wav"), stereo, 44100)

    # 2. Mono WAV at 16000 Hz (should be kept as-is)
    mono_16k = rng.standard_normal(16000).astype(np.float32) * 0.5
    sf.write(str(audio_dir / "mono_16k.wav"), mono_16k, 16000)

    # 3. Mono FLAC at 48000 Hz
    mono_48k = rng.standard_normal(48000).astype(np.float32) * 0.5
    sf.write(str(audio_dir / "mono_48k.flac"), mono_48k, 48000)

    # 4. A non-audio file (should be ignored)
    (audio_dir / "readme.txt").write_text("This is not audio.")

    # 5. Sub-directory with another WAV
    sub = audio_dir / "subdir"
    sub.mkdir()
    mono_22k = rng.standard_normal(22050).astype(np.float32) * 0.5
    sf.write(str(sub / "nested_22k.wav"), mono_22k, 22050)

    return audio_dir


# ── Tests ──────────────────────────────────────────────────────────────────

class TestDiscoverAudioFiles:
    def test_finds_wav_and_flac(self, tmp_audio_dir: Path):
        files = discover_audio_files(tmp_audio_dir)
        extensions = {f.suffix.lower() for f in files}
        assert ".wav" in extensions
        assert ".flac" in extensions

    def test_ignores_non_audio(self, tmp_audio_dir: Path):
        files = discover_audio_files(tmp_audio_dir)
        names = {f.name for f in files}
        assert "readme.txt" not in names

    def test_finds_nested_files(self, tmp_audio_dir: Path):
        files = discover_audio_files(tmp_audio_dir)
        names = {f.name for f in files}
        assert "nested_22k.wav" in names


class TestLoadAndResample:
    def test_stereo_to_mono(self, tmp_audio_dir: Path):
        """Stereo input should become 1-D mono output."""
        audio, sr = load_and_resample(tmp_audio_dir / "stereo_44k.wav")
        assert audio.ndim == 1
        assert sr == TARGET_SR

    def test_resampled_to_target_sr(self, tmp_audio_dir: Path):
        """Non-16kHz input should be resampled."""
        audio, sr = load_and_resample(tmp_audio_dir / "mono_48k.flac")
        assert sr == TARGET_SR
        # 48000 samples at 48kHz = 1 second → should be ~16000 samples at 16kHz
        # Allow ±100 samples for resampler filter delay
        assert abs(len(audio) - TARGET_SR) < 100

    def test_already_16k_unchanged_length(self, tmp_audio_dir: Path):
        """16kHz mono input should pass through with same length."""
        audio, sr = load_and_resample(tmp_audio_dir / "mono_16k.wav")
        assert sr == TARGET_SR
        assert len(audio) == 16000

    def test_output_is_float32(self, tmp_audio_dir: Path):
        audio, _ = load_and_resample(tmp_audio_dir / "mono_16k.wav")
        assert audio.dtype == np.float32

    def test_output_is_finite(self, tmp_audio_dir: Path):
        audio, _ = load_and_resample(tmp_audio_dir / "stereo_44k.wav")
        assert np.all(np.isfinite(audio))


class TestProcessDirectory:
    def test_creates_output_files(self, tmp_audio_dir: Path, tmp_path: Path):
        out_dir = tmp_path / "output"
        stats = process_directory(tmp_audio_dir, out_dir)
        assert stats["processed"] > 0
        assert stats["errors"] == 0
        # Check an output file exists
        assert (out_dir / "mono_16k.wav").exists()

    def test_idempotent_rerun(self, tmp_audio_dir: Path, tmp_path: Path):
        """Running twice should skip everything on the second run."""
        out_dir = tmp_path / "output"
        stats1 = process_directory(tmp_audio_dir, out_dir)
        stats2 = process_directory(tmp_audio_dir, out_dir)
        assert stats2["processed"] == 0
        assert stats2["skipped"] == stats1["processed"]

    def test_output_files_are_16k_mono(self, tmp_audio_dir: Path, tmp_path: Path):
        """Every output file should be 16kHz mono."""
        out_dir = tmp_path / "output"
        process_directory(tmp_audio_dir, out_dir)

        for wav in out_dir.rglob("*.wav"):
            info = sf.info(str(wav))
            assert info.samplerate == TARGET_SR, f"{wav} has sr={info.samplerate}"
            assert info.channels == 1, f"{wav} has {info.channels} channels"
