import numpy as np
import pytest

@pytest.fixture
def make_speech_like():
    """Returns a function that generates a speech-like signal (modulated noise) for tests."""
    def _make(duration_s: float = 2.0, sr: int = 16000, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = int(sr * duration_s)
        t = np.arange(n, dtype=np.float32) / sr
        # Mix of tones to simulate speech formants
        signal = (
            0.3 * np.sin(2 * np.pi * 200 * t)
            + 0.2 * np.sin(2 * np.pi * 800 * t)
            + 0.1 * np.sin(2 * np.pi * 1500 * t)
            + 0.05 * rng.standard_normal(n).astype(np.float32)
        ).astype(np.float32)
        # Amplitude modulation to simulate speech envelope
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
        return (signal * envelope).astype(np.float32)
    return _make

@pytest.fixture
def make_sine():
    """Returns a function that generates a sine wave."""
    def _make(freq_hz: float = 440.0, duration_s: float = 2.0, sr: int = 16000) -> np.ndarray:
        t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
        return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    return _make

@pytest.fixture
def make_noise():
    """Returns a function that generates white noise."""
    def _make(duration_s: float = 2.0, sr: int = 16000, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n_samples = int(sr * duration_s)
        return rng.standard_normal(n_samples).astype(np.float32) * 0.3
    return _make
