"""
test_defense_mixer.py — Unit tests for defense battlefield mixing.
"""
import numpy as np
import pytest
from sih26052.data.mixer import (
    synthesize_gunfire_track,
    mix_defense_sample,
    measure_snr,
)


def test_synthesize_gunfire_single():
    rng = np.random.default_rng(42)
    shot = np.ones(1000, dtype=np.float32) * 0.5
    track, meta = synthesize_gunfire_track([shot], target_len=16000, rng=rng, mode="single")
    assert len(track) == 16000
    assert meta["mode"] == "single"
    assert meta["shots"] == 1
    assert len(meta["offsets"]) == 1
    assert np.max(np.abs(track)) > 0.0


def test_synthesize_gunfire_multiple():
    rng = np.random.default_rng(42)
    shot = np.ones(500, dtype=np.float32) * 0.5
    track, meta = synthesize_gunfire_track([shot], target_len=16000, rng=rng, mode="multiple", min_shots=3, max_shots=3)
    assert len(track) == 16000
    assert meta["shots"] == 3
    assert len(meta["offsets"]) == 3


def test_synthesize_gunfire_burst():
    rng = np.random.default_rng(42)
    shot = np.ones(400, dtype=np.float32) * 0.5
    track, meta = synthesize_gunfire_track(
        [shot], target_len=16000, rng=rng, mode="burst", min_shots=3, max_shots=3, burst_interval_ms=(60.0, 60.0)
    )
    assert len(track) == 16000
    assert meta["mode"] == "burst"
    assert meta["shots"] >= 3


@pytest.mark.parametrize("m_type", ["A", "B", "C", "D", "E", "F"])
def test_mix_defense_sample_types(m_type):
    rng = np.random.default_rng(123)
    clean = rng.standard_normal(32000).astype(np.float32) * 0.2
    bg = rng.standard_normal(32000).astype(np.float32) * 0.2
    ev = rng.standard_normal(8000).astype(np.float32) * 0.3
    gf = rng.standard_normal(4000).astype(np.float32) * 0.5

    noisy, clean_out, meta = mix_defense_sample(
        clean, background=bg, event=ev, gunfire=gf, mixture_type=m_type, snr_db=5.0, rng=rng
    )
    assert len(noisy) == len(clean)
    assert len(clean_out) == len(clean)
    assert np.max(np.abs(noisy)) <= 1.0
    assert np.max(np.abs(clean_out)) <= 1.0
    assert meta["mixture_type"] == f"TYPE_{m_type}"
    assert len(meta["active_components"]) > 0


def test_mix_at_ratio_60_40():
    from sih26052.data.mixer import mix_at_ratio
    rng = np.random.default_rng(42)
    clean = rng.standard_normal(32000).astype(np.float32) * 0.2
    noise = rng.standard_normal(32000).astype(np.float32) * 0.2

    # 40% clean, 60% background
    noisy, clean_out = mix_at_ratio(clean, noise, clean_ratio=0.40, noise_ratio=0.60, rng=rng)
    assert len(noisy) == len(clean)
    assert len(clean_out) == len(clean)
    assert np.max(np.abs(noisy)) <= 1.0
    assert np.max(np.abs(clean_out)) <= 1.0

    actual_snr = measure_snr(clean_out, noisy)
    expected_snr = 20.0 * np.log10(0.40 / 0.60)  # -3.5218 dB
    assert abs(actual_snr - expected_snr) < 0.5


def test_mix_defense_sample_ratio_60_40():
    rng = np.random.default_rng(999)
    clean = rng.standard_normal(32000).astype(np.float32) * 0.2
    bg = rng.standard_normal(32000).astype(np.float32) * 0.2

    noisy, clean_out, meta = mix_defense_sample(
        clean, background=bg, clean_ratio=0.40, noise_ratio=0.60, rng=rng
    )
    assert len(noisy) == len(clean)
    assert len(clean_out) == len(clean)
    assert np.max(np.abs(noisy)) <= 1.0
    assert np.max(np.abs(clean_out)) <= 1.0
    assert "clean_ratio" in meta
    assert meta["clean_ratio"] == 0.40
    assert meta["noise_ratio"] == 0.60

    expected_snr = 20.0 * np.log10(0.40 / 0.60)
    assert abs(meta["actual_snr_db"] - expected_snr) < 0.5
