"""
test_ab_switch.py — Verify A/B crossfade produces no discontinuity.

Tests:
    1. No click at toggle point (max sample-to-sample delta is bounded).
    2. Steady state after crossfade completes.
    3. Multiple toggles work correctly.
    4. Vectorized and scalar versions produce same result.
"""
from __future__ import annotations

import numpy as np
import pytest

from sih26052.runtime.ab_switch import ABSwitch


def make_constant(value: float, n: int = 256) -> np.ndarray:
    return np.full(n, value, dtype=np.float32)


class TestCrossfade:
    def test_no_click_on_toggle(self):
        """The max sample-to-sample jump during crossfade should be small."""
        switch = ABSwitch(sr=16000, crossfade_ms=20.0)
        hop = 256

        raw = make_constant(0.0, hop)
        enhanced = make_constant(1.0, hop)

        # Start enhanced (fade=1.0), toggle to raw
        out_before = switch.apply(raw, enhanced)
        assert np.allclose(out_before, 1.0), "Should start fully enhanced"

        switch.toggle()  # → transitioning to raw

        # Process several frames during transition
        all_output = []
        for _ in range(5):
            out = switch.apply(raw, enhanced)
            all_output.append(out)

        full = np.concatenate(all_output)

        # Check: no sample-to-sample jump > 0.1
        # (320 samples for crossfade at 16kHz, value goes from 1.0 to 0.0
        #  so per-sample step ≈ 1/320 ≈ 0.003)
        diffs = np.abs(np.diff(full))
        max_jump = np.max(diffs)
        assert max_jump < 0.05, f"Max jump = {max_jump:.4f} — audible click!"

    def test_settles_after_crossfade(self):
        """After enough frames, output should be fully raw or fully enhanced."""
        switch = ABSwitch(sr=16000, crossfade_ms=20.0)
        hop = 256

        raw = make_constant(-0.5, hop)
        enhanced = make_constant(0.5, hop)

        switch.toggle()  # enhanced → raw

        # Process enough frames to complete the crossfade
        # 20ms at 16kHz = 320 samples.  At hop=256, that's ~2 frames.
        for _ in range(5):
            out = switch.apply(raw, enhanced)

        # Should now be fully raw
        np.testing.assert_allclose(out, -0.5, atol=0.01)

    def test_double_toggle(self):
        """Toggling twice should return to the original state."""
        switch = ABSwitch(sr=16000, crossfade_ms=20.0)
        hop = 256

        raw = make_constant(0.0, hop)
        enhanced = make_constant(1.0, hop)

        switch.toggle()  # → raw
        switch.toggle()  # → enhanced again

        # Process enough frames
        for _ in range(5):
            out = switch.apply(raw, enhanced)

        np.testing.assert_allclose(out, 1.0, atol=0.01)


class TestVectorized:
    def test_matches_scalar(self):
        """Vectorized and scalar versions should produce similar results."""
        hop = 256
        raw = np.random.default_rng(0).standard_normal(hop).astype(np.float32) * 0.5
        enhanced = np.random.default_rng(1).standard_normal(hop).astype(np.float32) * 0.5

        switch1 = ABSwitch(sr=16000, crossfade_ms=20.0)
        switch2 = ABSwitch(sr=16000, crossfade_ms=20.0)

        # Both start enhanced, toggle to raw
        switch1.toggle()
        switch2.toggle()

        out_scalar = switch1.apply(raw, enhanced)
        out_vector = switch2.apply_vectorized(raw, enhanced)

        np.testing.assert_allclose(out_scalar, out_vector, atol=0.01)


class TestState:
    def test_initial_state_is_enhanced(self):
        switch = ABSwitch()
        assert switch.is_enhanced is True

    def test_toggle_changes_state(self):
        switch = ABSwitch()
        assert switch.is_enhanced is True
        switch.toggle()
        assert switch.is_enhanced is False
        switch.toggle()
        assert switch.is_enhanced is True
