import numpy as np
import pytest

from sih26052.runtime.impulse_gate import ImpulseGate

class TestImpulseGate:
    def test_passthrough_on_silence(self):
        gate = ImpulseGate(sr=16000, hop=256, threshold_ratio=10.0, history_frames=10)
        silence = np.zeros(256, dtype=np.float32)
        
        for _ in range(20):
            out = gate.process(silence)
            assert np.array_equal(out, silence)
            assert gate.state == "idle"

    def test_fire_on_impulse(self):
        gate = ImpulseGate(sr=16000, hop=256, threshold_ratio=10.0, hold_ms=16.0, release_ms=16.0, history_frames=10)
        
        # Build history with low energy
        low_noise = np.random.randn(256).astype(np.float32) * 0.01
        for _ in range(15):
            gate.process(low_noise)
        assert gate.state == "idle"
        
        # Fire transient
        transient = np.random.randn(256).astype(np.float32) * 1.0
        out = gate.process(transient)
        
        assert gate.state == "fired"
        # Since hold_ms=16.0 and hop=256@16kHz is 16ms, it should hold for 1 frame
        assert gate.current_gain < 1.0
        
        # Next frame: hold counter expires, enters releasing
        gate.process(low_noise)
        assert gate.state == "releasing"
        
        # Next frame: release finishes, back to idle
        gate.process(low_noise)
        assert gate.state == "idle"
