import pytest
import numpy as np
from pathlib import Path
from sih26052.runtime.ola import OverlapAdd
from ._stub_model import create_stub_model

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")

@pytest.fixture
def noisy_wav(tmp_path):
    """Programmatically generate a noisy WAV fixture."""
    sr = 16000
    duration_s = 0.5
    hop = 256
    
    rng = np.random.default_rng(42)
    # Sine wave + white noise
    t = np.arange(int(sr * duration_s)) / sr
    clean = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.standard_normal(len(clean)) * 0.1
    noisy = (clean + noise).astype(np.float32)
    
    path = tmp_path / "fixture.wav"
    sf.write(str(path), noisy, sr)
    return path

def test_integration_streaming_state_propagation(noisy_wav):
    """
    Feed audio through OLA -> StubModel -> OLA and verify that hidden states 
    are properly propagated across frames (output n depends on state n-1).
    """
    sr = 16000
    nfft = 512
    hop = 256
    
    ola_continuous = OverlapAdd(nfft=nfft, hop=hop)
    ola_fresh = OverlapAdd(nfft=nfft, hop=hop)
    
    model = create_stub_model()
    model.eval()
    
    # Export stub to temporary ONNX
    onnx_path = noisy_wav.parent / "stub.onnx"
    spec_frame = torch.randn(1, 257, 1, 2)
    h0 = torch.zeros(1, 1, model.hidden_size)
    torch.onnx.export(
        model,
        (spec_frame, h0),
        str(onnx_path),
        input_names=["spec_frame", "state_h_0"],
        output_names=["enhanced_frame", "state_h_0_out"],
        opset_version=17,
    )
    
    from sih26052.runtime.enhancer import StreamingEnhancer
    enhancer_continuous = StreamingEnhancer(onnx_path)
    enhancer_fresh = StreamingEnhancer(onnx_path)
    
    audio, _ = sf.read(str(noisy_wav), dtype='float32')
    
    # Process continuous
    outputs_continuous = []
    for i in range(0, len(audio), hop):
        block = audio[i:i+hop]
        if len(block) < hop:
            block = np.pad(block, (0, hop - len(block)))
            
        spec = ola_continuous.analyze(block)
        out_spec = enhancer_continuous.process_frame(spec)
        out_block = ola_continuous.synthesize(out_spec)
        outputs_continuous.append(out_block)
            
    # Process fresh state every frame (simulating broken streaming)
    outputs_fresh = []
    for i in range(0, len(audio), hop):
        block = audio[i:i+hop]
        if len(block) < hop:
            block = np.pad(block, (0, hop - len(block)))
            
        spec = ola_fresh.analyze(block)
        out_spec = enhancer_fresh.process_frame(spec)
        out_block = ola_fresh.synthesize(out_spec)
        outputs_fresh.append(out_block)
        
        # ALWAYS reset to force a fresh zero state
        enhancer_fresh.reset()
            
    # Concatenate and verify difference
    audio_continuous = np.concatenate(outputs_continuous)
    audio_fresh = np.concatenate(outputs_fresh)
    
    # First frame should be exactly identical
    np.testing.assert_allclose(audio_continuous[:hop], audio_fresh[:hop], atol=1e-5)
    
    # Later frames should differ significantly because state wasn't propagated in the fresh run
    diff = np.max(np.abs(audio_continuous[hop:] - audio_fresh[hop:]))
    assert diff > 1e-4, f"States did not propagate! Max diff between continuous and fresh runs is only {diff}"
