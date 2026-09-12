"""
processor.py — Audio enhancement processor for the GTCRN model dashboard.

Handles model loading, inference (PyTorch and ONNX streaming),
spectrogram generation, and audio quality/performance metrics.
"""
from __future__ import annotations

import base64
import io
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.signal
import soundfile as sf

try:
    import torch
    from gtcrn import GTCRN
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    torch = None
    HAS_TORCH = False

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "models" / "gtcrn") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "models" / "gtcrn"))

logger = logging.getLogger(__name__)

# Optional quality metric libraries
try:
    from pesq import pesq as calc_pesq
except ImportError:
    calc_pesq = None

try:
    from pystoi import stoi as calc_stoi
except ImportError:
    calc_stoi = None


class ModelEngineManager:
    """Manages and caches loaded GTCRN model engines."""

    def __init__(self, repo_root: Path = REPO_ROOT):
        self.repo_root = repo_root
        self._pytorch_model = None
        self._onnx_fp32_enhancer = None
        self._onnx_int8_enhancer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else None

    def get_pytorch_model(self):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed in this environment. Please select an ONNX engine.")
        if self._pytorch_model is not None:
            return self._pytorch_model

        from gtcrn import GTCRN

        model = GTCRN().to(self.device)
        ckpt_path = self._find_best_checkpoint()
        if ckpt_path and ckpt_path.exists():
            logger.info("Loading PyTorch GTCRN checkpoint: %s", ckpt_path)
            ckpt = torch.load(str(ckpt_path), map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning("No checkpoint found, using random initialized model")
        model.eval()
        self._pytorch_model = model
        return self._pytorch_model

    def get_onnx_enhancer(self, int8: bool = True):
        if int8 and self._onnx_int8_enhancer is not None:
            return self._onnx_int8_enhancer
        if not int8 and self._onnx_fp32_enhancer is not None:
            return self._onnx_fp32_enhancer

        from sih26052.runtime.enhancer import StreamingEnhancer

        model_name = "gtcrn_finetuned_stream_int8.onnx" if int8 else "gtcrn_finetuned_stream.onnx"
        model_path = self.repo_root / "models" / model_name
        if not model_path.exists():
            # Fallback to standard ONNX models if fine-tuned doesn't exist
            fallback_name = "gtcrn_stream_int8.onnx" if int8 else "gtcrn_stream.onnx"
            model_path = self.repo_root / "models" / fallback_name

        try:
            logger.info("Loading ONNX Streaming Enhancer: %s", model_path)
            enhancer = StreamingEnhancer(model_path, n_freq=257)
        except Exception as err:
            logger.warning("Could not initialize %s (%s). Falling back to FP32 model...", model_path.name, err)
            fp32_path = self.repo_root / "models" / "gtcrn_finetuned_stream.onnx"
            if not fp32_path.exists():
                fp32_path = self.repo_root / "models" / "gtcrn_stream.onnx"
            logger.info("Loading fallback ONNX Enhancer: %s", fp32_path)
            enhancer = StreamingEnhancer(fp32_path, n_freq=257)

        if int8:
            self._onnx_int8_enhancer = enhancer
        else:
            self._onnx_fp32_enhancer = enhancer
        return enhancer

    def _find_best_checkpoint(self) -> Optional[Path]:
        ckpt_dir = self.repo_root / "models" / "checkpoints"
        if (ckpt_dir / "checkpoint_best.pth").exists():
            return ckpt_dir / "checkpoint_best.pth"
        if (ckpt_dir / "best_model.pth").exists():
            return ckpt_dir / "best_model.pth"
        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
        if ckpts:
            return ckpts[-1]
        default_tar = self.repo_root / "models" / "gtcrn" / "checkpoints" / "model_trained_on_dns3.tar"
        if default_tar.exists():
            return default_tar
        return None

    def list_available_engines(self) -> list[dict]:
        has_int8 = (self.repo_root / "models" / "gtcrn_finetuned_stream_int8.onnx").exists() or (
            self.repo_root / "models" / "gtcrn_stream_int8.onnx"
        ).exists()
        has_fp32 = (self.repo_root / "models" / "gtcrn_finetuned_stream.onnx").exists() or (
            self.repo_root / "models" / "gtcrn_stream.onnx"
        ).exists()

        engines = [
            {
                "id": "onnx_stream_int8",
                "name": "ONNX Streaming INT8 (Edge Optimized)",
                "description": "Quantized streaming ONNX runtime model for ultra-low latency (<0.10 RTF).",
                "available": has_int8,
                "is_default": True if (not HAS_TORCH or has_int8) else False,
            },
            {
                "id": "onnx_stream_fp32",
                "name": "ONNX Streaming FP32",
                "description": "Full-precision streaming ONNX runtime model with step-by-step state caching.",
                "available": has_fp32,
                "is_default": False,
            },
            {
                "id": "pytorch",
                "name": "PyTorch GTCRN (Highest Fidelity)",
                "description": "Full complex STFT-domain neural network with 2x DPGRNN and TRA modules.",
                "available": HAS_TORCH,
                "is_default": False if (not HAS_TORCH or has_int8) else True,
            },
        ]
        return engines


# Global model manager
model_manager = ModelEngineManager()


def read_audio_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode audio bytes (WAV, MP3, FLAC, OGG) to float32 numpy array and sample rate."""
    buf = io.BytesIO(audio_bytes)
    audio, sr = sf.read(buf, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16000 Hz if needed
    if sr != 16000:
        n_resampled = int(len(audio) * 16000 / sr)
        audio = scipy.signal.resample(audio, n_resampled)
        if isinstance(audio, tuple):
            audio = audio[0]
        audio = np.asarray(audio, dtype=np.float32)
        sr = 16000

    # Ensure finite values
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    return audio, sr


def write_audio_to_base64_wav(audio: np.ndarray, sr: int = 16000) -> str:
    """Encode float32 audio array to base64 data URI (data:audio/wav;base64,...)."""
    # Clip to [-1.0, 1.0] to prevent clipping distortion
    audio_clipped = np.clip(audio, -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, audio_clipped, sr, format="WAV", subtype="PCM_16")
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64_str}"


def compute_spectrogram_matrix(audio: np.ndarray, nfft: int = 512, hop: int = 256, target_width: int = 120, target_height: int = 64) -> list[list[float]]:
    """Compute downsampled 2D spectrogram (dB scale normalized to 0.0-1.0) for visual display."""
    if len(audio) < nfft:
        audio = np.pad(audio, (0, nfft - len(audio)))

    window = np.hanning(nfft)
    f, t, zxx = scipy.signal.stft(audio, fs=16000, window=window, nperseg=nfft, noverlap=nfft - hop)
    mag = np.abs(zxx)  # shape: (n_freq, n_frames)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-6))

    # Normalize roughly from -60 dB to 0 dB
    norm_db = np.clip((mag_db + 60.0) / 60.0, 0.0, 1.0)

    # Downsample to target_height (freq bins) x target_width (time frames)
    h, w = norm_db.shape
    if w == 0 or h == 0:
        return [[0.0] * target_width for _ in range(target_height)]

    # Interpolate using simple bin averaging
    res_freq = np.zeros((target_height, w), dtype=np.float32)
    step_y = h / target_height
    for i in range(target_height):
        y0 = int(i * step_y)
        y1 = max(y0 + 1, int((i + 1) * step_y))
        res_freq[i, :] = np.mean(norm_db[y0:y1, :], axis=0)

    res_final = np.zeros((target_height, target_width), dtype=np.float32)
    step_x = w / target_width
    for j in range(target_width):
        x0 = int(j * step_x)
        x1 = max(x0 + 1, int((j + 1) * step_x))
        res_final[:, j] = np.mean(res_freq[:, x0:x1], axis=1)

    # Round to 3 decimals to reduce JSON payload size
    return [[round(float(v), 3) for v in row] for row in res_final]


def estimate_snr_db(signal: np.ndarray, sr: int = 16000) -> float:
    """Estimate SNR using energy percentile ratio between speech frames and noise floor."""
    if len(signal) < 256:
        return 0.0
    # Frame energy in 20ms frames
    frame_len = int(0.02 * sr)
    hop = frame_len // 2
    n_frames = (len(signal) - frame_len) // hop + 1
    if n_frames < 2:
        return 10.0

    energies = []
    for i in range(n_frames):
        chunk = signal[i * hop : i * hop + frame_len]
        energies.append(float(np.mean(chunk**2)))

    energies = np.array(energies)
    # 90th percentile as speech energy, 10th percentile as noise floor
    p_speech = np.percentile(energies, 90)
    p_noise = max(np.percentile(energies, 10), 1e-9)

    snr = 10 * np.log10(max(p_speech / p_noise, 1e-4))
    return float(np.clip(snr, -10.0, 50.0))


def compute_metrics(raw: np.ndarray, enh: np.ndarray, clean: Optional[np.ndarray] = None, sr: int = 16000) -> Dict[str, Any]:
    """Compute comprehensive audio quality and noise suppression metrics."""
    # RMS and Peak
    raw_rms = float(np.sqrt(np.mean(raw**2)))
    enh_rms = float(np.sqrt(np.mean(enh**2)))
    raw_peak = float(np.max(np.abs(raw)))
    enh_peak = float(np.max(np.abs(enh)))

    raw_rms_db = round(20 * np.log10(max(raw_rms, 1e-6)), 1)
    enh_rms_db = round(20 * np.log10(max(enh_rms, 1e-6)), 1)

    # Estimated SNR
    snr_raw = round(estimate_snr_db(raw, sr), 1)
    snr_enh = round(estimate_snr_db(enh, sr), 1)
    snr_gain = round(snr_enh - snr_raw, 1)

    # Spectral noise reduction: average attenuation in quieter frames
    noise_reduction_db = round(max(0.0, raw_rms_db - enh_rms_db), 1)

    metrics: Dict[str, Any] = {
        "snr_raw_db": snr_raw,
        "snr_enhanced_db": snr_enh,
        "snr_improvement_db": max(0.0, snr_gain),
        "raw_rms_db": raw_rms_db,
        "enhanced_rms_db": enh_rms_db,
        "raw_peak": round(raw_peak, 3),
        "enhanced_peak": round(enh_peak, 3),
        "noise_reduction_db": noise_reduction_db,
        "pesq": None,
        "stoi": None,
    }

    # If clean ground-truth reference is available, compute PESQ and STOI
    if clean is not None and len(clean) > 0:
        min_len = min(len(clean), len(enh), len(raw))
        clean_c = clean[:min_len]
        enh_c = enh[:min_len]

        if calc_pesq is not None and min_len >= sr:
            try:
                score = calc_pesq(sr, clean_c, enh_c, "wb")
                metrics["pesq"] = round(float(score), 2)
            except Exception as e:
                logger.debug("PESQ calc error: %s", e)

        if calc_stoi is not None and min_len >= sr:
            try:
                score = calc_stoi(clean_c, enh_c, sr, extended=False)
                metrics["stoi"] = round(float(score), 3)
            except Exception as e:
                logger.debug("STOI calc error: %s", e)

    return metrics


def enhance_audio_pytorch(audio: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, float]:
    """Enhance audio using full PyTorch GTCRN model."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not available. Use ONNX streaming engine instead.")
    model = model_manager.get_pytorch_model()
    device = model_manager.device

    nfft = 512
    hop = 256
    window = torch.hann_window(nfft).to(device)

    noisy_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    stft = torch.view_as_real(torch.stft(noisy_t, nfft, hop, window=window, return_complex=True))

    with torch.no_grad():
        pred_stft = model(stft)

    pred_complex = torch.complex(pred_stft[..., 0], pred_stft[..., 1])
    enhanced = torch.istft(pred_complex, nfft, hop, window=window, length=len(audio))
    proc_time = time.perf_counter() - t0

    enh_np = enhanced.squeeze(0).cpu().numpy().astype(np.float32)
    return enh_np, proc_time


def enhance_audio_onnx_stream(audio: np.ndarray, int8: bool = True, sr: int = 16000) -> Tuple[np.ndarray, float]:
    """Enhance audio using streaming ONNX runtime model with OverlapAdd."""
    from sih26052.runtime.ola import OverlapAdd

    enhancer = model_manager.get_onnx_enhancer(int8=int8)
    enhancer.reset()

    nfft = 512
    hop = 256
    ola = OverlapAdd(nfft=nfft, hop=hop)

    n_samples = len(audio)
    pad_len = (hop - (n_samples % hop)) % hop
    if pad_len > 0:
        audio_padded = np.pad(audio, (0, pad_len))
    else:
        audio_padded = audio

    output = np.zeros_like(audio_padded)

    t0 = time.perf_counter()
    for i in range(0, len(audio_padded), hop):
        chunk = audio_padded[i : i + hop]
        spec_frame = ola.analyze(chunk)
        enh_spec = enhancer.process_frame(spec_frame)
        output[i : i + hop] = ola.synthesize(enh_spec)
    proc_time = time.perf_counter() - t0

    return output[:n_samples].astype(np.float32), proc_time


def process_audio_file(
    raw_audio_bytes: bytes,
    engine: str = "onnx_stream_int8" if not HAS_TORCH else "pytorch",
    clean_audio_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Full processing pipeline: load, enhance, compute metrics & spectrograms, return response dict."""
    raw_audio, sr = read_audio_from_bytes(raw_audio_bytes)
    audio_dur = len(raw_audio) / sr

    clean_audio = None
    if clean_audio_bytes:
        try:
            clean_audio, _ = read_audio_from_bytes(clean_audio_bytes)
        except Exception as e:
            logger.warning("Could not decode clean audio reference: %s", e)

    # Automatic fallback if PyTorch is requested but not installed
    if (engine == "pytorch" or not engine) and not HAS_TORCH:
        engine = "onnx_stream_int8"

    # Perform enhancement with selected engine
    if engine == "onnx_stream_int8":
        enhanced_audio, proc_time = enhance_audio_onnx_stream(raw_audio, int8=True, sr=sr)
        engine_label = "ONNX Streaming INT8"
    elif engine == "onnx_stream_fp32":
        enhanced_audio, proc_time = enhance_audio_onnx_stream(raw_audio, int8=False, sr=sr)
        engine_label = "ONNX Streaming FP32"
    else:
        enhanced_audio, proc_time = enhance_audio_pytorch(raw_audio, sr=sr)
        engine_label = "PyTorch GTCRN (Best Quality)"

    rtf = proc_time / max(audio_dur, 1e-4)

    # Compute metrics
    metrics = compute_metrics(raw_audio, enhanced_audio, clean=clean_audio, sr=sr)

    # Compute downsampled spectrograms for visual canvas
    spec_raw = compute_spectrogram_matrix(raw_audio)
    spec_enh = compute_spectrogram_matrix(enhanced_audio)

    # Calculate difference spectrogram (attenuation map)
    spec_diff = [
        [round(max(0.0, raw_val - enh_val), 3) for raw_val, enh_val in zip(r_row, e_row)]
        for r_row, e_row in zip(spec_raw, spec_enh)
    ]

    return {
        "status": "success",
        "engine": engine,
        "engine_label": engine_label,
        "duration_sec": round(audio_dur, 2),
        "sample_rate": sr,
        "processing_time_ms": round(proc_time * 1000, 1),
        "rtf": round(rtf, 4),
        "enhanced_audio_url": write_audio_to_base64_wav(enhanced_audio, sr),
        "raw_audio_url": write_audio_to_base64_wav(raw_audio, sr),
        "metrics": metrics,
        "spectrograms": {
            "raw": spec_raw,
            "enhanced": spec_enh,
            "difference": spec_diff,
        },
        "_raw_array": raw_audio,
        "_enh_array": enhanced_audio,
    }
