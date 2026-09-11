#!/usr/bin/env python3
"""
compare_fp32_vs_int8.py — Compare FP32 vs INT8 streaming ONNX models on real defense test audio.

Measures:
    - Model file size (KB)
    - Inference latency (ms/frame) & Real-Time Factor (RTF)
    - SI-SNR (dB), STOI, and PESQ
    - Accuracy retention after quantization
"""
import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import soundfile as sf

from sih26052.eval.metrics import compute_all_metrics
from sih26052.export.benchmark import benchmark_rtf
from sih26052.runtime.enhancer import StreamingEnhancer
from sih26052.runtime.ola import OverlapAdd


def run_streaming_audio(onnx_path: Path, noisy_wav: np.ndarray, nfft: int = 512, hop: int = 256) -> np.ndarray:
    ola = OverlapAdd(nfft=nfft, hop=hop)
    enhancer = StreamingEnhancer(onnx_path, n_freq=nfft // 2 + 1)

    out_chunks = []
    # Pad input so it's a multiple of hop
    rem = len(noisy_wav) % hop
    if rem > 0:
        noisy_padded = np.pad(noisy_wav, (0, hop - rem))
    else:
        noisy_padded = noisy_wav

    for i in range(0, len(noisy_padded), hop):
        block = noisy_padded[i : i + hop]
        spec = ola.analyze(block)  # (n_freq, 2)
        enh_spec = enhancer.process_frame(spec)
        out_block = ola.synthesize(enh_spec)
        out_chunks.append(out_block)

    out_full = np.concatenate(out_chunks)[: len(noisy_wav)]
    return out_full


def main():
    parser = argparse.ArgumentParser(description="Compare FP32 and INT8 ONNX models")
    parser.add_argument("--fp32", type=Path, default=repo_root / "models" / "gtcrn_finetuned_stream.onnx")
    parser.add_argument("--int8", type=Path, default=repo_root / "models" / "gtcrn_finetuned_stream_int8.onnx")
    parser.add_argument("--test-audio", type=Path, default=repo_root / "data" / "defense_test_audio" / "background_gunshot_000_noisy.wav")
    parser.add_argument("--clean-audio", type=Path, default=repo_root / "data" / "defense_test_audio" / "background_gunshot_000_clean.wav")
    args = parser.parse_args()

    print("=" * 80)
    print("FP32 vs INT8 QUANTIZATION BENCHMARK & COMPARISON")
    print("=" * 80)

    # 1. File sizes
    size_fp32_kb = args.fp32.stat().st_size / 1024.0
    size_int8_kb = args.int8.stat().st_size / 1024.0
    print("Model Sizes:")
    print(f"  FP32 Model: {size_fp32_kb:.1f} KB")
    print(f"  INT8 Model: {size_int8_kb:.1f} KB")
    print(f"  Size Reduction: {((size_fp32_kb - size_int8_kb) / size_fp32_kb) * 100:.1f}%")

    # 2. Benchmarks
    print("\nBenchmarking Latency & RTF...")
    bm_fp32 = benchmark_rtf(args.fp32, duration_s=10.0, n_runs=3, warmup=True)
    bm_int8 = benchmark_rtf(args.int8, duration_s=10.0, n_runs=3, warmup=True)

    print(f"  FP32: RTF = {bm_fp32['rtf_median']:.4f}, Latency = {bm_fp32['inference_ms_per_frame']:.3f} ms/frame")
    print(f"  INT8: RTF = {bm_int8['rtf_median']:.4f}, Latency = {bm_int8['inference_ms_per_frame']:.3f} ms/frame")

    # 3. Audio Quality
    if args.test_audio.exists() and args.clean_audio.exists():
        print("\nProcessing Audio Sample through Streaming Runtime...")
        noisy, _ = sf.read(str(args.test_audio), dtype="float32")
        clean, _ = sf.read(str(args.clean_audio), dtype="float32")

        out_fp32 = run_streaming_audio(args.fp32, noisy)
        out_int8 = run_streaming_audio(args.int8, noisy)

        m_noisy = compute_all_metrics(clean, noisy)
        m_fp32 = compute_all_metrics(clean, out_fp32)
        m_int8 = compute_all_metrics(clean, out_int8)

        print("\nQuality Evaluation:")
        print(f"  {'Metric':<12} | {'Noisy Input':>12} | {'FP32 Output':>12} | {'INT8 Output':>12} | {'Difference':>12}")
        print("-" * 72)
        print(f"  {'SI-SNR (dB)':<12} | {m_noisy.si_snr:>12.2f} | {m_fp32.si_snr:>12.2f} | {m_int8.si_snr:>12.2f} | {m_int8.si_snr - m_fp32.si_snr:>+12.2f}")
        print(f"  {'STOI':<12} | {m_noisy.stoi or 0.0:>12.3f} | {m_fp32.stoi or 0.0:>12.3f} | {m_int8.stoi or 0.0:>12.3f} | {(m_int8.stoi or 0.0) - (m_fp32.stoi or 0.0):>+12.3f}")
        p_noisy = f"{m_noisy.pesq:.3f}" if m_noisy.pesq else "N/A"
        p_fp32 = f"{m_fp32.pesq:.3f}" if m_fp32.pesq else "N/A"
        p_int8 = f"{m_int8.pesq:.3f}" if m_int8.pesq else "N/A"
        print(f"  {'PESQ':<12} | {p_noisy:>12} | {p_fp32:>12} | {p_int8:>12} | {'N/A':>12}")

    print("=" * 80)


if __name__ == "__main__":
    main()
