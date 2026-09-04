"""
benchmark.py — Measure Real-Time Factor (RTF) of the ONNX model.

RTF = (time to process N seconds of audio) / N seconds
RTF < 1.0 means real-time capable.  We target RTF < 0.5 for headroom.

The benchmark:
    1. Loads a 60-second audio file (or generates white noise).
    2. Runs one warm-up iteration (fills caches, triggers JIT).
    3. Runs 3 timed iterations.
    4. Reports median RTF.

Why median of 3?
    Mean is sensitive to one-off spikes from OS scheduler contention.
    Median is more representative of steady-state performance.

Usage:
    python -m sih26052.export.benchmark \\
        --onnx models/gtcrn_stream_int8.onnx \\
        --duration 60
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def benchmark_rtf(
    onnx_path: str | Path,
    duration_s: float = 60.0,
    sr: int = 16000,
    nfft: int = 512,
    hop: int = 256,
    n_runs: int = 3,
    warmup: bool = True,
    num_threads: int = 1,
    *,
    audio_path: str | Path | None = None,
) -> dict:
    """Measure the Real-Time Factor of an ONNX model.

    Parameters
    ----------
    onnx_path   : path to the ONNX model
    duration_s  : simulated audio duration in seconds
    sr          : sample rate
    nfft        : FFT size
    hop         : hop size (256 = 16ms at 16kHz)
    n_runs      : number of timed runs (takes median)
    warmup      : if True, do one untimed warm-up run first
    num_threads : intra_op_num_threads for ONNX Runtime (default: 1)

    Returns
    -------
    dict with:
        rtf_median : float — median RTF across runs
        rtf_all    : list[float] — RTF per run
        fps        : float — frames per second
        total_frames : int
        inference_ms_per_frame : float — median ms per frame
    """
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = num_threads
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    n_freq = nfft // 2 + 1
    total_samples = int(duration_s * sr)
    total_frames = total_samples // hop

    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]

    def init_states():
        states = {}
        for inp in sess.get_inputs():
            if inp.name != "spec_frame":
                shape = [d if isinstance(d, int) else 1 for d in inp.shape]
                states[inp.name] = np.zeros(shape, dtype=np.float32)
        return states

    rng = np.random.default_rng(42)
    # Pre-generate all frames to exclude generation time from benchmark
    frames = []
    
    if audio_path:
        import soundfile as sf
        from sih26052.runtime.ola import OverlapAdd
        
        ola = OverlapAdd(nfft=nfft, hop=hop)
        logger.info("Loading %s for benchmark...", audio_path)
        with sf.SoundFile(str(audio_path)) as sf_in:
            for block in sf_in.blocks(blocksize=hop, dtype='float32', fill_value=0.0):
                if block.ndim > 1:
                    block = block[:, 0]  # mono
                spec = ola.analyze(block)
                # spec is (n_freq, 2). ONNX model expects (1, n_freq, 1, 2)
                spec_onnx = spec[np.newaxis, :, np.newaxis, :]
                frames.append(spec_onnx)
                if len(frames) >= total_frames:
                    break
        # Pad with silence if audio was shorter than requested duration
        while len(frames) < total_frames:
            spec = ola.analyze(np.zeros(hop, dtype=np.float32))
            frames.append(spec[np.newaxis, :, np.newaxis, :])
    else:
        frames = [
            rng.standard_normal((1, n_freq, 1, 2)).astype(np.float32)
            for _ in range(total_frames)
        ]

    def run_inference():
        """Run all frames through the model, return elapsed time."""
        states = init_states()
        t0 = time.perf_counter()

        for spec in frames:
            feed = {"spec_frame": spec, **states}
            outputs = sess.run(output_names, feed)
            # Update states
            for i, name in enumerate(output_names):
                in_name = name.replace("_out", "")
                if in_name in states:
                    states[in_name] = outputs[i]

        t1 = time.perf_counter()
        return t1 - t0

    # ── Warm-up ──
    if warmup:
        logger.info("Warm-up run...")
        run_inference()

    # ── Timed runs ──
    rtf_all = []
    for run_idx in range(n_runs):
        elapsed = run_inference()
        rtf = elapsed / duration_s
        rtf_all.append(rtf)
        logger.info(
            "Run %d/%d: %.3f s for %.1f s audio → RTF=%.4f",
            run_idx + 1, n_runs, elapsed, duration_s, rtf,
        )

    rtf_median = float(np.median(rtf_all))
    ms_per_frame = float(np.median(rtf_all)) * duration_s * 1000 / total_frames

    logger.info("=" * 50)
    logger.info("RTF median: %.4f  (target < 0.5)", rtf_median)
    logger.info("Inference: %.3f ms/frame", ms_per_frame)
    logger.info("Throughput: %.0f frames/sec", total_frames / (rtf_median * duration_s))

    return {
        "rtf_median": rtf_median,
        "rtf_all": rtf_all,
        "fps": total_frames / (rtf_median * duration_s),
        "total_frames": total_frames,
        "inference_ms_per_frame": ms_per_frame,
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX model RTF.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--audio", type=Path, default=None, help="Optional real WAV file to use as input")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    benchmark_rtf(args.onnx, audio_path=args.audio, duration_s=args.duration, n_runs=args.runs)


if __name__ == "__main__":
    main()
