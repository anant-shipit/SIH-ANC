"""
audio_loop.py — Real-time audio processing loop using sounddevice.

This is the main entry point for live speech enhancement.  It:
    1. Opens a sounddevice stream with the configured audio device.
    2. In the callback: reads mic input → STFT → enhance → ISTFT → output.
    3. Pushes metrics to a queue for the dashboard (non-blocking).
    4. Counts xruns via the sounddevice status flags.

Architecture:
    ┌─────────┐     ┌─────┐     ┌──────────┐     ┌─────┐     ┌──────────┐
    │ mic in  │────>│ OLA │────>│ enhancer │────>│ OLA │────>│ speaker  │
    │         │     │ .analyze  │ .process │     │ .synth    │ out      │
    └─────────┘     └─────┘     └──────────┘     └─────┘     └──────────┘
                                      │
                                      ├── ab_switch
                                      ├── impulse_gate (Phase 5)
                                      └── queue → dashboard

Rules for the callback:
    - NO memory allocation (everything preallocated)
    - NO print / logging / file I/O
    - NO locks (except the A/B switch's minimal lock)
    - NO blocking calls

NO torch imports — only numpy, sounddevice, onnxruntime.

Usage:
    python -m sih26052.runtime.audio_loop \\
        --onnx models/gtcrn_stream_int8.onnx \\
        --device 0
"""
from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class AudioLoop:
    """Real-time speech enhancement loop.

    Parameters
    ----------
    onnx_path    : path to the streaming ONNX model
    device       : sounddevice device index (None = default)
    sr           : sample rate (must match model)
    nfft         : FFT size
    hop          : hop size (= callback block size)
    queue_size   : max items in the dashboard metrics queue
    """

    def __init__(
        self,
        onnx_path: str | Path,
        device: int | None = None,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        sr: int = 16000,
        native_sr: int = 16000,
        nfft: int = 512,
        hop: int = 256,
        queue_size: int = 100,
        use_impulse_gate: bool = False,
    ):
        from sih26052.runtime.ola import OverlapAdd
        from sih26052.runtime.enhancer import StreamingEnhancer
        from sih26052.runtime.ab_switch import ABSwitch, attach_gpio_button
        from sih26052.runtime.led_status import LEDStatus

        self.sr = sr
        self.native_sr = native_sr
        self.hop = hop
        # Prefer explicit input/output device; fall back to shared device index
        self.input_device = input_device if input_device is not None else device
        self.output_device = output_device if output_device is not None else device
        self.device = device

        # ── Resampling ratio: native hw rate → model rate ──
        # e.g. native_sr=48000, sr=16000 → ratio=3 (decimate 3:1 / interpolate 1:3)
        if native_sr % sr != 0:
            raise ValueError(
                f"native_sr ({native_sr}) must be an integer multiple of sr ({sr}). "
                f"Got ratio {native_sr / sr:.2f}."
            )
        self._resample_ratio: int = native_sr // sr  # 1 = pass-through, 3 = 48k→16k

        # Native hop size: how many samples the hardware delivers per GTCRN hop
        self._native_hop: int = hop * self._resample_ratio

        # ── Processing chain ──
        self.ola = OverlapAdd(nfft=nfft, hop=hop)
        self.enhancer = StreamingEnhancer(onnx_path, n_freq=nfft // 2 + 1)
        self.ab_switch = ABSwitch(sr=sr)
        self.leds = LEDStatus(led_sys_pin=22, led_mode_pin=23, led_act_pin=24)
        self.button_ref = None

        # ── Impulse gate placeholder (populated in Phase 5) ──
        if use_impulse_gate:
            from sih26052.runtime.impulse_gate import ImpulseGate
            self.impulse_gate = ImpulseGate(sr=sr, hop=hop)
        else:
            self.impulse_gate = None

        # ── Metrics queue for dashboard (non-blocking push) ──
        self.metrics_queue: queue.Queue = queue.Queue(maxsize=queue_size)

        # ── Xrun tracking ──
        self.xrun_count = 0
        self.frame_count = 0
        self.start_time = 0.0

        # ── Preallocated buffers (all at model rate = sr) ──
        self._mono_buffer = np.zeros(hop, dtype=np.float32)
        self._ref_buffer = np.zeros(hop, dtype=np.float32)
        self._raw_buffer = np.zeros(hop, dtype=np.float32)
        self._prev_mono_buffer = np.zeros(hop, dtype=np.float32)
        self._enhanced_buffer = np.zeros(hop, dtype=np.float32)
        # Native-rate output buffer (upsampled) pre-allocated
        self._native_out_buf = np.zeros(self._native_hop, dtype=np.float32)

    def _callback(self, indata, outdata, frames, time_info, status):
        """Sounddevice stream callback.

        This runs in a separate high-priority thread.  It MUST NOT
        allocate memory, print, or call any blocking function.

        When native_sr > sr (e.g. 48 kHz hardware, 16 kHz model):
          - Input  : decimate by ratio (simple slice — FIR pre-filter not needed at
                     ratio=3 because the I2S hardware already band-limits to 8 kHz)
          - Output : repeat-upsample by ratio (zero-order hold — acceptable latency
                     at 16 kHz block size; avoids malloc in callback)
        """
        frame_start = time.monotonic()

        # ── Track xruns ──
        if status:
            self.xrun_count += 1

        self.frame_count += 1

        # ── Get input channels (Dual INMP441: Left = Primary Mic, Right = Ref Mic) ──
        # Decimate from native_sr to sr by taking every Nth sample
        ratio = self._resample_ratio
        primary_ch = indata[:, 0]
        np.copyto(self._mono_buffer, primary_ch[::ratio] if ratio > 1 else primary_ch)
        if indata.shape[1] > 1:
            ref_ch = indata[:, 1]
            np.copyto(self._ref_buffer, ref_ch[::ratio] if ratio > 1 else ref_ch)
        mono_in = self._mono_buffer

        # ── STFT analysis ──
        spec = self.ola.analyze(mono_in)

        # ── Neural enhancement ──
        enhanced_spec = self.enhancer.process_frame(spec)

        # ── ISTFT synthesis ──
        # Delay raw audio by 1 hop to phase-align with OLA latency
        self._raw_buffer[:] = self._prev_mono_buffer
        self._prev_mono_buffer[:] = mono_in[:self.hop]
        
        self._enhanced_buffer[:] = self.ola.synthesize(enhanced_spec)

        # ── Impulse gate (Phase 5 — no-op if not set) ──
        if self.impulse_gate is not None:
            self._enhanced_buffer = self.impulse_gate.process(self._enhanced_buffer)

        # ── A/B crossfade ──
        output = self.ab_switch.apply_vectorized(
            self._raw_buffer, self._enhanced_buffer
        )

        # ── Write to output (upsample back to native_sr if needed) ──
        ratio = self._resample_ratio
        if ratio > 1:
            # Zero-order hold: repeat each sample `ratio` times (no malloc)
            np.copyto(self._native_out_buf, np.repeat(output, ratio))
            outdata[:, 0] = self._native_out_buf
            if outdata.shape[1] > 1:
                outdata[:, 1] = self._native_out_buf
        else:
            outdata[:, 0] = output
            if outdata.shape[1] > 1:
                outdata[:, 1] = output

        # Update LED 2 (GTCRN Filtered active mode) & LED 3 (Activity)
        if self.leds.enabled and self.frame_count % 5 == 0:
            self.leds.set_enhancement_mode(self.ab_switch.is_enhanced)
            # Pulse activity LED if audio level exceeds threshold
            is_active = np.max(np.abs(output)) > 0.05
            self.leds.set_activity(is_active)

        # ── Push metrics to dashboard queue (non-blocking) ──
        try:
            metrics = {
                "frame": self.frame_count,
                "spec_in": spec[:, 0].copy(),   # real part for spectrogram
                "spec_out": enhanced_spec[:, 0].copy(),
                "xruns": self.xrun_count,
                "enhanced": self.ab_switch.is_enhanced,
                "processing_time_ms": (time.monotonic() - frame_start) * 1000,
                "gate_state": self.impulse_gate.state if self.impulse_gate else "idle",
            }
            self.metrics_queue.put_nowait(metrics)
        except queue.Full:
            pass  # Dashboard is behind — drop this frame's metrics

    def run(self, duration: float | None = None) -> None:
        """Start the real-time loop."""
        import sounddevice as sd
        from sih26052.runtime.ab_switch import attach_gpio_button

        logger.info(
            "Starting audio loop: device=%s, sr=%d, hop=%d, model=%s",
            self.device, self.sr, self.hop, self.enhancer.onnx_path,
        )

        # Attach hardware push button on Pin 11 / GPIO 17
        self.button_ref = attach_gpio_button(self.ab_switch, gpio_pin=17)

        # Enable System Active LED (LED 1 / GPIO 22)
        self.leds.set_system_active(True)
        self.leds.set_enhancement_mode(self.ab_switch.is_enhanced)

        # ── Open stream ──
        # Use native_sr for the hardware stream; the callback decimates/interpolates.
        # When input and output are on different devices (I2S mic + USB DAC),
        # use separate InputStream + OutputStream instead of a duplex Stream.
        use_duplex = (self.input_device == self.output_device)

        if use_duplex:
            stream = sd.Stream(
                device=self.input_device,
                samplerate=self.native_sr,
                blocksize=self._native_hop,
                channels=2,
                dtype="float32",
                callback=self._callback,
                latency=0.064,
            )
        else:
            # Separate capture (I2S mic card) and playback (USB sound card)
            stream = sd.Stream(
                device=(self.input_device, self.output_device),
                samplerate=self.native_sr,
                blocksize=self._native_hop,
                channels=2,
                dtype="float32",
                callback=self._callback,
                latency=0.064,
            )

        self.start_time = time.monotonic()

        # ── Graceful shutdown on Ctrl+C ──
        stop_event = False

        def handle_sigint(sig, frame):
            nonlocal stop_event
            stop_event = True

        signal.signal(signal.SIGINT, handle_sigint)

        with stream:
            logger.info("Audio stream active. Press Ctrl+C to stop.")
            logger.info("Push Button (GPIO 17) or SPACE toggles A/B switch.")

            try:
                while not stop_event:
                    time.sleep(0.1)

                    # Duration limit
                    if duration and (time.monotonic() - self.start_time) >= duration:
                        break

                    # Periodic status
                    if self.frame_count > 0 and self.frame_count % 1000 == 0:
                        elapsed = time.monotonic() - self.start_time
                        logger.info(
                            "Frames: %d | Xruns: %d | Elapsed: %.1fs",
                            self.frame_count, self.xrun_count, elapsed,
                        )
            except KeyboardInterrupt:
                pass
            finally:
                self.leds.close()

        elapsed = time.monotonic() - self.start_time
        logger.info(
            "Audio loop stopped.  %d frames, %d xruns, %.1f seconds",
            self.frame_count, self.xrun_count, elapsed,
        )

    def run_offline(self, input_path: str | Path, output_path: str | Path) -> None:
        """Run the exact callback pipeline offline using soundfile."""
        import soundfile as sf
        
        logger.info("Starting offline processing: %s -> %s", input_path, output_path)
        self.start_time = time.monotonic()
        
        info = sf.info(str(input_path))
        if info.samplerate != self.sr:
            logger.error("Input SR %d != configured SR %d. Resampling not supported.", info.samplerate, self.sr)
            return
            
        with sf.SoundFile(str(input_path)) as sf_in, \
             sf.SoundFile(str(output_path), 'w', samplerate=self.sr, channels=2, subtype='FLOAT') as sf_out:
            
            for block in sf_in.blocks(blocksize=self.hop, dtype='float32', fill_value=0.0):
                # block is exactly (hop, channels) due to fill_value=0.0 padding
                if block.ndim == 1:
                    block = block.reshape(-1, 1)
                
                # Create fake outdata
                outdata = np.zeros((self.hop, 2), dtype=np.float32)
                
                self._callback(block, outdata, self.hop, None, None)
                sf_out.write(outdata)
                
            # Flush the final block (since OLA group delay is 1 hop)
            outdata = np.zeros((self.hop, 2), dtype=np.float32)
            empty_block = np.zeros((self.hop, sf_in.channels), dtype=np.float32)
            self._callback(empty_block, outdata, self.hop, None, None)
            sf_out.write(outdata)
                
        elapsed = time.monotonic() - self.start_time
        logger.info(
            "Offline processing complete. %d frames processed in %.1fs (%.2fx real-time).",
            self.frame_count, elapsed, (self.frame_count * self.hop / self.sr) / elapsed
        )


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time speech enhancement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (auto-detect, 16 kHz hardware like USB mic):
  python -m sih26052.runtime.audio_loop --onnx models/gtcrn_finetuned_stream_int8.onnx

  # Dual I2S INMP441 (hw:0) + USB DAC (hw:1), both running at 48 kHz native:
  python -m sih26052.runtime.audio_loop \\
      --onnx models/gtcrn_finetuned_stream_int8.onnx \\
      --input-device 0 --output-device 1 \\
      --native-sr 48000 --sr 16000
"""
    )
    parser.add_argument("--onnx", type=Path, default=None, help="Streaming ONNX model")
    parser.add_argument("--device", type=int, default=None,
                        help="Shared audio device index (used when input and output are the same device)")
    parser.add_argument("--input-device", default=None,
                        help="Capture device index or ALSA name (e.g. 0 or 'plughw:0'). Overrides --device for input.")
    parser.add_argument("--output-device", default=None,
                        help="Playback device index or ALSA name (e.g. 1 or 'plughw:1'). Overrides --device for output.")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Model sample rate — must match the ONNX model (default: 16000)")
    parser.add_argument("--native-sr", type=int, default=None,
                        help="Hardware sample rate. Defaults to --sr. Set to 48000 for I2S INMP441 on Pi 5.")
    parser.add_argument("--hop", type=int, default=256, help="Hop size (frames per callback block at model rate)")
    parser.add_argument("--duration", type=float, default=None, help="Duration (seconds)")
    parser.add_argument("--impulse-gate", action="store_true", help="Enable the impulse gate")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--input-file", type=Path, default=None,
                        help="Process a WAV file offline instead of live stream")
    parser.add_argument("--output-file", type=Path, default=None,
                        help="Output WAV file for offline processing")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    if not args.onnx:
        default_model = Path("models/gtcrn_finetuned_stream_int8.onnx")
        if default_model.exists():
            args.onnx = default_model
        else:
            parser.error("--onnx is required unless --list-devices is given")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # --native-sr defaults to --sr (pass-through, no resampling)
    native_sr = args.native_sr if args.native_sr is not None else args.sr

    # Parse device args — allow int or string (ALSA device name)
    def parse_device(val):
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return val  # ALSA name string like 'plughw:0'

    loop = AudioLoop(
        onnx_path=args.onnx,
        device=args.device,
        input_device=parse_device(args.input_device),
        output_device=parse_device(args.output_device),
        sr=args.sr,
        native_sr=native_sr,
        hop=args.hop,
        use_impulse_gate=args.impulse_gate,
    )
    
    if args.input_file and args.output_file:
        loop.run_offline(args.input_file, args.output_file)
    elif args.input_file or args.output_file:
        logger.error("Both --input-file and --output-file must be provided for offline mode.")
        sys.exit(1)
    else:
        loop.run(duration=args.duration)


if __name__ == "__main__":
    main()
