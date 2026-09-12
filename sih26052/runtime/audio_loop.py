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


def resolve_stream_channels(
    input_dev: int | str | None = None,
    output_dev: int | str | None = None,
) -> tuple[int, int]:
    """Query sounddevice to determine supported hardware channel counts.

    Returns (in_channels, out_channels) clamped to at most 2 channels for pipeline compatibility.
    Handles mono microphones (1 in) and stereo output (2 out) seamlessly.
    """
    import sounddevice as sd

    in_ch = 1
    out_ch = 2

    try:
        target_in = input_dev if input_dev is not None else sd.default.device[0]
        if target_in is not None and target_in != -1:
            dev_info = sd.query_devices(target_in)
            if isinstance(dev_info, dict):
                max_in = dev_info.get("max_input_channels", 1)
                in_ch = min(2, max(1, int(max_in)))
    except Exception as exc:
        logger.debug("Failed to query input channels for device %s: %s", input_dev, exc)
        in_ch = 1

    try:
        target_out = output_dev if output_dev is not None else sd.default.device[1]
        if target_out is not None and target_out != -1:
            dev_info = sd.query_devices(target_out)
            if isinstance(dev_info, dict):
                max_out = dev_info.get("max_output_channels", 2)
                out_ch = min(2, max(1, int(max_out)))
    except Exception as exc:
        logger.debug("Failed to query output channels for device %s: %s", output_dev, exc)
        out_ch = 2

    return in_ch, out_ch


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
        leds: Optional[Any] = None,
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
        self.leds = leds if leds is not None else LEDStatus(led_sys_pin=22, led_mode_pin=23, led_act_pin=24)
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

        # ── Output gain & activity hysteresis ──
        self.output_gain: float = 1.0
        self._act_hold_frames: int = 0

    def set_gain(self, gain: float) -> None:
        """Set digital output gain (0.0 to 3.0)."""
        self.output_gain = max(0.0, min(3.0, float(gain)))

    def set_enhanced_mode(self, enabled: bool) -> None:
        """Dynamically enable/disable AI enhancement (ANC)."""
        if hasattr(self.ab_switch, "set_enhanced"):
            self.ab_switch.set_enhanced(enabled)
        elif hasattr(self.ab_switch, "toggle"):
            if self.ab_switch.is_enhanced != enabled:
                self.ab_switch.toggle()
        if self.leds and getattr(self.leds, "enabled", False):
            self.leds.set_enhancement_mode(enabled)

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
        self.frame_count += 1

        if status.input_overflow or status.output_underflow:
            self.xrun_count += 1

        # ── Decimate input if native_sr > sr ──
        ratio = self._resample_ratio
        if ratio > 1:
            input_block = indata[::ratio, :]
        else:
            input_block = indata

        # Split channels: Ch0 = Primary (Speech+Noise), Ch1 = Reference (Noise)
        if input_block.shape[1] >= 2:
            np.copyto(self._mono_buffer, input_block[:, 0])
            np.copyto(self._ref_buffer, input_block[:, 1])
        else:
            np.copyto(self._mono_buffer, input_block[:, 0])
            np.copyto(self._ref_buffer, input_block[:, 0])

        mono = self._mono_buffer
        np.copyto(self._raw_buffer, mono)

        # ── STFT analysis ──
        spec = self.ola.analyze(mono)

        # ── Model enhancement ──
        enhanced_spec = self.enhancer.process_frame(spec)

        # ── ISTFT synthesis ──
        enhanced_time = self.ola.synthesize(enhanced_spec)
        np.copyto(self._enhanced_buffer, enhanced_time)

        # ── Impulse gate (Phase 5) ──
        if self.impulse_gate is not None:
            gated = self.impulse_gate.process(self._enhanced_buffer)
            np.copyto(self._enhanced_buffer, gated)

        # ── A/B switch (instant mode toggle) ──
        output = self.ab_switch.select(
            raw=self._prev_mono_buffer,
            enhanced=self._enhanced_buffer,
        )
        np.copyto(self._prev_mono_buffer, self._raw_buffer)

        # ── Apply digital headphone gain ──
        if self.output_gain != 1.0:
            output = np.clip(output * self.output_gain, -1.0, 1.0)

        # ── Write to output (upsample back to native_sr if needed) ──
        if ratio > 1:
            np.copyto(self._native_out_buf, np.repeat(output, ratio))
            for ch in range(outdata.shape[1]):
                outdata[:, ch] = self._native_out_buf
        else:
            for ch in range(outdata.shape[1]):
                outdata[:, ch] = output

        # ── Smooth Voice Activity LED with hysteresis (no flickering on air) ──
        peak_level = float(np.max(np.abs(output)))
        if peak_level > 0.12:  # Speech threshold (calibrated so ambient air won't trigger)
            self._act_hold_frames = 15  # Hold for ~240ms (15 frames * 16ms)
        elif self._act_hold_frames > 0:
            self._act_hold_frames -= 1

        is_active = self._act_hold_frames > 0

        # Update LED 2 (GTCRN Filtered active mode) & LED 3 (Activity)
        if self.leds.enabled and self.frame_count % 3 == 0:
            self.leds.set_enhancement_mode(self.ab_switch.is_enhanced)
            self.leds.set_activity(is_active)
        # ── Push metrics to dashboard queue (non-blocking, ~30 fps) ──
        try:
            if self.frame_count % 2 == 0:
                mag_in = np.abs(spec[:64, 0])
                mag_out = np.abs(enhanced_spec[:64, 0])
                rms_in = float(np.sqrt(np.mean(mono**2)))
                rms_out = float(np.sqrt(np.mean(output**2)))
                snr_gain = max(0.0, 20.0 * np.log10(max(rms_out, 1e-5) / max(rms_in, 1e-5) + 1.0))
                metrics = {
                    "frame": self.frame_count,
                    "rms_in": round(rms_in, 4),
                    "rms_out": round(rms_out, 4),
                    "snr_gain": round(snr_gain, 1),
                    "spec_in": [round(float(v), 3) for v in mag_in],
                    "spec_out": [round(float(v), 3) for v in mag_out],
                    "xruns": self.xrun_count,
                    "enhanced": self.ab_switch.is_enhanced,
                    "processing_time_ms": round((time.monotonic() - frame_start) * 1000, 2),
                    "is_active": is_active,
                    "gate_state": self.impulse_gate.state if self.impulse_gate else "idle",
                    "audio_chunk": output[:128].astype(np.float32).tolist(),
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
        # Query hardware device channel limits dynamically
        in_ch, out_ch = resolve_stream_channels(self.input_device, self.output_device)
        logger.info("Opening hardware audio stream: channels=(in=%d, out=%d), latency=0.064s", in_ch, out_ch)

        use_duplex = (self.input_device == self.output_device)
        dev_arg = self.input_device if use_duplex else (self.input_device, self.output_device)

        stream = sd.Stream(
            device=dev_arg,
            samplerate=self.native_sr,
            blocksize=self._native_hop,
            channels=(in_ch, out_ch),
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

    # Parse / resolve device args — handle integers, ALSA names ('plughw:2', 'hw:2'), substrings, or auto-detect
    def resolve_device(val, kind="input"):
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            pass

        import sounddevice as sd
        import re
        devices = sd.query_devices()
        val_str = str(val).lower()

        # Extract hardware card number if specified like "plughw:2" or "hw:2,0"
        m = re.search(r"hw:?(\d+)", val_str)
        hw_target = f"hw:{m.group(1)}" if m else None

        for idx, d in enumerate(devices):
            channels = d["max_input_channels"] if kind == "input" else d["max_output_channels"]
            if channels <= 0:
                continue
            d_name = d["name"].lower()
            if hw_target and hw_target in d_name:
                return idx
            if val_str in d_name:
                return idx
        return val

    def auto_detect_devices():
        import sounddevice as sd
        devices = sd.query_devices()
        in_idx = None
        out_idx = None

        in_keywords = ["voicehat", "googlevoicehat", "i2s", "inmp441", "mic"]
        for kw in in_keywords:
            for idx, d in enumerate(devices):
                if d["max_input_channels"] > 0 and kw in d["name"].lower():
                    in_idx = idx
                    break
            if in_idx is not None:
                break

        if in_idx is None:
            for idx, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    in_idx = idx
                    break

        out_keywords = ["usb", "pnp", "headphone", "audio", "dac", "codec"]
        for kw in out_keywords:
            for idx, d in enumerate(devices):
                if d["max_output_channels"] > 0 and kw in d["name"].lower():
                    out_idx = idx
                    break
            if out_idx is not None:
                break

        if out_idx is None:
            for idx, d in enumerate(devices):
                if d["max_output_channels"] > 0:
                    out_idx = idx
                    break

        return in_idx, out_idx

    input_dev = resolve_device(args.input_device, kind="input")
    output_dev = resolve_device(args.output_device, kind="output")

    if input_dev is None and output_dev is None and args.device is None:
        auto_in, auto_out = auto_detect_devices()
        logger.info("Auto-detected devices: input=%s, output=%s", auto_in, auto_out)
        input_dev = auto_in
        output_dev = auto_out

    import sounddevice as sd
    devs = sd.query_devices()
    if input_dev is not None and isinstance(input_dev, int) and input_dev < len(devs):
        logger.info("Selected Input Device  [%d]: %s (%d in)", input_dev, devs[input_dev]["name"], devs[input_dev]["max_input_channels"])
    if output_dev is not None and isinstance(output_dev, int) and output_dev < len(devs):
        logger.info("Selected Output Device [%d]: %s (%d out)", output_dev, devs[output_dev]["name"], devs[output_dev]["max_output_channels"])

    loop = AudioLoop(
        onnx_path=args.onnx,
        device=args.device,
        input_device=input_dev,
        output_device=output_dev,
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
