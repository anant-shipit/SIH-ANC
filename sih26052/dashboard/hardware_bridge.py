"""
hardware_bridge.py — Hardware Bridge between Pi AudioLoop and Dashboard WebSockets.

Controls:
  - Starting/stopping the live physical audio loop from the web UI
  - Real-time telemetry broadcast (spectrum, SNR gain, latency, xruns)
  - Real-time audio chunk streaming to browser speakers
  - A/B ANC mode toggle (GTCRN filtering vs raw bypass)
  - Headphone volume control via ALSA mixer and software digital gain
  - Physical LED test routine
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class HardwareBridge:
    """Singleton bridge managing Pi hardware audio pipeline and WebSocket clients."""

    _instance: Optional[HardwareBridge] = None

    def __init__(self):
        self.loop_thread: Optional[threading.Thread] = None
        self.audio_loop: Optional[Any] = None
        self.is_running: bool = False
        self._stop_event = threading.Event()
        self.clients: Set[Any] = set()
        self.current_volume: int = 85
        self.is_enhanced: bool = True
        self._latest_telemetry: Dict[str, Any] = {
            "is_running": False,
            "enhanced": True,
            "volume": 85,
            "rms_in": 0.0,
            "rms_out": 0.0,
            "snr_gain": 0.0,
            "spec_in": [0.0] * 64,
            "spec_out": [0.0] * 64,
            "xruns": 0,
            "processing_time_ms": 0.0,
            "is_active": False,
            "cpu_temp": 0.0,
        }

    @classmethod
    def get_instance(cls) -> HardwareBridge:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_cpu_temp(self) -> float:
        """Read Raspberry Pi SoC temperature in Celsius."""
        temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if temp_path.exists():
            try:
                mc = int(temp_path.read_text().strip())
                return round(mc / 1000.0, 1)
            except Exception:
                pass
        return 42.5  # Fallback for dev environment

    def set_volume(self, volume_pct: int) -> int:
        """Set headphone volume (0-100%) via ALSA mixer and software gain."""
        volume_pct = max(0, min(100, int(volume_pct)))
        self.current_volume = volume_pct

        # 1. Update software digital gain
        if self.audio_loop is not None:
            gain = (volume_pct / 100.0) * 1.5
            self.audio_loop.set_gain(gain)

        # 2. Try updating ALSA hardware mixer
        for card_idx in (3, 1, 0):
            try:
                subprocess.run(
                    ["amixer", "-c", str(card_idx), "sset", "Speaker", f"{volume_pct}%"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                subprocess.run(
                    ["amixer", "-c", str(card_idx), "sset", "PCM", f"{volume_pct}%"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except Exception:
                pass

        logger.info("Volume set to %d%%", volume_pct)
        return self.current_volume

    def toggle_enhanced_mode(self) -> bool:
        """Toggle between GTCRN AI Enhancement and Raw Bypass."""
        self.is_enhanced = not self.is_enhanced
        if self.audio_loop is not None:
            self.audio_loop.set_enhanced_mode(self.is_enhanced)
        logger.info("ANC Mode toggled: is_enhanced=%s", self.is_enhanced)
        return self.is_enhanced

    def test_physical_leds(self) -> bool:
        """Flash physical GPIO status LEDs (22, 23, 24) for 2 seconds."""
        def _flash():
            try:
                from sih26052.runtime.led_status import LEDStatus
                leds = LEDStatus(22, 23, 24)
                leds.set_system_active(True)
                leds.set_enhancement_mode(True)
                leds.set_activity(True)
                time.sleep(2.0)
                if not self.is_running:
                    leds.set_system_active(False)
                leds.set_enhancement_mode(self.is_enhanced)
                leds.set_activity(False)
            except Exception as e:
                logger.warning("LED test error: %s", e)

        threading.Thread(target=_flash, daemon=True).start()
        return True

    def start_pipeline(
        self,
        onnx_model: str = "models/gtcrn_finetuned_stream_int8.onnx",
        native_sr: int = 48000,
    ) -> Dict[str, Any]:
        """Start the background hardware audio loop."""
        if self.is_running:
            return {"status": "already_running"}

        model_path = REPO_ROOT / onnx_model
        if not model_path.exists():
            fallback = REPO_ROOT / "models" / "gtcrn_stream_int8.onnx"
            if fallback.exists():
                model_path = fallback
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")

        from sih26052.runtime.audio_loop import AudioLoop

        # Resolve devices using auto-detection
        self.audio_loop = AudioLoop(
            onnx_path=model_path,
            native_sr=native_sr,
            sr=16000,
            hop=256,
        )
        self.audio_loop.set_gain((self.current_volume / 100.0) * 1.5)
        self.audio_loop.set_enhanced_mode(self.is_enhanced)

        self._stop_event.clear()
        self.is_running = True

        def _run_worker():
            logger.info("Hardware AudioLoop worker started.")
            try:
                import sounddevice as sd
                use_duplex = (self.audio_loop.input_device == self.audio_loop.output_device)
                dev_arg = self.audio_loop.input_device if use_duplex else (self.audio_loop.input_device, self.audio_loop.output_device)

                stream = sd.Stream(
                    device=dev_arg,
                    samplerate=self.audio_loop.native_sr,
                    blocksize=self.audio_loop._native_hop,
                    channels=2,
                    dtype="float32",
                    callback=self.audio_loop._callback,
                    latency=0.064,
                )

                self.audio_loop.leds.set_system_active(True)
                self.audio_loop.leds.set_enhancement_mode(self.is_enhanced)

                with stream:
                    while not self._stop_event.is_set():
                        time.sleep(0.05)

            except Exception as exc:
                logger.error("Error in AudioLoop worker: %s", exc)
            finally:
                self.is_running = False
                if self.audio_loop and self.audio_loop.leds:
                    self.audio_loop.leds.set_system_active(False)
                logger.info("Hardware AudioLoop worker stopped.")

        self.loop_thread = threading.Thread(target=_run_worker, daemon=True)
        self.loop_thread.start()

        return {"status": "started", "model": str(model_path.name)}

    def stop_pipeline(self) -> Dict[str, Any]:
        """Stop the background hardware audio loop."""
        if not self.is_running:
            return {"status": "not_running"}

        self._stop_event.set()
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=2.0)

        self.is_running = False
        return {"status": "stopped"}

    def get_status(self) -> Dict[str, Any]:
        """Return current status dictionary."""
        status = dict(self._latest_telemetry)
        status.update({
            "is_running": self.is_running,
            "enhanced": self.is_enhanced,
            "volume": self.current_volume,
            "cpu_temp": self.get_cpu_temp(),
        })
        return status

    async def telemetry_broadcaster(self):
        """Async task continuously pushing telemetry from metrics_queue to WebSockets."""
        while True:
            if self.is_running and self.audio_loop is not None:
                try:
                    # Drain queue to get the freshest metric
                    metric = None
                    while not self.audio_loop.metrics_queue.empty():
                        metric = self.audio_loop.metrics_queue.get_nowait()

                    if metric is not None:
                        metric["cpu_temp"] = self.get_cpu_temp()
                        metric["volume"] = self.current_volume
                        metric["is_running"] = True
                        self._latest_telemetry = metric

                        if self.clients:
                            payload = json.dumps(metric)
                            dead_clients = set()
                            for ws in list(self.clients):
                                try:
                                    await ws.send_text(payload)
                                except Exception:
                                    dead_clients.add(ws)
                            self.clients -= dead_clients
                except Exception as e:
                    logger.debug("Telemetry broadcast error: %s", e)

            await asyncio.sleep(0.033)  # ~30 fps broadcast rate


hardware_bridge = HardwareBridge.get_instance()
