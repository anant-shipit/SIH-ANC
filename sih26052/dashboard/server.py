"""
server.py — FastAPI + WebSocket dashboard server for the GTCRN Speech Enhancement Model.

Serves the interactive Model Dashboard UI, handles audio upload/enhancement requests,
provides preset samples, and supports low-latency real-time microphone streaming.

Usage:
    python -m sih26052.dashboard.server --port 8080
    python scripts/run_dashboard.py
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from sih26052.dashboard.processor import (
    model_manager,
    process_audio_file,
    read_audio_from_bytes,
    write_audio_to_base64_wav,
)
from sih26052.dashboard.hardware_bridge import hardware_bridge

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).parent / "static"
TEST_AUDIO_DIR = STATIC_DIR / "eval_samples"
if not TEST_AUDIO_DIR.exists():
    TEST_AUDIO_DIR = REPO_ROOT / "data" / "test_audio"


def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(title="GTCRN Speech Enhancement Studio", version="1.0.0")

    # Allow CORS for development flexibility
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static assets
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(hardware_bridge.telemetry_broadcaster())

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    # ── Hardware AudioLoop & Diagnostics API ─────────────────────────────────

    @app.get("/api/hardware/status")
    async def hardware_status():
        """Retrieve real-time Pi hardware pipeline status & diagnostics."""
        return hardware_bridge.get_status()

    @app.post("/api/hardware/start")
    async def hardware_start(model: str = Form("models/gtcrn_finetuned_stream_int8.onnx")):
        """Start the background hardware AudioLoop on Raspberry Pi."""
        try:
            res = hardware_bridge.start_pipeline(onnx_model=model)
            return JSONResponse(content=res)
        except Exception as e:
            logger.exception("Hardware start error: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/hardware/stop")
    async def hardware_stop():
        """Stop the background hardware AudioLoop."""
        return hardware_bridge.stop_pipeline()

    @app.post("/api/hardware/toggle-mode")
    async def hardware_toggle_mode():
        """Toggle ANC Active vs Passthrough Bypass mode."""
        is_enhanced = hardware_bridge.toggle_enhanced_mode()
        return {"enhanced": is_enhanced}

    @app.post("/api/hardware/volume")
    async def hardware_set_volume(volume: int = Form(...)):
        """Set headphone output volume (0-100%)."""
        vol = hardware_bridge.set_volume(volume)
        return {"volume": vol}

    @app.post("/api/hardware/test-leds")
    async def hardware_test_leds():
        """Flash physical GPIO status LEDs for 2 seconds."""
        hardware_bridge.test_physical_leds()
        return {"status": "testing_leds"}

    @app.post("/api/hardware/play-sample")
    async def hardware_play_sample(mode: str = Form("enhanced")):
        """Play processed audio (enhanced or raw) directly through the Pi's physical headphones."""
        try:
            res = hardware_bridge.play_on_headphones(audio_type=mode)
            return JSONResponse(content=res)
        except Exception as e:
            logger.error("Pi headphone playback error: %s", e)
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/hardware/stop-playback")
    async def hardware_stop_playback():
        """Stop sound playing on the Pi's physical headphones."""
        return hardware_bridge.stop_headphone_playback()

    @app.websocket("/ws/hardware")
    async def websocket_hardware_endpoint(websocket: WebSocket):
        """High-speed 30fps telemetry and audio stream from Raspberry Pi hardware."""
        await websocket.accept()
        hardware_bridge.clients.add(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    cmd = json.loads(data)
                    action = cmd.get("action")
                    if action == "toggle_mode":
                        hardware_bridge.toggle_enhanced_mode()
                    elif action == "set_volume":
                        hardware_bridge.set_volume(cmd.get("volume", 85))
                    elif action == "test_leds":
                        hardware_bridge.test_physical_leds()
                except Exception:
                    pass
        except WebSocketDisconnect:
            hardware_bridge.clients.discard(websocket)

    # ── Model API ─────────────────────────────────────────────────────────────

    @app.get("/api/models")
    async def get_models():
        """List available enhancement model engines."""
        return {"engines": model_manager.list_available_engines()}

    @app.get("/api/presets")
    async def list_presets():
        """List sample test audio files available for instant demonstration."""
        presets = []
        if TEST_AUDIO_DIR.exists():
            noisy_files = sorted(TEST_AUDIO_DIR.glob("*_noisy.wav"))
            labels = {
                "sample_0": "Battlefield Heavy Noise (-5 dB SNR)",
                "sample_1": "Armored Vehicle Engine Noise (+7 dB SNR)",
                "sample_2": "Command Center Radio Babble (+12 dB SNR)",
                "sample_3": "High-Wind & Rotor Blade Noise (-4 dB SNR)",
                "sample_4": "Tactical Radio Static & Gunfire (+7 dB SNR)",
            }
            for i, noisy_path in enumerate(noisy_files):
                clean_path = TEST_AUDIO_DIR / noisy_path.name.replace("_noisy.wav", "_clean.wav")
                preset_id = noisy_path.stem.replace("_noisy", "")
                presets.append({
                    "id": preset_id,
                    "title": labels.get(preset_id, f"Defense Sample #{i+1:02d} ({preset_id})"),
                    "filename": noisy_path.name,
                    "has_clean": clean_path.exists(),
                    "size_bytes": noisy_path.stat().st_size,
                })
        return {"presets": presets}

    @app.get("/api/preset/{preset_id}")
    async def get_preset_audio(preset_id: str):
        """Retrieve audio data for a given preset ID and cache for Pi headphone output."""
        noisy_path = TEST_AUDIO_DIR / f"{preset_id}_noisy.wav"
        if not noisy_path.exists():
            noisy_path = TEST_AUDIO_DIR / preset_id
        if not noisy_path.exists():
            raise HTTPException(status_code=404, detail="Preset file not found")

        clean_path = TEST_AUDIO_DIR / f"{preset_id}_clean.wav"
        enh_path = TEST_AUDIO_DIR / f"{preset_id}_enhanced.wav"

        noisy_bytes = noisy_path.read_bytes()
        clean_bytes = clean_path.read_bytes() if clean_path.exists() else None
        enh_bytes = enh_path.read_bytes() if enh_path.exists() else None

        noisy_arr, sr = read_audio_from_bytes(noisy_bytes)
        clean_arr, _ = read_audio_from_bytes(clean_bytes) if clean_bytes else (None, 16000)
        enh_arr, _ = read_audio_from_bytes(enh_bytes) if enh_bytes else (None, 16000)

        # Cache samples in hardware bridge so user can immediately listen on Pi headphones
        target_enh = enh_arr if enh_arr is not None else (clean_arr if clean_arr is not None else noisy_arr)
        hardware_bridge.cache_samples(noisy_arr, target_enh, sr=sr)

        return {
            "id": preset_id,
            "sample_rate": sr,
            "duration_sec": round(len(noisy_arr) / sr, 2),
            "noisy_audio_url": write_audio_to_base64_wav(noisy_arr, sr),
            "clean_audio_url": write_audio_to_base64_wav(clean_arr, sr) if clean_arr is not None else None,
            "enhanced_audio_url": write_audio_to_base64_wav(enh_arr, sr) if enh_arr is not None else None,
            "has_clean": clean_arr is not None,
            "has_enhanced": enh_arr is not None,
        }

    @app.post("/api/enhance")
    async def enhance_audio(
        file: Optional[UploadFile] = File(None),
        engine: str = Form("onnx_stream_int8"),
        clean_file: Optional[UploadFile] = File(None),
        preset_id: Optional[str] = Form(None),
    ):
        """Process an audio file (uploaded or preset) and cache for Pi headphone playback."""
        try:
            if file is not None:
                audio_bytes = await file.read()
            elif preset_id:
                preset_file = TEST_AUDIO_DIR / f"{preset_id}_noisy.wav"
                if not preset_file.exists():
                    preset_file = TEST_AUDIO_DIR / preset_id
                if not preset_file.exists():
                    raise HTTPException(status_code=404, detail=f"Preset {preset_id} not found")
                audio_bytes = preset_file.read_bytes()
            else:
                raise HTTPException(status_code=400, detail="No audio file or preset ID provided")

            clean_bytes = None
            if clean_file:
                clean_bytes = await clean_file.read()
            elif preset_id:
                clean_path = TEST_AUDIO_DIR / f"{preset_id}_clean.wav"
                if clean_path.exists():
                    clean_bytes = clean_path.read_bytes()

            result = process_audio_file(
                raw_audio_bytes=audio_bytes,
                engine=engine,
                clean_audio_bytes=clean_bytes,
            )

            # Pop numpy arrays to avoid JSON serialization errors and cache for Pi headphone output
            raw_arr = result.pop("_raw_array", None)
            enh_arr = result.pop("_enh_array", None)
            if raw_arr is not None and enh_arr is not None:
                hardware_bridge.cache_samples(raw_arr, enh_arr, sr=result.get("sample_rate", 16000))

            return JSONResponse(content=result)
        except Exception as e:
            logger.exception("Error enhancing audio: %s", e)
            raise HTTPException(status_code=500, detail=f"Enhancement error: {str(e)}")

    # ── Real-Time Streaming WebSocket ─────────────────────────────────────────

    @app.websocket("/ws/stream")
    async def stream_endpoint(websocket: WebSocket):
        """Low-latency real-time streaming audio enhancement over WebSocket."""
        await websocket.accept()
        logger.info("Real-time audio streaming client connected")

        from sih26052.runtime.enhancer import StreamingEnhancer
        from sih26052.runtime.ola import OverlapAdd

        # Initialize streaming pipeline per client session
        nfft = 512
        hop = 256
        ola = OverlapAdd(nfft=nfft, hop=hop)
        enhancer = model_manager.get_onnx_enhancer(int8=True)
        enhancer.reset()

        try:
            while True:
                # Expect binary PCM float32 or int16 data (256 samples = 16ms at 16kHz)
                message = await websocket.receive()
                if "bytes" in message:
                    raw_bytes = message["bytes"]
                    # Assume 256 samples of float32 (1024 bytes) or int16 (512 bytes)
                    if len(raw_bytes) == hop * 4:
                        chunk = np.frombuffer(raw_bytes, dtype=np.float32)
                    elif len(raw_bytes) == hop * 2:
                        chunk = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    else:
                        # Reshape or pad
                        chunk = np.frombuffer(raw_bytes, dtype=np.float32)
                        if len(chunk) < hop:
                            chunk = np.pad(chunk, (0, hop - len(chunk)))
                        else:
                            chunk = chunk[:hop]

                    # Process through OLA and ONNX model
                    spec_frame = ola.analyze(chunk)
                    enh_spec = enhancer.process_frame(spec_frame)
                    out_chunk = ola.synthesize(enh_spec)

                    # Send back enhanced float32 PCM chunk
                    out_bytes = out_chunk.astype(np.float32).tobytes()
                    await websocket.send_bytes(out_bytes)
                elif "text" in message:
                    # Client control commands (e.g. reset state or ping)
                    msg_text = message["text"]
                    if msg_text == "ping":
                        await websocket.send_text("pong")
                    elif msg_text == "reset":
                        enhancer.reset()
                        ola.reset()
                        await websocket.send_text("reset_ack")
        except WebSocketDisconnect:
            logger.info("Streaming client disconnected")
        except Exception as e:
            logger.warning("Streaming websocket exception: %s", e)

    # ── Legacy Telemetry WebSocket ───────────────────────────────────────────

    @app.websocket("/ws")
    async def legacy_websocket(websocket: WebSocket):
        """Unified telemetry WebSocket endpoint."""
        await websocket.accept()
        hardware_bridge.clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            hardware_bridge.clients.discard(websocket)

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "telemetry_clients": len(hardware_bridge.clients),
            "is_running": hardware_bridge.is_running,
            "model_ready": True,
            "available_engines": [e["id"] for e in model_manager.list_available_engines()],
        }

    app.state.clients = hardware_bridge.clients

    return app


async def broadcast(app, message: dict) -> None:
    """Broadcast a telemetry message to all connected clients."""
    clients = getattr(app.state, "clients", set())
    if not clients:
        return

    data = json.dumps(message)
    disconnected = set()

    for ws in list(clients):
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.add(ws)

    clients -= disconnected


def main():
    parser = argparse.ArgumentParser(description="GTCRN Model Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind server to (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Starting GTCRN Speech Enhancement Studio on http://%s:%d", args.host, args.port)

    import uvicorn
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
