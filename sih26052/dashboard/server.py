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

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).parent / "static"
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

    # Set of connected telemetry WebSocket clients
    telemetry_clients: set[WebSocket] = set()

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

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
            for i, noisy_path in enumerate(noisy_files[:15]):  # Provide top 15 presets
                clean_path = TEST_AUDIO_DIR / noisy_path.name.replace("_noisy.wav", "_clean.wav")
                # Generate a descriptive title based on index or name
                preset_id = noisy_path.stem.replace("_noisy", "")
                presets.append({
                    "id": preset_id,
                    "title": f"Sample #{i+1:02d} ({preset_id})",
                    "filename": noisy_path.name,
                    "has_clean": clean_path.exists(),
                    "size_bytes": noisy_path.stat().st_size,
                })
        return {"presets": presets}

    @app.get("/api/preset/{preset_id}")
    async def get_preset_audio(preset_id: str):
        """Retrieve audio data for a given preset ID."""
        noisy_path = TEST_AUDIO_DIR / f"{preset_id}_noisy.wav"
        if not noisy_path.exists():
            # Try raw name
            noisy_path = TEST_AUDIO_DIR / preset_id
        if not noisy_path.exists():
            raise HTTPException(status_code=404, detail="Preset file not found")

        clean_path = TEST_AUDIO_DIR / f"{preset_id}_clean.wav"

        noisy_bytes = noisy_path.read_bytes()
        clean_bytes = clean_path.read_bytes() if clean_path.exists() else None

        noisy_arr, sr = read_audio_from_bytes(noisy_bytes)
        clean_arr, _ = read_audio_from_bytes(clean_bytes) if clean_bytes else (None, 16000)

        return {
            "id": preset_id,
            "sample_rate": sr,
            "duration_sec": round(len(noisy_arr) / sr, 2),
            "noisy_audio_url": write_audio_to_base64_wav(noisy_arr, sr),
            "clean_audio_url": write_audio_to_base64_wav(clean_arr, sr) if clean_arr is not None else None,
            "has_clean": clean_arr is not None,
        }

    @app.post("/api/enhance")
    async def enhance_audio(
        file: UploadFile = File(...),
        engine: str = Form("pytorch"),
        clean_file: Optional[UploadFile] = File(None),
        preset_id: Optional[str] = Form(None),
    ):
        """Process an uploaded noisy audio file and return enhanced audio + metrics."""
        try:
            audio_bytes = await file.read()
            if not audio_bytes:
                raise HTTPException(status_code=400, detail="Empty audio file provided")

            clean_bytes = None
            if clean_file:
                clean_bytes = await clean_file.read()
            elif preset_id:
                # If preset_id is passed, auto-load its clean pair if present
                clean_path = TEST_AUDIO_DIR / f"{preset_id}_clean.wav"
                if clean_path.exists():
                    clean_bytes = clean_path.read_bytes()

            result = process_audio_file(
                raw_audio_bytes=audio_bytes,
                engine=engine,
                clean_audio_bytes=clean_bytes,
            )
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
    async def websocket_telemetry_endpoint(websocket: WebSocket):
        """Telemetry WebSocket endpoint for Raspberry Pi audio loop metrics."""
        await websocket.accept()
        telemetry_clients.add(websocket)
        logger.info("Telemetry client connected (%d total)", len(telemetry_clients))

        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            telemetry_clients.discard(websocket)
            logger.info("Telemetry client disconnected (%d remaining)", len(telemetry_clients))

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "telemetry_clients": len(telemetry_clients),
            "model_ready": True,
            "available_engines": [e["id"] for e in model_manager.list_available_engines()],
        }

    # Store telemetry clients on app state for bridge
    app.state.clients = telemetry_clients

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
