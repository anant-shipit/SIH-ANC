"""
server.py — FastAPI + WebSocket dashboard server.

Serves the static dashboard UI and provides a WebSocket endpoint
for real-time metrics from the audio loop.

Architecture:
    Audio callback → queue.put_nowait() → [bridge thread] → WebSocket → browser
                     ↑
             NO network I/O here

Usage:
    python -m sih26052.dashboard.server --port 8080
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app():
    """Create and configure the FastAPI application."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    app = FastAPI(title="SIH26052 Dashboard", version="0.1.0")

    # ── Serve static files ──
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ── Connected WebSocket clients ──
    clients: set[WebSocket] = set()

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        clients.add(websocket)
        logger.info("Dashboard client connected (%d total)", len(clients))

        try:
            while True:
                # Keep connection alive — client sends pings
                await websocket.receive_text()
        except WebSocketDisconnect:
            clients.discard(websocket)
            logger.info("Dashboard client disconnected (%d remaining)", len(clients))

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "clients": len(clients)}

    # Store clients set on app for the bridge to access
    app.state.clients = clients

    return app


async def broadcast(app, message: dict) -> None:
    """Broadcast a message to all connected WebSocket clients."""
    clients = app.state.clients
    if not clients:
        return

    data = json.dumps(message)
    disconnected = set()

    for ws in clients:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.add(ws)

    clients -= disconnected


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dashboard server.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import uvicorn
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
