"""
bridge.py — Queue consumer → WebSocket broadcaster.

Runs in a separate thread.  Reads metrics from the audio loop's queue
and broadcasts them to all connected dashboard clients via the FastAPI
WebSocket endpoint.

Rate limited to ~10 Hz to avoid overwhelming the browser.

Architecture:
    [audio callback] → queue.put_nowait() → [bridge thread] → WebSocket → browser

The bridge is the ONLY component that does network I/O.  The audio
callback NEVER touches the network.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)


class DashboardBridge:
    """Bridge between audio loop queue and WebSocket clients.

    Parameters
    ----------
    metrics_queue : the queue that the audio callback pushes metrics to
    app           : the FastAPI app (to access connected clients)
    target_hz     : target broadcast rate (default: 10 Hz)
    """

    def __init__(
        self,
        metrics_queue: queue.Queue,
        app,
        target_hz: float = 10.0,
    ):
        self._queue = metrics_queue
        self._app = app
        self._interval = 1.0 / target_hz
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the bridge thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="dashboard-bridge")
        self._thread.start()
        logger.info("Dashboard bridge started (%.0f Hz)", 1.0 / self._interval)

    def stop(self) -> None:
        """Stop the bridge thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Dashboard bridge stopped")

    def _run(self) -> None:
        """Main bridge loop — runs in a separate thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self._running:
            t0 = time.monotonic()

            # Drain the queue — take the latest metrics, discard older ones
            latest = None
            try:
                while True:
                    latest = self._queue.get_nowait()
            except queue.Empty:
                pass

            if latest is not None:
                # Convert numpy arrays to lists for JSON serialization
                message = self._serialize(latest)
                # Broadcast to clients
                try:
                    loop.run_until_complete(
                        self._broadcast(message)
                    )
                except Exception as exc:
                    logger.debug("Broadcast error: %s", exc)

            # Rate limit
            elapsed = time.monotonic() - t0
            sleep_time = max(0.0, self._interval - elapsed)
            time.sleep(sleep_time)

        loop.close()

    def _serialize(self, metrics: dict) -> dict:
        """Convert metrics dict to JSON-serializable format."""
        serialized = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                # Downsample spectrum for transmission (64 bins is enough for display)
                if len(value) > 64:
                    # Simple downsampling by averaging adjacent bins
                    n_bins = 64
                    bin_size = len(value) // n_bins
                    downsampled = [
                        float(np.mean(value[i * bin_size:(i + 1) * bin_size]))
                        for i in range(n_bins)
                    ]
                    serialized[key] = downsampled
                else:
                    serialized[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                serialized[key] = float(value)
            else:
                serialized[key] = value
        return serialized

    async def _broadcast(self, message: dict) -> None:
        """Send message to all connected WebSocket clients."""
        from sih26052.dashboard.server import broadcast
        await broadcast(self._app, message)
