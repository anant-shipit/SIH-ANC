/**
 * app.js — Dashboard WebSocket client + canvas spectrogram rendering.
 *
 * Connects to the FastAPI WebSocket endpoint and renders:
 *   - Scrolling spectrograms (before/after enhancement)
 *   - Latency, xrun count, frame count
 *   - Transient indicator state (idle/fired/releasing)
 */

// ── Constants ────────────────────────────────────────────────────────────
const WS_URL = `ws://${window.location.host}/ws`;
const RECONNECT_INTERVAL_MS = 2000;
const SPECTROGRAM_BINS = 64;  // frequency bins to display
const SPECTROGRAM_WIDTH = 800;
const SPECTROGRAM_HEIGHT = 200;

// Viridis-inspired colormap (dark purple → blue → green → yellow)
const COLORMAP = [
    [68, 1, 84],    // dark purple
    [72, 35, 116],
    [64, 67, 135],
    [52, 94, 141],
    [33, 145, 140], // teal
    [53, 183, 121],
    [109, 205, 89],
    [180, 222, 44],
    [253, 231, 37], // yellow
];

// ── State ────────────────────────────────────────────────────────────────
let ws = null;
let reconnectTimer = null;

// Spectrogram image data (scrolling left)
let spectrogramInData = [];
let spectrogramOutData = [];

// ── DOM Elements ─────────────────────────────────────────────────────────
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const latencyValue = document.getElementById('latency-value');
const xrunValue = document.getElementById('xrun-value');
const xrunCard = document.getElementById('xrun-card');
const modeValue = document.getElementById('mode-value');
const frameValue = document.getElementById('frame-value');
const transientIndicator = document.getElementById('transient-indicator');
const transientText = document.getElementById('transient-text');
const canvasIn = document.getElementById('spectrogram-in');
const canvasOut = document.getElementById('spectrogram-out');
const ctxIn = canvasIn.getContext('2d');
const ctxOut = canvasOut.getContext('2d');

// ── WebSocket Connection ─────────────────────────────────────────────────

let pingTimer = null;

function connect() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        statusDot.className = 'status-dot connected';
        statusText.textContent = 'Connected';
        if (reconnectTimer) {
            clearInterval(reconnectTimer);
            reconnectTimer = null;
        }
        if (pingTimer) {
            clearInterval(pingTimer);
        }
        // Send periodic pings to keep connection alive
        pingTimer = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send('ping');
            }
        }, 5000);
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        } catch (e) {
            console.error('Failed to parse message:', e);
        }
    };

    ws.onclose = () => {
        statusDot.className = 'status-dot disconnected';
        statusText.textContent = 'Disconnected';
        if (pingTimer) {
            clearInterval(pingTimer);
            pingTimer = null;
        }
        scheduleReconnect();
    };

    ws.onerror = () => {
        statusDot.className = 'status-dot disconnected';
        statusText.textContent = 'Error';
    };
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setInterval(connect, RECONNECT_INTERVAL_MS);
}

// ── Dashboard Update ─────────────────────────────────────────────────────

function updateDashboard(data) {
    // Frame count
    if (data.frame !== undefined) {
        frameValue.textContent = data.frame.toLocaleString();
    }

    // Xrun count
    if (data.xruns !== undefined) {
        xrunValue.textContent = data.xruns;
        if (data.xruns > 0) {
            xrunCard.classList.add('has-xruns');
        }
    }

    // Mode
    if (data.enhanced !== undefined) {
        if (data.enhanced) {
            modeValue.textContent = 'ENHANCED';
            modeValue.className = 'stat-value enhanced';
        } else {
            modeValue.textContent = 'BYPASS';
            modeValue.className = 'stat-value bypass';
        }
    }

    // Transient state
    if (data.gate_state !== undefined) {
        transientIndicator.className = 'transient-indicator ' + data.gate_state;
        transientText.textContent = data.gate_state.toUpperCase();
    }

    // Latency and Compute Budget
    if (data.processing_time_ms !== undefined) {
        latencyValue.textContent = data.processing_time_ms.toFixed(1);
        
        // Update compute gauge (scale to 20ms max, 16ms is 80%)
        const computeVal = data.processing_time_ms;
        const computeValEl = document.getElementById('compute-val');
        if (computeValEl) {
            computeValEl.textContent = computeVal.toFixed(1);
        }
        
        const gauge = document.getElementById('compute-gauge');
        if (gauge) {
            let pct = (computeVal / 20.0) * 100;
            pct = Math.min(100, Math.max(0, pct));
            gauge.style.width = pct + '%';
            
            if (computeVal > 16.0) {
                gauge.style.backgroundColor = 'var(--accent-red)';
            } else if (computeVal > 12.0) {
                gauge.style.backgroundColor = 'var(--accent-yellow)';
            } else {
                gauge.style.backgroundColor = 'var(--accent-green)';
            }
        }
    }

    // Spectrograms
    if (data.spec_in) {
        spectrogramInData.push(data.spec_in);
        if (spectrogramInData.length > SPECTROGRAM_WIDTH) {
            spectrogramInData.shift();
        }
        drawSpectrogram(ctxIn, canvasIn, spectrogramInData);
    }

    if (data.spec_out) {
        spectrogramOutData.push(data.spec_out);
        if (spectrogramOutData.length > SPECTROGRAM_WIDTH) {
            spectrogramOutData.shift();
        }
        drawSpectrogram(ctxOut, canvasOut, spectrogramOutData);
    }
}

// ── Spectrogram Rendering ────────────────────────────────────────────────

function drawSpectrogram(ctx, canvas, data) {
    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);

    if (data.length === 0) return;

    const nBins = data[0].length;
    const nFrames = data.length;
    const colWidth = Math.max(1, width / SPECTROGRAM_WIDTH);
    const rowHeight = height / nBins;

    // Find global min/max for normalization
    let minVal = Infinity, maxVal = -Infinity;
    for (let i = 0; i < nFrames; i++) {
        for (let j = 0; j < nBins; j++) {
            const v = Math.abs(data[i][j]);
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }
    }

    const range = maxVal - minVal || 1;

    // Draw columns (time) from right to left (newest on the right)
    for (let i = 0; i < nFrames; i++) {
        const x = width - (nFrames - i) * colWidth;
        if (x < 0) continue;

        for (let j = 0; j < nBins; j++) {
            const normalized = (Math.abs(data[i][j]) - minVal) / range;
            // Convert to dB scale for better visibility
            const dbNorm = Math.max(0, Math.min(1, 1 + Math.log10(normalized + 1e-6) / 3));
            const color = getColor(dbNorm);

            ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
            // Flip y-axis: low frequencies at bottom
            const y = height - (j + 1) * rowHeight;
            ctx.fillRect(x, y, colWidth + 1, rowHeight + 1);
        }
    }
}

function getColor(t) {
    // Interpolate through colormap
    t = Math.max(0, Math.min(1, t));
    const idx = t * (COLORMAP.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, COLORMAP.length - 1);
    const frac = idx - lo;

    return [
        Math.round(COLORMAP[lo][0] + frac * (COLORMAP[hi][0] - COLORMAP[lo][0])),
        Math.round(COLORMAP[lo][1] + frac * (COLORMAP[hi][1] - COLORMAP[lo][1])),
        Math.round(COLORMAP[lo][2] + frac * (COLORMAP[hi][2] - COLORMAP[lo][2])),
    ];
}

// ── Initialize ───────────────────────────────────────────────────────────

// Set canvas resolution to match CSS size
function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
}

window.addEventListener('load', () => {
    resizeCanvas(canvasIn);
    resizeCanvas(canvasOut);
    connect();
});

window.addEventListener('resize', () => {
    resizeCanvas(canvasIn);
    resizeCanvas(canvasOut);
});
