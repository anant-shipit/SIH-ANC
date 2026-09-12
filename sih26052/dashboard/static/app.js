/**
 * app.js — SIH-26052 Minimalist Live Hardware Audio Studio Client
 * Real-time WebSocket telemetry, 60fps spectrum analyzer, live audio streaming.
 */

// ── State ──────────────────────────────────────────────────────────────────
const state = {
    ws: null,
    wsConnected: false,
    isPipelineRunning: false,
    isEnhanced: true,
    volume: 85,
    isListeningBrowser: false,
    browserAudioCtx: null,
    audioQueueTime: 0,
    
    // Waveform & Spectrum data buffers
    specIn: new Float32Array(64),
    specOut: new Float32Array(64),
    targetSpecIn: new Float32Array(64),
    targetSpecOut: new Float32Array(64),
    oscBuffer: new Float32Array(256),
    
    // Studio player state
    studioAudioCtx: null,
    studioRawBuffer: null,
    studioEnhBuffer: null,
    studioSource: null,
    studioIsPlaying: false,
    studioMode: 'enh', // 'enh' or 'raw'
    studioStartTime: 0,
    studioPauseOffset: 0,
    studioDuration: 0,
};

// ── DOM References ─────────────────────────────────────────────────────────
const dom = {
    // Top pills
    wsDot: document.getElementById('ws-dot'),
    wsStatusText: document.getElementById('ws-status-text'),
    valCpuTemp: document.getElementById('val-cpu-temp'),

    // Hero controls
    btnPipelineToggle: document.getElementById('btn-pipeline-toggle'),
    pipelineIndicator: document.getElementById('pipeline-indicator'),
    pipelineBtnText: document.getElementById('pipeline-btn-text'),
    pipelineStatusText: document.getElementById('pipeline-status-text'),
    btnModeToggle: document.getElementById('btn-mode-toggle'),
    modeIcon: document.getElementById('mode-icon'),
    modeText: document.getElementById('mode-text'),
    volumeSlider: document.getElementById('volume-slider'),
    volumeValDisplay: document.getElementById('volume-val-display'),
    btnBrowserAudio: document.getElementById('btn-browser-audio'),
    browserAudioText: document.getElementById('browser-audio-text'),

    // Metrics
    metricLatency: document.getElementById('metric-latency'),
    metricRtf: document.getElementById('metric-rtf'),
    metricSnr: document.getElementById('metric-snr'),
    metricXruns: document.getElementById('metric-xruns'),
    xrunBadge: document.getElementById('xrun-badge'),

    // Visualizer Canvases
    canvasSpectrum: document.getElementById('canvas-spectrum'),
    canvasOscilloscope: document.getElementById('canvas-oscilloscope'),

    // Virtual LEDs
    vledSys: document.getElementById('vled-sys'),
    vledSysState: document.getElementById('vled-sys-state'),
    vledMode: document.getElementById('vled-mode'),
    vledModeState: document.getElementById('vled-mode-state'),
    vledAct: document.getElementById('vled-act'),
    vledActState: document.getElementById('vled-act-state'),
    btnTestLeds: document.getElementById('btn-test-leds'),

    // Studio section
    studioCollapseTrigger: document.getElementById('studio-collapse-trigger'),
    studioCollapseIcon: document.getElementById('studio-collapse-icon'),
    studioBody: document.getElementById('studio-body'),
    fileInput: document.getElementById('file-input'),
    btnBrowseFile: document.getElementById('btn-browse-file'),
    fileChosenName: document.getElementById('file-chosen-name'),
    presetDropdown: document.getElementById('preset-dropdown'),
    btnLoadPreset: document.getElementById('btn-load-preset'),
    btnEnhanceFile: document.getElementById('btn-enhance-file'),
    enhanceBtnText: document.getElementById('enhance-btn-text'),
    studioPlayer: document.getElementById('studio-player'),
    btnStudioPlay: document.getElementById('btn-studio-play'),
    abBtnEnh: document.getElementById('ab-btn-enh'),
    abBtnRaw: document.getElementById('ab-btn-raw'),
    studioTimeDisplay: document.getElementById('studio-time-display'),
    stProcTime: document.getElementById('st-proc-time'),
    stRtf: document.getElementById('st-rtf'),
    stSnr: document.getElementById('st-snr'),
    piHeadphoneStatus: document.getElementById('pi-headphone-status'),
    btnPiPlayEnh: document.getElementById('btn-pi-play-enh'),
    btnPiPlayRaw: document.getElementById('btn-pi-play-raw'),
    btnPiStopPlayback: document.getElementById('btn-pi-stop-playback'),
};

// ── Utility: Safe Base64 to ArrayBuffer (No fetch CORS issues) ─────────────
function dataUriToArrayBuffer(dataUri) {
    const base64Index = dataUri.indexOf(',');
    const base64 = base64Index >= 0 ? dataUri.slice(base64Index + 1) : dataUri;
    const binaryStr = window.atob(base64);
    const len = binaryStr.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
    }
    return bytes.buffer;
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// ── WebSocket Telemetry & Live Audio Stream ────────────────────────────────
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/hardware`;

    try {
        state.ws = new WebSocket(wsUrl);

        state.ws.onopen = () => {
            state.wsConnected = true;
            dom.wsDot.className = 'dot-indicator online';
            dom.wsStatusText.textContent = 'Live Telemetry';
        };

        state.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleTelemetryUpdate(data);
            } catch (err) {
                console.debug('WS parse error:', err);
            }
        };

        state.ws.onclose = () => {
            state.wsConnected = false;
            dom.wsDot.className = 'dot-indicator';
            dom.wsStatusText.textContent = 'Reconnecting...';
            setTimeout(initWebSocket, 2000);
        };

        state.ws.onerror = () => {
            state.ws.close();
        };
    } catch (e) {
        setTimeout(initWebSocket, 3000);
    }
}

function handleTelemetryUpdate(data) {
    // 1. Update pipeline running state
    if (typeof data.is_running === 'boolean' && data.is_running !== state.isPipelineRunning) {
        updatePipelineUI(data.is_running);
    }

    // 2. Update ANC mode
    if (typeof data.enhanced === 'boolean' && data.enhanced !== state.isEnhanced) {
        updateModeUI(data.enhanced);
    }

    // 3. Update Metrics
    if (data.processing_time_ms !== undefined) {
        dom.metricLatency.textContent = data.processing_time_ms.toFixed(1);
        const rtf = data.processing_time_ms / 16.0;
        dom.metricRtf.textContent = rtf.toFixed(2);
    }
    if (data.snr_gain !== undefined) {
        dom.metricSnr.textContent = `+${data.snr_gain.toFixed(1)}`;
    }
    if (data.xruns !== undefined) {
        dom.metricXruns.textContent = data.xruns;
        dom.xrunBadge.textContent = data.xruns === 0 ? 'Optimal' : `${data.xruns} Dropouts`;
        dom.xrunBadge.className = data.xruns === 0 ? 'metric-badge text-emerald' : 'metric-badge text-rose';
    }
    if (data.cpu_temp !== undefined) {
        dom.valCpuTemp.textContent = `${data.cpu_temp}°C`;
    }

    // 4. Update Virtual LEDs
    const isSys = Boolean(data.is_running);
    const isMode = Boolean(data.enhanced && data.is_running);
    const isAct = Boolean(data.is_active && data.is_running);

    dom.vledSys.classList.toggle('active', isSys);
    dom.vledSysState.textContent = isSys ? 'ACTIVE' : 'OFF';
    dom.vledSysState.className = isSys ? 'led-live-state text-emerald' : 'led-live-state text-muted';

    dom.vledMode.classList.toggle('active', isMode);
    dom.vledModeState.textContent = isMode ? 'FILTERING ON' : (isSys ? 'PASSTHROUGH' : 'OFF');
    dom.vledModeState.className = isMode ? 'led-live-state text-accent' : 'led-live-state text-muted';

    dom.vledAct.classList.toggle('active', isAct);
    dom.vledActState.textContent = isAct ? 'VOICE DETECTED' : 'QUIET';
    dom.vledActState.className = isAct ? 'led-live-state text-amber' : 'led-live-state text-muted';

    // 5. Update Frequency Spectrum Targets
    if (Array.isArray(data.spec_in)) {
        for (let i = 0; i < Math.min(64, data.spec_in.length); i++) {
            state.targetSpecIn[i] = data.spec_in[i];
        }
    }
    if (Array.isArray(data.spec_out)) {
        for (let i = 0; i < Math.min(64, data.spec_out.length); i++) {
            state.targetSpecOut[i] = data.spec_out[i];
        }
    }

    // 6. Push audio chunk to live oscilloscope & browser audio player
    if (Array.isArray(data.audio_chunk) && data.audio_chunk.length > 0) {
        // Shift oscilloscope buffer
        state.oscBuffer.copyWithin(0, data.audio_chunk.length);
        state.oscBuffer.set(data.audio_chunk, state.oscBuffer.length - data.audio_chunk.length);

        // Play in browser if enabled
        if (state.isListeningBrowser && state.browserAudioCtx) {
            playAudioChunkInBrowser(data.audio_chunk);
        }
    }
}

// ── Live Browser Audio Stream Player ───────────────────────────────────────
function playAudioChunkInBrowser(samples) {
    const ctx = state.browserAudioCtx;
    if (!ctx || ctx.state === 'suspended') {
        if (ctx) ctx.resume();
        return;
    }

    const buffer = ctx.createBuffer(1, samples.length, 16000);
    const channelData = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) {
        channelData[i] = samples[i];
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const currentTime = ctx.currentTime;
    if (state.audioQueueTime < currentTime) {
        state.audioQueueTime = currentTime + 0.02;
    }

    source.start(state.audioQueueTime);
    state.audioQueueTime += buffer.duration;
}

dom.btnBrowserAudio.addEventListener('click', () => {
    state.isListeningBrowser = !state.isListeningBrowser;
    if (state.isListeningBrowser) {
        if (!state.browserAudioCtx) {
            state.browserAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        }
        if (state.browserAudioCtx.state === 'suspended') {
            state.browserAudioCtx.resume();
        }
        dom.browserAudioText.textContent = 'Stop Listening in Browser';
        dom.btnBrowserAudio.classList.add('running');
    } else {
        dom.browserAudioText.textContent = 'Listen Live in Browser';
        dom.btnBrowserAudio.classList.remove('running');
    }
});

// ── Hardware Controls ──────────────────────────────────────────────────────
function updatePipelineUI(running) {
    state.isPipelineRunning = running;
    if (running) {
        dom.btnPipelineToggle.classList.add('running');
        dom.pipelineBtnText.textContent = 'Stop Hardware Stream';
        dom.pipelineStatusText.textContent = 'Hardware Active • Streaming 48kHz I2S ➔ 16kHz GTCRN';
    } else {
        dom.btnPipelineToggle.classList.remove('running');
        dom.pipelineBtnText.textContent = 'Start Hardware Stream';
        dom.pipelineStatusText.textContent = 'Ready to stream (48 kHz I2S ➔ 16 kHz GTCRN)';
    }
}

function updateModeUI(enhanced) {
    state.isEnhanced = enhanced;
    if (enhanced) {
        dom.btnModeToggle.className = 'btn btn-mode active';
        dom.modeIcon.textContent = '🛡️';
        dom.modeText.textContent = 'ANC Filtering: ACTIVE';
    } else {
        dom.btnModeToggle.className = 'btn btn-mode bypass';
        dom.modeIcon.textContent = '⚠️';
        dom.modeText.textContent = 'Raw Passthrough: BYPASS';
    }
}

dom.btnPipelineToggle.addEventListener('click', async () => {
    try {
        if (!state.isPipelineRunning) {
            dom.pipelineBtnText.textContent = 'Starting...';
            const res = await fetch('/api/hardware/start', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'started' || data.status === 'already_running') {
                updatePipelineUI(true);
            }
        } else {
            dom.pipelineBtnText.textContent = 'Stopping...';
            const res = await fetch('/api/hardware/stop', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'stopped' || data.status === 'not_running') {
                updatePipelineUI(false);
            }
        }
    } catch (err) {
        alert(`Hardware control error: ${err.message}`);
        updatePipelineUI(state.isPipelineRunning);
    }
});

dom.btnModeToggle.addEventListener('click', async () => {
    try {
        const res = await fetch('/api/hardware/toggle-mode', { method: 'POST' });
        const data = await res.json();
        updateModeUI(data.enhanced);
    } catch (err) {
        console.error('Mode toggle error:', err);
    }
});

// Headphone Volume Slider
let volumeDebounce = null;
dom.volumeSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    dom.volumeValDisplay.textContent = `${val}%`;
    state.volume = val;

    clearTimeout(volumeDebounce);
    volumeDebounce = setTimeout(async () => {
        const formData = new FormData();
        formData.append('volume', val);
        try {
            await fetch('/api/hardware/volume', { method: 'POST', body: formData });
        } catch (e) {
            console.error('Volume adjust error:', e);
        }
    }, 100);
});

// Test Physical LEDs Button
dom.btnTestLeds.addEventListener('click', async () => {
    try {
        dom.btnTestLeds.disabled = true;
        await fetch('/api/hardware/test-leds', { method: 'POST' });
        // Flash virtual LEDs on UI as feedback
        dom.vledSys.classList.add('active');
        dom.vledMode.classList.add('active');
        dom.vledAct.classList.add('active');
        setTimeout(() => {
            dom.btnTestLeds.disabled = false;
        }, 2100);
    } catch (err) {
        dom.btnTestLeds.disabled = false;
    }
});

// ── 60fps Smooth Canvas Visualizers ────────────────────────────────────────
function renderLoop() {
    drawSpectrumCanvas();
    drawOscilloscopeCanvas();
    requestAnimationFrame(renderLoop);
}

function drawSpectrumCanvas() {
    const canvas = dom.canvasSpectrum;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    const numBars = 64;
    const barWidth = (w / numBars) - 2;

    for (let i = 0; i < numBars; i++) {
        // Smooth lerp towards target magnitudes
        state.specIn[i] += (state.targetSpecIn[i] - state.specIn[i]) * 0.25;
        state.specOut[i] += (state.targetSpecOut[i] - state.specOut[i]) * 0.25;

        const x = i * (barWidth + 2);

        // Normalize bar height (clamp to height - 20)
        const inH = Math.min(h - 15, Math.max(2, state.specIn[i] * (h * 0.7)));
        const outH = Math.min(h - 15, Math.max(2, state.specOut[i] * (h * 0.7)));

        // 1. Draw Raw Input Bar (Subtle slate grey / amber)
        ctx.fillStyle = 'rgba(100, 116, 139, 0.4)';
        ctx.fillRect(x, h - inH, barWidth, inH);

        // 2. Draw Cleaned Output Bar (Vibrant Emerald)
        ctx.fillStyle = '#10b981';
        ctx.fillRect(x, h - outH, barWidth, outH);
    }

    // Grid baseline
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.beginPath();
    ctx.moveTo(0, h - 1);
    ctx.lineTo(w, h - 1);
    ctx.stroke();
}

function drawOscilloscopeCanvas() {
    const canvas = dom.canvasOscilloscope;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const midY = h / 2;

    ctx.clearRect(0, 0, w, h);

    // Center baseline
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();

    // Waveform line
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const buf = state.oscBuffer;
    const sliceWidth = w / buf.length;
    let x = 0;

    for (let i = 0; i < buf.length; i++) {
        const v = buf[i];
        const y = midY + (v * (h * 0.45));

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
        x += sliceWidth;
    }

    ctx.stroke();
}

// ── Studio Section (Collapsible & Robust) ──────────────────────────────────
let selectedStudioFile = null;
let selectedStudioPresetId = null;

dom.studioCollapseTrigger.addEventListener('click', () => {
    const isHidden = dom.studioBody.style.display === 'none';
    dom.studioBody.style.display = isHidden ? 'block' : 'none';
    dom.studioCollapseIcon.textContent = isHidden ? '▾' : '▸';
});

dom.btnBrowseFile.addEventListener('click', () => dom.fileInput.click());

dom.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
        selectedStudioFile = e.target.files[0];
        selectedStudioPresetId = null;
        dom.fileChosenName.textContent = selectedStudioFile.name;
        dom.btnEnhanceFile.disabled = false;
    }
});

// Load Presets
async function loadPresets() {
    try {
        const res = await fetch('/api/presets');
        const data = await res.json();
        if (data.presets && data.presets.length > 0) {
            dom.presetDropdown.innerHTML = data.presets.map(p => 
                `<option value="${p.id}">${p.title}</option>`
            ).join('');
        } else {
            dom.presetDropdown.innerHTML = '<option value="">No sample presets available</option>';
        }
    } catch (e) {
        dom.presetDropdown.innerHTML = '<option value="">Error loading samples</option>';
    }
}

dom.btnLoadPreset.addEventListener('click', async () => {
    const presetId = dom.presetDropdown.value;
    if (!presetId) return;

    try {
        dom.btnLoadPreset.disabled = true;
        dom.btnLoadPreset.textContent = 'Loading...';
        const res = await fetch(`/api/preset/${presetId}`);
        if (!res.ok) throw new Error('Preset not found');
        const data = await res.json();

        // Convert base64 data to blob
        const audioBuf = dataUriToArrayBuffer(data.noisy_audio_url);
        selectedStudioFile = new Blob([audioBuf], { type: 'audio/wav' });
        selectedStudioPresetId = presetId;
        dom.fileChosenName.textContent = `${presetId}.wav (Loaded)`;
        dom.btnEnhanceFile.disabled = false;

        // If preset already contains enhanced audio, setup the player right away!
        if (data.enhanced_audio_url) {
            await setupStudioPlayerFromData(data);
        }
    } catch (err) {
        alert('Failed to load sample: ' + err.message);
    } finally {
        dom.btnLoadPreset.disabled = false;
        dom.btnLoadPreset.textContent = 'Load Sample';
    }
});

async function setupStudioPlayerFromData(data) {
    try {
        if (!state.studioAudioCtx) {
            state.studioAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }

        const rawUrl = data.raw_audio_url || data.noisy_audio_url;
        const enhUrl = data.enhanced_audio_url;

        if (rawUrl) {
            const rawBuf = dataUriToArrayBuffer(rawUrl);
            state.studioRawBuffer = await state.studioAudioCtx.decodeAudioData(rawBuf);
        }
        if (enhUrl) {
            const enhBuf = dataUriToArrayBuffer(enhUrl);
            state.studioEnhBuffer = await state.studioAudioCtx.decodeAudioData(enhBuf);
        }

        const activeBuf = state.studioEnhBuffer || state.studioRawBuffer;
        state.studioDuration = activeBuf ? activeBuf.duration : 0;

        if (data.processing_time_ms !== undefined) {
            dom.stProcTime.textContent = data.processing_time_ms.toFixed(0);
        }
        if (data.rtf !== undefined) {
            dom.stRtf.textContent = `${data.rtf.toFixed(2)}x`;
        }
        if (data.metrics && data.metrics.snr_improvement_db !== undefined) {
            dom.stSnr.textContent = `+${data.metrics.snr_improvement_db.toFixed(1)}`;
        }

        dom.studioPlayer.style.display = 'flex';
        dom.studioTimeDisplay.textContent = `00:00 / ${formatTime(state.studioDuration)}`;
        dom.piHeadphoneStatus.textContent = `Sample Ready (${state.studioDuration.toFixed(1)}s) • USB DAC 3.5mm`;
        dom.piHeadphoneStatus.className = 'headphone-status-badge text-emerald';

        state.studioMode = 'enh';
        dom.abBtnEnh.classList.add('active');
        dom.abBtnRaw.classList.remove('active');
    } catch (err) {
        console.error('Audio decode error:', err);
    }
}

dom.btnEnhanceFile.addEventListener('click', async () => {
    if (!selectedStudioFile && !selectedStudioPresetId) return;

    try {
        dom.btnEnhanceFile.disabled = true;
        dom.enhanceBtnText.textContent = '⚡ Processing on Pi...';

        const formData = new FormData();
        if (selectedStudioFile && !selectedStudioPresetId) {
            formData.append('file', selectedStudioFile, selectedStudioFile.name || 'input.wav');
        } else if (selectedStudioPresetId) {
            formData.append('preset_id', selectedStudioPresetId);
        }
        formData.append('engine', 'onnx_stream_int8');

        const res = await fetch('/api/enhance', { method: 'POST', body: formData });
        if (!res.ok) {
            const errJson = await res.json();
            throw new Error(errJson.detail || 'Enhancement failed');
        }

        const data = await res.json();
        await setupStudioPlayerFromData(data);

    } catch (err) {
        alert('Enhancement error: ' + err.message);
    } finally {
        dom.btnEnhanceFile.disabled = false;
        dom.enhanceBtnText.textContent = '⚡ Process on Pi';
    }
});

// ── Physical Pi Headphone Output Handlers ───────────────────────────────────
dom.btnPiPlayEnh.addEventListener('click', async () => {
    try {
        dom.btnPiPlayEnh.disabled = true;
        dom.piHeadphoneStatus.textContent = '🔊 Outputting Enhanced Audio to Pi Headphones...';
        dom.piHeadphoneStatus.className = 'headphone-status-badge text-accent';

        const formData = new FormData();
        formData.append('mode', 'enhanced');
        const res = await fetch('/api/hardware/play-sample', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Playback failed');

        setTimeout(() => {
            dom.btnPiPlayEnh.disabled = false;
            dom.piHeadphoneStatus.textContent = `Playing Cleaned Audio on Headphones (${data.duration}s)...`;
        }, 500);
    } catch (err) {
        alert('Pi Headphone Output Error: ' + err.message);
        dom.piHeadphoneStatus.textContent = 'Error: ' + err.message;
        dom.btnPiPlayEnh.disabled = false;
    }
});

dom.btnPiPlayRaw.addEventListener('click', async () => {
    try {
        dom.btnPiPlayRaw.disabled = true;
        dom.piHeadphoneStatus.textContent = '🔊 Outputting Raw Noisy Audio to Pi Headphones...';
        dom.piHeadphoneStatus.className = 'headphone-status-badge';

        const formData = new FormData();
        formData.append('mode', 'raw');
        const res = await fetch('/api/hardware/play-sample', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Playback failed');

        setTimeout(() => {
            dom.btnPiPlayRaw.disabled = false;
            dom.piHeadphoneStatus.textContent = `Playing Raw Audio on Headphones (${data.duration}s)...`;
        }, 500);
    } catch (err) {
        alert('Pi Headphone Output Error: ' + err.message);
        dom.piHeadphoneStatus.textContent = 'Error: ' + err.message;
        dom.btnPiPlayRaw.disabled = false;
    }
});

dom.btnPiStopPlayback.addEventListener('click', async () => {
    try {
        await fetch('/api/hardware/stop-playback', { method: 'POST' });
        dom.piHeadphoneStatus.textContent = 'Headphone output stopped';
        dom.btnPiPlayEnh.disabled = false;
        dom.btnPiPlayRaw.disabled = false;
    } catch (err) {
        console.warn('Stop error:', err);
    }
});

// ── In-Browser Studio Playback ──────────────────────────────────────────────
dom.btnStudioPlay.addEventListener('click', () => {
    if (state.studioIsPlaying) {
        stopStudioPlayback();
    } else {
        startStudioPlayback();
    }
});

function startStudioPlayback() {
    if (!state.studioAudioCtx || (!state.studioEnhBuffer && !state.studioRawBuffer)) return;

    if (state.studioAudioCtx.state === 'suspended') {
        state.studioAudioCtx.resume();
    }

    const activeBuffer = state.studioMode === 'enh' ? state.studioEnhBuffer : state.studioRawBuffer;
    if (!activeBuffer) return;

    state.studioSource = state.studioAudioCtx.createBufferSource();
    state.studioSource.buffer = activeBuffer;
    state.studioSource.connect(state.studioAudioCtx.destination);

    state.studioSource.onended = () => {
        state.studioIsPlaying = false;
        dom.btnStudioPlay.textContent = '▶ Play';
        state.studioPauseOffset = 0;
    };

    state.studioStartTime = state.studioAudioCtx.currentTime - state.studioPauseOffset;
    state.studioSource.start(0, state.studioPauseOffset);

    state.studioIsPlaying = true;
    dom.btnStudioPlay.textContent = '⏸ Pause';
}

function stopStudioPlayback() {
    if (state.studioSource) {
        try { state.studioSource.stop(); } catch (e) {}
    }
    state.studioIsPlaying = false;
    dom.btnStudioPlay.textContent = '▶ Play';
}

dom.abBtnEnh.addEventListener('click', () => {
    if (state.studioMode === 'enh') return;
    state.studioMode = 'enh';
    dom.abBtnEnh.classList.add('active');
    dom.abBtnRaw.classList.remove('active');
    if (state.studioIsPlaying) {
        state.studioPauseOffset = state.studioAudioCtx.currentTime - state.studioStartTime;
        stopStudioPlayback();
        startStudioPlayback();
    }
});

dom.abBtnRaw.addEventListener('click', () => {
    if (state.studioMode === 'raw') return;
    state.studioMode = 'raw';
    dom.abBtnRaw.classList.add('active');
    dom.abBtnEnh.classList.remove('active');
    if (state.studioIsPlaying) {
        state.studioPauseOffset = state.studioAudioCtx.currentTime - state.studioStartTime;
        stopStudioPlayback();
        startStudioPlayback();
    }
});

// ── Startup ────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    loadPresets();
    requestAnimationFrame(renderLoop);

    // Initial status fetch
    fetch('/api/hardware/status')
        .then(r => r.json())
        .then(d => {
            updatePipelineUI(d.is_running);
            updateModeUI(d.enhanced);
            if (d.volume) {
                dom.volumeSlider.value = d.volume;
                dom.volumeValDisplay.textContent = `${d.volume}%`;
            }
        })
        .catch(() => {});
});
