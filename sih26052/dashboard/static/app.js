/**
 * app.js — GTCRN Speech Enhancement Studio Controller
 *
 * Handles:
 *   - Audio Studio: File Upload, Mic Recording, Presets
 *   - Model Enhancement API & Performance / SNR Metrics
 *   - Seamless Sample-Accurate A/B Audio Player with Synchronized Scrubber
 *   - High-Resolution Spectrogram & Waveform Canvas Rendering
 *   - Live WebSocket Microphone Streaming
 *   - Edge Hardware Telemetry Monitor
 */

// ── Application State ───────────────────────────────────────────────────────
const state = {
    activeTab: 'studio',
    inputMode: 'upload',
    selectedEngine: 'pytorch',
    currentPresetId: null,

    // Audio blobs and buffers
    rawAudioBlob: null,
    rawAudioBuffer: null,
    enhAudioBlob: null,
    enhAudioBuffer: null,

    // Web Audio Player
    audioCtx: null,
    rawSourceNode: null,
    enhSourceNode: null,
    rawGainNode: null,
    enhGainNode: null,
    masterGainNode: null,
    activeSource: 'raw', // 'raw' or 'enh'
    isPlaying: false,
    isLooping: false,
    playbackStartTime: 0,
    playbackPauseOffset: 0,
    audioDuration: 0,
    animFrameId: null,

    // Microphone Recording
    mediaRecorder: null,
    recordedChunks: [],
    recStartTime: null,
    recTimerInterval: null,
    recStream: null,
    recAnalyser: null,
    recVuAnimId: null,

    // Live Streaming Mode
    streamWs: null,
    streamAudioCtx: null,
    streamWorklet: null,
    isStreaming: false,
    streamFrameCount: 0,

    // Edge Telemetry WS
    telemetryWs: null,

    // Spectrogram data matrices
    spectrograms: {
        raw: null,
        enh: null,
        diff: null,
    }
};

// ── Colormaps for Spectrograms ──────────────────────────────────────────────
// Viridis-inspired colormap (purple -> teal -> yellow) for speech spectrograms
const COLORMAP_VIRIDIS = [
    [10, 8, 28],
    [38, 16, 68],
    [64, 43, 116],
    [53, 94, 142],
    [33, 144, 140],
    [43, 175, 120],
    [93, 201, 81],
    [170, 220, 50],
    [253, 231, 37],
];

// Inferno-inspired colormap (black -> red -> orange -> yellow) for difference
const COLORMAP_INFERNO = [
    [0, 0, 4],
    [40, 11, 84],
    [101, 21, 110],
    [159, 42, 99],
    [212, 72, 66],
    [245, 125, 21],
    [250, 193, 39],
    [252, 255, 164],
];

// ── DOM References ──────────────────────────────────────────────────────────
const dom = {
    // Navigation
    tabBtns: document.querySelectorAll('.nav-tab'),
    tabPanes: {
        studio: document.getElementById('pane-studio'),
        stream: document.getElementById('pane-stream'),
        telemetry: document.getElementById('pane-telemetry'),
    },
    activeEngineLabel: document.getElementById('active-engine-label'),

    // Input Mode
    modeBtnUpload: document.getElementById('mode-btn-upload'),
    modeBtnRecord: document.getElementById('mode-btn-record'),
    modeBtnPreset: document.getElementById('mode-btn-preset'),
    subpanelUpload: document.getElementById('subpanel-upload'),
    subpanelRecord: document.getElementById('subpanel-record'),
    subpanelPreset: document.getElementById('subpanel-preset'),

    // Upload
    dropzone: document.getElementById('audio-dropzone'),
    fileInput: document.getElementById('audio-file-input'),
    browseBtn: document.getElementById('browse-btn'),
    fileLoadedInfo: document.getElementById('file-loaded-info'),
    loadedFileName: document.getElementById('loaded-file-name'),
    loadedFileMeta: document.getElementById('loaded-file-meta'),
    clearFileBtn: document.getElementById('clear-file-btn'),

    // Record
    recordToggleBtn: document.getElementById('record-toggle-btn'),
    recordStatusText: document.getElementById('record-status-text'),
    recordTimer: document.getElementById('record-timer'),
    micVuBar: document.getElementById('mic-vu-bar'),

    // Presets
    presetSelect: document.getElementById('preset-select'),
    loadPresetBtn: document.getElementById('load-preset-btn'),

    // Engine & Action
    engineSelect: document.getElementById('engine-select'),
    enhanceBtn: document.getElementById('enhance-btn'),
    enhanceBtnText: document.getElementById('enhance-btn-text'),
    enhanceSpinner: document.getElementById('enhance-spinner'),

    // Player
    abBtnRaw: document.getElementById('ab-btn-raw'),
    abBtnEnh: document.getElementById('ab-btn-enh'),
    downloadBtn: document.getElementById('download-btn'),
    waveformCanvas: document.getElementById('waveform-canvas'),
    waveformPlayhead: document.getElementById('waveform-playhead'),
    waveformEmpty: document.getElementById('waveform-empty'),
    waveformContainer: document.getElementById('waveform-container'),
    btnPlay: document.getElementById('btn-play'),
    btnRestart: document.getElementById('btn-restart'),
    btnLoop: document.getElementById('btn-loop'),
    currentTime: document.getElementById('current-time'),
    totalTime: document.getElementById('total-time'),
    volumeSlider: document.getElementById('volume-slider'),

    // Metrics
    valRtf: document.getElementById('val-rtf'),
    subRtf: document.getElementById('sub-rtf'),
    valProcTime: document.getElementById('val-proc-time'),
    valSnrGain: document.getElementById('val-snr-gain'),
    valNoiseRed: document.getElementById('val-noise-red'),
    valPesq: document.getElementById('val-pesq'),

    // Spectrograms
    specTabs: document.querySelectorAll('.spec-tab'),
    specSideBySide: document.getElementById('spec-side-by-side'),
    specDiffWrapper: document.getElementById('spec-diff-wrapper'),
    canvasSpecRaw: document.getElementById('canvas-spec-raw'),
    canvasSpecEnh: document.getElementById('canvas-spec-enh'),
    canvasSpecDiff: document.getElementById('canvas-spec-diff'),
    cursorSpecRaw: document.getElementById('cursor-spec-raw'),
    cursorSpecEnh: document.getElementById('cursor-spec-enh'),

    // Live Stream
    streamToggleBtn: document.getElementById('stream-toggle-btn'),
    streamToggleText: document.getElementById('stream-toggle-text'),
    liveStreamStatus: document.getElementById('live-stream-status'),
    liveStreamFrames: document.getElementById('live-stream-frames'),
    liveStreamLatency: document.getElementById('live-stream-latency'),
    streamSpeakerToggle: document.getElementById('stream-speaker-toggle'),
    canvasStreamSpec: document.getElementById('canvas-stream-spec'),

    // Edge Telemetry
    edgeLatencyVal: document.getElementById('edge-latency-val'),
    edgeXrunVal: document.getElementById('edge-xrun-val'),
    edgeModeVal: document.getElementById('edge-mode-val'),
    edgeFrameVal: document.getElementById('edge-frame-val'),
    computeGauge: document.getElementById('compute-gauge'),
    computeVal: document.getElementById('compute-val'),
};

// ── Audio Context Initialization ───────────────────────────────────────────

function getAudioContext() {
    if (!state.audioCtx) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        state.audioCtx = new AudioContext({ sampleRate: 16000 });
    }
    if (state.audioCtx.state === 'suspended') {
        state.audioCtx.resume();
    }
    return state.audioCtx;
}

// ── Pure-JS PCM 16-bit WAV Encoder & Decoder Utilities ─────────────────────

function audioBufferToWav(buffer) {
    const numChannels = 1;
    const sampleRate = buffer.sampleRate || 16000;
    const format = 1; // PCM
    const bitDepth = 16;
    
    // Mix to mono if multiple channels
    let channelData;
    if (buffer.numberOfChannels > 1) {
        const c0 = buffer.getChannelData(0);
        const c1 = buffer.getChannelData(1);
        channelData = new Float32Array(c0.length);
        for (let i = 0; i < c0.length; i++) {
            channelData[i] = (c0[i] + c1[i]) * 0.5;
        }
    } else {
        channelData = buffer.getChannelData(0);
    }
    
    const dataLength = channelData.length * 2;
    const bufferLength = 44 + dataLength;
    const arrayBuffer = new ArrayBuffer(bufferLength);
    const view = new DataView(arrayBuffer);
    
    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }
    
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, bitDepth, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataLength, true);
    
    let offset = 44;
    for (let i = 0; i < channelData.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, channelData[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    
    return new Blob([view], { type: 'audio/wav' });
}

function dataUriToBlob(dataUri) {
    try {
        const parts = dataUri.split(',');
        const mime = parts[0].match(/:(.*?);/)[1] || 'audio/wav';
        const binary = atob(parts[1]);
        const len = binary.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return new Blob([bytes], { type: mime });
    } catch (e) {
        return null;
    }
}


// ── Navigation Tabs ────────────────────────────────────────────────────────

dom.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        dom.tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        Object.keys(dom.tabPanes).forEach(paneKey => {
            if (paneKey === tab) {
                dom.tabPanes[paneKey].style.display = 'flex';
                dom.tabPanes[paneKey].classList.add('active');
            } else {
                dom.tabPanes[paneKey].style.display = 'none';
                dom.tabPanes[paneKey].classList.remove('active');
            }
        });

        state.activeTab = tab;
        if (tab === 'studio') {
            redrawWaveform();
            renderSpectrograms();
        }
    });
});

// ── Input Mode Tabs ────────────────────────────────────────────────────────

function setInputMode(mode) {
    state.inputMode = mode;
    [dom.modeBtnUpload, dom.modeBtnRecord, dom.modeBtnPreset].forEach(b => b.classList.remove('active'));
    [dom.subpanelUpload, dom.subpanelRecord, dom.subpanelPreset].forEach(p => p.style.display = 'none');

    if (mode === 'upload') {
        dom.modeBtnUpload.classList.add('active');
        dom.subpanelUpload.style.display = 'block';
    } else if (mode === 'record') {
        dom.modeBtnRecord.classList.add('active');
        dom.subpanelRecord.style.display = 'block';
    } else if (mode === 'preset') {
        dom.modeBtnPreset.classList.add('active');
        dom.subpanelPreset.style.display = 'block';
    }
}

dom.modeBtnUpload.addEventListener('click', () => setInputMode('upload'));
dom.modeBtnRecord.addEventListener('click', () => setInputMode('record'));
dom.modeBtnPreset.addEventListener('click', () => setInputMode('preset'));

// ── File Upload Handling ───────────────────────────────────────────────────

dom.browseBtn.addEventListener('click', () => dom.fileInput.click());
dom.dropzone.addEventListener('click', (e) => {
    if (e.target !== dom.clearFileBtn) {
        dom.fileInput.click();
    }
});

dom.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
        handleSelectedAudioFile(e.target.files[0]);
    }
});

dom.dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dom.dropzone.classList.add('dragover');
});

dom.dropzone.addEventListener('dragleave', () => {
    dom.dropzone.classList.remove('dragover');
});

dom.dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dom.dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleSelectedAudioFile(e.dataTransfer.files[0]);
    }
});

dom.clearFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetLoadedAudio();
});

async function handleSelectedAudioFile(file) {
    state.currentPresetId = null;

    dom.loadedFileName.textContent = file.name;
    const sizeKB = (file.size / 1024).toFixed(1);
    dom.loadedFileMeta.textContent = `${sizeKB} KB • Loading waveform...`;

    dom.dropzone.querySelector('.dropzone-content').style.display = 'none';
    dom.fileLoadedInfo.style.display = 'flex';

    await loadAudioData(file);
    if (state.rawAudioBuffer) {
        state.rawAudioBlob = audioBufferToWav(state.rawAudioBuffer);
    } else {
        state.rawAudioBlob = file;
    }
    dom.loadedFileMeta.textContent = `16.0 kHz • ${state.audioDuration.toFixed(1)}s • ${sizeKB} KB`;
    dom.enhanceBtn.disabled = false;
    dom.downloadBtn.disabled = false;
}

function resetLoadedAudio() {
    stopPlayback();
    state.rawAudioBlob = null;
    state.rawAudioBuffer = null;
    state.enhAudioBlob = null;
    state.enhAudioBuffer = null;
    state.audioDuration = 0;
    state.currentPresetId = null;

    dom.dropzone.querySelector('.dropzone-content').style.display = 'block';
    dom.fileLoadedInfo.style.display = 'none';
    dom.fileInput.value = '';
    dom.enhanceBtn.disabled = true;
    dom.btnPlay.disabled = true;
    dom.btnRestart.disabled = true;
    dom.downloadBtn.disabled = true;

    dom.waveformEmpty.style.display = 'flex';
    clearCanvas(dom.waveformCanvas);
    clearCanvas(dom.canvasSpecRaw);
    clearCanvas(dom.canvasSpecEnh);
    clearCanvas(dom.canvasSpecDiff);
    dom.currentTime.textContent = '0:00.0';
    dom.totalTime.textContent = '0:00.0';

    // Reset metrics
    dom.valRtf.textContent = '—';
    dom.valProcTime.textContent = '—';
    dom.valSnrGain.textContent = '—';
    dom.valNoiseRed.textContent = '—';
    dom.valPesq.textContent = '—';
}

// ── Decode and Load Audio ──────────────────────────────────────────────────

async function loadAudioData(blobOrUrl) {
    const ctx = getAudioContext();
    let arrayBuffer;

    if (blobOrUrl instanceof Blob) {
        arrayBuffer = await blobOrUrl.arrayBuffer();
    } else if (typeof blobOrUrl === 'string') {
        const response = await fetch(blobOrUrl);
        arrayBuffer = await response.arrayBuffer();
    }

    state.rawAudioBuffer = await ctx.decodeAudioData(arrayBuffer);
    state.audioDuration = state.rawAudioBuffer.duration;

    dom.totalTime.textContent = formatTime(state.audioDuration);
    dom.waveformEmpty.style.display = 'none';
    dom.btnPlay.disabled = false;
    dom.btnRestart.disabled = false;
    dom.downloadBtn.disabled = false;

    // Redraw waveform and initial preview
    redrawWaveform();
}

// ── Presets Loader ─────────────────────────────────────────────────────────

async function fetchPresets() {
    try {
        const res = await fetch('/api/presets');
        const data = await res.json();
        dom.presetSelect.innerHTML = '';

        if (data.presets && data.presets.length > 0) {
            data.presets.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.id;
                opt.textContent = `${p.title} ${p.has_clean ? '✓ (Clean Reference)' : ''}`;
                dom.presetSelect.appendChild(opt);
            });
        } else {
            dom.presetSelect.innerHTML = '<option value="">No presets found</option>';
        }
    } catch (e) {
        console.error('Failed to load presets:', e);
    }
}

dom.loadPresetBtn.addEventListener('click', async () => {
    const presetId = dom.presetSelect.value;
    if (!presetId) return;

    dom.loadPresetBtn.disabled = true;
    dom.loadPresetBtn.textContent = 'Loading...';

    try {
        const res = await fetch(`/api/preset/${presetId}`);
        const data = await res.json();

        state.currentPresetId = presetId;
        const noisyBlob = await (await fetch(data.noisy_audio_url)).blob();
        state.rawAudioBlob = noisyBlob;

        await loadAudioData(noisyBlob);
        dom.enhanceBtn.disabled = false;

        // Auto-enhance preset sample for instant gratification!
        enhanceAudio();
    } catch (e) {
        console.error('Failed to load preset audio:', e);
        alert('Could not load preset audio.');
    } finally {
        dom.loadPresetBtn.disabled = false;
        dom.loadPresetBtn.textContent = 'Load Sample';
    }
});

// ── Microphone Recording ───────────────────────────────────────────────────

dom.recordToggleBtn.addEventListener('click', toggleRecording);

async function toggleRecording() {
    if (state.mediaRecorder && state.mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        state.recStream = stream;

        const ctx = getAudioContext();
        const src = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        src.connect(analyser);
        state.recAnalyser = analyser;

        state.recordedChunks = [];
        state.mediaRecorder = new MediaRecorder(stream);

        state.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                state.recordedChunks.push(e.data);
            }
        };

        state.mediaRecorder.onstop = async () => {
            const rawWebmBlob = new Blob(state.recordedChunks, { type: 'audio/webm' });
            state.currentPresetId = null;

            await loadAudioData(rawWebmBlob);
            if (state.rawAudioBuffer) {
                state.rawAudioBlob = audioBufferToWav(state.rawAudioBuffer);
            } else {
                state.rawAudioBlob = rawWebmBlob;
            }
            dom.enhanceBtn.disabled = false;
            dom.downloadBtn.disabled = false;
            dom.recordStatusText.textContent = 'Recorded Audio Ready (PCM WAV)!';
        };

        state.mediaRecorder.start();
        dom.recordToggleBtn.classList.add('recording');
        dom.recordStatusText.textContent = 'Recording in progress...';
        state.recStartTime = Date.now();

        // Update timer
        state.recTimerInterval = setInterval(() => {
            const elapsed = (Date.now() - state.recStartTime) / 1000;
            dom.recordTimer.textContent = formatTime(elapsed);
            if (elapsed >= 30) {
                stopRecording();
            }
        }, 100);

        // Update VU meter
        updateVuMeter();
    } catch (e) {
        console.error('Microphone error:', e);
        alert('Microphone access denied or not available.');
    }
}

function stopRecording() {
    if (state.mediaRecorder && state.mediaRecorder.state === 'recording') {
        state.mediaRecorder.stop();
    }
    if (state.recStream) {
        state.recStream.getTracks().forEach(t => t.stop());
    }
    clearInterval(state.recTimerInterval);
    cancelAnimationFrame(state.recVuAnimId);
    dom.recordToggleBtn.classList.remove('recording');
    dom.micVuBar.style.width = '0%';
}

function updateVuMeter() {
    if (!state.recAnalyser) return;
    const data = new Uint8Array(state.recAnalyser.frequencyBinCount);
    state.recAnalyser.getByteFrequencyData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
        sum += data[i];
    }
    const avg = sum / data.length;
    const pct = Math.min(100, (avg / 128) * 100);
    dom.micVuBar.style.width = pct + '%';
    state.recVuAnimId = requestAnimationFrame(updateVuMeter);
}

// ── Model Enhancement Execution ───────────────────────────────────────────

dom.engineSelect.addEventListener('change', (e) => {
    state.selectedEngine = e.target.value;
    const selectedOpt = dom.engineSelect.options[dom.engineSelect.selectedIndex];
    dom.activeEngineLabel.textContent = selectedOpt.text.split(' (')[0];
});

dom.enhanceBtn.addEventListener('click', enhanceAudio);

async function enhanceAudio() {
    if (!state.rawAudioBlob) return;

    dom.enhanceBtn.disabled = true;
    dom.enhanceBtnText.textContent = 'Processing with GTCRN...';
    dom.enhanceSpinner.style.display = 'inline-block';

    const formData = new FormData();
    formData.append('file', state.rawAudioBlob, 'input.wav');
    formData.append('engine', state.selectedEngine);
    if (state.currentPresetId) {
        formData.append('preset_id', state.currentPresetId);
    }

    try {
        const t0 = performance.now();
        const res = await fetch('/api/enhance', {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Enhancement failed');
        }

        const data = await res.json();

        // Load enhanced audio into buffer via direct Blob decode
        const ctx = getAudioContext();
        let enhBlob = dataUriToBlob(data.enhanced_audio_url);
        if (!enhBlob) {
            enhBlob = await (await fetch(data.enhanced_audio_url)).blob();
        }
        state.enhAudioBlob = enhBlob;
        const enhBuf = await enhBlob.arrayBuffer();
        state.enhAudioBuffer = await ctx.decodeAudioData(enhBuf);

        // Update metrics
        dom.valRtf.textContent = `${data.rtf.toFixed(2)}x`;
        dom.subRtf.textContent = data.rtf < 0.20 ? '⚡ Ultra-Fast (<0.20x)' : 'Real-Time Capable';
        dom.valProcTime.textContent = `${data.processing_time_ms.toFixed(0)}`;
        dom.valSnrGain.textContent = `+${data.metrics.snr_improvement_db.toFixed(1)}`;
        dom.valNoiseRed.textContent = `-${data.metrics.noise_reduction_db.toFixed(1)}`;
        dom.valPesq.textContent = data.metrics.pesq ? `${data.metrics.pesq.toFixed(2)}` : '—';

        // Update spectrogram data
        state.spectrograms = data.spectrograms;
        renderSpectrograms();

        // Enable download and switch to enhanced mode
        dom.downloadBtn.disabled = false;
        dom.downloadBtn.innerHTML = '<span>⬇️</span> Download Filtered Audio';
        setActiveAudioSource('enh');

        // Play if not playing
        if (!state.isPlaying) {
            startPlayback();
        }
    } catch (e) {
        console.error('Enhancement error:', e);
        alert(`Enhancement error: ${e.message}`);
    } finally {
        dom.enhanceBtn.disabled = false;
        dom.enhanceBtnText.textContent = 'Enhance Audio';
        dom.enhanceSpinner.style.display = 'none';
    }
}

// Download Button
dom.downloadBtn.addEventListener('click', () => {
    const isEnh = state.activeSource === 'enh';
    const blobToDownload = (isEnh && state.enhAudioBlob) ? state.enhAudioBlob : (state.enhAudioBlob || state.rawAudioBlob);
    if (!blobToDownload) {
        alert('Please record or load an audio file first.');
        return;
    }
    const url = URL.createObjectURL(blobToDownload);
    const a = document.createElement('a');
    a.href = url;
    const prefix = (blobToDownload === state.enhAudioBlob) ? 'gtcrn_enhanced' : 'noisy_raw';
    a.download = `${prefix}_${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1500);
});

// ── Sample-Accurate A/B Audio Player ───────────────────────────────────────

function setupAudioNodes() {
    const ctx = getAudioContext();

    // Master Volume Gain
    if (!state.masterGainNode) {
        state.masterGainNode = ctx.createGain();
        state.masterGainNode.connect(ctx.destination);
    }
    state.masterGainNode.gain.setValueAtTime(parseFloat(dom.volumeSlider.value), ctx.currentTime);

    // Stop old sources if any
    if (state.rawSourceNode) {
        try { state.rawSourceNode.stop(); } catch (e) {}
        state.rawSourceNode.disconnect();
    }
    if (state.enhSourceNode) {
        try { state.enhSourceNode.stop(); } catch (e) {}
        state.enhSourceNode.disconnect();
    }

    // Create raw and enhanced source nodes
    state.rawSourceNode = ctx.createBufferSource();
    state.rawSourceNode.buffer = state.rawAudioBuffer;
    state.rawSourceNode.loop = state.isLooping;

    state.rawGainNode = ctx.createGain();
    state.rawSourceNode.connect(state.rawGainNode);
    state.rawGainNode.connect(state.masterGainNode);

    if (state.enhAudioBuffer) {
        state.enhSourceNode = ctx.createBufferSource();
        state.enhSourceNode.buffer = state.enhAudioBuffer;
        state.enhSourceNode.loop = state.isLooping;

        state.enhGainNode = ctx.createGain();
        state.enhSourceNode.connect(state.enhGainNode);
        state.enhGainNode.connect(state.masterGainNode);
    }

    // Set initial gains with immediate values
    applySourceGains(true);

    // Handle audio completion
    state.rawSourceNode.onended = () => {
        if (!state.isLooping && state.isPlaying) {
            stopPlayback();
            state.playbackPauseOffset = 0;
            updatePlayhead(0);
        }
    };
}

function applySourceGains(immediate = false) {
    if (!state.rawGainNode) return;
    const ctx = getAudioContext();
    const now = ctx.currentTime;
    const fadeDuration = immediate ? 0.001 : 0.025; // 25ms click-free crossfade!

    if (state.activeSource === 'raw' || !state.enhAudioBuffer) {
        state.rawGainNode.gain.setTargetAtTime(1.0, now, fadeDuration);
        if (state.enhGainNode) {
            state.enhGainNode.gain.setTargetAtTime(0.0, now, fadeDuration);
        }
    } else {
        state.rawGainNode.gain.setTargetAtTime(0.0, now, fadeDuration);
        if (state.enhGainNode) {
            state.enhGainNode.gain.setTargetAtTime(1.0, now, fadeDuration);
        }
    }
}

function setActiveAudioSource(source) {
    state.activeSource = source;
    if (source === 'raw') {
        dom.abBtnRaw.classList.add('active');
        dom.abBtnEnh.classList.remove('active');
        if (dom.downloadBtn && (state.rawAudioBlob || state.enhAudioBlob)) {
            dom.downloadBtn.innerHTML = '<span>⬇️</span> Download Noisy Audio';
        }
    } else {
        dom.abBtnEnh.classList.add('active');
        dom.abBtnRaw.classList.remove('active');
        if (dom.downloadBtn && state.enhAudioBlob) {
            dom.downloadBtn.innerHTML = '<span>⬇️</span> Download Filtered Audio';
        }
    }
    applySourceGains(false);
}

dom.abBtnRaw.addEventListener('click', () => setActiveAudioSource('raw'));
dom.abBtnEnh.addEventListener('click', () => {
    if (!state.enhAudioBuffer) {
        alert('Please click "Enhance Audio" first to generate the filtered sound!');
        return;
    }
    setActiveAudioSource('enh');
});

// Transport Controls
dom.btnPlay.addEventListener('click', () => {
    if (state.isPlaying) {
        pausePlayback();
    } else {
        startPlayback();
    }
});

dom.btnRestart.addEventListener('click', () => {
    state.playbackPauseOffset = 0;
    if (state.isPlaying) {
        startPlayback();
    } else {
        updatePlayhead(0);
        dom.currentTime.textContent = '0:00.0';
    }
});

dom.btnLoop.addEventListener('click', () => {
    state.isLooping = !state.isLooping;
    dom.btnLoop.classList.toggle('active', state.isLooping);
    if (state.rawSourceNode) state.rawSourceNode.loop = state.isLooping;
    if (state.enhSourceNode) state.enhSourceNode.loop = state.isLooping;
});

dom.volumeSlider.addEventListener('input', (e) => {
    if (state.masterGainNode) {
        state.masterGainNode.gain.setValueAtTime(parseFloat(e.target.value), getAudioContext().currentTime);
    }
});

function startPlayback() {
    if (!state.rawAudioBuffer) return;
    const ctx = getAudioContext();

    setupAudioNodes();

    const offset = state.playbackPauseOffset % state.audioDuration;
    state.playbackStartTime = ctx.currentTime - offset;

    state.rawSourceNode.start(0, offset);
    if (state.enhSourceNode) {
        state.enhSourceNode.start(0, offset);
    }

    state.isPlaying = true;
    dom.btnPlay.textContent = '⏸';
    dom.btnPlay.title = 'Pause';

    tickPlayback();
}

function pausePlayback() {
    if (!state.isPlaying) return;
    const ctx = getAudioContext();
    state.playbackPauseOffset = ctx.currentTime - state.playbackStartTime;

    if (state.rawSourceNode) {
        try { state.rawSourceNode.stop(); } catch (e) {}
    }
    if (state.enhSourceNode) {
        try { state.enhSourceNode.stop(); } catch (e) {}
    }

    state.isPlaying = false;
    dom.btnPlay.textContent = '▶';
    dom.btnPlay.title = 'Play';
    cancelAnimationFrame(state.animFrameId);
}

function stopPlayback() {
    pausePlayback();
    state.playbackPauseOffset = 0;
}

function tickPlayback() {
    if (!state.isPlaying) return;
    const ctx = getAudioContext();
    let currentSec = (ctx.currentTime - state.playbackStartTime);

    if (state.isLooping && state.audioDuration > 0) {
        currentSec = currentSec % state.audioDuration;
    }

    currentSec = Math.min(currentSec, state.audioDuration);
    dom.currentTime.textContent = formatTime(currentSec);

    const pct = state.audioDuration > 0 ? (currentSec / state.audioDuration) * 100 : 0;
    updatePlayhead(pct);

    state.animFrameId = requestAnimationFrame(tickPlayback);
}

function updatePlayhead(pct) {
    const clamped = Math.max(0, Math.min(100, pct));
    dom.waveformPlayhead.style.left = `${clamped}%`;
    dom.cursorSpecRaw.style.left = `${clamped}%`;
    dom.cursorSpecEnh.style.left = `${clamped}%`;
}

// Waveform Scrubber Click
dom.waveformContainer.addEventListener('click', (e) => {
    if (!state.audioDuration) return;
    const rect = dom.waveformContainer.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const pct = clickX / rect.width;
    const seekSec = pct * state.audioDuration;

    state.playbackPauseOffset = seekSec;
    updatePlayhead(pct * 100);
    dom.currentTime.textContent = formatTime(seekSec);

    if (state.isPlaying) {
        startPlayback();
    }
});

// ── Waveform Canvas Rendering ──────────────────────────────────────────────

function redrawWaveform() {
    const canvas = dom.waveformCanvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    clearCanvas(canvas);
    if (!state.rawAudioBuffer) return;

    const data = state.rawAudioBuffer.getChannelData(0);
    const step = Math.ceil(data.length / width);
    const amp = height / 2;

    // Draw center line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.beginPath();
    ctx.moveTo(0, amp);
    ctx.lineTo(width, amp);
    ctx.stroke();

    // Draw Raw audio waveform (amber/subtle)
    ctx.fillStyle = 'rgba(245, 158, 11, 0.45)';
    for (let i = 0; i < width; i++) {
        let min = 1.0, max = -1.0;
        for (let j = 0; j < step; j++) {
            const val = data[(i * step) + j];
            if (val < min) min = val;
            if (val > max) max = val;
        }
        ctx.fillRect(i, amp + (min * amp), 1, Math.max(1, (max - min) * amp));
    }

    // If enhanced exists, overlay enhanced peaks in emerald
    if (state.enhAudioBuffer) {
        const enhData = state.enhAudioBuffer.getChannelData(0);
        const enhStep = Math.ceil(enhData.length / width);
        ctx.fillStyle = 'rgba(16, 185, 129, 0.75)';
        for (let i = 0; i < width; i++) {
            let min = 1.0, max = -1.0;
            for (let j = 0; j < enhStep; j++) {
                const val = enhData[(i * enhStep) + j];
                if (val < min) min = val;
                if (val > max) max = val;
            }
            ctx.fillRect(i, amp + (min * amp), 1, Math.max(1, (max - min) * amp));
        }
    }
}

// ── Spectrogram Canvas Rendering ───────────────────────────────────────────

function renderSpectrograms() {
    if (state.spectrograms.raw) {
        drawSpectrogramMatrix(dom.canvasSpecRaw, state.spectrograms.raw, COLORMAP_VIRIDIS);
    }
    if (state.spectrograms.enh) {
        drawSpectrogramMatrix(dom.canvasSpecEnh, state.spectrograms.enh, COLORMAP_VIRIDIS);
    }
    if (state.spectrograms.diff) {
        drawSpectrogramMatrix(dom.canvasSpecDiff, state.spectrograms.diff, COLORMAP_INFERNO);
    }
}

function drawSpectrogramMatrix(canvas, matrix, colormap) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#020408';
    ctx.fillRect(0, 0, width, height);

    if (!matrix || matrix.length === 0) return;

    const nFreq = matrix.length;
    const nTime = matrix[0].length;
    const colW = width / nTime;
    const rowH = height / nFreq;

    for (let i = 0; i < nFreq; i++) {
        const y = height - ((i + 1) * rowH); // High frequencies at top
        for (let j = 0; j < nTime; j++) {
            const val = matrix[i][j]; // 0.0 - 1.0
            const color = interpolateColor(val, colormap);
            ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
            ctx.fillRect(j * colW, y, colW + 1, rowH + 1);
        }
    }
}

function interpolateColor(t, map) {
    const val = Math.max(0, Math.min(1, t));
    const idx = val * (map.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, map.length - 1);
    const f = idx - lo;

    return [
        Math.round(map[lo][0] + f * (map[hi][0] - map[lo][0])),
        Math.round(map[lo][1] + f * (map[hi][1] - map[lo][1])),
        Math.round(map[lo][2] + f * (map[hi][2] - map[lo][2])),
    ];
}

// Spectrogram View Tabs
dom.specTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        dom.specTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const view = tab.dataset.view;

        if (view === 'side-by-side') {
            dom.specSideBySide.style.display = 'grid';
            dom.specDiffWrapper.style.display = 'none';
        } else {
            dom.specSideBySide.style.display = 'none';
            dom.specDiffWrapper.style.display = 'block';
            if (state.spectrograms.diff) {
                drawSpectrogramMatrix(dom.canvasSpecDiff, state.spectrograms.diff, COLORMAP_INFERNO);
            }
        }
    });
});

// ── Real-Time Streaming WebSocket ──────────────────────────────────────────

dom.streamToggleBtn.addEventListener('click', toggleLiveStream);

async function toggleLiveStream() {
    if (state.isStreaming) {
        stopLiveStream();
    } else {
        startLiveStream();
    }
}

async function startLiveStream() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProto}//${window.location.host}/ws/stream`;

        state.streamWs = new WebSocket(wsUrl);
        state.streamWs.binaryType = 'arraybuffer';

        state.streamWs.onopen = () => {
            state.isStreaming = true;
            state.streamFrameCount = 0;
            dom.liveStreamStatus.textContent = 'STREAMING';
            dom.liveStreamStatus.className = 'stat-value text-green';
            dom.streamToggleText.textContent = 'Stop Streaming';
            dom.streamToggleBtn.classList.add('recording');

            // Setup Web Audio Capture at 16kHz
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            state.streamAudioCtx = new AudioContext({ sampleRate: 16000 });
            const src = state.streamAudioCtx.createMediaStreamSource(stream);

            // ScriptProcessorNode buffers 256 samples per 16ms
            const processor = state.streamAudioCtx.createScriptProcessor(256, 1, 1);
            src.connect(processor);
            processor.connect(state.streamAudioCtx.destination); // Required in Chrome

            processor.onaudioprocess = (e) => {
                if (!state.isStreaming || state.streamWs.readyState !== WebSocket.OPEN) return;
                const inputData = e.inputBuffer.getChannelData(0);
                // Send raw float32 bytes
                state.streamWs.send(inputData.buffer);
            };

            state.streamWorklet = processor;
        };

        state.streamWs.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                state.streamFrameCount++;
                dom.liveStreamFrames.textContent = state.streamFrameCount;

                // Optionally play enhanced sound back through speakers
                if (dom.streamSpeakerToggle.checked && state.streamAudioCtx) {
                    const enhancedFloat = new Float32Array(event.data);
                    playStreamChunk(enhancedFloat);
                }
            }
        };

        state.streamWs.onclose = () => {
            stopLiveStream();
        };
    } catch (e) {
        console.error('Live streaming failed:', e);
        alert(`Could not start live stream: ${e.message}`);
        stopLiveStream();
    }
}

function playStreamChunk(floatArr) {
    if (!state.streamAudioCtx) return;
    const buf = state.streamAudioCtx.createBuffer(1, floatArr.length, 16000);
    buf.copyToChannel(floatArr, 0);
    const src = state.streamAudioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(state.streamAudioCtx.destination);
    src.start();
}

function stopLiveStream() {
    state.isStreaming = false;
    if (state.streamWs) {
        state.streamWs.close();
        state.streamWs = null;
    }
    if (state.streamWorklet) {
        state.streamWorklet.disconnect();
        state.streamWorklet = null;
    }
    if (state.streamAudioCtx) {
        state.streamAudioCtx.close();
        state.streamAudioCtx = null;
    }

    dom.liveStreamStatus.textContent = 'STOPPED';
    dom.liveStreamStatus.className = 'stat-value text-muted';
    dom.streamToggleText.textContent = 'Start Live Streaming';
    dom.streamToggleBtn.classList.remove('recording');
}

// ── Edge Telemetry WebSocket Client ────────────────────────────────────────

function connectTelemetry() {
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProto}//${window.location.host}/ws`;

    state.telemetryWs = new WebSocket(wsUrl);

    state.telemetryWs.onopen = () => {
        document.getElementById('connection-indicator').querySelector('.indicator-dot').className = 'indicator-dot online';
        document.getElementById('connection-text').textContent = 'Online';
    };

    state.telemetryWs.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            if (data.latency_ms !== undefined) {
                dom.edgeLatencyVal.textContent = data.latency_ms.toFixed(1);
            }
            if (data.xruns !== undefined) {
                dom.edgeXrunVal.textContent = data.xruns;
            }
            if (data.frame !== undefined) {
                dom.edgeFrameVal.textContent = data.frame;
            }
            if (data.processing_time_ms !== undefined) {
                const ms = data.processing_time_ms;
                dom.computeVal.textContent = ms.toFixed(1);
                const pct = Math.min(100, (ms / 20.0) * 100);
                dom.computeGauge.style.width = `${pct}%`;
                dom.computeGauge.style.backgroundColor = ms > 16.0 ? 'var(--accent-red)' : (ms > 12.0 ? 'var(--accent-amber)' : 'var(--accent-emerald)');
            }
        } catch (err) {}
    };

    state.telemetryWs.onclose = () => {
        document.getElementById('connection-indicator').querySelector('.indicator-dot').className = 'indicator-dot offline';
        document.getElementById('connection-text').textContent = 'Offline';
        setTimeout(connectTelemetry, 3000);
    };
}

// ── Utility Helpers ────────────────────────────────────────────────────────

function formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) return '0:00.0';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 10);
    return `${m}:${s < 10 ? '0' : ''}${s}.${ms}`;
}

function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ── App Initialization ─────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
    fetchPresets();
    connectTelemetry();
});
