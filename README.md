# SIH26052 — Real-Time Speech Enhancement for Defense Environments

Real-time speech enhancement using **GTCRN (Grouped Temporal Convolutional Recurrent Network)** deployed on **Raspberry Pi 4B** (ARM Cortex-A72).

Developed for Smart India Hackathon (SIH 2026) / DRDO defense communications challenge.

---

## Architecture Overview

```
Physical Audio Input (Mic)
         │
         ▼
┌─────────────────────────────────┐
│ Overlap-Add Engine (ola.py)     │  512-pt FFT, 256-pt hop (16 ms)
│ sqrt-Hann Analysis Window       │  COLA-compliant 50% overlap
└─────────────────────────────────┘
         │ STFT Frame (257, 2)
         ▼
┌─────────────────────────────────┐
│ Streaming GTCRN (enhancer.py)   │  Quantized dynamic int8 ONNX
│ Recurrent Hidden State Caches   │  ~48.2K params, < 0.5 RTF on Pi 4B
└─────────────────────────────────┘
         │ Enhanced Frame (257, 2)
         ▼
┌─────────────────────────────────┐
│ Overlap-Add Engine (ola.py)     │  sqrt-Hann Synthesis Window
│ ISTFT Reconstruction            │
└─────────────────────────────────┘
         │ Enhanced Audio Samples
         ▼
┌─────────────────────────────────┐
│ Impulse Gate (impulse_gate.py)  │  Transient detector + hold-and-release
│ Transient Attenuator            │  Suppresses gunshots / weapon blasts
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ A/B Switch (ab_switch.py)       │  Hardware GPIO / Keyboard toggle
│ 20 ms Smooth Crossfade          │  Instant A/B bypass without clicks
└─────────────────────────────────┘
         │
         ├──────────────────────────► Live Audio Output (Headphones / Radio)
         │
         ▼ (non-blocking queue)
┌─────────────────────────────────┐
│ Dashboard Bridge (bridge.py)    │  FastAPI + WebSocket @ 10 Hz
│ Live Web UI (server.py)         │  Canvas spectrograms & latency readout
└─────────────────────────────────┘
```

---

## Key Performance Specifications

| Metric | Target | Achieved / Budgeted |
|---|---|---|
| **Total Algorithmic Latency** | ≤ 150 ms | **~69 ms** (32ms STFT window + 16ms hop + 16ms ALSA buffers + ~5ms inference) |
| **Model Size** | Edge-friendly | **~48.2K parameters** (~0.2 MB int8 ONNX) |
| **Compute / RTF** | RTF < 0.5 on Pi 4B | **33.0 MMACs/s**, tested in real-time |
| **Speech Quality (clean)** | Identity pass | **PESQ 4.64**, **SI-SNR > 100 dB**, **STOI 1.00** |
| **Hardware Target** | Low-cost edge board | **Raspberry Pi 4B (4GB)**, 64-bit OS |

---

## Repository Structure

```
.
├── sih26052/
│   ├── data/                 # Phase 1: Dataset preprocessing & mixing
│   │   ├── preprocess.py     # Resample audio to 16 kHz mono WAV
│   │   ├── mixer.py          # SNR mixing, clipping protection, impulse placement
│   │   ├── manifest.py       # JSONL manifest reader/writer with statistics
│   │   └── augment.py        # Gain, band-limiting, soft clipping, synthetic reverb
│   ├── eval/                 # Phase 2: Evaluation harness
│   │   ├── metrics.py        # PESQ, STOI, and Scale-Invariant SNR (SI-SNR)
│   │   ├── alignment.py      # Cross-correlation delay finder
│   │   └── harness.py        # Automated test harness across subsets
│   ├── export/               # Phase 3: Streaming export & quantization
│   │   ├── to_onnx.py        # PyTorch to streaming stateful ONNX (opset 17)
│   │   ├── quantize.py       # Dynamic int8 quantization
│   │   ├── verify.py         # Numerical PyTorch vs ONNX consistency checker
│   │   └── benchmark.py      # Real-Time Factor (RTF) measurement
│   ├── runtime/              # Phase 4+5: Real-time inference (NO torch)
│   │   ├── audio_loop.py     # sounddevice callback streaming loop
│   │   ├── ola.py            # Overlap-add engine with sqrt-Hann windowing
│   │   ├── enhancer.py       # ONNX Runtime stateful frame processor
│   │   ├── ab_switch.py      # 20ms crossfade A/B switch (GPIO/keyboard)
│   │   ├── impulse_gate.py   # Transient detector with hold-and-release
│   │   └── nlms.py           # Dual-mic adaptive filter (optional)
│   ├── dashboard/            # Phase 5: Live web dashboard
│   │   ├── server.py         # FastAPI WebSocket server
│   │   ├── bridge.py         # Audio callback to WebSocket broadcaster
│   │   └── static/           # HTML5 Canvas spectrogram UI
│   └── train/                # Phase 6: Fine-tuning pipeline
│       ├── dataset.py        # On-the-fly mixing DataLoader (40/40/20 composition)
│       ├── loss.py           # SI-SNR + Compressed Spectral Loss (|X|^0.3)
│       ├── train.py          # Fine-tuning loop with pretrained weights loading
│       ├── validate.py       # Per-epoch validation & forgetting detector
│       └── select_checkpoint.py # Best-by-PESQ checkpoint selector
├── scripts/                  # CLI execution scripts
│   ├── eval_identity.py      # Quick identity verification test
│   ├── eval_baseline.py      # Benchmark baseline audio on test manifest
│   ├── verify_export.py      # Verify ONNX model validity & quantization
│   └── benchmark_rtf.py      # Benchmark RTF on current device
├── tests/                    # 59 unit and integration tests (100% passing)
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation script
└── README.md
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/anant-shipit/SIH-ANC.git
cd SIH-ANC
```

### 2. Create virtual environment & install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Verification & Testing

Run the full pytest suite (59 tests covering all modules):
```bash
python3 -m pytest tests/ -v
```

Run the identity evaluation:
```bash
python3 scripts/eval_identity.py
```

---

## Usage Guide

### 1. Data Preprocessing
Resample raw audio directories (clean speech, stationary noise, impulsive noise) to 16 kHz mono:
```bash
python3 -m sih26052.data.preprocess \
    --input-dirs /path/to/VoiceBank /path/to/DEMAND /path/to/ESC-50 \
    --output-dir data/processed/
```

### 2. Model Export to Streaming ONNX
Convert a trained PyTorch GTCRN checkpoint to streaming ONNX:
```bash
python3 -m sih26052.export.to_onnx \
    --checkpoint models/checkpoints/model.pth \
    --output models/gtcrn_stream.onnx
```

Quantize to dynamic int8:
```bash
python3 -m sih26052.export.quantize \
    --input models/gtcrn_stream.onnx \
    --output models/gtcrn_stream_int8.onnx
```

Benchmark Real-Time Factor (RTF):
```bash
python3 scripts/benchmark_rtf.py --onnx models/gtcrn_stream_int8.onnx
```

### 3. Real-Time Processing Loop
Run real-time enhancement on mic input and stream to audio output:
```bash
python3 -m sih26052.runtime.audio_loop --onnx models/gtcrn_stream_int8.onnx
```

### 4. Launch Monitoring Dashboard
Run the web dashboard:
```bash
python3 -m sih26052.dashboard.server --port 8080
```
Open `http://localhost:8080` in any browser to view live spectrograms, latency, xruns, and transient detection.

### 5. Model Fine-Tuning
Fine-tune GTCRN with combined SI-SNR + compressed spectral loss:
```bash
python3 -m sih26052.train.train \
    --checkpoint models/checkpoints/pretrained.pth \
    --clean-dir data/processed/clean \
    --noise-dirs broadband=data/processed/DEMAND impulsive=data/processed/ESC-50 \
    --epochs 50 \
    --lr 1e-4
```

---

## License
MIT License. Developed for SIH 2026.
