# SIH26052 — Real-Time Edge Speech Enhancement & Active Noise Cancellation

[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205%20(ARM%20Cortex--A76)-crimson.svg)](#hardware-architecture)
[![Latency](https://img.shields.io/badge/Algorithmic%20Latency-16%20ms-emerald.svg)](#key-performance-specifications)
[![Inference](https://img.shields.io/badge/Inference%20Time-1.8%20ms%20(RTF%200.08x)-blue.svg)](#key-performance-specifications)
[![Precision](https://img.shields.io/badge/Defense%20Precision-97.4%25-darkviolet.svg)](#model-benchmarks)
[![Model Size](https://img.shields.io/badge/Model%20Size-1.3%20MB%20(23.6K%20params)-orange.svg)](#model-benchmarks)

**SIH-26052** is a production-grade, low-latency edge AI speech enhancement and active noise cancellation system engineered for defense and tactical communications. Built for the **Raspberry Pi 5** (ARM Cortex-A76 64-bit), the system captures multi-channel environmental audio via dual digital I2S microphones, suppresses harsh battlefield, vehicular, and radio noise using a fine-tuned **Grouped Temporal Convolutional Recurrent Network (GTCRN)**, and delivers clean speech through physical headphones and live web consoles with a total end-to-end roundtrip latency of under 37 ms.

Developed for Smart India Hackathon (SIH 2026) / DRDO defense communications.

---

## System Architecture

```
Physical Acoustic Field (Speech + Defense Noise)
                  │
                  ▼
┌──────────────────────────────────────────────────┐
│ Dual INMP441 I2S MEMS Microphones                │
│ BCLK: GPIO 18  │  WS: GPIO 19  │  SD: GPIO 20    │
└──────────────────────────────────────────────────┘
                  │ 48 kHz / 32-bit I2S PCM
                  ▼
┌──────────────────────────────────────────────────┐
│ PortAudio Native Capture Engine (audio_loop.py) │
│ 3:1 Decimation Filter (48 kHz ➔ 16 kHz Mono)     │
└──────────────────────────────────────────────────┘
                  │ 16 kHz Mono PCM (256 samples / 16ms Hop)
         ┌────────┴─────────────────────────────────┐
         │                                          │
         ▼                                          ▼
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│ Overlap-Add Analysis (ola.py)    │    │ 1-Hop Delay Alignment Buffer     │
│ 512-pt FFT • sqrt-Hann Window   │    │ 16 ms group delay compensation   │
└──────────────────────────────────┘    └──────────────────────────────────┘
         │ STFT Complex Frame (257, 2)              │
         ▼                                          │
┌──────────────────────────────────┐               │
│ Streaming GTCRN (enhancer.py)    │               │
│ INT8 Quantized • Stateful Caches │               │
│ Inference: 1.8 ms on Cortex-A76  │               │
└──────────────────────────────────┘               │
         │ Cleaned Complex Frame (257, 2)           │
         ▼                                          │
┌──────────────────────────────────┐               │
│ Overlap-Add Synthesis (ola.py)   │               │
│ 512-pt Inverse FFT (ISTFT)       │               │
└──────────────────────────────────┘               │
         │ Cleaned Audio Frame                      │
         ▼                                          │
┌──────────────────────────────────┐               │
│ Transient Gate (impulse_gate.py) │               │
│ Gunfire / Blast Attenuation      │               │
└──────────────────────────────────┘               │
         │                                          │
         ▼                                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Click-Free A/B Crossfade Switch (ab_switch.py)                           │
│ Hardware Tactile Button (GPIO 17) • 20 ms Equal-Power Crossfade          │
└──────────────────────────────────────────────────────────────────────────┘
                  │ Selected Stream (Enhanced or Raw Bypass)
                  ├──────────────────────────────────┐
                  ▼                                  ▼
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│ 1:3 Interpolation Engine         │    │ Dashboard Telemetry Queue        │
│ 16 kHz ➔ 48 kHz Resampling       │    │ Non-blocking 30 fps ringbuffer   │
└──────────────────────────────────┘    └──────────────────────────────────┘
                  │ 48 kHz PCM                       │
                  ▼                                  ▼
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│ USB DAC / Headphone Amp          │    │ Executive Web Console            │
│ 3.5mm Headphone Jack             │    │ 60 fps STFT Spectrum & Osc       │
│ ALSA Mixer + Digital Gain        │    │ Pi Headphone Studio + Live Audio │
└──────────────────────────────────┘    └──────────────────────────────────┘
```

---

## Hardware Architecture

The system executes entirely on-device using commercial-off-the-shelf (COTS) edge components with no cloud dependencies:

![Hardware Overview](media_1789159066269.png)

### 1. Component Specification
* **Edge Compute**: Raspberry Pi 5 Model B (Quad-core ARM Cortex-A76 @ 2.4 GHz, 4GB/8GB LPDDR4X).
* **Primary Audio Input**: Dual INMP441 Omnidirectional MEMS Microphones with 24-bit I2S digital output (SNR: 61 dBA, Sensitivity: -26 dBFS).
* **Audio Output**: USB PnP Stereo Audio DAC / 3.5mm Headphone Amplifier.
* **Tactile Mode Controller**: Momentary Pushbutton connected between GPIO 17 and GND.
* **Visual Status Array**: Three high-visibility 3mm/5mm status LEDs with current-limiting resistors.

### 2. Complete 40-Pin GPIO Pinout Table

| Peripheral | Signal / Function | Raspberry Pi 5 Header Pin | BCM GPIO | Notes |
|---|---|---|---|---|
| **INMP441 Mic** | VDD (Power) | **Pin 1** | 3.3V Power | Clean regulated rail |
| **INMP441 Mic** | GND (Ground) | **Pin 6, 14, or 20** | Ground | Common reference ground |
| **INMP441 Mic** | SCK / BCLK (Clock) | **Pin 12** | **GPIO 18** | I2S Bit Clock |
| **INMP441 Mic** | WS / LRCLK (Word) | **Pin 35** | **GPIO 19** | I2S Frame / Word Select |
| **INMP441 Mic** | SD (Serial Data) | **Pin 38** | **GPIO 20** | I2S Data Input |
| **INMP441 Mic** | L/R (Channel) | Pin 9 (GND) / Pin 17 (3.3V) | — | GND for Left channel; 3.3V for Right |
| **Push Button** | ANC / Bypass Toggle | **Pin 11** | **GPIO 17** | To Ground (internal pull-up enabled) |
| **Status LED 1** | System Power & Loop | **Pin 15** | **GPIO 22** | **Green**: Solid when audio stream is running |
| **Status LED 2** | ANC Mode Active | **Pin 16** | **GPIO 23** | **Blue**: Solid when GTCRN filtering is ON |
| **Status LED 3** | Voice Activity (VAD) | **Pin 18** | **GPIO 24** | **Amber**: Voice activity with 240ms hold |

---

## Key Performance Specifications

| Metric | Target Specification | Measured On Raspberry Pi 5 |
|---|---|---|
| **Algorithmic Latency** | ≤ 20 ms | **16.0 ms** (256-sample hop @ 16 kHz) |
| **Total Roundtrip Latency** | ≤ 50 ms | **~37 ms** (16ms OLA group delay + 16ms ALSA ringbuffer + 5ms compute) |
| **Inference Time per Hop** | ≤ 16 ms | **1.8 ms** (11.2% core load per frame) |
| **Real-Time Factor (RTF)** | < 0.25x | **0.08x** (12.5× faster than real-time) |
| **Clean Speech PESQ** | ≥ 4.00 | **4.64** (Transparent identity pass-through) |
| **Noisy Speech PESQ** | ≥ 2.50 | **2.71** (Evaluated across DRDO defense noise sets) |
| **Signal-to-Noise Ratio (SI-SNR)** | ≥ +6.0 dB | **+8.4 dB average gain** (Up to +17.4 dB in high noise) |
| **Extended STOI** | ≥ 0.85 | **0.905** |
| **Model Size / Footprint** | ≤ 5.0 MB | **1.3 MB** (INT8 ONNX) / **2.7 MB** (FP32 ONNX) |
| **Model Parameter Count** | ≤ 50K params | **23,616 parameters** |
| **Target Defense Precision** | ≥ 95.0% | **97.4%** |
| **Target Defense Recall** | ≥ 85.0% | **88.4%** |

---

## Repository Structure

```
SIH-ANC/
├── sih26052/
│   ├── runtime/                      # Real-time CPython edge runtime (Zero PyTorch)
│   │   ├── audio_loop.py             # PortAudio capture/playback, 48k↔16k resampling
│   │   ├── enhancer.py               # ONNX Runtime stateful frame processor
│   │   ├── ola.py                    # 512-pt STFT / ISTFT Overlap-Add engine
│   │   ├── ab_switch.py              # Click-free 20ms equal-power crossfader
│   │   ├── impulse_gate.py           # Transient blast and gunfire suppressor
│   │   └── led_status.py             # Hardware GPIO LED driver with hold hysteresis
│   ├── dashboard/                    # Executive monitoring & evaluation dashboard
│   │   ├── server.py                 # FastAPI server with WebSockets & REST endpoints
│   │   ├── hardware_bridge.py        # Background loop manager, ALSA mixer, headphone cache
│   │   ├── processor.py              # Offline/online audio evaluation pipeline
│   │   └── static/
│   │       ├── index.html            # Minimalist industrial executive console
│   │       ├── style.css             # Monospaced technical dark theme
│   │       ├── app.js                # 60 fps Canvas spectrum, oscilloscope, headphone controls
│   │       └── eval_samples/         # Pre-loaded military/defense audio benchmarks
│   ├── export/                       # ONNX export and INT8 quantization
│   │   ├── to_onnx.py                # Stateful recurrent streaming ONNX exporter
│   │   ├── quantize.py               # Dynamic INT8 weight quantizer
│   │   ├── verify.py                 # Numerical PyTorch vs ONNX equivalence verifier
│   │   └── benchmark.py              # RTF & frame latency benchmark harness
│   ├── train/                        # PyTorch model training and fine-tuning
│   │   ├── dataset.py                # Real-time 40/40/20 multi-condition data loader
│   │   ├── loss.py                   # Combined SI-SNR + Compressed Spectral loss
│   │   └── train.py                  # Fine-tuning loop with checkpointing
│   ├── eval/                         # Objective metric evaluation suite
│   │   ├── metrics.py                # PESQ, STOI, SI-SNR, and SNR estimators
│   │   └── harness.py                # Multi-dataset automated benchmark suite
│   └── data/                         # Audio manifests and preprocessing tools
├── models/                           # Exported models and checkpoint summaries
│   ├── gtcrn_finetuned_stream_int8.onnx # Primary INT8 streaming model (1.3 MB)
│   ├── gtcrn_finetuned_stream.onnx   # Primary FP32 streaming model (2.7 MB)
│   └── epochs_and_combos_summary.json# Comprehensive metric validation log
├── scripts/                          # Automated verification and benchmarking scripts
├── tests/                            # Pytest unit and integration test suite
├── requirements-pi.txt               # Lightweight Raspberry Pi dependencies
├── requirements.txt                  # Full workstation / training dependencies
└── README.md                         # Authoritative system documentation
```

---

## Executive Live Web Dashboard

The executive web console provides real-time operational telemetry, hardware controls, and a sample evaluation studio:

1. **Hardware Stream Control**: Start and stop the physical audio loop directly from the interface.
2. **Tactile ANC Toggle**: Toggle between GTCRN AI filtering and raw microphone bypass with visual indicator confirmation.
3. **Headphone Volume Slider**: 0–100% linear control synchronizing the ALSA hardware mixer and digital gain stages.
4. **Dual Spectral Frequency Analyzer**: 60 fps Canvas rendering comparing the raw input spectrum against the cleaned speech spectrum across 0–8 kHz.
5. **Real-Time Oscilloscope**: Continuous waveform monitoring at 16,000 samples per second.
6. **Live Browser Monitor**: Stream the cleaned audio directly to the operator's computer speakers over WebSockets.
7. **Sample Audio Evaluation & Headphone Studio**:
   - Upload local audio files (WAV, MP3, FLAC) or select from pre-loaded tactical military defense recordings (Battlefield Noise, Armored Vehicle Engine, Command Center Radio Babble, High-Wind Rotor Noise, Tactical Static).
   - Instant on-Pi processing with inference time, RTF, and SNR gain readouts.
   - **Physical Headphone Playback**: Output **both clean enhanced audio and raw noisy audio directly through the headphones** connected to the Raspberry Pi's 3.5mm jack.
   - **In-Browser Playback**: Instant A/B comparative audio playback within the browser.

---

## Installation & Setup Guide

### 1. Raspberry Pi 5 Preparation

1. Flash **Raspberry Pi OS (64-bit Bookworm)** onto a high-speed micro-SD card or NVMe SSD.
2. Enable the Google voiceHAT / generic I2S sound card overlay by editing `/boot/firmware/config.txt`:
   ```bash
   sudo nano /boot/firmware/config.txt
   ```
   Add the following line to the end of the file:
   ```ini
   dtoverlay=googlevoicehat-soundcard
   ```
3. Save the file (`Ctrl+O`, `Enter`, `Ctrl+X`) and reboot the board:
   ```bash
   sudo reboot
   ```
4. Verify that the I2S capture sound card is detected:
   ```bash
   arecord -l
   ```
   You will see:
   ```
   card 2: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar], device 0: Google voiceHAT SoundCard HiFi voicehat-hifi-0
   ```
5. Plug the USB DAC into any USB port and verify playback hardware:
   ```bash
   aplay -l
   ```
   You will see:
   ```
   card 3: Device [USB PnP Sound Device], device 0: USB Audio
   ```

### 2. Environment Setup

Install ALSA system libraries and clone the repository:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv libportaudio2 alsa-utils git

git clone https://github.com/anant-shipit/SIH-ANC.git
cd SIH-ANC
```

Create a virtual environment and install dependencies:
```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements-pi.txt
pip install -e .
```

---

## Execution Modes

### Mode 1: Standalone Real-Time Audio Loop (CLI)

Run the autonomous streaming audio engine directly in the terminal:
```bash
python3 -m sih26052.runtime.audio_loop \
    --onnx models/gtcrn_finetuned_stream_int8.onnx \
    --native-sr 48000
```

* **Pushbutton Control**: Press the physical button connected to GPIO 17 (or hit `SPACE` in the terminal) to instantly toggle between cleaned speech and raw bypass.
* **Status LEDs**: 
  - **SYS (GPIO 22)**: Solid Green indicates the audio engine is running.
  - **MODE (GPIO 23)**: Solid Blue indicates GTCRN filtering is active.
  - **ACT (GPIO 24)**: Amber glows during voice activity with calibrated 240ms hold time.

### Mode 2: Executive Live Web Dashboard

Launch the web console:
```bash
python3 -m sih26052.dashboard.server --host 0.0.0.0 --port 8080
```

Access the dashboard by navigating to `http://<pi-ip-address>:8080` (or `http://sih-pi.local:8080`) from any device on the local network.

* Click **Start Hardware Stream** to activate live microphone capture and processing.
* Adjust the **Headphone Volume** slider to calibrate listening levels.
* Use the **Sample Audio Evaluation & Headphone Studio** to process audio files and toggle between raw and enhanced playback directly through the physical headphones connected to the Raspberry Pi.

---

## Model Benchmarks & Validation

The GTCRN architecture utilizes dual-path grouped recurrent units and complex-valued spectral mapping. Below are the verified evaluation metrics across target defense scenarios:

| Model Variant | Quantization | Model Size | PESQ | SI-SNR Gain | Defense Precision | Defense Recall | F1-Score |
|---|---|---|---|---|---|---|---|
| **Stage-3 Ep10 (Final Best)** | INT8 | 1.3 MB | 2.71 | +8.4 dB | **97.4%** | **88.4%** | **0.926** |
| **Combo-2way Ensemble** | INT8 | 1.3 MB | 2.68 | +8.1 dB | **96.6%** | **95.9%** | **0.963** |
| **Core Run Ep45** | INT8 | 1.3 MB | 2.64 | +7.8 dB | **96.2%** | **97.1%** | **0.966** |
| **Clean Baseline** | — | — | 4.64 | +0.0 dB | 100.0% | 100.0% | 1.000 |

All evaluation manifests, checkpoints, and confusion matrices are located in the `models/` directory.

---

## Automated Verification Tests

Run the complete 59-test verification suite to validate all mathematical, streaming, and hardware components:
```bash
pytest tests/ -v
```

Run the identity check to confirm transparent pass-through:
```bash
python3 scripts/eval_identity.py
```

---

## License

This project is licensed under the MIT License. Developed for Smart India Hackathon (SIH 2026).
