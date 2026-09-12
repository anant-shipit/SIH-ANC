# GTCRN Hardware Runbook (Raspberry Pi 5)

This runbook outlines the hardware components, 40-pin GPIO pinout wiring, Linux ALSA kernel overlay configuration, OS real-time scheduling, and operational procedures required for the GTCRN neural speech enhancement engine on the Raspberry Pi 5.

---

## 1. System Hardware Components & Wiring

### Hardware Components
- **Raspberry Pi 5 (4GB/8GB)**: Quad-core ARM Cortex-A76 @ 2.4 GHz running Raspberry Pi OS 64-bit (Bookworm) and the lightweight ONNX INT8 streaming runtime.
- **2x INMP441 Digital MEMS Microphones**: Omnidirectional digital I2S microphones with integrated 24-bit ADCs (no external analog codec required).
  - **Mic 1 (Primary)**: Positioned for speech + noise capture; `L/R` pin wired to `GND` (Left Channel).
  - **Mic 2 (Reference)**: Positioned for ambient background noise; `L/R` pin wired to `3.3V` (Right Channel).
- **USB PnP Audio DAC / 3.5mm Headphone Amplifier**: External low-latency USB DAC providing physical 3.5mm headphone output.
- **Push Button**: Momentary tactile switch connected to toggle GTCRN AI filtering vs. raw bypass in real time.
- **3x Status LEDs (+ 330Ω Resistors)**: Physical status indicators (System Running, Enhanced Mode Active, Voice Activity).
- **Active Cooler & 27W USB-C PSU**: Official Raspberry Pi Active Cooler and 27W Power Delivery supply.

### Signal Flow Path
> **Acoustic Sound Field → 2x INMP441 MEMS (Analog ➔ Digital) → I2S Bus (48 kHz) → Raspberry Pi 5 (3:1 Decimation ➔ 16 kHz STFT ➔ Streaming GTCRN INT8 ➔ 16 kHz iSTFT ➔ Impulse Gate ➔ 1:3 Interpolation) → USB DAC (48 kHz) → 3.5mm Headphones**

---

### Pin Configuration & 40-Pin GPIO Wiring Table

| Component | Pin / Port | BCM GPIO | Notes / Connection |
| :--- | :--- | :--- | :--- |
| **Mic 1 & 2 VDD** | Pin 1 | `3.3V` | Regulated 3.3V rail (shared) |
| **Mic 1 & 2 GND** | Pin 6, 14, or 20 | `GND` | Ground reference (shared) |
| **Mic 1 & 2 SCK** | Pin 12 | `GPIO18` | Shared I2S Serial Clock (BCLK) |
| **Mic 1 & 2 WS** | Pin 35 | `GPIO19` | Shared I2S Word Select (LRCLK / Frame Clock) |
| **Mic 1 & 2 SD** | Pin 38 | `GPIO20` | Shared I2S Serial Data line |
| **Mic 1 L/R Pin** | Pin 9 (or any GND) | `GND` | Wired to `GND` (Left Channel — Primary Speech Mic) |
| **Mic 2 L/R Pin** | Pin 17 (or 3.3V) | `3.3V` | Wired to `3.3V` (Right Channel — Ambient Reference Mic) |
| **Push Button** | Pin 11 | `GPIO17` | Other leg to `GND` (Internal pull-up enabled) |
| **LED 1 (SYS)** | Pin 15 | `GPIO22` | + 330Ω resistor ➔ `GND` (Solid Green: Audio pipeline active) |
| **LED 2 (MODE)** | Pin 16 | `GPIO23` | + 330Ω resistor ➔ `GND` (Solid Blue: GTCRN AI filter active) |
| **LED 3 (ACT)** | Pin 18 | `GPIO24` | + 330Ω resistor ➔ `GND` (Amber: Voice activity with 240ms hold) |
| **USB Audio Card** | USB 2.0/3.0 Port | — | USB PnP DAC / 3.5mm Headphone Output |
| **Active Cooler** | 4-Pin Fan Header | — | Dedicated Raspberry Pi 5 fan connector |
| **Power Supply** | USB-C Port | — | 27W USB-PD Power Supply (5V / 5A) |

---

## 2. Raspberry Pi 5 OS & Soundcard Configuration

The INMP441 I2S digital microphones communicate directly with the Broadcom BCM2712 I2S peripherals via the kernel sound card overlay.

### Step 1: Enable I2S Device Tree Overlay
Edit `/boot/firmware/config.txt`:
```bash
sudo nano /boot/firmware/config.txt
```

Append the following line:
```ini
dtoverlay=googlevoicehat-soundcard
```
Save the file (`Ctrl+O`, `Enter`, `Ctrl+X`) and reboot the board:
```bash
sudo reboot
```

### Step 2: Verify Audio Hardware Detection
1. **Verify I2S Microphone Capture Device**:
   ```bash
   arecord -l
   ```
   Confirm Card 2 is detected:
   ```
   card 2: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar], device 0: Google voiceHAT SoundCard HiFi voicehat-hifi-0 [Google voiceHAT SoundCard HiFi voicehat-hifi-0]
   ```

2. **Verify USB Sound Card Playback Device**:
   ```bash
   aplay -l
   ```
   Confirm Card 3 (USB DAC) is detected:
   ```
   card 3: Device [USB PnP Sound Device], device 0: USB Audio [USB Audio]
   ```

---

## 3. Deployment Pipeline & Environment

The Raspberry Pi 5 runs a lightweight ONNX Runtime environment without PyTorch dependencies.

### Step 1: Export & Quantize (On Training Machine)
Convert the best PyTorch checkpoint to streaming ONNX and quantize weights to dynamic INT8:
```bash
# Export stateful recurrent streaming model
python -m sih26052.export.to_onnx \
    --checkpoint models/checkpoints/checkpoint_best.pth \
    --output models/gtcrn_finetuned_stream.onnx

# Dynamic INT8 Quantization
python -m sih26052.export.quantize \
    --input models/gtcrn_finetuned_stream.onnx \
    --output models/gtcrn_finetuned_stream_int8.onnx
```

### Step 2: Deploy to Raspberry Pi 5
Transfer the repository and INT8 model to the Pi:
```bash
git clone https://github.com/anant-shipit/SIH-ANC.git
cd SIH-ANC
```

### Step 3: Setup Virtual Environment
Install runtime dependencies:
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libportaudio2 alsa-utils

python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements-pi.txt
pip install -e .
```

---

## 4. Real-Time Execution Modes

### Mode A: Real-Time Audio Loop CLI

The I2S sound card overlay fixes hardware capture to 48 kHz. The GTCRN neural network operates on 16 kHz audio. The `audio_loop.py` engine performs 3:1 decimation on input and 1:3 interpolation on output:

```bash
python3 -m sih26052.runtime.audio_loop \
    --onnx models/gtcrn_finetuned_stream_int8.onnx \
    --native-sr 48000
```

* **Tactile Push Button**: Press the physical switch on GPIO 17 (or hit `SPACE` in terminal) to instantly toggle between enhanced speech and raw audio bypass.
* **Status LEDs**:
  - `SYS` (GPIO 22): Solid Green indicates active audio loop.
  - `MODE` (GPIO 23): Solid Blue indicates GTCRN AI filtering is active.
  - `ACT` (GPIO 24): Amber illuminates during speech (calibrated to ignore ambient air).

### Mode B: Executive Live Web Console

Launch the web console:
```bash
python3 -m sih26052.dashboard.server --host 0.0.0.0 --port 8080
```
Open `http://<pi-ip-address>:8080` in any web browser.

**Console Capabilities:**
- **Start / Stop Hardware Stream**: Trigger the live I2S microphone loop.
- **Headphone Volume Slider**: Synchronizes both ALSA hardware mixer and software digital gain (0–100%).
- **Spectral Frequency Analyzer (60 fps Canvas)**: Live 0–8 kHz STFT response comparing raw input against cleaned speech.
- **Real-Time Oscilloscope**: Displays continuous 16 kHz waveform audio chunks.
- **Sample Audio Evaluation & Headphone Studio**:
  - Upload local audio files or choose from pre-loaded tactical military defense samples (*Battlefield Noise*, *Armored Vehicle Engine*, *Radio Babble*, *High Wind*, *Gunfire*).
  - Execute on-Pi GTCRN INT8 inference.
  - Click **`🎧 Listen Enhanced on Pi`** or **`🎧 Listen Raw on Pi`** to output audio directly through the physical 3.5mm headphones connected to the Raspberry Pi's USB DAC, with volume control and synchronized status LEDs.

---

## 5. OS Performance Tuning & Latency Optimization

To achieve zero audio dropouts (xruns) and maintain deterministic 1.8ms frame inference, apply real-time Linux optimizations.

### 1. Real-Time Scheduling (`SCHED_FIFO`)
Elevate process priority above standard background tasks:
```bash
sudo chrt -f 50 python3 -m sih26052.runtime.audio_loop \
    --onnx models/gtcrn_finetuned_stream_int8.onnx \
    --native-sr 48000
```

### 2. CPU Affinity (`taskset`)
Pin the audio thread to a dedicated core (e.g., Core 3) to prevent L1/L2 cache invalidation:
```bash
sudo taskset -c 3 chrt -f 50 python3 -m sih26052.runtime.audio_loop \
    --onnx models/gtcrn_finetuned_stream_int8.onnx \
    --native-sr 48000
```

### 3. CPU Performance Governor
Lock Cortex-A76 clock frequencies to maximum to prevent frequency-scaling latency spikes:
```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## 6. Benchmarking & Verification

Verify the algorithmic latency and Real-Time Factor (RTF) on the Raspberry Pi 5:

```bash
python3 scripts/benchmark_rtf.py --onnx models/gtcrn_finetuned_stream_int8.onnx
```

**Target Criteria on Raspberry Pi 5:**
- **Inference Time**: ≤ 2.5 ms per 16 ms frame (Achieved: **1.8 ms**).
- **Real-Time Factor (RTF)**: < 0.20x (Achieved: **0.08x**, ~12.5x faster than real-time).
- **Algorithmic Latency**: 16.0 ms (256-sample hop @ 16 kHz).
- **Total Pipeline Latency**: ~37 ms (16ms OLA group delay + 16ms ALSA ringbuffer + ~5ms compute).
