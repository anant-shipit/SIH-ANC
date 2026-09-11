# GTCRN Hardware Runbook (Raspberry Pi 5)

This runbook outlines the hardware components, GPIO pinout wiring, and OS scheduling configuration required to achieve real-time, zero-xrun GTCRN neural speech enhancement on the Raspberry Pi 5.

---

## 1. System Hardware Components & Wiring

### **Hardware Components**
- **Raspberry Pi 5 (8GB)**: Runs Raspberry Pi OS 64-bit (Bookworm) and the ONNX INT8 neural noise-suppression runtime.
- **2x INMP441 Digital MEMS Microphones**: Digital I2S mics with integrated 24/32-bit ADCs (no external codec required).
  - **Mic 1 (Primary)**: Mounted near mouth (speech + noise), `L/R` pin wired to `GND` (Left Channel).
  - **Mic 2 (Reference)**: Noise-facing (ref ambient noise), `L/R` pin wired to `3.3V` (Right Channel).
- **Quantron QSC-260 USB Sound Card**: External USB DAC + headphone amplifier for zero-latency audio output.
- **Push Button**: Momentary tactile switch to toggle Raw vs. GTCRN Enhanced audio mode in real time.
- **3x Status LEDs (+ Resistors)**: Physical status indicators (System Active, Enhanced Mode, Audio Peak Activity).
- **Active Cooler & 27W PSU**: Dedicated 4-pin fan header & 27W USB-C power supply.

### **Signal Flow Path**
> **Mics → 2x INMP441 (Analog ➔ Digital) → Stereo I2S → Raspberry Pi 5 (STFT ➔ GTCRN ➔ iSTFT ➔ Impulse Gate) → USB ➔ Quantron QSC-260 Sound Card (Digital ➔ Analog) → Headphones**

---

### **Pin Configuration & Wiring Table**

| Component | Pin / Port | GPIO | Notes / Connection |
| :--- | :--- | :--- | :--- |
| **Mic 1 VDD** | Pin 1 / 17 | `3.3V` | Shared 3.3V power supply with Mic 2 |
| **Mic 1 & 2 GND** | Any GND | `GND` | Shared Ground connection |
| **Mic 1 & 2 SCK** | Pin 12 | `GPIO18` | Shared I2S Serial Clock |
| **Mic 1 & 2 WS** | Pin 35 | `GPIO19` | Shared I2S Word Select (LRCLK) |
| **Mic 1 & 2 SD** | Pin 38 | `GPIO20` | Shared I2S Serial Data line |
| **Mic 1 L/R Pin** | — | — | Wired to `GND` (Selects **Left** Channel — Primary Mic) |
| **Mic 2 L/R Pin** | — | — | Wired to `3.3V` (Selects **Right** Channel — Reference Mic) |
| **Push Button** | Pin 11 | `GPIO17` | Other leg to `GND` (Uses internal pull-up) |
| **LED 1 (System Status)** | Pin 15 | `GPIO22` | + resistor ➔ `GND` (Solid ON when streaming) |
| **LED 2 (Enhanced Mode)** | Pin 16 | `GPIO23` | + resistor ➔ `GND` (ON = Filtered, OFF = Raw) |
| **LED 3 (Audio Activity)** | Pin 18 | `GPIO24` | + resistor ➔ `GND` (Pulses during speech/activity) |
| **USB Sound Card** | USB-A Port | — | Quantron QSC-260 DAC/Amp |
| **Active Cooler** | 4-Pin Fan Port | — | Dedicated cooling connector |
| **27W Power Supply** | USB-C Port | — | Primary power input |

---

## 2. Deployment Pipeline (Mac → Pi)

Before tuning the OS, you must package your trained model and deploy it to the Pi. The Pi **should not** have PyTorch installed; it only needs ONNX Runtime.

### Step 1: Export & Quantize (On Training Machine)
Convert your best `.pth` checkpoint to a streaming ONNX graph, then quantize it to INT8. This shrinks the model to ~0.2 MB and makes it fast enough for the Pi's CPU.
```bash
# Export
python -m sih26052.export.to_onnx \
    --checkpoint models/checkpoints/checkpoint_epoch_050.pth \
    --output models/gtcrn_stream.onnx

# Quantize
python -m sih26052.export.quantize \
    --input models/gtcrn_stream.onnx \
    --output models/gtcrn_stream_int8.onnx
```

### Step 2: Transfer (To Pi)
Copy the `SIH--ANC` repository and your new `models/gtcrn_stream_int8.onnx` file to the Raspberry Pi. You do **not** need to copy the datasets or `.pth` files.

### Step 3: Setup Environment (On Pi)
Install the lightweight Pi-specific dependencies (which include `onnxruntime` instead of `torch`).
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-pi.txt
```

---

## 2. Real-Time Scheduling (`SCHED_FIFO`)

Standard processes run under `SCHED_OTHER` which optimizes for overall throughput, not latency. You must elevate the audio loop script to `SCHED_FIFO`.

`SCHED_FIFO` (First-In, First-Out) is a real-time policy. A `SCHED_FIFO` thread will preempt any normal thread and run until it yields or is preempted by a higher-priority real-time thread.

**Action:**
Run the audio script with `chrt`. Priority 50 is generally sufficient:
```bash
sudo chrt -f 50 python -m sih26052.runtime.audio_loop
```
*Note: Because `SCHED_FIFO` can lock up your system if the process enters an infinite loop without yielding, we strongly recommend deploying this only on a dedicated hardware unit or running it cautiously during development.*

## 3. CPU Affinity (`taskset`)

Even with `SCHED_FIFO`, the kernel might migrate the audio thread between CPU cores to balance thermal loads. Thread migration causes L1/L2 cache invalidation, inducing severe multi-millisecond latency spikes that easily cause xruns.

You must pin the audio thread to a specific, dedicated core. On a Raspberry Pi 5, cores 2 and 3 are typically best to isolate from OS background tasks (which often default to core 0).

**Action:**
Combine `taskset` (CPU pinning) with `chrt` (Real-time scheduling). To pin the process exclusively to Core 3:
```bash
sudo taskset -c 3 chrt -f 50 python -m sih26052.runtime.audio_loop
```

## 4. Disable CPU Frequency Scaling (Governor)

The default CPU frequency governor (`ondemand` or `powersave`) aggressively downclocks the CPU during idle moments. When an audio frame arrives, the CPU takes several milliseconds to ramp up its clock speed—often missing the 16ms deadline.

You must lock the CPU frequency to its maximum using the `performance` governor.

**Action:**
Apply the performance governor to all cores:
```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## 5. Benchmarking the Setup

Before running live audio, use the offline benchmarking tool to verify the algorithmic capability of your hardware.

```bash
python -m sih26052.export.benchmark --onnx models/gtcrn_stream_int8.onnx --audio
```

**What to look for:**
- **Algorithmic Budget:** The benchmark should report processing times well below 16ms per frame. 
- **Real-Time Factor (RTF):** Must be `< 1.0` (ideally `< 0.3` to leave headroom for OS jitter).

If your processing time exceeds 16ms in the benchmark, the hardware is fundamentally too slow, and no amount of OS tuning will prevent xruns. You must use a smaller model (e.g., quantize the ONNX model to INT8).
