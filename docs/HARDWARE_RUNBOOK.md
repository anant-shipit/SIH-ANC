# GTCRN Hardware Runbook

This runbook outlines the required system configuration to achieve real-time, zero-xrun performance for GTCRN on constrained edge devices like the Raspberry Pi 4B. 

Audio processing on Linux is highly sensitive to OS scheduling jitter. A 16ms audio budget means missing a single thread wake-up by 2-3ms will cause an xrun (buffer underrun) resulting in audible clicking. To achieve reliable streaming, you must bypass the standard Linux completely fair scheduler (CFS) for the audio thread.

## 1. Deployment Pipeline (Mac → Pi)

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

You must pin the audio thread to a specific, dedicated core. On a Raspberry Pi 4B, cores 2 and 3 are typically best to isolate from OS background tasks (which often default to core 0).

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
