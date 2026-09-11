#!/usr/bin/env bash
# =============================================================================
# setup_pi.sh — Run this ON the Raspberry Pi 5 to:
#   1. Install system packages (ALSA, portaudio, git, etc.)
#   2. Clone/update the SIH-ANC repo
#   3. Create a Python venv and install requirements-pi.txt
#   4. Configure I2S MEMS microphone (SPH0645) device tree overlay
#   5. Write ALSA config for low-latency 16kHz audio
#   6. Install systemd services for auto-start on boot
#
# Called automatically by deploy_to_pi.sh on Mac,
# OR run manually on the Pi:
#   chmod +x setup_pi.sh && ./setup_pi.sh
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/anant-shipit/SIH-ANC.git"
REPO_DIR="$HOME/SIH-ANC"
VENV_DIR="$REPO_DIR/venv"
MODEL_PATH="$REPO_DIR/models/gtcrn_stream_int8.onnx"

# ── Color output ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
step()    { echo -e "\n${BOLD}━━━ $* ━━━${RESET}"; }

# ── Sanity check ─────────────────────────────────────────────────────────────
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    warn "This script is designed for Raspberry Pi. Continuing anyway..."
fi

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║    SIH26052 — Pi 5 Setup Script         ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Step 1: System packages ───────────────────────────────────────────────────
step "Step 1: Installing System Packages"

sudo apt-get update -q
sudo apt-get install -y -q \
    git \
    python3-venv \
    python3-dev \
    python3-pip \
    libportaudio2 \
    portaudio19-dev \
    libsndfile1 \
    libsndfile1-dev \
    libasound2-dev \
    python3-lgpio \
    python3-gpiozero \
    alsa-utils \
    htop \
    screen

success "System packages installed."

# ── Step 2: Clone or update repo ─────────────────────────────────────────────
step "Step 2: Cloning / Updating Repository"

if [ -d "$REPO_DIR/.git" ]; then
    info "Repo already exists. Pulling latest..."
    git -C "$REPO_DIR" pull --ff-only
    success "Repo updated."
else
    info "Cloning $REPO_URL..."
    git clone "$REPO_URL" "$REPO_DIR"
    success "Repo cloned to $REPO_DIR"
fi

# ── Step 3: Python virtual environment ───────────────────────────────────────
step "Step 3: Setting Up Python Virtual Environment"

if [ ! -d "$VENV_DIR" ]; then
    info "Creating venv with --system-site-packages (for lgpio/gpiozero)..."
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

info "Installing Pi dependencies (no torch)..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$REPO_DIR/requirements-pi.txt" -q

info "Installing dashboard extras (fastapi, uvicorn, websockets)..."
"$VENV_DIR/bin/pip" install fastapi uvicorn websockets -q

info "Installing package in editable mode..."
"$VENV_DIR/bin/pip" install -e "$REPO_DIR" -q

success "Python environment ready."

# ── Step 4: Verify ONNX model ────────────────────────────────────────────────
step "Step 4: Verifying ONNX Model"

if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
    success "Model found: $MODEL_PATH ($MODEL_SIZE)"
else
    warn "Model not found at $MODEL_PATH"
    warn "Copy it from your Mac with:"
    warn "  scp /path/to/gtcrn_stream_int8.onnx grovestreet@<PI_IP>:$MODEL_PATH"
fi

# ── Step 5: Configure I2S MEMS microphone ────────────────────────────────────
step "Step 5: Configuring I2S MEMS Microphone (Adafruit SPH0645)"

CONFIG_FILE="/boot/firmware/config.txt"
# Older Pi OS puts it in /boot/config.txt
[ -f "$CONFIG_FILE" ] || CONFIG_FILE="/boot/config.txt"

I2S_MARKER="# SIH26052 I2S mic config"

if grep -q "$I2S_MARKER" "$CONFIG_FILE" 2>/dev/null; then
    success "I2S config already present in $CONFIG_FILE"
else
    info "Adding I2S overlay to $CONFIG_FILE..."
    sudo tee -a "$CONFIG_FILE" > /dev/null <<EOF

$I2S_MARKER
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
EOF
    success "I2S config added. Reboot required for it to take effect."
    NEEDS_REBOOT=true
fi

# ── Step 6: ALSA low-latency config ──────────────────────────────────────────
step "Step 6: Writing ALSA Configuration"

ASOUND_CONF="/etc/asound.conf"
ASOUND_MARKER="# SIH26052 ALSA config"

if grep -q "$ASOUND_MARKER" "$ASOUND_CONF" 2>/dev/null; then
    success "ALSA config already present."
else
    info "Writing ALSA config to $ASOUND_CONF..."
    sudo tee "$ASOUND_CONF" > /dev/null <<'EOF'
# SIH26052 ALSA config — Dual INMP441 I2S Mics + Quantron QSC-260 USB DAC

pcm.!default {
    type asym
    capture.pcm "mic_capture"
    playback.pcm "headphone_out"
}

# Dual INMP441 I2S MEMS microphone input (Left=Primary, Right=Reference)
pcm.mic_capture {
    type plug
    slave {
        pcm "hw:0,0"
        rate 16000
        channels 2
        format S32_LE
    }
}

# Quantron QSC-260 USB sound card output (Headphones)
pcm.headphone_out {
    type plug
    slave {
        pcm "hw:1,0"
        rate 16000
        channels 2
    }
}

ctl.!default {
    type hw
    card 1
}
EOF
    success "ALSA config written."
fi

# ── Step 7: Systemd service — Audio Loop ─────────────────────────────────────
step "Step 7: Installing Systemd Services"

# Audio loop service
AUDIO_SERVICE="/etc/systemd/system/sih-audio.service"
info "Creating audio loop service..."
sudo tee "$AUDIO_SERVICE" > /dev/null <<EOF
[Unit]
Description=SIH26052 Real-Time Speech Enhancer
Documentation=https://github.com/anant-shipit/SIH-ANC
After=sound.target
Wants=sound.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR
ExecStart=$VENV_DIR/bin/python3 -m sih26052.runtime.audio_loop \\
    --onnx $MODEL_PATH \\
    --impulse-gate
Restart=on-failure
RestartSec=5
Nice=-10
LimitRTPRIO=95
LimitMEMLOCK=infinity
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Dashboard service
DASH_SERVICE="/etc/systemd/system/sih-dashboard.service"
info "Creating dashboard service..."
sudo tee "$DASH_SERVICE" > /dev/null <<EOF
[Unit]
Description=SIH26052 Live Dashboard
Documentation=https://github.com/anant-shipit/SIH-ANC
After=network-online.target sih-audio.service
Wants=network-online.target
BindsTo=sih-audio.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$REPO_DIR
ExecStart=$VENV_DIR/bin/python3 -m sih26052.dashboard.server \\
    --host 0.0.0.0 \\
    --port 8080
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
success "Systemd services installed."

# ── Step 8: Enable/disable services based on model availability ───────────────
step "Step 8: Enabling Services"

if [ -f "$MODEL_PATH" ]; then
    info "Enabling and starting services..."
    sudo systemctl enable sih-audio.service sih-dashboard.service
    sudo systemctl start sih-audio.service || warn "Audio service failed to start — check 'journalctl -u sih-audio.service'"
    sudo systemctl start sih-dashboard.service || warn "Dashboard service failed to start — check 'journalctl -u sih-dashboard.service'"
    success "Services enabled and started."
else
    info "Enabling services (will start after model is copied + reboot)..."
    sudo systemctl enable sih-audio.service sih-dashboard.service
    warn "Services enabled but NOT started — model not found yet."
    warn "After copying the model, start with:"
    warn "  sudo systemctl start sih-audio.service sih-dashboard.service"
fi

# ── Step 9: CPU performance governor ─────────────────────────────────────────
step "Step 9: CPU Performance Tweaks"

info "Setting CPU governor to 'performance'..."
if ls /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor &>/dev/null; then
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
    success "CPU governor set to 'performance'."

    # Make it persistent across reboots
    GOVERNOR_CONF="/etc/systemd/system/cpu-performance.service"
    sudo tee "$GOVERNOR_CONF" > /dev/null <<'EOF'
[Unit]
Description=Set CPU governor to performance
After=sysinit.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl enable cpu-performance.service
    success "CPU governor set to 'performance' on every boot."
else
    warn "Could not set CPU governor (not a Pi or not supported)."
fi

# ── Step 10: Print audio device info ─────────────────────────────────────────
step "Step 10: Audio Device Summary"

echo ""
info "Detected ALSA playback devices:"
aplay -l 2>/dev/null || warn "aplay not available yet — devices will show after reboot"
echo ""
info "Detected ALSA capture devices:"
arecord -l 2>/dev/null || warn "arecord not available yet — devices will show after reboot"
echo ""

# ── Final Summary ─────────────────────────────────────────────────────────────
echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║        Pi Setup Complete! ✓              ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${RESET}"

echo -e "  ${BOLD}Useful commands on the Pi:${RESET}"
echo ""
echo "  # Check service status"
echo "  sudo systemctl status sih-audio.service"
echo "  sudo systemctl status sih-dashboard.service"
echo ""
echo "  # View live logs"
echo "  journalctl -u sih-audio.service -f"
echo ""
echo "  # Run manually (for debugging)"
echo "  cd $REPO_DIR && source venv/bin/activate"
echo "  python3 -m sih26052.runtime.audio_loop --list-devices"
echo "  python3 -m sih26052.runtime.audio_loop --onnx models/gtcrn_stream_int8.onnx"
echo ""
echo "  # Dashboard: open in browser"
echo "  http://$(hostname -I | awk '{print $1}'):8080"
echo ""

if [ "${NEEDS_REBOOT:-false}" = true ]; then
    echo -e "${YELLOW}${BOLD}  ⚠ REBOOT REQUIRED for I2S mic overlay to take effect!${RESET}"
    echo ""
    read -r -p "  Reboot now? [y/N]: " REBOOT_CHOICE
    if [[ "$REBOOT_CHOICE" =~ ^[Yy]$ ]]; then
        info "Rebooting..."
        sudo reboot
    else
        warn "Remember to reboot before testing audio: sudo reboot"
    fi
fi
