#!/usr/bin/env bash
# =============================================================================
# deploy_to_pi.sh — Run this on your MAC to deploy SIH26052 to the Pi.
#
#   Clones the `pi` branch (includes all INT8 ONNX models) directly on the
#   Pi over SSH, then runs setup_pi.sh to install system packages + services.
#   No checkpoint export or model scp required — models ship in the branch.
#
# Usage:
#   chmod +x scripts/deploy_to_pi.sh
#   ./scripts/deploy_to_pi.sh
#
# Optional flags:
#   --pi-host   <hostname>   default: sih-pi.local
#   --pi-user   <username>   default: grovestreet
#   --model     <filename>   INT8 ONNX file to use (default: gtcrn_finetuned_stream_int8.onnx)
# =============================================================================
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
PI_HOST="sih-pi.local"
PI_USER="grovestreet"
REPO_URL="https://github.com/anant-shipit/SIH-ANC.git"
BRANCH="pi"
REMOTE_DIR="~/SIH-ANC"
MODEL="gtcrn_finetuned_stream_int8.onnx"

# ── Color output ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
step()    { echo -e "\n${BOLD}━━━ $* ━━━${RESET}"; }

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --pi-host) PI_HOST="$2"; shift 2 ;;
        --pi-user) PI_USER="$2"; shift 2 ;;
        --model)   MODEL="$2";   shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--pi-host <host>] [--pi-user <user>] [--model <int8.onnx>]"
            exit 0 ;;
        *) error "Unknown argument: $1" ;;
    esac
done

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   SIH26052 — Pi 5 Deployment Script     ║"
echo "  ║   Branch: pi  •  No export required     ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${RESET}"
info "Target:  $PI_USER@$PI_HOST"
info "Branch:  $BRANCH (includes all INT8 ONNX models)"
info "Model:   $MODEL"

# ── Phase 1: SSH Key Setup ───────────────────────────────────────────────────
step "Phase 1: SSH Connection"

info "Testing SSH connection..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$PI_USER@$PI_HOST" "echo ok" &>/dev/null; then
    warn "Passwordless SSH not configured. Setting up key auth..."

    if [ ! -f "$HOME/.ssh/id_rsa" ] && [ ! -f "$HOME/.ssh/id_ed25519" ]; then
        info "Generating SSH key..."
        ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519" -N "" -C "sih26052-deploy"
    fi

    info "Copying SSH key to Pi (enter Pi password once)..."
    ssh-copy-id "$PI_USER@$PI_HOST"
    success "SSH key installed — future connections will be passwordless."
fi
success "SSH connection OK"

# ── Phase 2: Install system deps on Pi ──────────────────────────────────────
step "Phase 2: System Dependencies (apt)"

info "Installing libportaudio2 and system packages on Pi..."
ssh "$PI_USER@$PI_HOST" "sudo apt-get update -qq && sudo apt-get install -y \
    libportaudio2 portaudio19-dev \
    git python3-venv python3-dev \
    python3-lgpio python3-gpiozero \
    alsa-utils libasound2-dev \
    i2c-tools"
success "System packages installed"

# ── Phase 3: Clone pi branch on Pi ──────────────────────────────────────────
step "Phase 3: Clone / Update Repository (pi branch)"

ssh "$PI_USER@$PI_HOST" "
    if [ -d $REMOTE_DIR/.git ]; then
        echo '[INFO] Repo exists — pulling latest pi branch...'
        cd $REMOTE_DIR
        git fetch origin
        git checkout pi
        git reset --hard origin/pi
    else
        echo '[INFO] Cloning pi branch...'
        git clone --branch pi --single-branch $REPO_URL $REMOTE_DIR
    fi
"
success "Repository ready at $REMOTE_DIR"

# ── Phase 4: Python venv + pip install ──────────────────────────────────────
step "Phase 4: Python Environment"

ssh "$PI_USER@$PI_HOST" "
    cd $REMOTE_DIR
    if [ ! -d venv ]; then
        python3 -m venv --system-site-packages venv
    fi
    source venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements-pi.txt
    pip install --quiet -e .
    echo '[OK] pip install complete'
"
success "Python venv ready"

# ── Phase 5: I2S + ALSA Config ──────────────────────────────────────────────
step "Phase 5: I2S Overlay & ALSA Config"

info "Copying ALSA config..."
ssh "$PI_USER@$PI_HOST" "
    if ! grep -q 'dtparam=i2s=on' /boot/firmware/config.txt 2>/dev/null; then
        echo 'dtparam=i2s=on' | sudo tee -a /boot/firmware/config.txt
        echo 'dtoverlay=googlevoicehat-soundcard' | sudo tee -a /boot/firmware/config.txt
        echo '[INFO] I2S overlay added — reboot required'
    else
        echo '[INFO] I2S overlay already enabled'
    fi
"

ssh "$PI_USER@$PI_HOST" "
sudo tee /etc/asound.conf > /dev/null << 'EOF'
# SIH26052 ALSA config — Dual INMP441 I2S + Quantron QSC-260 USB DAC
pcm.!default {
    type asym
    capture.pcm  \"mic_capture\"
    playback.pcm \"headphone_out\"
}
pcm.mic_capture {
    type plug
    slave { pcm \"hw:0,0\"; rate 16000; channels 2; format S32_LE }
}
pcm.headphone_out {
    type plug
    slave { pcm \"hw:1,0\"; rate 16000; channels 2 }
}
ctl.!default { type hw; card 1 }
EOF
echo '[OK] /etc/asound.conf written'
"
success "ALSA config applied"

# ── Phase 6: CPU Performance Governor ───────────────────────────────────────
step "Phase 6: CPU Governor"

ssh "$PI_USER@$PI_HOST" "
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
    echo '[OK] CPU governor set to performance'
"
success "CPU governor: performance"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║         Deployment Complete! ✓           ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "  ${BOLD}Next steps on Pi ($PI_USER@$PI_HOST):${RESET}"
echo "  1. Reboot if I2S overlay was just added:  sudo reboot"
echo "  2. Verify mics:  arecord -l && aplay -l"
echo "  3. Run audio loop:"
echo "       cd $REMOTE_DIR && source venv/bin/activate"
echo "       python3 -m sih26052.runtime.audio_loop --onnx models/$MODEL --native-sr 48000"
echo "  4. Run dashboard: python3 -m sih26052.dashboard.server --port 8080"
echo "  5. Open browser:  http://$PI_HOST:8080"
echo ""
