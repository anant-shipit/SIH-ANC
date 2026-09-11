#!/usr/bin/env bash
# =============================================================================
# deploy_to_pi.sh — Run this on your MAC to:
#   1. Export your .pth checkpoint to streaming ONNX
#   2. Quantize to int8
#   3. SCP the model + setup script to the Pi
#   4. SSH into the Pi and run the setup automatically
#
# Usage:
#   chmod +x scripts/deploy_to_pi.sh
#   ./scripts/deploy_to_pi.sh --checkpoint models/checkpoints/model.pth
#
# Optional flags:
#   --pi-host   <hostname>   default: sih-pi.local
#   --pi-user   <username>   default: pi
#   --skip-export            skip ONNX export if model already exists
# =============================================================================
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
PI_HOST="sih-pi.local"
PI_USER="grovestreet"
CHECKPOINT=""
SKIP_EXPORT=false
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ONNX_FP32="$REPO_DIR/models/gtcrn_stream.onnx"
ONNX_INT8="$REPO_DIR/models/gtcrn_stream_int8.onnx"

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
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --pi-host)    PI_HOST="$2";    shift 2 ;;
        --pi-user)    PI_USER="$2";    shift 2 ;;
        --skip-export) SKIP_EXPORT=true; shift ;;
        -h|--help)
            echo "Usage: $0 --checkpoint <path.pth> [--pi-host <host>] [--pi-user <user>] [--skip-export]"
            exit 0 ;;
        *) error "Unknown argument: $1" ;;
    esac
done

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   SIH26052 — Pi 5 Deployment Script     ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Phase 1: Export ONNX (on Mac) ───────────────────────────────────────────
step "Phase 1: ONNX Export & Quantization (Mac)"

if [ "$SKIP_EXPORT" = true ]; then
    warn "--skip-export set. Skipping ONNX export."
    [ -f "$ONNX_INT8" ] || error "No int8 model found at $ONNX_INT8. Cannot skip."
    success "Using existing model: $ONNX_INT8"
else
    [ -n "$CHECKPOINT" ] || error "No --checkpoint provided. Run with --checkpoint path/to/model.pth"
    [ -f "$CHECKPOINT" ] || error "Checkpoint not found: $CHECKPOINT"

    info "Activating virtual environment..."
    VENV="$REPO_DIR/venv"
    [ -d "$VENV" ] || error "No venv found at $VENV. Run: python3 -m venv venv && pip install -r requirements.txt"
    source "$VENV/bin/activate"

    info "Exporting checkpoint → streaming ONNX..."
    python3 -m sih26052.export.to_onnx \
        --checkpoint "$CHECKPOINT" \
        --output "$ONNX_FP32"
    success "Exported: $ONNX_FP32"

    info "Quantizing to dynamic int8..."
    python3 -m sih26052.export.quantize \
        --input "$ONNX_FP32" \
        --output "$ONNX_INT8"
    success "Quantized: $ONNX_INT8"

    ONNX_SIZE=$(du -sh "$ONNX_INT8" | cut -f1)
    info "Final model size: $ONNX_SIZE"
fi

# ── Phase 2: Check Pi Connectivity ──────────────────────────────────────────
step "Phase 2: Connecting to Pi ($PI_USER@$PI_HOST)"

info "Testing SSH connection..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$PI_USER@$PI_HOST" "echo ok" &>/dev/null; then
    echo ""
    warn "Cannot SSH into $PI_USER@$PI_HOST without a password prompt."
    warn "Setting up SSH key authentication first..."
    echo ""

    # Generate SSH key if not present
    if [ ! -f "$HOME/.ssh/id_rsa" ] && [ ! -f "$HOME/.ssh/id_ed25519" ]; then
        info "Generating SSH key..."
        ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519" -N "" -C "sih26052-deploy"
    fi

    info "Copying SSH key to Pi (you'll need to enter the Pi password once)..."
    ssh-copy-id "$PI_USER@$PI_HOST"
    success "SSH key installed. Future connections will be passwordless."
fi

success "SSH connection to $PI_USER@$PI_HOST works!"

# ── Phase 3: Copy Files to Pi ────────────────────────────────────────────────
step "Phase 3: Copying Files to Pi"

info "Creating directory structure on Pi..."
ssh "$PI_USER@$PI_HOST" "mkdir -p ~/SIH-ANC/models ~/SIH-ANC/scripts"

info "Copying int8 ONNX model..."
scp "$ONNX_INT8" "$PI_USER@$PI_HOST:~/SIH-ANC/models/"
success "Model copied."

info "Copying Pi setup script..."
scp "$REPO_DIR/scripts/setup_pi.sh" "$PI_USER@$PI_HOST:~/SIH-ANC/scripts/"
ssh "$PI_USER@$PI_HOST" "chmod +x ~/SIH-ANC/scripts/setup_pi.sh"
success "Setup script copied."

# ── Phase 4: Run Setup on Pi ─────────────────────────────────────────────────
step "Phase 4: Running Setup on Pi"

info "Launching Pi setup script over SSH..."
info "(This will take a few minutes — installing packages, cloning repo, setting up services)"
echo ""

ssh -t "$PI_USER@$PI_HOST" "~/SIH-ANC/scripts/setup_pi.sh"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║         Deployment Complete! ✓           ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "  ${BOLD}Next steps:${RESET}"
echo "  1. Wire your I2S mic if you haven't already (see README or guide)"
echo "  2. SSH into the Pi:  ssh $PI_USER@$PI_HOST"
echo "  3. Run audio loop:   python3 -m sih26052.runtime.audio_loop --onnx models/gtcrn_stream_int8.onnx"
echo "  4. Run dashboard:    python3 -m sih26052.dashboard.server --port 8080"
echo "  5. Open browser:     http://$PI_HOST:8080"
echo ""
