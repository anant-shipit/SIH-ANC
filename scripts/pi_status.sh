#!/usr/bin/env bash
# =============================================================================
# pi_status.sh — Quick health check for the SIH26052 system on the Pi.
#
# Run on the Pi (or via: ssh pi@sih-pi.local "~/SIH-ANC/scripts/pi_status.sh")
#
# Checks:
#   - ONNX model present and size
#   - Audio devices (aplay/arecord)
#   - sounddevice device list
#   - Service status (sih-audio, sih-dashboard)
#   - Recent xrun count from logs
#   - CPU temperature and frequency
#   - Memory usage
# =============================================================================
set -euo pipefail

REPO_DIR="$HOME/SIH-ANC"
VENV_DIR="$REPO_DIR/venv"
MODEL_PATH="$REPO_DIR/models/gtcrn_stream_int8.onnx"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'
YELLOW='\033[1;33m'; BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "  ${GREEN}✓${RESET} $*"; }
fail() { echo -e "  ${RED}✗${RESET} $*"; }
info() { echo -e "  ${CYAN}·${RESET} $*"; }
section() { echo -e "\n${BOLD}── $* ──${RESET}"; }

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║    SIH26052 — System Health Check       ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Model ────────────────────────────────────────────────────────────────────
section "ONNX Model"
if [ -f "$MODEL_PATH" ]; then
    SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
    ok "Model found: $MODEL_PATH ($SIZE)"
else
    fail "Model NOT found at $MODEL_PATH"
    info "Copy it from Mac: scp /path/to/gtcrn_stream_int8.onnx pi@$(hostname):$MODEL_PATH"
fi

# ── Audio devices ─────────────────────────────────────────────────────────────
section "Audio Devices (ALSA)"
echo "  Playback:"
aplay -l 2>/dev/null | grep "^card" | sed 's/^/    /' || info "  (none found)"
echo "  Capture:"
arecord -l 2>/dev/null | grep "^card" | sed 's/^/    /' || info "  (none found)"

# ── sounddevice device list ───────────────────────────────────────────────────
section "sounddevice Device List"
if [ -f "$VENV_DIR/bin/python3" ]; then
    "$VENV_DIR/bin/python3" -c "
import sounddevice as sd
devs = sd.query_devices()
for i, d in enumerate(devs):
    tag = '  IN ' if d['max_input_channels'] > 0 else '  OUT'
    tag = ' I/O' if d['max_input_channels'] > 0 and d['max_output_channels'] > 0 else tag
    print(f'  [{i:2d}]{tag}  {d[\"name\"]}  ({d[\"max_input_channels\"]}in/{d[\"max_output_channels\"]}out)')
" 2>/dev/null || info "  sounddevice not available in venv"
else
    info "  venv not found at $VENV_DIR"
fi

# ── Systemd services ─────────────────────────────────────────────────────────
section "Systemd Services"
for SVC in sih-audio sih-dashboard; do
    STATUS=$(systemctl is-active "${SVC}.service" 2>/dev/null || echo "not-found")
    ENABLED=$(systemctl is-enabled "${SVC}.service" 2>/dev/null || echo "not-found")
    if [ "$STATUS" = "active" ]; then
        ok "${SVC}: active (enabled: $ENABLED)"
    elif [ "$STATUS" = "not-found" ]; then
        info "${SVC}: not installed yet"
    else
        fail "${SVC}: $STATUS (enabled: $ENABLED)"
        info "  Check logs: journalctl -u ${SVC}.service --no-pager -n 20"
    fi
done

# ── Recent xruns ─────────────────────────────────────────────────────────────
section "Recent Xruns (last 50 log lines)"
XRUN_COUNT=$(journalctl -u sih-audio.service --no-pager -n 50 2>/dev/null | grep -c "Xruns:" || true)
if [ "$XRUN_COUNT" -gt 0 ]; then
    LAST_XRUN=$(journalctl -u sih-audio.service --no-pager -n 50 2>/dev/null | grep "Xruns:" | tail -1 || true)
    info "$LAST_XRUN"
else
    info "No xrun logs found (service may not be running)"
fi

# ── CPU temp & frequency ──────────────────────────────────────────────────────
section "CPU Temperature & Frequency"
if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
    TEMP_MC=$(cat /sys/class/thermal/thermal_zone0/temp)
    TEMP_C=$(echo "scale=1; $TEMP_MC / 1000" | bc)
    if (( TEMP_MC > 80000 )); then
        fail "CPU temp: ${TEMP_C}°C  ⚠ THROTTLING RISK"
    elif (( TEMP_MC > 70000 )); then
        info "CPU temp: ${YELLOW}${TEMP_C}°C${RESET}  (warm but OK)"
    else
        ok "CPU temp: ${TEMP_C}°C"
    fi
fi

if ls /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq &>/dev/null; then
    FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
    FREQ_GHZ=$(echo "scale=2; $FREQ / 1000000" | bc)
    GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
    ok "CPU: ${FREQ_GHZ} GHz  (governor: $GOV)"
fi

# ── Memory ────────────────────────────────────────────────────────────────────
section "Memory"
free -h | grep "Mem:" | awk '{printf "  Used: %s / %s (Available: %s)\n", $3, $2, $7}'

# ── Network (for dashboard access) ───────────────────────────────────────────
section "Network (Dashboard URL)"
IP=$(hostname -I | awk '{print $1}')
ok "Dashboard: http://${IP}:8080"
ok "SSH from Mac: ssh $(whoami)@${IP}"

echo ""
