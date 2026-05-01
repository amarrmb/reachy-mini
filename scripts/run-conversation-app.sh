#!/usr/bin/env bash
# Launch the Reachy Mini conversation app on the Pi5.
#
# All inference is on-LAN (no cloud):
#   - Pi5: STT (distil-whisper), TTS (Kokoro), VAD, audio I/O via
#          reachy.media + fastrtc, motion via the daemon
#   - Thor (10.0.0.2): vLLM (Qwen2.5-VL-7B-Instruct-NVFP4) for the LLM
#
# The .env tells the app which endpoints to use; this launcher just
# activates the venv and invokes the app's CLI with the right flags
# for Reachy Mini Wireless on-device.

set -euo pipefail

REPO="${REPO:-/home/pollen/dn/conversation-app}"

if [ ! -d "$REPO/.venv" ]; then
    echo "ERROR: $REPO/.venv missing. Run install first." >&2
    exit 1
fi

if [ ! -f "$REPO/.env" ]; then
    echo "ERROR: $REPO/.env missing. Copy from /home/pollen/dn/reachy-mini/configs/conversation-app.env." >&2
    exit 1
fi

# Cleanup trap. The Reachy SDK opens a WebSocket that streams motor
# commands; if we get Ctrl-C'd mid-write, the motor serial bus can
# latch into a state where the daemon can't talk to motors anymore
# ("No motors detected on bus" → 503 from /api/motors/status until a
# physical power-cycle). Putting motors back into 'disabled' mode
# explicitly via the daemon's HTTP API before we exit avoids that.
DAEMON_BASE="${DAEMON_BASE:-http://localhost:8000}"
shutdown_clean() {
    echo "[run-conversation-app] cleanup: parking motors + releasing media..."
    # goto_sleep first (animates head to neutral while motors still enabled).
    curl -sS -m 3 -X POST "${DAEMON_BASE}/api/move/play/goto_sleep" >/dev/null 2>&1 || true
    sleep 1
    # Disable motor torque so they don't fight gravity at rest.
    curl -sS -m 3 -X POST "${DAEMON_BASE}/api/motors/set_mode/disabled" >/dev/null 2>&1 || true
    # Hand media back to the daemon so its own pipeline can resume.
    curl -sS -m 3 -X POST "${DAEMON_BASE}/api/media/acquire" >/dev/null 2>&1 || true
    echo "[run-conversation-app] cleanup done."
}
trap shutdown_clean EXIT INT TERM

# Wake the robot before the app takes over media. The conversation app
# never enables motors itself — without this, the daemon stays in
# `motors=disabled` / `backend.ready=false`, so the head never moves
# and tool calls like play_emotion/dance silently no-op.
echo "[run-conversation-app] waking robot..."
curl -sS -m 5 -X POST "${DAEMON_BASE}/api/motors/set_mode/enabled" >/dev/null 2>&1 || true
curl -sS -m 8 -X POST "${DAEMON_BASE}/api/move/play/wake_up" >/dev/null 2>&1 || true
sleep 1

# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
cd "$REPO"

# --wireless-version --on-device tells the app to use the GStreamer media
# backend that talks to the local Reachy daemon (no WebRTC indirection).
# Note: do NOT `exec` here — exec replaces the bash process and the EXIT
# trap above never fires, leaving motors latched after Ctrl-C.
reachy-mini-conversation-app \
    --wireless-version \
    --on-device \
    "$@"
