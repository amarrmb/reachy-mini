#!/usr/bin/env bash
# Run the voice assistant ON the Reachy Mini Pi5.
#
# This is the "A-OnPi" deployment:
#   - Reachy's onboard mic, camera, speaker do all I/O
#   - Heavy compute (STT/TTS/LLM) lives on a remote jetson-assistant server
#     (e.g. Jetson Thor at $JA_SERVER_HOST)
#
# Prerequisites on the Pi:
#   1. reachy-mini daemon running (it auto-starts as a systemd service)
#   2. Orchestrator venv installed at $ORCH_VENV (default: /home/pollen/dn/orch-venv)
#   3. Both repos rsync'd to /home/pollen/dn/{jetson-assistant,reachy-mini}
#
# Prerequisites on the server (Thor):
#   1. `jetson-assistant serve --port 8080` running with vLLM/Nemotron/Kokoro
#      preloaded.

set -euo pipefail

# --- config ---
ORCH_VENV="${ORCH_VENV:-/home/pollen/dn/orch-venv}"
REPO_DIR="${REPO_DIR:-/home/pollen/dn/reachy-mini}"
JA_DIR="${JA_DIR:-/home/pollen/dn/jetson-assistant}"
JA_SERVER_HOST="${JA_SERVER_HOST:-10.0.0.2}"   # Thor by default
JA_SERVER_PORT="${JA_SERVER_PORT:-8080}"
DAEMON_BASE="${DAEMON_BASE:-http://localhost:8000}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_DIR}/configs/pi-orchestrator.yaml}"

# --- daemon media handoff ---
release_media() {
    echo "[run-on-pi] POST ${DAEMON_BASE}/api/media/release"
    curl -sS -X POST "${DAEMON_BASE}/api/media/release" || {
        echo "[run-on-pi] WARNING: /api/media/release failed; mic/camera may already be released"
    }
    echo
}

acquire_media() {
    echo "[run-on-pi] POST ${DAEMON_BASE}/api/media/acquire"
    curl -sS -X POST "${DAEMON_BASE}/api/media/acquire" || true
    echo
}

trap acquire_media EXIT INT TERM

release_media

# --- env ---
# bot/ on PYTHONPATH so jetson-assistant's `--external-tools reachy_tools`
# (importlib.import_module) resolves to bot/reachy_tools.py.
export PYTHONPATH="${REPO_DIR}/bot:${PYTHONPATH:-}"

source "${ORCH_VENV}/bin/activate"

# Route TTS / chime aplay calls through the daemon's shared dmix sink.
# The Pi's ~/.asoundrc points the ALSA `default` PCM straight at hw:0,0
# (exclusive) — which the daemon already holds. `plug:reachymini_audio_sink`
# is a daemon-provided dmix node with sample-rate conversion, so multiple
# processes can stream into it simultaneously.
export JA_APLAY_DEVICE="${JA_APLAY_DEVICE:-plug:reachymini_audio_sink}"

echo "[run-on-pi] config:    ${CONFIG_FILE}"
echo "[run-on-pi] server:    ${JA_SERVER_HOST}:${JA_SERVER_PORT}"
echo "[run-on-pi] aplay:     ${JA_APLAY_DEVICE}"
echo

exec jetson-assistant assistant \
    --config "${CONFIG_FILE}" \
    --server \
    --server-host "${JA_SERVER_HOST}" \
    --server-port "${JA_SERVER_PORT}" \
    --camera-backend picamera2 \
    "$@"
