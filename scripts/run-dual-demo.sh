#!/usr/bin/env bash
#
# Dual Reachy Mini Demo — Two robots, one voice
#
# Primary robot speaks and interacts. Follower mirrors emotions,
# dances, and listening poses with complementary motions.
#
# USAGE:
#   # Primary on daemon at 192.168.0.29, follower at 192.168.0.30:
#   REACHY_HOST=192.168.0.29 REACHY_HOST_2=192.168.0.30 ./run-dual-demo.sh
#
#   # Simulation mode (both on localhost, different prefixes):
#   ./run-dual-demo.sh --sim
#
# PREREQUISITES:
#   1. Two reachy-mini-daemon instances running (different machines or different prefixes)
#   2. vLLM container running on :8001
#   3. pip install reachy-mini[mujoco] (in jetson-speech venv)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

REACHY_HOST_2="${REACHY_HOST_2:-}"

if [ -z "$REACHY_HOST_2" ]; then
    echo "ERROR: Set REACHY_HOST_2=<ip> for the follower robot daemon."
    echo ""
    echo "Usage:"
    echo "  REACHY_HOST=<primary-ip> REACHY_HOST_2=<follower-ip> $0"
    exit 1
fi

# Activate venv
VENV_PATH="${JETSON_SPEECH_VENV:-$HOME/jetson-speech/.venv}"
if [ -d "$VENV_PATH" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "ERROR: jetson-speech venv not found"
    exit 1
fi

export PYTHONPATH="${PROJECT_DIR}/bot:${PYTHONPATH:-}"

if [ -d "/usr/local/lib/ollama/cuda_v12" ]; then
    export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH:-}"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        DUAL REACHY DEMO — Two Robots, One Voice             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Primary:  ${REACHY_HOST:-localhost} (speaks + interacts)"
echo "  Follower: ${REACHY_HOST_2} (mirrors + reacts)"
echo "  Broadcast: UDP :5555"
echo ""

# Start follower in background
echo "Starting follower (${REACHY_HOST_2})..."
python3 "${PROJECT_DIR}/bot/follower.py" --host "${REACHY_HOST_2}" &
FOLLOWER_PID=$!
echo "  Follower PID: ${FOLLOWER_PID}"

# Give follower time to connect
sleep 2

# Cleanup on exit
cleanup() {
    echo ""
    echo "Stopping follower (PID ${FOLLOWER_PID})..."
    kill "$FOLLOWER_PID" 2>/dev/null || true
    wait "$FOLLOWER_PID" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# Start primary with broadcast enabled
echo "Starting primary (${REACHY_HOST:-localhost})..."
echo ""
export REACHY_BROADCAST=1
exec "${SCRIPT_DIR}/run-gtc-demo.sh" "$@"
