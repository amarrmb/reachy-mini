#!/usr/bin/env bash
# One-shot installer for the Pi5-side orchestrator (A-OnPi deployment).
#
# Run this on the Reachy Mini's onboard Pi5 (login: pollen).
# Prerequisite: the reachy-mini and jetson-assistant repos already
# rsync'd to /home/pollen/dn/.
#
# Usage:
#     /home/pollen/dn/reachy-mini/scripts/install-on-pi.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/pollen/dn/reachy-mini}"
JA_DIR="${JA_DIR:-/home/pollen/dn/jetson-assistant}"
ORCH_VENV="${ORCH_VENV:-/home/pollen/dn/orch-venv}"

if [ ! -d "$REPO_DIR" ] || [ ! -d "$JA_DIR" ]; then
    echo "ERROR: Source dirs missing. Expected $REPO_DIR and $JA_DIR." >&2
    echo "Rsync them from your dev machine first." >&2
    exit 1
fi

echo "[install] system packages (need root)"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-picamera2 \
    python3-libcamera

echo "[install] creating venv: $ORCH_VENV"
python3 -m venv --system-site-packages "$ORCH_VENV"

# shellcheck disable=SC1091
source "$ORCH_VENV/bin/activate"

echo "[install] pip deps (no-cache to spare disk)"
pip install --no-cache-dir -U pip
pip install --no-cache-dir \
    "reachy-mini>=1.7.0" \
    httpx \
    sounddevice \
    webrtcvad \
    openai \
    pydantic \
    fastapi \
    "uvicorn[standard]" \
    websockets \
    typer \
    pyyaml \
    scipy \
    opencv-python-headless \
    Pillow

echo "[install] editable installs (no transitive deps — already pinned above)"
pip install --no-cache-dir --no-deps -e "$JA_DIR"
pip install --no-cache-dir --no-deps -e "$REPO_DIR"

echo "[install] sanity check"
PYTHONPATH="$REPO_DIR/bot" python -c "
import reachy_mini, sounddevice, picamera2, httpx, cv2
import jetson_assistant, reachy_tools, motion_manager
print('OK — reachy_mini', reachy_mini.__version__,
      'sounddevice', sounddevice.__version__,
      'picamera2', picamera2.__version__,
      'jetson_assistant', jetson_assistant.__file__)
"

echo
echo "[install] DONE."
echo "  Run with:  JA_SERVER_HOST=<thor-ip> $REPO_DIR/scripts/run-on-pi.sh"
