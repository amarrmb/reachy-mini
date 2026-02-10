#!/usr/bin/env bash
# Run Reachy Mini voice assistant demo.
#
# Prerequisites:
#   1. pip install reachy-mini[mujoco] Pillow
#   2. reachy-mini-daemon --sim  (in another terminal)
#   3. vLLM container running on port 8001 (see jetson-speech CLAUDE.md)
#   4. pip install -e ".[kokoro,nemotron,assistant,vision]" (in jetson-speech/)
#
# Usage: ./run.sh [extra jetson-speech args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Add bot/ to PYTHONPATH so reachy_tools is importable
export PYTHONPATH="${PROJECT_DIR}/bot:${PYTHONPATH:-}"

# CUDA 12 libs for flash-attn (if on Jetson with Ollama installed)
if [ -d "/usr/local/lib/ollama/cuda_v12" ]; then
    export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH:-}"
fi

echo "=== Reachy Mini Voice Assistant ==="
echo "Config: ${PROJECT_DIR}/configs/default.yaml"
echo "PYTHONPATH includes: ${PROJECT_DIR}/bot"
echo ""

exec jetson-speech assistant --config "${PROJECT_DIR}/configs/default.yaml" "$@"
