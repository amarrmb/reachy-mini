#!/usr/bin/env bash
#
# Reachy Mini — Full Demo Launcher
#
# Starts everything needed for the complete Reachy Mini voice assistant demo:
#   1. Checks prerequisites (vLLM, Aether Hub, reachy-mini-daemon)
#   2. Starts reachy-mini-daemon if not running (MuJoCo sim or real robot)
#   3. Starts Aether Hub if not running (for phone camera + alerts)
#   4. Launches jetson-speech assistant with full config
#
# ┌──────────────────────────────────────────────────────────────────┐
# │                        ARCHITECTURE                              │
# │                                                                  │
# │  JETSON THOR (this machine)                                      │
# │  ├── vLLM container (:8001) — Qwen2.5-VL-7B NVFP4              │
# │  ├── Aether Hub (:8000) — phone camera relay + alerts           │
# │  └── jetson-speech assistant                                     │
# │      ├── Nemotron STT (in-process, ~24ms)                       │
# │      ├── Kokoro TTS (in-process, <300ms, af_heart voice)        │
# │      ├── vLLM VLM backend (vision + text)                       │
# │      ├── Local USB camera → CameraPool "local"                  │
# │      ├── Phone camera (UDP :5001) → CameraPool "phone"          │
# │      ├── Built-in tools (time, search, remember, cameras, etc.) │
# │      └── Reachy tools (look, express, dance, see, nod, power)   │
# │          └── Zenoh → Reachy daemon (local or remote)            │
# │                                                                  │
# │  LAPTOP / DESKTOP (optional, for MuJoCo sim)                    │
# │  └── reachy-mini-daemon --sim --no-localhost-only                │
# │      └── MuJoCo window showing Reachy Mini                      │
# │                                                                  │
# │  PHONE (optional)                                                │
# │  └── Aether mobile app → WebRTC → Hub → UDP :5001               │
# └──────────────────────────────────────────────────────────────────┘
#
# USAGE:
#   # Simulation mode (daemon on this machine):
#   ./run-full-demo.sh
#
#   # Simulation on remote laptop (e.g., 192.168.0.29):
#   REACHY_HOST=192.168.0.29 ./run-full-demo.sh
#
#   # Real robot (daemon runs on robot's machine):
#   REACHY_HOST=<robot-ip> ./run-full-demo.sh --no-daemon
#
#   # Skip specific components:
#   ./run-full-demo.sh --no-daemon      # Don't start reachy-mini-daemon
#   ./run-full-demo.sh --no-hub         # Don't start Aether Hub
#   ./run-full-demo.sh --no-phone       # Disable phone camera (Aether WebRTC)
#
# PREREQUISITES:
#   1. vLLM container running on :8001 (see below)
#   2. pip install -e ".[kokoro,nemotron,assistant,vision]" (in jetson-speech/)
#   3. pip install reachy-mini[mujoco] Pillow (for sim mode)
#   4. apt-get install espeak-ng (required by Kokoro TTS)
#
# START VLLM (run once, takes ~5min to load):
#   sudo sysctl -w vm.drop_caches=3
#   docker run -d --rm --runtime=nvidia --network host --ipc=host \
#       --ulimit memlock=-1 --ulimit stack=67108864 \
#       -v ~/.cache/huggingface:/root/.cache/huggingface \
#       --name vllm ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
#       vllm serve nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 \
#       --host 0.0.0.0 --port 8001 \
#       --max-model-len 4096 --gpu-memory-utilization 0.3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Parse flags ──
START_DAEMON=true
START_HUB=true
ENABLE_PHONE=true
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --no-daemon)  START_DAEMON=false ;;
        --no-hub)     START_HUB=false ;;
        --no-phone)   ENABLE_PHONE=false ;;
        *)            EXTRA_ARGS+=("$arg") ;;
    esac
done

# ── Environment ──

# Activate jetson-speech venv (provides the `jetson-speech` CLI command)
VENV_PATH="${JETSON_SPEECH_VENV:-$HOME/jetson-speech/.venv}"
if [ -d "$VENV_PATH" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "ERROR: jetson-speech venv not found at ~/jetson-speech/.venv"
    echo "Install: cd ~/jetson-speech && python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[kokoro,nemotron,assistant,vision]'"
    exit 1
fi

# Add bot/ to PYTHONPATH (for reachy_tools import)
export PYTHONPATH="${PROJECT_DIR}/bot:${PYTHONPATH:-}"

# CUDA 12 libs for flash-attn / Kokoro (Jetson with Ollama installed)
if [ -d "/usr/local/lib/ollama/cuda_v12" ]; then
    export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH:-}"
fi

echo "============================================================"
echo "  Reachy Mini — Full Demo"
echo "============================================================"
echo ""

# ── Check 1: vLLM ──
echo -n "Checking vLLM on :8001... "
if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    MODEL=$(curl -s http://localhost:8001/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")
    echo "OK ($MODEL)"
else
    echo "FAILED"
    echo ""
    echo "  vLLM is not running. Start it first:"
    echo ""
    echo "    sudo sysctl -w vm.drop_caches=3"
    echo "    docker run -d --rm --runtime=nvidia --network host --ipc=host \\"
    echo "        --ulimit memlock=-1 --ulimit stack=67108864 \\"
    echo "        -v ~/.cache/huggingface:/root/.cache/huggingface \\"
    echo "        --name vllm ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \\"
    echo "        vllm serve nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 \\"
    echo "        --host 0.0.0.0 --port 8001 \\"
    echo "        --max-model-len 4096 --gpu-memory-utilization 0.3"
    echo ""
    echo "  Wait ~5 minutes for model loading, then re-run this script."
    exit 1
fi

# ── Check 2: Aether Hub ──
if [ "$START_HUB" = true ]; then
    echo -n "Checking Aether Hub on :8000... "
    if pgrep -f "aether.*hub" > /dev/null 2>&1; then
        echo "OK (already running)"
    else
        echo "starting..."
        AETHER_DIR="${AETHER_DIR:-$HOME/aether}"
        if [ -d "$AETHER_DIR" ]; then
            (cd "$AETHER_DIR" && nohup go run ./cmd/hub > /tmp/aether-hub.log 2>&1 &)
            sleep 2
            if pgrep -f "aether.*hub" > /dev/null 2>&1; then
                echo "  Aether Hub started (log: /tmp/aether-hub.log)"
            else
                echo "  WARNING: Aether Hub failed to start. Phone camera and alerts will be unavailable."
                echo "  Check /tmp/aether-hub.log for details."
                ENABLE_PHONE=false
            fi
        else
            echo "  WARNING: $AETHER_DIR not found. Skipping Aether Hub."
            ENABLE_PHONE=false
        fi
    fi
fi

# ── Check 3: Reachy Mini daemon ──
REACHY_HOST_VAL="${REACHY_HOST:-}"
if [ "$START_DAEMON" = true ]; then
    echo -n "Checking reachy-mini-daemon... "
    if pgrep -f "reachy-mini-daemon" > /dev/null 2>&1; then
        echo "OK (already running, PID $(pgrep -f reachy-mini-daemon | head -1))"
    elif [ -n "$REACHY_HOST_VAL" ]; then
        echo "REMOTE (REACHY_HOST=$REACHY_HOST_VAL)"
        echo "  Make sure reachy-mini-daemon is running on $REACHY_HOST_VAL"
    else
        echo "starting (MuJoCo sim)..."
        nohup reachy-mini-daemon --sim > /tmp/reachy-daemon.log 2>&1 &
        DAEMON_PID=$!
        sleep 3
        if kill -0 "$DAEMON_PID" 2>/dev/null; then
            echo "  MuJoCo sim started (PID $DAEMON_PID, log: /tmp/reachy-daemon.log)"
        else
            echo "  WARNING: reachy-mini-daemon failed to start."
            echo "  Check /tmp/reachy-daemon.log for details."
            echo "  Install with: pip install reachy-mini[mujoco]"
        fi
    fi
fi

# ── Build CLI overrides ──
CLI_OVERRIDES=()

# Aether Hub for camera alerts + remote commands (not supported via YAML resolution)
if [ "$START_HUB" = true ]; then
    CLI_OVERRIDES+=(--aether-hub "localhost:8000")
fi

# Remote camera port override (--remote-camera-port 0 disables phone camera)
if [ "$ENABLE_PHONE" = false ]; then
    CLI_OVERRIDES+=(--remote-camera-port 0)
fi

echo ""
echo "────────────────────────────────────────────────────────────"
echo "  Pipeline: Nemotron STT → vLLM VLM → Kokoro TTS"
echo "  Voice:    af_heart (Kokoro, 24kHz)"
echo "  Tools:    built-in + reachy_tools (8 robot actions)"
echo "  Vision:   local USB camera + phone camera (Aether WebRTC)"
echo "  Preview:  http://$(hostname -I | awk '{print $1}'):9090/"
echo "  REACHY:   ${REACHY_HOST_VAL:-localhost} (Zenoh)"
echo "────────────────────────────────────────────────────────────"
echo ""
echo "Voice commands to try:"
echo '  "Look to the left"'
echo '  "Show me you'\''re happy!"'
echo '  "Do a dance"'
echo '  "Nod yes"'
echo '  "What do you see?" (uses Reachy camera)'
echo '  "Check the local camera" (uses Jetson USB camera)'
echo '  "Check the phone camera" (uses phone via Aether)'
echo '  "Watch the local camera for a person"'
echo '  "What time is it?"'
echo '  "Remember my name is Alex"'
echo '  "Search the web for NVIDIA GTC 2026"'
echo '  "Go to sleep"'
echo '  "Wake up!"'
echo ""
echo "Press Ctrl+C to stop."
echo ""

# ── Launch assistant ──
exec jetson-speech assistant \
    --config "${PROJECT_DIR}/configs/full-demo.yaml" \
    ${CLI_OVERRIDES[@]+"${CLI_OVERRIDES[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
