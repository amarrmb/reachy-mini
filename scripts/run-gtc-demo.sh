#!/usr/bin/env bash
#
# Reachy Mini — GTC 2026 Demo Launcher
#
# "Same Robot. Better Brain. Zero Cloud."
#
# Single-command launcher for the complete GTC demo:
#   1. Validates vLLM container is ready
#   2. Starts reachy-mini-daemon (sim or real)
#   3. Starts Aether Hub (phone camera + alerts)
#   4. Launches jetson-assistant with GTC config + MotionManager
#
# USAGE:
#   # Simulation mode (MuJoCo on this machine):
#   ./run-gtc-demo.sh
#
#   # Remote sim (e.g., laptop at 192.168.0.29):
#   REACHY_HOST=192.168.0.29 ./run-gtc-demo.sh
#
#   # Real robot (daemon on robot's machine):
#   REACHY_HOST=<robot-ip> ./run-gtc-demo.sh --no-daemon
#
#   # Skip components:
#   ./run-gtc-demo.sh --no-daemon     # Don't start reachy-mini-daemon
#   ./run-gtc-demo.sh --no-hub        # Don't start Aether Hub
#   ./run-gtc-demo.sh --no-phone      # Disable phone camera
#
# PREREQUISITES:
#   1. vLLM container running on :8001 (start with recover-demo.sh if needed)
#   2. pip install -e ".[kokoro,nemotron,assistant,vision]" (in jetson-assistant/)
#   3. pip install reachy-mini[mujoco] Pillow (for sim mode)
#   4. apt-get install espeak-ng (required by Kokoro TTS)

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

# Activate jetson-assistant venv
VENV_PATH="${JETSON_SPEECH_VENV:-$HOME/jetson-assistant/.venv}"
if [ -d "$VENV_PATH" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "ERROR: jetson-assistant venv not found at ~/jetson-assistant/.venv"
    echo "Install: cd ~/jetson-assistant && python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[kokoro,nemotron,assistant,vision]'"
    exit 1
fi

# Add bot/ to PYTHONPATH (for reachy_tools + motion_manager imports)
export PYTHONPATH="${PROJECT_DIR}/bot:${PYTHONPATH:-}"

# CUDA 12 libs for flash-attn / Kokoro (Jetson with Ollama installed)
if [ -d "/usr/local/lib/ollama/cuda_v12" ]; then
    export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH:-}"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           BASKD × REACHY MINI — GTC 2026 DEMO              ║"
echo "║       \"Same Robot. Better Brain. Zero Cloud.\"               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Check 1: vLLM ──
echo -n "  [1/3] vLLM on :8001... "
VLLM_RETRIES=0
while ! curl -s http://localhost:8001/v1/models > /dev/null 2>&1; do
    if [ $VLLM_RETRIES -ge 3 ]; then
        echo "NOT READY"
        echo ""
        echo "  vLLM is not running or still loading. Options:"
        echo "    1. Wait for it to finish loading (~5 min)"
        echo "    2. Start it: scripts/recover-demo.sh --vllm-only"
        echo "    3. Start everything: scripts/recover-demo.sh"
        exit 1
    fi
    echo -n "."
    sleep 2
    VLLM_RETRIES=$((VLLM_RETRIES + 1))
done
MODEL=$(curl -s http://localhost:8001/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")
echo "OK ($MODEL)"

# ── Check 2: Aether Hub ──
if [ "$START_HUB" = true ]; then
    echo -n "  [2/3] Aether Hub on :8000... "
    if pgrep -f "aether.*hub" > /dev/null 2>&1; then
        echo "OK (running)"
    else
        echo -n "starting... "
        AETHER_DIR="${AETHER_DIR:-$HOME/aether}"
        if [ -d "$AETHER_DIR" ]; then
            (cd "$AETHER_DIR" && nohup go run ./cmd/hub > /tmp/aether-hub.log 2>&1 &)
            sleep 2
            if pgrep -f "aether.*hub" > /dev/null 2>&1; then
                echo "OK"
            else
                echo "WARN (failed — phone camera unavailable)"
                ENABLE_PHONE=false
            fi
        else
            echo "SKIP ($AETHER_DIR not found)"
            ENABLE_PHONE=false
        fi
    fi
else
    echo "  [2/3] Aether Hub... SKIP (--no-hub)"
fi

# ── Check 3: Reachy Mini daemon ──
REACHY_HOST_VAL="${REACHY_HOST:-}"
if [ "$START_DAEMON" = true ]; then
    echo -n "  [3/3] Reachy daemon... "
    if pgrep -f "reachy-mini-daemon" > /dev/null 2>&1; then
        echo "OK (PID $(pgrep -f reachy-mini-daemon | head -1))"
    elif [ -n "$REACHY_HOST_VAL" ]; then
        echo "REMOTE ($REACHY_HOST_VAL)"
    else
        echo -n "starting MuJoCo sim... "
        nohup reachy-mini-daemon --sim > /tmp/reachy-daemon.log 2>&1 &
        DAEMON_PID=$!
        sleep 3
        if kill -0 "$DAEMON_PID" 2>/dev/null; then
            echo "OK (PID $DAEMON_PID)"
        else
            echo "WARN (failed — install: pip install reachy-mini[mujoco])"
        fi
    fi
else
    echo "  [3/3] Reachy daemon... SKIP (--no-daemon)"
fi

# ── Build CLI overrides ──
CLI_OVERRIDES=()

if [ "$START_HUB" = true ]; then
    CLI_OVERRIDES+=(--aether-hub "localhost:8000")
fi

if [ "$ENABLE_PHONE" = false ]; then
    CLI_OVERRIDES+=(--remote-camera-port 0)
fi

IP_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│  Pipeline: Nemotron STT → vLLM VLM → Kokoro TTS             │"
echo "│  Voice:    af_heart (Kokoro, 24kHz, near-human)              │"
echo "│  Motion:   MotionManager 50Hz (breathing + sway + listening) │"
echo "│  Tools:    built-in (20+) + reachy_tools (8 robot actions)   │"
echo "│  Vision:   USB camera + phone (Aether WebRTC)                │"
echo "│  Preview:  http://${IP_ADDR}:9090/                           │"
echo "│  Reachy:   ${REACHY_HOST_VAL:-localhost} (Zenoh)             │"
echo "│  VRAM:     ~8GB / 128GB                                      │"
echo "│  Network:  NOT REQUIRED                                      │"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  GTC DEMO SCRIPT — 5 Acts, ~3 minutes"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  ACT 1 — ALIVE (30s):"
echo "    Robot wakes, breathes, sways while speaking."
echo '    Say: "Hey Reachy!" → notice head sway + breathing'
echo ""
echo "  ACT 2 — SMART (60s):"
echo '    "What do you see?"          → VLM vision'
echo '    "Set a timer for 30 seconds" → concurrent tool'
echo '    "Watch for someone approaching" → proactive vision'
echo ""
echo "  ACT 3 — POLYGLOT (30s):"
echo '    "Can you speak Japanese?"   → Kokoro switches language'
echo '    "Say hello in Japanese"     → near-human Japanese TTS'
echo ""
echo "  ACT 4 — UNKILLABLE (30s):"
echo '    [Unplug WiFi / disable network]'
echo '    "What time is it?"          → answers instantly, no cloud'
echo ""
echo "  ACT 5 — EXTENSIBLE (30s):"
echo '    Show reachy_tools.py — "Adding a tool is one function."'
echo '    "This works on any robot. Reachy today, yours tomorrow."'
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Recovery: scripts/recover-demo.sh"
echo "  Press Ctrl+C to stop."
echo ""

# ── Launch assistant ──
exec jetson-assistant assistant \
    --config "${PROJECT_DIR}/configs/gtc-demo.yaml" \
    ${CLI_OVERRIDES[@]+"${CLI_OVERRIDES[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
