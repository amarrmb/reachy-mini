#!/usr/bin/env bash
#
# Reachy Mini — GTC Demo Recovery Script
#
# One-command restart after any failure. Kills stale processes,
# clears GPU memory, restarts services in correct order.
#
# USAGE:
#   ./recover-demo.sh              # Full recovery (vLLM + daemon + assistant)
#   ./recover-demo.sh --vllm-only  # Only restart vLLM container
#   ./recover-demo.sh --quick      # Skip vLLM (restart daemon + assistant only)
#
# After running, wait for "ALL SERVICES READY" then run:
#   ./run-gtc-demo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

RESTART_VLLM=true
RESTART_OTHER=true

for arg in "$@"; do
    case "$arg" in
        --vllm-only)  RESTART_OTHER=false ;;
        --quick)      RESTART_VLLM=false ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           GTC DEMO — RECOVERY                           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Kill stale processes ──
echo "  [1/4] Killing stale processes..."

if [ "$RESTART_OTHER" = true ]; then
    # Kill assistant (jetson-assistant)
    if pkill -f "jetson-assistant assistant" 2>/dev/null; then
        echo "    Killed: jetson-assistant assistant"
    fi

    # Kill reachy-mini-daemon
    if pkill -f "reachy-mini-daemon" 2>/dev/null; then
        echo "    Killed: reachy-mini-daemon"
    fi

    # Kill Aether Hub
    if pkill -f "aether.*hub" 2>/dev/null; then
        echo "    Killed: Aether Hub"
    fi
fi

if [ "$RESTART_VLLM" = true ]; then
    # Stop vLLM container
    if docker stop vllm 2>/dev/null; then
        echo "    Stopped: vLLM container"
    fi
    sleep 2
fi

echo "    Done."

# ── Step 2: Clear GPU memory ──
if [ "$RESTART_VLLM" = true ]; then
    echo ""
    echo "  [2/4] Clearing GPU memory..."
    sudo sysctl -w vm.drop_caches=3 > /dev/null 2>&1 || true
    echo "    Cache cleared."
fi

# ── Step 3: Restart vLLM ──
if [ "$RESTART_VLLM" = true ]; then
    echo ""
    echo "  [3/4] Starting vLLM container..."
    echo "    Model: nvidia/Qwen2.5-VL-7B-Instruct-NVFP4"
    echo "    This takes ~5 minutes on first load."

    docker run -d --rm --runtime=nvidia --network host --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --name vllm ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
        vllm serve nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 \
        --host 0.0.0.0 --port 8001 \
        --max-model-len 4096 --gpu-memory-utilization 0.3

    echo ""
    echo "    Waiting for vLLM to load..."
    WAIT_COUNT=0
    MAX_WAIT=120  # 4 minutes max (120 × 2s)
    while ! curl -s http://localhost:8001/v1/models > /dev/null 2>&1; do
        if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
            echo ""
            echo "    ERROR: vLLM did not become ready in 4 minutes."
            echo "    Check: docker logs vllm"
            exit 1
        fi
        echo -n "."
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 1))
    done
    echo ""

    MODEL=$(curl -s http://localhost:8001/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "unknown")
    echo "    vLLM ready: $MODEL"
else
    echo ""
    echo "  [2/4] GPU memory... SKIP (--quick)"
    echo "  [3/4] vLLM... SKIP (--quick)"
fi

# ── Step 4: Verify readiness ──
echo ""
echo "  [4/4] Verifying services..."

READY=true

# Check vLLM
echo -n "    vLLM (:8001)... "
if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo "OK"
else
    echo "NOT READY"
    READY=false
fi

echo ""
if [ "$READY" = true ]; then
    echo "  ════════════════════════════════════════════"
    echo "  ALL SERVICES READY. Run the demo:"
    echo ""
    echo "    cd ${PROJECT_DIR}"
    echo "    scripts/run-gtc-demo.sh"
    echo ""
    echo "  (run-gtc-demo.sh will start daemon + Hub)"
    echo "  ════════════════════════════════════════════"
else
    echo "  WARNING: Some services not ready."
    echo "  Check: docker logs vllm"
fi
echo ""
