# Reachy Mini Voice Assistant

Fully offline voice control for [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) robots. Sub-second STT + LLM + TTS pipeline running entirely on NVIDIA Jetson Thor — zero cloud, zero cost per query.

```
Mic → Nemotron STT (24ms) → Qwen2.5-VL-7B (vLLM) → Kokoro TTS → Speaker
                                    ↓ tool calls
                              bot/reachy_tools.py
                                    ↓ Zenoh
                            Reachy Mini (sim or real)
```

## Features

- **9 robot tools** — look, express, dance, nod, reachy_see, reachy_power, set_antennas, look_at_point, reachy_status
- **50Hz MotionManager** — breathing, sway, and listening animations
- **Multi-camera vision** — Reachy camera, USB camera, phone via Aether WebRTC
- **Extensible** — add a tool in one function (see `examples/weather_tool.py`)
- **Dual-robot mode** — primary speaks, follower mirrors emotions + dances

## Prerequisites

| Component | Install |
|-----------|---------|
| vLLM container | See [Start vLLM](#start-vllm) below |
| jetson-assistant | `cd ~/jetson-assistant && pip install -e ".[kokoro,nemotron,assistant,vision]"` |
| reachy-mini SDK | `pip install reachy-mini[mujoco] Pillow` |
| espeak-ng | `apt-get install espeak-ng` |

## Quick Start

### Terminal 1 — Start vLLM

```bash
sudo sysctl -w vm.drop_caches=3
docker run -d --rm --runtime=nvidia --network host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --name vllm ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor \
    vllm serve nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 \
    --host 0.0.0.0 --port 8001 \
    --max-model-len 4096 --gpu-memory-utilization 0.3
```

Wait ~5 minutes, then verify: `curl http://localhost:8001/v1/models`

### Terminal 2 — Start Reachy daemon

```bash
reachy-mini-daemon --sim          # MuJoCo simulation
# or: REACHY_HOST=<ip> for remote real robot
```

### Terminal 3 — Run the assistant

```bash
# Basic mode
scripts/run.sh

# Full demo (auto-starts daemon + Aether Hub)
scripts/run-full-demo.sh

# GTC booth demo (MotionManager + proactive greetings)
scripts/run-gtc-demo.sh
```

## Project Structure

```
reachy-mini/
├── README.md                 # This file
├── .env.template             # Environment variables template
├── .gitignore
├── requirements.txt          # Python dependencies
│
├── bot/                      # Robot control modules
│   ├── reachy_tools.py       # 9 tools: look, express, dance, nod, see, ...
│   ├── motion_manager.py     # 50Hz MotionManager (breathing, sway, listen)
│   ├── reachy_connect.py     # Shared SDK connection helper (Zenoh patches)
│   └── follower.py           # Dual-robot follower (mirrors primary's state)
│
├── configs/                  # Configuration profiles
│   ├── default.yaml          # Basic config (Kokoro + Nemotron + vLLM)
│   ├── full-demo.yaml        # Full demo (+ Aether Hub, phone camera)
│   └── gtc-demo.yaml         # GTC booth (+ MotionManager, greetings)
│
├── scripts/                  # Launcher scripts
│   ├── run.sh                # Basic launcher
│   ├── run-full-demo.sh      # Full demo (starts all services)
│   ├── run-gtc-demo.sh       # GTC demo (5-act booth script)
│   ├── run-dual-demo.sh      # Dual-robot launcher
│   └── recover-demo.sh       # Kill + restart everything
│
├── examples/                 # Extensibility demos
│   └── weather_tool.py       # "Add a tool in 60 seconds"
│
├── docs/                     # Supporting docs
│   ├── NVIDIA_POSITIONING.md # GTC booth positioning
│   └── LIVE_TOOL_DEMO.md     # Live tool demo script
│
└── tests/
    └── test_e2e.py           # NL → vLLM → tool → robot test
```

## How It Works

This demo uses jetson-assistant's **external tool plugin system**. The `bot/reachy_tools.py` module registers tools with the assistant's tool registry. The LLM decides when to call them based on conversation context.

### Tool Plugin Contract

```python
# bot/reachy_tools.py (simplified)
from typing import Annotated

def register_tools(registry, context=None):
    @registry.register("Move Reachy's head to look in a direction")
    def look(
        direction: Annotated[str, "left, right, up, down, or center"],
    ) -> str:
        # Control robot via Zenoh → reachy-mini-daemon
        return "Looking left"

def cleanup():
    """Called on assistant shutdown."""
    pass
```

Every tool:
- Takes typed, annotated parameters
- Returns a string (success message or error)
- Never raises exceptions

### Adding a Custom Tool

1. Create a Python file with `register_tools(registry, context)`:

```python
# my_tool.py
from typing import Annotated

def register_tools(registry, context=None):
    @registry.register("Description shown to the LLM")
    def my_action(param: Annotated[str, "What this param does"]) -> str:
        # Your logic here
        return "Done"
```

2. Add it to your config YAML:

```yaml
external_tools:
  - reachy_tools
  - my_tool
```

3. Make sure the file is on PYTHONPATH and restart.

See `examples/weather_tool.py` for a complete example.

## Configuration

Configs live in `configs/`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `tts_backend` | `kokoro` | TTS engine (`kokoro`, `vllm`) |
| `tts_voice` | `af_heart` | Kokoro voice ID |
| `stt_backend` | `nemotron` | STT engine (`nemotron`, `whisper`, `vllm`) |
| `llm_backend` | `vllm` | LLM backend |
| `llm_model` | `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4` | Model name |
| `external_tools` | `[reachy_tools]` | Tool plugin modules |
| `stream_llm` | `true` | Stream LLM output for pipelined TTS |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `REACHY_HOST` | Remote Reachy daemon IP (empty = localhost) |
| `BOOTH_MODE=1` | Enable proactive booth greetings |
| `REACHY_BROADCAST=1` | Enable UDP state broadcast for dual-robot mode |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Cannot connect to Reachy Mini` | Make sure `reachy-mini-daemon --sim` is running |
| `ModuleNotFoundError: reachy_tools` | Use launcher scripts, or set `PYTHONPATH=bot` |
| vLLM not responding | `curl http://localhost:8001/v1/models` — takes ~5min to start |
| No audio output | Run from a real terminal (not SSH nohup). Check `python -m sounddevice` |
| Camera in use | Kill `jetson-assistant` if it holds the camera: `pkill -f jetson-assistant` |
| Demo crashed mid-run | `scripts/recover-demo.sh` kills + restarts everything |
| vLLM OOM | `scripts/recover-demo.sh --vllm-only` clears GPU memory and restarts |

## Voice Commands

```
"Look to the left"              → head movement
"Show me you're happy!"         → emotional expression
"Do a dance"                    → choreographed sequence
"What do you see?"              → VLM vision (Reachy camera)
"Check the local camera"        → VLM vision (USB camera)
"Watch for someone approaching" → proactive vision monitoring
"Nod yes" / "Shake your head"   → gesture
"Go to sleep" / "Wake up"       → power control
"Set a timer for 30 seconds"    → concurrent tool
"Can you speak Japanese?"       → Kokoro multi-language TTS
```
