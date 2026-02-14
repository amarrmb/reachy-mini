# Reachy Mini Voice Assistant

Give your [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) a voice.
Fully offline on NVIDIA Jetson Thor — zero cloud, zero cost per query.

<!-- TODO: replace with actual demo GIF recorded on Thor -->
<!-- ![Demo](docs/demo.gif) -->

```
Mic → Nemotron STT (24ms) → Qwen2.5-VL-7B (vLLM) → Kokoro TTS → Speaker
                                    ↓ tool calls
                              bot/reachy_tools.py
                                    ↓ Zenoh
                            Reachy Mini (sim or real)
```

9 robot tools, 50Hz MotionManager, multi-camera vision. Sub-second end-to-end.

## Try It

### What you need

| Machine | Runs | Why |
|---------|------|-----|
| **Jetson Thor** | Voice AI (STT + LLM + TTS) | GPU required for inference |
| **Your laptop** | Reachy Mini daemon (sim or real) | Needs a display for MuJoCo UI |

### Step 1: Start the voice AI on Jetson

```bash
# SSH into your Jetson Thor
git clone https://github.com/amarrmb/reachy-mini.git
cd reachy-mini

sudo sysctl -w vm.drop_caches=3

docker compose up -d
# vLLM loads the model (~5 min), then the assistant starts
docker compose logs -f reachy
```

### Step 2: Start the robot (pick one)

**Option A — MuJoCo simulation (no robot needed):**

On your laptop:
```bash
pip install reachy-mini[mujoco]
reachy-mini-daemon --sim
```

A MuJoCo window opens showing the simulated Reachy Mini.

**Option B — Real Reachy Mini:**

The daemon runs on the robot itself. Just power it on — it auto-starts.

### Step 3: Connect them

Set `REACHY_HOST` on the Jetson to point at your laptop/robot IP:

```bash
# Stop and restart with the host set
docker compose down
REACHY_HOST=<laptop-ip> docker compose up -d
```

Or for localhost (if daemon runs on the same Jetson):
```bash
docker compose up -d    # auto-discovers on localhost
```

Plug in a mic and speaker on the Jetson. Say "Look to the left",
"Show me you're happy!", or "Do a dance".

To stop: `docker compose down`

## Like It? Make It Yours

```bash
# Clone both repos
git clone https://github.com/amarrmb/jetson-assistant.git
git clone https://github.com/amarrmb/reachy-mini.git

# Install jetson-assistant first (dependency)
cd jetson-assistant && pip install -e ".[kokoro,nemotron,assistant,vision]"

# Install reachy-mini
cd ../reachy-mini && pip install -e .

# Start vLLM + reachy daemon, then run
docker compose up -d vllm
reachy-mini-daemon --sim &
./scripts/run.sh
```

## Voice Commands

```
"Look to the left"              → head movement
"Show me you're happy!"         → emotional expression
"Do a dance"                    → choreographed sequence
"What do you see?"              → VLM vision (Reachy camera)
"Nod yes" / "Shake your head"   → gesture
"Go to sleep" / "Wake up"       → power control
"Set a timer for 30 seconds"    → concurrent tool
```

## Adding Custom Tools

Create a Python file with `register_tools()`:

```python
# my_tool.py
from typing import Annotated

def register_tools(registry, context=None):
    @registry.register("Description shown to the LLM")
    def my_action(param: Annotated[str, "What this param does"]) -> str:
        return "Done"
```

Add it to your config YAML and restart:

```yaml
external_tools:
  - reachy_tools
  - my_tool
```

See [`examples/weather_tool.py`](examples/weather_tool.py) for a complete example.

## Platforms

| Platform | Docker Tag | Status |
|----------|-----------|--------|
| Jetson Thor (JetPack 7+) | `:thor` | Tested |
| AGX Orin 64GB | — | Community welcome |

### Adapting for Your Hardware

This project depends on [jetson-assistant](https://github.com/amarrmb/jetson-assistant).
Adapt that Dockerfile first, then this one layers on top.

---

## Features

- **9 robot tools** — look, express, dance, nod, reachy_see, reachy_power, set_antennas, look_at_point, reachy_status
- **50Hz MotionManager** — breathing, sway, and listening animations
- **Multi-camera vision** — Reachy camera, USB camera, phone via Aether WebRTC
- **Extensible** — add a tool in one function
- **Dual-robot mode** — primary speaks, follower mirrors emotions + dances

## Project Structure

```
reachy-mini/
├── bot/                      # Robot control modules
│   ├── reachy_tools.py       # 9 tools: look, express, dance, nod, see, ...
│   ├── motion_manager.py     # 50Hz MotionManager (breathing, sway, listen)
│   ├── reachy_connect.py     # Shared SDK connection helper
│   └── follower.py           # Dual-robot follower
├── configs/                  # Configuration profiles
│   ├── default.yaml          # Basic config (Kokoro + Nemotron + vLLM)
│   ├── full-demo.yaml        # + Aether Hub, phone camera
│   └── gtc-demo.yaml         # + MotionManager, booth greetings
├── scripts/                  # Launcher scripts
├── examples/                 # Tool plugin examples
└── tests/
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `tts_backend` | `kokoro` | TTS engine |
| `stt_backend` | `nemotron` | STT engine |
| `llm_backend` | `vllm` | LLM backend |
| `external_tools` | `[reachy_tools]` | Tool plugin modules |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `REACHY_HOST` | Remote Reachy daemon IP (empty = localhost) |
| `BOOTH_MODE=1` | Enable proactive booth greetings |
| `REACHY_BROADCAST=1` | Enable UDP state broadcast for dual-robot mode |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Cannot connect to Reachy Mini` | Make sure `reachy-mini-daemon --sim` is running on your laptop or robot |
| `ModuleNotFoundError: reachy_tools` | Use launcher scripts, or set `PYTHONPATH=bot` |
| vLLM not responding | `curl http://localhost:8001/v1/models` — takes ~5min to start |
| No audio output | Check `python -m sounddevice`. Run from a real terminal. |
| Camera in use | `pkill -f jetson-assistant` if it holds the camera |
| Demo crashed | `scripts/recover-demo.sh` kills + restarts everything |

## License

Apache 2.0 — See [LICENSE](LICENSE)

Built with [jetson-assistant](https://github.com/amarrmb/jetson-assistant) by [DeviceNexus.ai](https://devicenexus.ai).
