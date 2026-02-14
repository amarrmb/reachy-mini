# Reachy Mini Voice Assistant

Give your [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) a voice.
Fully offline on NVIDIA Jetson Thor — zero cloud, zero cost per query.

```
Mic → Nemotron STT (24ms) → Qwen2.5-VL-7B (vLLM) → Kokoro TTS → Speaker
                                    ↓ tool calls
                              bot/reachy_tools.py
                                    ↓ Zenoh
                            Reachy Mini (sim or real)
```

9 robot tools, 50Hz MotionManager, multi-camera vision. Sub-second end-to-end.

<!-- TODO: replace with actual demo -->
<!-- ![Demo](docs/demo.gif) -->

## Try It

Two machines required: Jetson Thor runs voice AI (needs GPU), your laptop runs the robot daemon (needs a display for MuJoCo).

### Step 1: Voice AI on Jetson

```bash
git clone https://github.com/amarrmb/reachy-mini.git
cd reachy-mini

sudo sysctl -w vm.drop_caches=3

docker compose up -d                    # pulls ~12GB, vLLM loads model (~5 min)
docker compose logs -f reachy           # watch it come up
```

### Step 2: Robot daemon on your laptop

**Simulation (no robot needed):**
```bash
pip install reachy-mini[mujoco]
reachy-mini-daemon --sim
```

**Real Reachy Mini:** Just power it on — daemon auto-starts.

### Step 3: Connect them

```bash
docker compose down
REACHY_HOST=<laptop-ip> docker compose up -d
```

Plug in a mic and speaker on the Jetson. Say "Look to the left", "Show me you're happy!", or "Do a dance".

To stop: `docker compose down`

## Make It Yours

### Options

Configured via YAML files in `configs/`:

| Setting | Default | Description |
|---------|---------|-------------|
| `tts_backend` | `kokoro` | TTS engine |
| `stt_backend` | `nemotron` | STT engine |
| `llm_backend` | `vllm` | LLM backend |
| `external_tools` | `[reachy_tools]` | Tool plugin modules |

Environment variables:

| Variable | Description |
|----------|-------------|
| `REACHY_HOST` | Remote daemon IP (empty = localhost) |
| `BOOTH_MODE=1` | Proactive booth greetings |
| `REACHY_BROADCAST=1` | UDP broadcast for dual-robot mode |

Voice commands:

```
"Look to the left"              → head movement
"Show me you're happy!"         → emotional expression
"Do a dance"                    → choreographed sequence
"What do you see?"              → VLM vision
"Nod yes" / "Shake your head"   → gesture
"Go to sleep" / "Wake up"       → power control
```

### Build Locally

```bash
# Clone both repos (reachy-mini depends on jetson-assistant)
git clone https://github.com/amarrmb/jetson-assistant.git
git clone https://github.com/amarrmb/reachy-mini.git

# Install jetson-assistant first
cd jetson-assistant && pip install -e ".[kokoro,nemotron,assistant,vision]"

# Install reachy-mini
cd ../reachy-mini && pip install -e .

# Start vLLM + daemon, then run
docker compose up -d vllm
reachy-mini-daemon --sim &
./scripts/run.sh
```

### Build Docker

```bash
# On Jetson Thor (aarch64 only)
# Requires jetson-assistant:thor as base image
docker build -t reachy-mini:thor .
```

This layers on top of `ghcr.io/amarrmb/jetson-assistant:thor`. To adapt for other hardware, adapt that image first.

### Extend It

Add a tool in one function:

```python
# my_tool.py
from typing import Annotated

def register_tools(registry, context=None):
    @registry.register("Description shown to the LLM")
    def my_action(param: Annotated[str, "What this param does"]) -> str:
        return "Done"
```

Add to your config YAML:
```yaml
external_tools:
  - reachy_tools
  - my_tool
```

See `examples/weather_tool.py` for a complete example.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Can't connect to Reachy | Make sure `reachy-mini-daemon --sim` is running on your laptop |
| `ModuleNotFoundError: reachy_tools` | Use launcher scripts, or set `PYTHONPATH=bot` |
| vLLM not responding | `curl http://localhost:8001/v1/models` — takes ~5min to start |
| No audio | Check `python -m sounddevice` — run from a real terminal |
| Camera in use | `pkill -f jetson-assistant` if another process holds it |

## License

Apache 2.0 — See [LICENSE](LICENSE)

## Acknowledgments

- [Pollen Robotics](https://www.pollen-robotics.com/) — Reachy Mini hardware + SDK
- [jetson-assistant](https://github.com/amarrmb/jetson-assistant) — voice + vision engine

Built by [DeviceNexus.ai](https://devicenexus.ai).
