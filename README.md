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

Jetson Thor runs voice AI (needs GPU). The robot daemon runs separately — either as a simulator on your laptop or on the real robot.

### Step 1: Start the robot daemon

**Option A — Simulation (no robot needed):**
```bash
# On your laptop
pip install reachy-mini[mujoco]
reachy-mini-daemon --sim
```

**Option B — Real Reachy Mini:** Power it on — the daemon auto-starts on the robot.

### Step 2: Start voice AI on Jetson

Set `REACHY_HOST` to wherever the daemon is running (your laptop IP for sim, or the robot IP for real hardware).

```bash
sudo sysctl -w vm.drop_caches=3

curl -fLO https://raw.githubusercontent.com/amarrmb/reachy-mini/main/docker-compose.yml
REACHY_HOST=<daemon-ip> docker compose up -d    # pulls ~12GB, vLLM loads model (~5 min)
docker compose logs -f reachy                    # watch it come up
```

Plug in a mic and speaker on the Jetson, and start talking:

```
"Look to the left"                → head movement (left, right, up, down, center)
"Show me you're happy!"           → emotional expression (happy, sad, surprised, angry, ...)
"Do a dance"                      → choreographed dance sequence
"What do you see?"                → VLM describes what Reachy's camera sees
"Nod yes" / "Shake your head"     → agree/disagree gesture
"Go to sleep" / "Wake up"         → power on/off
"Set your antennas up"            → direct antenna control (-90 to 90 degrees)
"Look at the object on the table" → precise 3D gaze control
"Are you connected?"              → robot connection status
"Set a timer for 30 seconds"      → spoken countdown alert
"What time is it?"                → built-in clock
```

All [jetson-assistant](https://github.com/amarrmb/jetson-assistant) tools work too (web search, memory, language switching, multi-camera).

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
| Can't connect to Reachy | Ensure `reachy-mini-daemon --sim` is running and `REACHY_HOST` points to it |
| vLLM not responding | Takes ~5 min to load. Check: `curl http://localhost:8001/v1/models` |
| No audio output | Verify `/dev/snd` is accessible: `docker exec reachy-assistant python -m sounddevice` |
| Camera in use | Another process holds it — `pkill -f jetson-assistant` on the host |
| Container keeps restarting | Check logs: `docker compose logs reachy` |

## License

Apache 2.0 — See [LICENSE](LICENSE)

## Acknowledgments

- [Pollen Robotics](https://www.pollen-robotics.com/) — Reachy Mini hardware + SDK
- [jetson-assistant](https://github.com/amarrmb/jetson-assistant) — voice + vision engine

Built by [DeviceNexus.ai](https://devicenexus.ai).
