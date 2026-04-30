# Reachy Mini Voice Assistant

Give your [Reachy Mini](https://www.pollen-robotics.com/reachy-mini/) a
voice — and a body. Fully offline on NVIDIA Jetson, zero cloud, zero
cost per query.

Built on top of [`amarrmb/jetson-assistant`](https://github.com/amarrmb/jetson-assistant) (the modular
voice + vision brain) and the upstream Pollen [`reachy_mini`](https://github.com/pollen-robotics/reachy_mini)
SDK. This repo is the Reachy integration: tool plugin + configs +
launchers + character profiles. About 400 lines of Python on top of
those two.

What you get:
- 9 motion tools (`look`, `dance`, `express`, `nod`, `set_antennas`,
  `look_at_point`, `reachy_power`, `reachy_see`, `reachy_status`)
  driving Pollen's curated emotion + dance libraries
- 5 character profiles out of the box: `curious_reachy`, `desk_philosopher`,
  `eager_explorer`, `deadpan_reachy`, `language_learner`
- Two deployment topologies (embodied or standalone — see below)
- Cleanup trap that puts motors in a safe state on Ctrl-C (a real
  failure mode without it)

<!-- TODO: replace with actual demo. See docs/demos/ for the four
     unique-capability shot lists. -->
<!-- ![Demo](docs/demo.gif) -->

## Architecture (two topologies, same brain)

**Embodied (recommended for real Reachy Mini Wireless):**
```
Reachy mic ─┐                                              ┌─→ Reachy speaker
            │  WS/HTTP                              WS/HTTP│
Reachy cam ─┼─────► Jetson Thor: Nemotron STT ─→ vLLM ─→ Kokoro TTS
            │       (jetson-assistant serve, port 8080)   │
Reachy Pi5 ─┴── tool calls ─→ Reachy SDK (localhost) ─→ motors
   (orchestrator + lifecycle + picamera2 + sounddevice)
```

**Standalone (any robot via WebSocket — sim or remote daemon):**
```
Jetson mic → Nemotron STT → Qwen2.5-VL-7B (vLLM) → Kokoro TTS → Jetson speaker
                                    ↓ tool calls
                              bot/reachy_tools.py
                                    ↓ WebSocket (reachy-mini SDK 1.7.0)
                            Reachy Mini daemon (sim or real)
```

## Try It

### Embodied — brain on Thor, I/O on Reachy (A-OnPi)

This is the most natural deployment for a real Reachy Mini Wireless: the
robot's onboard Pi5 captures audio + video, talks to Thor over the LAN
for inference, and plays the response back through Reachy's own speaker.
Thor stays mic/camera/speaker-free — pure compute.

**On Jetson Thor (model server):**
```bash
sudo sysctl -w vm.drop_caches=3
docker compose -f docker-compose.thor-serve.yml up -d   # ~12GB image already cached;
                                                         # vLLM/Nemotron/Kokoro preload ~5min
docker compose -f docker-compose.thor-serve.yml logs -f assistant-server
```

**On the Reachy Mini Pi5 (orchestrator):**
```bash
# One-time setup (rsync this repo + jetson-assistant to /home/pollen/dn/,
# then create the venv — see scripts/run-on-pi.sh for details).
sudo apt install python3-picamera2
python3 -m venv --system-site-packages /home/pollen/dn/orch-venv
/home/pollen/dn/orch-venv/bin/pip install httpx sounddevice openai opencv-python-headless \
    "reachy-mini>=1.7.0" pydantic fastapi typer pyyaml
/home/pollen/dn/orch-venv/bin/pip install --no-deps -e /home/pollen/dn/jetson-assistant
/home/pollen/dn/orch-venv/bin/pip install --no-deps -e /home/pollen/dn/reachy-mini

# Each session:
JA_SERVER_HOST=<thor-ip> /home/pollen/dn/reachy-mini/scripts/run-on-pi.sh
```

The launcher posts `POST /api/media/release` to the Reachy daemon (so it
can grab the camera + mic), runs the assistant pointed at Thor, and
posts `/api/media/acquire` on shutdown.

Now talk to Reachy directly — its onboard mic listens, its onboard
speaker replies, its head camera answers "what do you see?". Thor is
just a model server on the network.

### Standalone — Jetson does everything (legacy / sim)

Use this when you want the brain and I/O on the same Jetson and Reachy
is just a remote motion endpoint. Works with the MuJoCo simulator too.

**Step 1 — start the robot daemon:**

*Option A — Simulation (no robot needed):*
```bash
# On your laptop
pip install reachy-mini[mujoco]
reachy-mini-daemon --sim
```

*Option B — Real Reachy Mini:* Power it on — the daemon auto-starts on the robot.

**Step 2 — start the voice AI on Jetson:**

Set `REACHY_HOST` to wherever the daemon is running.

```bash
sudo sysctl -w vm.drop_caches=3

curl -fLO https://raw.githubusercontent.com/amarrmb/reachy-mini/main/docker-compose.yml
REACHY_HOST=<daemon-ip> docker compose up -d    # pulls ~12GB, vLLM loads model (~5 min)
docker compose logs -f reachy                    # watch it come up
```

> **Note:** vLLM may restart once on first boot due to CUDA graph memory allocation.
> This is normal — `restart: unless-stopped` handles it automatically. Wait ~5 minutes.

**Audio setup:** Set `ALSA_CARD` to your audio device name (find it with `aplay -l`):
```bash
ALSA_CARD=USB REACHY_HOST=<daemon-ip> docker compose up -d
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
| `camera_backend` | `v4l2` | `v4l2` (USB / cv2.VideoCapture) or `picamera2` (Pi CSI cameras — Reachy head cam) |
| `use_server` | `false` | When `true`, route STT/TTS/LLM to a remote `jetson-assistant serve` (used by the embodied A-OnPi deployment) |

Environment variables:

| Variable | Description |
|----------|-------------|
| `ALSA_CARD` | Audio device name from `aplay -l` (e.g. `USB`, `Jabra`) — Jetson-side audio |
| `REACHY_HOST` | Remote Reachy daemon hostname/IP (empty = `localhost`, used by the orchestrator on the Pi5) |
| `JA_SERVER_HOST` | Remote `jetson-assistant serve` host for the Pi5 launcher (default `10.0.0.2`) |
| `JA_SERVER_PORT` | Remote `jetson-assistant serve` port (default `8080`) |
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

The pre-built image (`ghcr.io/amarrmb/reachy-mini:thor`) works out of the box. If you want to modify the code and rebuild:

```bash
# On Jetson Thor (aarch64 only)

# Option A: Use the pre-built base image from ghcr.io
docker build -t reachy-mini:thor .

# Option B: Rebuild the base image too (if you modified jetson-assistant)
cd ../jetson-assistant
gh release download v0.1.0 -p 'flash_attn-*.whl' -D wheels/
docker build -t jetson-assistant:thor .
cd ../reachy-mini
docker build --build-arg BASE_IMAGE=jetson-assistant:thor -t reachy-mini:thor .
```

reachy-mini layers on top of `ghcr.io/amarrmb/jetson-assistant:thor`. The flash-attn wheel needed for building that base image is available from the [jetson-assistant v0.1.0 release](https://github.com/amarrmb/jetson-assistant/releases/tag/v0.1.0). To adapt for other hardware, adapt the base image first — see [jetson-assistant build docs](https://github.com/amarrmb/jetson-assistant#build-docker).

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

## PersonaPlex Mode (Full-Duplex Conversation)

Run [PersonaPlex](https://github.com/amarrmb/personaplex-oss) (Moshi 7B speech-to-speech) with audio-reactive Reachy Mini animations. No tool calling — the robot breathes, tilts attentively while listening, and sways its head in sync with PersonaPlex's speech.

```
Browser ──WebSocket──► PersonaPlex (Jetson GPU)
                           │ on_audio_frame
                           ▼
                       MotionManager (50Hz)
                           │ WebSocket (reachy-mini SDK)
                           ▼
                       Reachy Mini (sim or real)
```

### Quick Start

**Terminal 1 — Robot daemon** (laptop for sim, or real robot):
```bash
pip install "reachy-mini[mujoco]"
reachy-mini-daemon --sim --scene minimal
```

**Terminal 2 — PersonaPlex + Reachy bridge** (Jetson Thor):
```bash
python scripts/personaplex_reachy.py \
    --personaplex-dir ~/personaplex-oss \
    --port 8998 --fp8 \
    --reachy-host <laptop-ip>
```

Open the PersonaPlex Web UI at `https://<jetson-ip>:8998` and start talking. The robot reacts in real time:
- **Breathing** — subtle idle animation (always on)
- **Listening pose** — attentive head tilt when you speak
- **Audio-reactive sway** — head and antenna movement synced to PersonaPlex's speech

### Custom Backend (MuJoCo, Mock, etc.)

The bridge module is importable for custom integrations:

```python
from personaplex_bridge import setup_motion_manager, create_audio_bridge

# Any object with goto_target(head, antennas, duration) works
mm = setup_motion_manager(lambda: your_mujoco_reachy)
callback = create_audio_bridge(mm)

# Pass to PersonaPlex ServerState
state = ServerState(..., on_audio_frame=callback)
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Can't connect to Reachy | Ensure `reachy-mini-daemon --sim` is running and `REACHY_HOST` points to it |
| vLLM not responding | Takes ~5 min to load. Check: `curl http://localhost:8001/v1/models` |
| vLLM restarts on first boot | Normal — CUDA graph allocation may OOM once. It auto-recovers. |
| No audio output | Verify `/dev/snd` is accessible: `docker exec reachy-assistant python -m sounddevice` |
| Camera in use | Another process holds it — `pkill -f jetson-assistant` on the host |
| Container keeps restarting | Check logs: `docker compose logs reachy` |

## License

Apache 2.0 — See [LICENSE](LICENSE)

## Acknowledgments

- [Pollen Robotics](https://www.pollen-robotics.com/) — Reachy Mini hardware + SDK
- [jetson-assistant](https://github.com/amarrmb/jetson-assistant) — voice + vision engine
