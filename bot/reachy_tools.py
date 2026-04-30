"""
External tool plugin for Reachy Mini robot control via jetson-assistant.

Provides voice-controlled robot actions: look, express emotions, dance,
see through camera, power on/off, nod/shake, set antennas, look at point.

All motion goes through MotionManager (50Hz background thread) for:
- Non-blocking tool execution (dance doesn't freeze speech)
- Idle breathing animation (always-on secondary motion)
- Audio-reactive head sway (during SPEAKING state)
- Reactive listening poses (during LISTENING/PROCESSING)

Usage:
    jetson-assistant assistant --external-tools reachy_tools ...

Or in config.yaml:
    external_tools:
      - reachy_tools

Requires: pip install reachy-mini[mujoco]

SDK API notes (reachy-mini 1.3.0):
  - goto_target(head=<4x4 np matrix>, antennas=[right_rad, left_rad], duration=0.5)
  - create_head_pose(yaw=, pitch=, roll=, degrees=True) -> 4x4 matrix
  - Antennas are [right, left] in RADIANS
  - look_at_world(x, y, z, duration=1.0) -- 3D gaze control
  - set_target_antenna_joint_positions([right_rad, left_rad])
"""

import base64
import math
import os
import random
import sys
import threading
import time
from typing import Annotated, Optional

import urllib.request
import urllib.error

from motion_manager import (
    AudioReactiveSway,
    BreathingMotion,
    MotionManager,
    MotionSequence,
    Pose,
    ReactiveListeningMotion,
    single_pose,
)
from moves_catalog import (
    DANCES_DATASET,
    EMOTIONS_DATASET,
    MovesCatalog,
    encoded_dataset_path,
)


_DAEMON_BASE = os.environ.get("REACHY_DAEMON_BASE", "http://localhost:8000")
_moves = MovesCatalog()


def _daemon_post(path: str, timeout: float = 2.0) -> Optional[dict]:
    """Fire-and-forget POST to the daemon's HTTP API.

    The daemon's WebSocket SDK doesn't expose recorded-move playback yet,
    so we go through HTTP. The orchestrator already has a connection to
    the daemon for motor / pose commands; this is just a parallel channel
    for the move library.
    """
    url = _DAEMON_BASE + path
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            try:
                import json
                return json.loads(data) if data else {}
            except Exception:
                return {"raw": data.decode(errors="replace")}
    except urllib.error.HTTPError as e:
        logger_msg = f"daemon POST {path} -> {e.code} {e.reason}"
        print(logger_msg, file=sys.stderr)
        return None
    except Exception as e:
        print(f"daemon POST {path} -> {e}", file=sys.stderr)
        return None


def _ensure_motors_enabled() -> bool:
    """Enable motors if currently disabled. Returns True if motors are now enabled.

    Recorded moves only render visibly when motors are in 'enabled' mode.
    The daemon launches with --no-wake-up-on-start so motors are off until
    we wake. Calling express/dance before wake_up should still produce
    motion, so we lift them silently here.
    """
    try:
        with urllib.request.urlopen(_DAEMON_BASE + "/api/motors/status", timeout=1.0) as r:
            import json
            data = json.loads(r.read())
        if data.get("mode") == "enabled":
            return True
    except Exception:
        pass
    res = _daemon_post("/api/motors/set_mode/enabled")
    return res is not None

# Lazy connection -- avoid import errors if reachy-mini not installed
_reachy = None
_reachy_lock = threading.Lock()
_llm_ref = None  # Stored reference to LLM backend for reachy_see
_camera_pool_ref = None  # CameraPool from jetson-assistant context
_say_ref = None  # say() callback from assistant for proactive speech
_motion_manager: Optional[MotionManager] = None
_booth_greeter = None  # BoothGreeter instance (when BOOTH_MODE=1)
_broadcast_sock = None  # UDP socket for dual-Reachy state broadcast
_broadcast_addr = None  # ("255.255.255.255", 5555)


def _get_reachy():
    """Lazy-connect to Reachy Mini SDK. Thread-safe, connects once.

    Set REACHY_HOST=<ip> to connect to a remote daemon (WebSocket transport).
    Without REACHY_HOST, connects to the local daemon on localhost:8000.
    """
    global _reachy
    if _reachy is not None:
        return _reachy
    with _reachy_lock:
        if _reachy is not None:
            return _reachy
        try:
            from reachy_connect import connect_reachy
            _reachy = connect_reachy()
            return _reachy
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Reachy Mini: {e}")


# ── UDP broadcast for dual-Reachy follower mode ──

def _broadcast(msg: dict):
    """Send state update to follower via UDP broadcast (non-blocking, fire-and-forget)."""
    if _broadcast_sock is None:
        return
    try:
        import json
        data = json.dumps(msg).encode()
        _broadcast_sock.sendto(data, _broadcast_addr)
    except Exception:
        pass  # Never block primary on broadcast failure


def _init_broadcast():
    """Initialize UDP broadcast socket if REACHY_BROADCAST=1."""
    global _broadcast_sock, _broadcast_addr
    if os.environ.get("REACHY_BROADCAST", "").strip() not in ("1", "true", "yes"):
        return
    import socket
    _broadcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _broadcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    port = int(os.environ.get("REACHY_BROADCAST_PORT", "5555"))
    _broadcast_addr = ("255.255.255.255", port)
    print(f"Reachy broadcast enabled (UDP :{port})", file=sys.stderr)


# ── Head pose presets (yaw, pitch, roll in degrees) ──

_LOOK_PRESETS = {
    "left":   Pose(yaw=-25),
    "right":  Pose(yaw=25),
    "up":     Pose(pitch=-20),
    "down":   Pose(pitch=20),
    "center": Pose(),
}

# ── Emotion presets (exaggerated for GTC booth — visible from 3 meters) ──

_EMOTION_PRESETS = {
    "happy":     Pose(pitch=-15, left_antenna=-60, right_antenna=-60),
    "sad":       Pose(pitch=25, left_antenna=50, right_antenna=50),
    "curious":   Pose(yaw=15, pitch=-10, roll=-10, left_antenna=-45, right_antenna=0),
    "excited":   Pose(pitch=-20, left_antenna=-80, right_antenna=-80),
    "surprised": Pose(pitch=-25, left_antenna=-90, right_antenna=-90),
}

# ── Dance sequences ──
# Each dance is a list of (Pose, duration) keyframes

_DANCES = {
    "nod_groove": [
        (Pose(pitch=-15, left_antenna=-40, right_antenna=-40), 0.3),
        (Pose(pitch=10), 0.3),
        (Pose(pitch=-15, left_antenna=-40, right_antenna=-40), 0.3),
        (Pose(pitch=10), 0.3),
        (Pose(yaw=-15, pitch=-10, roll=-10, left_antenna=-60, right_antenna=0), 0.4),
        (Pose(yaw=15, pitch=-10, roll=10, left_antenna=0, right_antenna=-60), 0.4),
        (Pose(), 0.3),
    ],
    "wiggle": [
        (Pose(yaw=-20, roll=-15, left_antenna=-30, right_antenna=30), 0.25),
        (Pose(yaw=20, roll=15, left_antenna=30, right_antenna=-30), 0.25),
        (Pose(yaw=-20, roll=-15, left_antenna=-30, right_antenna=30), 0.25),
        (Pose(yaw=20, roll=15, left_antenna=30, right_antenna=-30), 0.25),
        (Pose(yaw=-10, pitch=-10, left_antenna=-50, right_antenna=-50), 0.3),
        (Pose(yaw=10, pitch=-10, left_antenna=-50, right_antenna=-50), 0.3),
        (Pose(), 0.3),
    ],
    "look_around": [
        (Pose(yaw=-30, pitch=-10, left_antenna=-20, right_antenna=0), 0.5),
        (Pose(pitch=-15, left_antenna=-40, right_antenna=-40), 0.4),
        (Pose(yaw=30, pitch=-10, left_antenna=0, right_antenna=-20), 0.5),
        (Pose(), 0.4),
    ],
}


# ── Booth Greeter (P3.1 — audience interaction mode) ──

_BOOTH_GREETINGS = [
    "Hi there! Want to see something cool? Ask me anything — or tell me to dance!",
    "Hey! I'm Reachy, running 100% on this little Jetson. Ask me to show you what I can do!",
    "Welcome! I speak 9 languages, have 20 tools, and zero cloud. Want a demo?",
    "Hello! I'm an AI robot that runs entirely on-device. Try asking me anything!",
]

_STARTUP_ANIMATION = [
    (Pose(pitch=-10), 0.4),                                     # Wake — lift head
    (Pose(yaw=-20, pitch=-5, left_antenna=-30), 0.5),           # Look left curiously
    (Pose(yaw=20, pitch=-5, right_antenna=-30), 0.5),           # Look right curiously
    (Pose(pitch=-15, left_antenna=-60, right_antenna=-60), 0.4),  # Happy!
    (Pose(), 0.3),                                                # Return to neutral
]


class BoothGreeter:
    """Background greeter for GTC booth — proactive animation + speech on idle.

    When enabled (BOOTH_MODE=1), periodically triggers:
    - Look-around animation every ~45s of silence
    - Spoken greeting every ~90s of silence (uses say() callback)
    """

    def __init__(self, motion_manager: MotionManager, say_fn=None,
                 idle_anim_interval: float = 45.0, greet_interval: float = 90.0):
        self._mm = motion_manager
        self._say = say_fn
        self._idle_anim_interval = idle_anim_interval
        self._greet_interval = greet_interval
        self._last_activity = time.monotonic()
        self._last_greet = time.monotonic()
        self._running = False
        self._thread = None
        self._greet_index = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._last_activity = time.monotonic()
        self._last_greet = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="BoothGreeter")
        self._thread.start()
        print("BoothGreeter started", file=sys.stderr)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def on_activity(self):
        """Call when user speaks or interacts — resets idle timers."""
        self._last_activity = time.monotonic()
        self._last_greet = time.monotonic()

    def _loop(self):
        while self._running:
            time.sleep(2.0)
            now = time.monotonic()
            idle_time = now - self._last_activity

            # Idle animation (look around) every ~45s
            if idle_time > self._idle_anim_interval:
                self._mm.submit_primary(MotionSequence([
                    (Pose(yaw=-15, pitch=-5), 0.5),
                    (Pose(yaw=15, pitch=-5), 0.5),
                    (Pose(), 0.3),
                ]))
                self._last_activity = now  # Reset so we don't spam

            # Spoken greeting every ~90s
            greet_idle = now - self._last_greet
            if greet_idle > self._greet_interval and self._say is not None:
                greeting = _BOOTH_GREETINGS[self._greet_index % len(_BOOTH_GREETINGS)]
                self._greet_index += 1
                try:
                    # Express happy + speak
                    self._mm.submit_pose(
                        Pose(pitch=-15, left_antenna=-60, right_antenna=-60), duration=0.4
                    )
                    self._say(greeting)
                except Exception as e:
                    print(f"BoothGreeter: greeting error: {e}", file=sys.stderr)
                self._last_greet = time.monotonic()
                self._last_activity = time.monotonic()


def _ensure_motion_manager():
    """Initialize and start MotionManager if not already running."""
    global _motion_manager
    if _motion_manager is not None:
        return _motion_manager

    _motion_manager = MotionManager(_get_reachy)
    _motion_manager.register_secondary("breathing", BreathingMotion())
    _motion_manager.register_secondary("listening", ReactiveListeningMotion())
    _motion_manager.register_secondary("audio_sway", AudioReactiveSway())
    _motion_manager.start()
    print("MotionManager initialized with breathing + listening + audio_sway", file=sys.stderr)

    # Startup animation (look around + happy)
    _motion_manager.submit_primary(MotionSequence(_STARTUP_ANIMATION))

    # Register atexit as safety net for cleanup on interpreter shutdown
    import atexit
    atexit.register(cleanup)

    return _motion_manager


def on_state_change(old_state: str, new_state: str):
    """Called by jetson-assistant core.py on every assistant state transition.

    Drives reactive listening poses and audio-reactive sway activation.
    """
    if _motion_manager is None:
        return

    # Update reactive listening pose
    listening = _motion_manager.get_secondary("listening")
    if isinstance(listening, ReactiveListeningMotion):
        listening.set_state(new_state)

    # Activate/deactivate audio-reactive sway
    sway = _motion_manager.get_secondary("audio_sway")
    if isinstance(sway, AudioReactiveSway):
        sway.set_active(new_state == "speaking")

    # Reset booth greeter idle timer on user activity
    if _booth_greeter is not None and new_state in ("listening", "processing"):
        _booth_greeter.on_activity()

    # Broadcast state to follower
    _broadcast({"type": "state", "state": new_state})


def on_audio_chunk(audio_chunk, sample_rate: int):
    """Called by jetson-assistant AudioOutput during TTS playback.

    Feeds loudness to the audio-reactive sway oscillator.
    """
    if _motion_manager is None:
        return

    sway = _motion_manager.get_secondary("audio_sway")
    if isinstance(sway, AudioReactiveSway):
        import numpy as np
        # Compute RMS loudness normalized to 0-1
        if audio_chunk.dtype == np.int16:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk
        rms = float(np.sqrt(np.mean(audio_float ** 2)))
        # Normalize: typical speech RMS is ~0.05-0.2 for int16→float
        loudness = min(1.0, rms / 0.15)
        sway.feed_loudness(loudness)
        # Forward loudness to follower
        _broadcast({"type": "loudness", "value": round(loudness, 3)})


def register_tools(registry, context=None):
    """Register Reachy Mini tools with the assistant's ToolRegistry."""
    global _llm_ref, _camera_pool_ref, _say_ref, _booth_greeter

    if context and "llm" in context:
        _llm_ref = context["llm"]
    if context and "camera_pool" in context:
        _camera_pool_ref = context["camera_pool"]
    if context and "say" in context:
        _say_ref = context["say"]

    # Start MotionManager (connects to Reachy lazily on first motion)
    _ensure_motion_manager()

    # Initialize UDP broadcast for dual-Reachy follower mode
    _init_broadcast()

    # Start booth greeter if BOOTH_MODE is enabled
    if os.environ.get("BOOTH_MODE", "").strip() in ("1", "true", "yes"):
        _booth_greeter = BoothGreeter(
            _motion_manager,
            say_fn=_say_ref,
            idle_anim_interval=float(os.environ.get("BOOTH_IDLE_ANIM_S", "45")),
            greet_interval=float(os.environ.get("BOOTH_GREET_S", "90")),
        )
        _booth_greeter.start()

    @registry.register(
        "Move Reachy Mini's head to look in a direction. "
        "Use when the user says 'look left', 'look up', 'look at me', etc."
    )
    def look(
        direction: Annotated[str, "Direction: left, right, up, down, or center"],
    ) -> str:
        direction = direction.lower().strip()
        preset = _LOOK_PRESETS.get(direction)
        if preset is None:
            return f"Unknown direction '{direction}'. Try: left, right, up, down, center."
        _motion_manager.submit_pose(preset, duration=0.5)
        _broadcast({"type": "look", "direction": direction})
        return f"Looking {direction}."

    @registry.register(
        "Express an emotion with Reachy Mini's head, antennas, and voice. "
        "Plays a Pollen-curated emotion animation (~80 to choose from). "
        "Use when the user wants the robot to show feelings."
    )
    def express(
        emotion: Annotated[
            str,
            "Emotion word (happy, sad, curious, excited, surprised, "
            "scared, proud, loving, frustrated, sorry, welcoming, ...). "
            "A specific recorded-move name (e.g. 'cheerful1') also works.",
        ],
    ) -> str:
        emotion = emotion.lower().strip()
        move = _moves.lookup_emotion(emotion)
        if move is None:
            # Fall back to the legacy hand-crafted poses if the catalog
            # missed (e.g. HF unreachable on first run).
            preset = _EMOTION_PRESETS.get(emotion)
            if preset is None:
                return f"Unknown emotion '{emotion}'."
            _motion_manager.submit_pose(preset, duration=0.6)
            _broadcast({"type": "emotion", "emotion": emotion})
            return f"Expressing {emotion}."

        _ensure_motors_enabled()
        _daemon_post(encoded_dataset_path(EMOTIONS_DATASET, move))
        _broadcast({"type": "emotion", "emotion": emotion, "move": move})
        return f"Expressing {emotion} ({move})."

    @registry.register(
        "Make Reachy Mini do a dance. Pulls from the Pollen-curated "
        "dance library (~20 choreographed moves). "
        "Use when the user asks the robot to dance, move, or be playful.",
    )
    def dance(
        name: Annotated[
            str,
            "Dance name (groovy_sway_and_roll, polyrhythm_combo, "
            "head_tilt_roll, simple_nod, etc.) or 'random'.",
        ] = "random",
    ) -> str:
        move = _moves.lookup_dance(name)
        if move is None:
            # Catalog unavailable — fall back to legacy hand-coded sequences.
            name_l = name.lower().strip()
            if name_l in ("random", "", "any"):
                name_l = random.choice(list(_DANCES.keys()))
            seq = _DANCES.get(name_l)
            if seq is None:
                return f"No dance library available and unknown fallback '{name}'."
            _motion_manager.submit_primary(MotionSequence(seq))
            _broadcast({"type": "dance", "name": name_l})
            return f"Dancing: {name_l}!"

        _ensure_motors_enabled()
        _daemon_post(encoded_dataset_path(DANCES_DATASET, move))
        _broadcast({"type": "dance", "name": move})
        return f"Dancing: {move}!"

    @registry.register(
        "Capture what Reachy Mini's camera sees and describe it. "
        "Use when the user asks 'what do you see?', 'look around', or 'describe what's in front of you'."
    )
    def reachy_see(
        question: Annotated[str, "What to look for or describe (e.g., 'what do you see?')"] = "Describe what you see briefly.",
    ) -> str:
        if _llm_ref is None:
            return "Vision not available -- no LLM backend configured."
        try:
            frame = None

            # Prefer the assistant's CameraPool. On the Pi5 orchestrator this
            # is a picamera2-backed source for the Reachy head camera (the
            # daemon has released /api/media/release, so reachy.media.get_frame
            # would not work anyway). On Jetson with a remote daemon this
            # is whatever USB camera the assistant was configured with.
            if _camera_pool_ref is not None:
                for cam in _camera_pool_ref.list_cameras():
                    frame = _camera_pool_ref.capture_frame(cam.name)
                    if frame is not None:
                        break

            # Fallback: ask the SDK directly. Only meaningful when the daemon
            # still owns media (media_backend != "no_media").
            if frame is None:
                try:
                    reachy = _get_reachy()
                    frame = reachy.media.get_frame()
                except Exception:
                    pass

            if frame is None:
                return "Could not capture camera frame. No camera available."

            # Encode frame as JPEG base64
            import io
            try:
                from PIL import Image
            except ImportError:
                return "Pillow not installed -- cannot encode camera frame."

            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode()

            response = _llm_ref.generate(question, images=[b64])
            text = getattr(response, "text", response) or ""
            return text or "I couldn't describe what I see."
        except Exception as e:
            return f"Camera error: {e}"

    @registry.register(
        "Wake up or put Reachy Mini to sleep. "
        "Use when the user says 'wake up', 'turn on', 'go to sleep', 'turn off'."
    )
    def reachy_power(
        action: Annotated[str, "Action: wake or sleep"],
    ) -> str:
        action = action.lower().strip()
        try:
            reachy = _get_reachy()
            if action in ("wake", "on", "wake_up"):
                # The SDK's wake_up() only plays the wake animation; motors
                # must be enabled separately, otherwise the move renders to
                # nothing and the robot stays slumped.
                reachy.enable_motors()
                reachy.wake_up()
                _motion_manager.submit_pose(Pose(), duration=0.5)
                return "Reachy is awake and ready."
            elif action in ("sleep", "off", "go_to_sleep"):
                # goto_sleep animates to the sleep pose. Disable motors
                # afterward so the head doesn't fight gravity while parked.
                reachy.goto_sleep()
                reachy.disable_motors()
                return "Reachy is going to sleep."
            else:
                return f"Unknown action '{action}'. Try: wake or sleep."
        except Exception as e:
            return f"Power control error: {e}"

    @registry.register(
        "Nod yes or shake head no. "
        "Use when the user asks the robot to agree or disagree."
    )
    def nod(
        response: Annotated[str, "Response: yes (nod) or no (shake)"],
    ) -> str:
        response = response.lower().strip()
        if response in ("yes", "nod", "agree"):
            # yeah_nod is a livelier choice than the metronome-style nod
            # we used to hand-roll; fall back to simple_nod or a yes
            # emotion if the dance library missed.
            move = _moves.lookup_dance("yeah_nod") or _moves.lookup_emotion("yes")
            if move:
                _ensure_motors_enabled()
                ds = DANCES_DATASET if move in _moves.dances else EMOTIONS_DATASET
                _daemon_post(encoded_dataset_path(ds, move))
                _broadcast({"type": "nod", "value": "yes", "move": move})
                return f"Nodding yes ({move})."
            # Hand-rolled fallback.
            keyframes = []
            for _ in range(3):
                keyframes.append((Pose(pitch=-15), 0.15))
                keyframes.append((Pose(pitch=10), 0.15))
            keyframes.append((Pose(), 0.2))
            _motion_manager.submit_primary(MotionSequence(keyframes))
            return "Nodding yes."
        elif response in ("no", "shake", "disagree"):
            move = _moves.lookup_emotion("no")
            if move:
                _ensure_motors_enabled()
                _daemon_post(encoded_dataset_path(EMOTIONS_DATASET, move))
                _broadcast({"type": "nod", "value": "no", "move": move})
                return f"Shaking head no ({move})."
            keyframes = []
            for _ in range(3):
                keyframes.append((Pose(yaw=-15), 0.15))
                keyframes.append((Pose(yaw=15), 0.15))
            keyframes.append((Pose(), 0.2))
            _motion_manager.submit_primary(MotionSequence(keyframes))
            return "Shaking head no."
        else:
            return f"Unknown response '{response}'. Try: yes or no."

    @registry.register(
        "Set Reachy Mini's antenna positions for expressiveness. "
        "Antennas range from -90 (back/happy) to 90 (forward/droopy) degrees."
    )
    def set_antennas(
        left: Annotated[float, "Left antenna angle in degrees (-90 to 90)"],
        right: Annotated[float, "Right antenna angle in degrees (-90 to 90)"],
    ) -> str:
        left = max(-90, min(90, float(left)))
        right = max(-90, min(90, float(right)))
        _motion_manager.submit_pose(Pose(left_antenna=left, right_antenna=right), duration=0.3)
        return f"Antennas set to left={left:.0f}, right={right:.0f}."

    @registry.register(
        "Make Reachy Mini look at a specific 3D point in the world. "
        "Use when you want precise gaze control (coordinates in meters)."
    )
    def look_at_point(
        x: Annotated[float, "X coordinate in meters (forward from robot)"],
        y: Annotated[float, "Y coordinate in meters (left of robot)"],
        z: Annotated[float, "Z coordinate in meters (up from robot base)"],
    ) -> str:
        try:
            reachy = _get_reachy()
            reachy.look_at_world(float(x), float(y), float(z))
            return f"Looking at point ({x:.2f}, {y:.2f}, {z:.2f})."
        except Exception as e:
            return f"Look-at error: {e}"

    @registry.register(
        "Check Reachy Mini's connection status. "
        "Use when the user asks 'are you connected?' or 'robot status'."
    )
    def reachy_status() -> str:
        try:
            reachy = _get_reachy()
            alive = getattr(reachy, '_is_alive', None)
            host = os.environ.get("REACHY_HOST", "localhost")
            mm = "running" if _motion_manager and _motion_manager._running else "stopped"
            return f"Connected to Reachy at {host}. MotionManager: {mm}."
        except Exception as e:
            return f"Reachy disconnected: {e}"


def cleanup():
    """Called on assistant shutdown. Stop MotionManager, disconnect from Reachy Mini."""
    global _reachy, _motion_manager, _booth_greeter
    print("Reachy cleanup: stopping motion + sending to sleep...", file=sys.stderr)
    if _booth_greeter is not None:
        _booth_greeter.stop()
        _booth_greeter = None
    if _motion_manager is not None:
        _motion_manager.stop()
        _motion_manager = None
    if _reachy is not None:
        # Run goto_sleep + disconnect with timeout — SDK can hang on remote disconnect
        def _shutdown():
            try:
                _reachy.goto_sleep()
            except Exception:
                pass
            try:
                _reachy.disconnect()
            except Exception:
                pass
        t = threading.Thread(target=_shutdown, daemon=True)
        t.start()
        t.join(timeout=3.0)
        _reachy = None
    print("Reachy cleanup: done.", file=sys.stderr)
