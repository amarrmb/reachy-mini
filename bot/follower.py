#!/usr/bin/env python3
"""
Reachy Mini Follower — mirrors a primary Reachy's emotions and state.

For GTC dual-robot demo: two Reachy Minis on the same network.
Primary speaks and interacts. Follower mirrors emotions, dances,
and listening poses with slight delay and variation.

Architecture:
    Primary (reachy_tools.py) broadcasts state via UDP on port 5555.
    Follower receives state, maps to complementary motions, and drives
    a second Reachy daemon via its own MotionManager.

Usage:
    # Terminal 1: Primary (normal GTC demo)
    REACHY_HOST=<primary-ip> REACHY_BROADCAST=1 ./run-gtc-demo.sh

    # Terminal 2: Follower (this script)
    python follower.py --host <follower-ip>

    # Or use the combined launcher:
    ./run-dual-demo.sh
"""

import argparse
import json
import math
import socket
import sys
import threading
import time

from motion_manager import (
    AudioReactiveSway,
    BreathingMotion,
    MotionManager,
    MotionSequence,
    Pose,
    ReactiveListeningMotion,
    single_pose,
)

# ── Follower-specific emotion reactions ──
# Complementary to primary's emotions (not identical — more interesting)

_FOLLOWER_REACTIONS = {
    "happy":     [
        (Pose(yaw=10, pitch=-10, left_antenna=-50, right_antenna=-50), 0.3),
        (Pose(yaw=-10, pitch=-10, left_antenna=-50, right_antenna=-50), 0.3),
        (Pose(pitch=-15, left_antenna=-60, right_antenna=-60), 0.3),
    ],
    "sad":       [
        (Pose(pitch=15, left_antenna=30, right_antenna=30), 0.5),
        (Pose(pitch=20, yaw=-5, left_antenna=40, right_antenna=40), 0.4),
    ],
    "curious":   [
        (Pose(yaw=-15, pitch=-10, roll=10, left_antenna=0, right_antenna=-45), 0.4),
        (Pose(yaw=-10, pitch=-8), 0.3),
    ],
    "excited":   [
        (Pose(pitch=-20, left_antenna=-80, right_antenna=-80), 0.2),
        (Pose(pitch=-10, left_antenna=-40, right_antenna=-40), 0.2),
        (Pose(pitch=-20, left_antenna=-80, right_antenna=-80), 0.2),
        (Pose(), 0.2),
    ],
    "surprised": [
        (Pose(pitch=-20, left_antenna=-80, right_antenna=-80), 0.3),
        (Pose(pitch=-25, roll=-5, left_antenna=-90, right_antenna=-90), 0.3),
        (Pose(pitch=-15, left_antenna=-60, right_antenna=-60), 0.3),
    ],
}

# Complementary dances — follower does opposite-side motions
_FOLLOWER_DANCES = {
    "nod_groove": [
        (Pose(pitch=10), 0.3),
        (Pose(pitch=-15, left_antenna=-40, right_antenna=-40), 0.3),
        (Pose(pitch=10), 0.3),
        (Pose(pitch=-15, left_antenna=-40, right_antenna=-40), 0.3),
        (Pose(yaw=15, pitch=-10, roll=10, left_antenna=0, right_antenna=-60), 0.4),
        (Pose(yaw=-15, pitch=-10, roll=-10, left_antenna=-60, right_antenna=0), 0.4),
        (Pose(), 0.3),
    ],
    "wiggle": [
        (Pose(yaw=20, roll=15, left_antenna=30, right_antenna=-30), 0.25),
        (Pose(yaw=-20, roll=-15, left_antenna=-30, right_antenna=30), 0.25),
        (Pose(yaw=20, roll=15, left_antenna=30, right_antenna=-30), 0.25),
        (Pose(yaw=-20, roll=-15, left_antenna=-30, right_antenna=30), 0.25),
        (Pose(yaw=10, pitch=-10, left_antenna=-50, right_antenna=-50), 0.3),
        (Pose(yaw=-10, pitch=-10, left_antenna=-50, right_antenna=-50), 0.3),
        (Pose(), 0.3),
    ],
    "look_around": [
        (Pose(yaw=30, pitch=-10, left_antenna=0, right_antenna=-20), 0.5),
        (Pose(pitch=-15, left_antenna=-40, right_antenna=-40), 0.4),
        (Pose(yaw=-30, pitch=-10, left_antenna=-20, right_antenna=0), 0.5),
        (Pose(), 0.4),
    ],
}


class ReachyFollower:
    """Connects to a second Reachy daemon and mirrors the primary's state."""

    def __init__(self, host: str, listen_port: int = 5555):
        self._host = host
        self._listen_port = listen_port
        self._running = False
        self._mm = None
        self._reachy = None
        self._last_state = "idle"
        self._last_emotion = None
        self._last_dance = None

    def start(self):
        self._connect_reachy()
        self._init_motion_manager()
        self._running = True

        # Start UDP listener
        listener = threading.Thread(target=self._udp_listen, daemon=True, name="FollowerUDP")
        listener.start()

        print(f"Follower running — listening for primary on UDP :{self._listen_port}", file=sys.stderr)
        print(f"Connected to Reachy at {self._host}", file=sys.stderr)

        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self._running = False
        if self._mm:
            self._mm.stop()
        if self._reachy:
            try:
                self._reachy.goto_sleep()
            except Exception:
                pass
            try:
                self._reachy.disconnect()
            except Exception:
                pass
        print("Follower stopped.", file=sys.stderr)

    def _connect_reachy(self):
        from reachy_connect import connect_reachy
        self._reachy = connect_reachy(host=self._host)

    def _init_motion_manager(self):
        self._mm = MotionManager(lambda: self._reachy)
        self._mm.register_secondary("breathing", BreathingMotion())
        self._mm.register_secondary("listening", ReactiveListeningMotion())
        self._mm.register_secondary("audio_sway", AudioReactiveSway())
        self._mm.start()

    def _udp_listen(self):
        """Listen for state broadcasts from primary on UDP."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self._listen_port))
        sock.settimeout(1.0)

        while self._running:
            try:
                data, addr = sock.recvfrom(4096)
                msg = json.loads(data.decode())
                self._handle_message(msg)
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Follower UDP error: {e}", file=sys.stderr)

        sock.close()

    def _handle_message(self, msg: dict):
        msg_type = msg.get("type")

        if msg_type == "state":
            new_state = msg.get("state", "idle")
            if new_state != self._last_state:
                self._on_state_change(self._last_state, new_state)
                self._last_state = new_state

        elif msg_type == "emotion":
            emotion = msg.get("emotion")
            if emotion and emotion != self._last_emotion:
                self._last_emotion = emotion
                self._on_emotion(emotion)

        elif msg_type == "dance":
            dance_name = msg.get("name")
            if dance_name:
                self._on_dance(dance_name)

        elif msg_type == "look":
            direction = msg.get("direction")
            if direction:
                self._on_look(direction)

        elif msg_type == "loudness":
            loudness = msg.get("value", 0.0)
            sway = self._mm.get_secondary("audio_sway")
            if isinstance(sway, AudioReactiveSway):
                sway.set_active(True)
                sway.feed_loudness(loudness * 0.7)  # Slightly dampened

    def _on_state_change(self, old_state: str, new_state: str):
        listening = self._mm.get_secondary("listening")
        if isinstance(listening, ReactiveListeningMotion):
            listening.set_state(new_state)

        sway = self._mm.get_secondary("audio_sway")
        if isinstance(sway, AudioReactiveSway):
            sway.set_active(new_state == "speaking")

    def _on_emotion(self, emotion: str):
        keyframes = _FOLLOWER_REACTIONS.get(emotion)
        if keyframes:
            # Slight delay for natural feel
            time.sleep(0.15)
            self._mm.submit_primary(MotionSequence(keyframes))

    def _on_dance(self, name: str):
        keyframes = _FOLLOWER_DANCES.get(name)
        if keyframes:
            # Staggered start — follower enters half a beat later
            time.sleep(0.2)
            self._mm.submit_primary(MotionSequence(keyframes))
        else:
            # Generic mirror: same dance from _DANCES but with mirrored yaw
            from reachy_tools import _DANCES
            original = _DANCES.get(name)
            if original:
                mirrored = [
                    (Pose(-p.yaw, p.pitch, -p.roll, p.right_antenna, p.left_antenna), d)
                    for p, d in original
                ]
                time.sleep(0.2)
                self._mm.submit_primary(MotionSequence(mirrored))

    def _on_look(self, direction: str):
        # Mirror look direction (left ↔ right, others same)
        mirror_map = {"left": "right", "right": "left"}
        mirrored = mirror_map.get(direction, direction)
        from reachy_tools import _LOOK_PRESETS
        preset = _LOOK_PRESETS.get(mirrored)
        if preset:
            time.sleep(0.1)
            self._mm.submit_pose(preset, duration=0.5)


def main():
    parser = argparse.ArgumentParser(description="Reachy Mini Follower — mirror mode")
    parser.add_argument("--host", required=True, help="Reachy daemon host for follower robot")
    parser.add_argument("--port", type=int, default=5555, help="UDP port to listen on (default: 5555)")
    args = parser.parse_args()

    follower = ReachyFollower(host=args.host, listen_port=args.port)
    follower.start()


if __name__ == "__main__":
    main()
