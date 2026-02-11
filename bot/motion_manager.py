"""
MotionManager — 50Hz background thread that composes primary + secondary motions.

Brev's Reachy demo uses a 100Hz MovementManager with audio-reactive sway,
idle breathing, and reactive listening layered on top of explicit commands.
This is our equivalent, designed for the jetson-assistant external tool plugin.

Architecture:
    Primary motion: explicit commands (look, dance, express) — exclusive, one at a time
    Secondary motions: breathing, audio-reactive sway, listening pose — additive, concurrent
    Compose: primary_pose + sum(secondary_offsets) → goto_target()

Thread safety:
    MotionManager is the SOLE caller of goto_target(). Tools submit motions
    to its queue instead of calling the SDK directly. All SDK access is
    mutex-protected.

Usage:
    manager = MotionManager(get_reachy_fn)
    manager.start()
    manager.register_secondary("breathing", BreathingMotion())
    manager.submit_primary(MotionSequence([
        (Pose(yaw=-25), 0.5),
        (Pose(), 0.3),
    ]))
    ...
    manager.stop()
"""

import math
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional


# ── Pose ──

@dataclass
class Pose:
    """Head + antenna pose in degrees."""
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    left_antenna: float = 0.0
    right_antenna: float = 0.0

    def __add__(self, other: "Pose") -> "Pose":
        return Pose(
            self.yaw + other.yaw,
            self.pitch + other.pitch,
            self.roll + other.roll,
            self.left_antenna + other.left_antenna,
            self.right_antenna + other.right_antenna,
        )

    def lerp(self, target: "Pose", t: float) -> "Pose":
        """Linear interpolation toward target. t=0 → self, t=1 → target."""
        t = max(0.0, min(1.0, t))
        return Pose(
            self.yaw + (target.yaw - self.yaw) * t,
            self.pitch + (target.pitch - self.pitch) * t,
            self.roll + (target.roll - self.roll) * t,
            self.left_antenna + (target.left_antenna - self.left_antenna) * t,
            self.right_antenna + (target.right_antenna - self.right_antenna) * t,
        )

    def clamp(
        self,
        yaw_range=(-45, 45),
        pitch_range=(-35, 35),
        roll_range=(-25, 25),
        antenna_range=(-90, 90),
    ) -> "Pose":
        """Clamp all axes to safe ranges."""
        return Pose(
            max(yaw_range[0], min(yaw_range[1], self.yaw)),
            max(pitch_range[0], min(pitch_range[1], self.pitch)),
            max(roll_range[0], min(roll_range[1], self.roll)),
            max(antenna_range[0], min(antenna_range[1], self.left_antenna)),
            max(antenna_range[0], min(antenna_range[1], self.right_antenna)),
        )

    def close_to(self, other: "Pose", threshold: float = 0.3) -> bool:
        """Check if two poses are close enough to skip an update."""
        return (
            abs(self.yaw - other.yaw) < threshold
            and abs(self.pitch - other.pitch) < threshold
            and abs(self.roll - other.roll) < threshold
            and abs(self.left_antenna - other.left_antenna) < threshold
            and abs(self.right_antenna - other.right_antenna) < threshold
        )


# ── Motion Sequences (primary) ──

class MotionSequence:
    """A sequence of keyframes for primary motion. Interpolates between poses."""

    def __init__(self, keyframes: list[tuple[Pose, float]], hold_last: bool = True):
        """
        Args:
            keyframes: list of (target_pose, duration_seconds)
            hold_last: if True, hold the last keyframe pose after sequence ends
        """
        self._keyframes = keyframes
        self._hold_last = hold_last
        self._start_time = 0.0
        self._total_duration = sum(d for _, d in keyframes)
        self._started = False

    def start(self, t: float):
        self._start_time = t
        self._started = True

    def is_done(self, t: float) -> bool:
        if not self._started:
            return False
        return (t - self._start_time) >= self._total_duration

    def get_pose(self, t: float) -> Pose:
        if not self._started or not self._keyframes:
            return Pose()

        elapsed = t - self._start_time

        # Walk through keyframes, interpolating within current segment
        cumulative = 0.0
        prev_pose = Pose()
        for target_pose, duration in self._keyframes:
            if elapsed < cumulative + duration:
                progress = (elapsed - cumulative) / duration if duration > 0 else 1.0
                return prev_pose.lerp(target_pose, progress)
            cumulative += duration
            prev_pose = target_pose

        # Past end — return last pose
        if self._hold_last and self._keyframes:
            return self._keyframes[-1][0]
        return Pose()


def single_pose(pose: Pose, duration: float = 0.5) -> MotionSequence:
    """Convenience: a single-keyframe primary motion."""
    return MotionSequence([(pose, duration)])


# ── Secondary Motions (additive) ──

class SecondaryMotion(ABC):
    """Base class for additive secondary motions (breathing, sway, etc.)."""

    @abstractmethod
    def is_active(self) -> bool:
        ...

    @abstractmethod
    def get_offset(self, t: float) -> Pose:
        ...


class BreathingMotion(SecondaryMotion):
    """Subtle idle breathing — sinusoidal pitch + antenna sway."""

    def __init__(
        self,
        pitch_amplitude: float = 0.8,
        pitch_freq: float = 0.25,
        antenna_amplitude: float = 2.0,
        antenna_freq: float = 0.15,
    ):
        self._pitch_amp = pitch_amplitude
        self._pitch_freq = pitch_freq
        self._ant_amp = antenna_amplitude
        self._ant_freq = antenna_freq
        self._active = True

    def is_active(self) -> bool:
        return self._active

    def set_active(self, active: bool):
        self._active = active

    def get_offset(self, t: float) -> Pose:
        pitch = self._pitch_amp * math.sin(2 * math.pi * self._pitch_freq * t)
        # Slight phase offset between antennas for organic feel
        left_ant = self._ant_amp * math.sin(2 * math.pi * self._ant_freq * t)
        right_ant = self._ant_amp * math.sin(2 * math.pi * self._ant_freq * t + 0.4)
        return Pose(pitch=pitch, left_antenna=left_ant, right_antenna=right_ant)


class ReactiveListeningMotion(SecondaryMotion):
    """State-driven secondary motion: attentive during LISTENING, curious during PROCESSING."""

    def __init__(self):
        self._state = "idle"  # idle, listening, processing, speaking
        self._transition_time = 0.0
        self._blend_duration = 0.3  # seconds to blend to new pose

    _POSES = {
        "idle": Pose(),
        "listening": Pose(pitch=-2, left_antenna=0, right_antenna=-25),  # one ear up, one cocked back (attentive)
        "processing": Pose(pitch=-3, yaw=3, roll=-2, left_antenna=-15, right_antenna=5),  # curious head tilt
        "speaking": Pose(),  # audio-reactive sway handles this
    }

    def set_state(self, state: str):
        if state != self._state:
            self._state = state
            self._transition_time = time.monotonic()

    def is_active(self) -> bool:
        return self._state in ("listening", "processing")

    def get_offset(self, t: float) -> Pose:
        target = self._POSES.get(self._state, Pose())
        # Smooth blend into target
        elapsed = t - self._transition_time
        blend = min(1.0, elapsed / self._blend_duration) if self._blend_duration > 0 else 1.0
        return Pose().lerp(target, blend)


class AudioReactiveSway(SecondaryMotion):
    """
    Audio-reactive head sway during speech. 4-axis oscillators modulated by loudness.

    Feed audio data via feed_audio() before/during playback.
    The oscillator amplitudes are modulated by the loudness envelope.
    """

    def __init__(self):
        self._active = False
        self._loudness = 0.0  # 0-1 normalized loudness
        self._lock = threading.Lock()

        # Oscillator parameters: (frequency_hz, max_amplitude_degrees)
        self._pitch_osc = (2.2, 2.0)
        self._yaw_osc = (0.6, 3.0)
        self._roll_osc = (1.3, 1.5)
        self._antenna_osc = (0.8, 5.0)

        # Random phase offsets for non-repetitive motion
        import random
        self._phase_offsets = {
            "pitch": random.uniform(0, 2 * math.pi),
            "yaw": random.uniform(0, 2 * math.pi),
            "roll": random.uniform(0, 2 * math.pi),
            "antenna": random.uniform(0, 2 * math.pi),
        }

    def set_active(self, active: bool):
        with self._lock:
            self._active = active
            if not active:
                self._loudness = 0.0

    def feed_loudness(self, loudness: float):
        """Feed normalized loudness (0-1) from audio callback."""
        with self._lock:
            # Smooth with exponential moving average
            self._loudness = 0.7 * self._loudness + 0.3 * max(0.0, min(1.0, loudness))

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def get_offset(self, t: float) -> Pose:
        with self._lock:
            loudness = self._loudness

        if loudness < 0.01:
            return Pose()

        def osc(freq, amp, phase_key):
            phase = self._phase_offsets[phase_key]
            return amp * loudness * math.sin(2 * math.pi * freq * t + phase)

        return Pose(
            pitch=osc(self._pitch_osc[0], self._pitch_osc[1], "pitch"),
            yaw=osc(self._yaw_osc[0], self._yaw_osc[1], "yaw"),
            roll=osc(self._roll_osc[0], self._roll_osc[1], "roll"),
            left_antenna=osc(self._antenna_osc[0], self._antenna_osc[1], "antenna"),
            right_antenna=osc(self._antenna_osc[0], -self._antenna_osc[1], "antenna"),
        )


# ── MotionManager ──

class MotionManager:
    """
    50Hz background thread that composes primary + secondary motions and
    sends the result to Reachy Mini via goto_target().

    Thread safety: This is the SOLE caller of goto_target(). All tools submit
    motions to queues/registries. The background thread reads them and sends
    one composed pose per tick.
    """

    def __init__(self, get_reachy: Callable, tick_rate: float = 50.0):
        """
        Args:
            get_reachy: callable that returns the Reachy Mini SDK instance
            tick_rate: updates per second (50Hz default)
        """
        self._get_reachy = get_reachy
        self._tick_interval = 1.0 / tick_rate
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Primary motion (exclusive)
        self._primary_queue: queue.Queue = queue.Queue()
        self._current_primary: Optional[MotionSequence] = None
        self._last_primary_pose = Pose()  # hold last primary pose when sequence ends

        # Secondary motions (additive)
        self._secondary_motions: dict[str, SecondaryMotion] = {}

        # Last sent pose (skip update if unchanged)
        # Initialize to None so the first update always goes through
        self._last_sent_pose: Optional[Pose] = None

        # SDK import cache
        self._create_head_pose = None
        self._consecutive_errors = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="MotionManager")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit_primary(self, sequence: MotionSequence):
        """Submit a primary motion. Replaces any current primary immediately."""
        self._primary_queue.put(sequence)

    def submit_pose(self, pose: Pose, duration: float = 0.5):
        """Convenience: submit a single pose as primary motion."""
        self.submit_primary(single_pose(pose, duration))

    def register_secondary(self, name: str, motion: SecondaryMotion):
        with self._lock:
            self._secondary_motions[name] = motion

    def unregister_secondary(self, name: str):
        with self._lock:
            self._secondary_motions.pop(name, None)

    def get_secondary(self, name: str) -> Optional[SecondaryMotion]:
        with self._lock:
            return self._secondary_motions.get(name)

    def _loop(self):
        import sys
        print("MotionManager started (50Hz)", file=sys.stderr)
        while self._running:
            t_start = time.monotonic()
            try:
                self._tick(t_start)
            except Exception as e:
                import sys
                print(f"MotionManager tick error: {e}", file=sys.stderr)
            elapsed = time.monotonic() - t_start
            sleep_time = self._tick_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _tick(self, t: float):
        # Drain primary queue — latest submission wins
        new_primary = None
        try:
            while True:
                new_primary = self._primary_queue.get_nowait()
        except queue.Empty:
            pass

        if new_primary is not None:
            self._current_primary = new_primary
            new_primary.start(t)

        # Get primary pose
        if self._current_primary is not None and not self._current_primary.is_done(t):
            primary = self._current_primary.get_pose(t)
            self._last_primary_pose = primary
        elif self._current_primary is not None and self._current_primary.is_done(t):
            # Hold last pose
            primary = self._last_primary_pose
            self._current_primary = None
        else:
            primary = self._last_primary_pose

        # Sum secondary offsets
        secondary = Pose()
        with self._lock:
            for motion in self._secondary_motions.values():
                if motion.is_active():
                    offset = motion.get_offset(t)
                    secondary = secondary + offset

        # Compose and clamp
        final = (primary + secondary).clamp()

        # Skip if pose hasn't changed enough (0.5° threshold reduces SDK chatter)
        if self._last_sent_pose is not None and final.close_to(self._last_sent_pose, threshold=0.5):
            return

        self._last_sent_pose = final
        self._send_pose(final)

    def _send_pose(self, pose: Pose):
        """Send composed pose to Reachy Mini via SDK."""
        try:
            if self._create_head_pose is None:
                from reachy_mini.utils import create_head_pose
                self._create_head_pose = create_head_pose

            reachy = self._get_reachy()

            head_pose = self._create_head_pose(
                yaw=pose.yaw, pitch=pose.pitch, roll=pose.roll, degrees=True
            )
            antennas = [
                math.radians(pose.right_antenna),
                math.radians(pose.left_antenna),
            ]  # SDK expects [right, left]

            # Short duration — we're updating at 50Hz, SDK interpolates
            reachy.goto_target(head=head_pose, antennas=antennas, duration=0.04)
            self._consecutive_errors = 0
        except Exception:
            self._consecutive_errors += 1
            if self._consecutive_errors == 10:
                import sys
                print("MotionManager: 10 consecutive SDK errors — robot may be disconnected", file=sys.stderr)
            # Back off after many failures to avoid flooding logs
            if self._consecutive_errors > 50:
                time.sleep(0.5)
