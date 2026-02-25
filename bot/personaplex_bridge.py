"""PersonaPlex <-> Reachy Mini audio-reactive bridge.

Connects PersonaPlex's decoded PCM audio output to Reachy Mini's
MotionManager for real-time audio-reactive robot animations.

No tool calling — pure audio-reactive motions:
  - BreathingMotion: always-on subtle idle breathing
  - ReactiveListeningMotion: attentive head tilt when user speaks
  - AudioReactiveSway: 4-axis head/antenna sway modulated by speech loudness

Usage:
    from personaplex_bridge import setup_motion_manager, create_audio_bridge

    # Works with any get_reachy callable (real robot, MuJoCo, stub):
    mm = setup_motion_manager(lambda: my_reachy)
    callback = create_audio_bridge(mm)

    # Pass callback to PersonaPlex ServerState:
    state = ServerState(..., on_audio_frame=callback)
"""

import numpy as np

from motion_manager import (
    AudioReactiveSway,
    BreathingMotion,
    MotionManager,
    MotionSequence,
    Pose,
    ReactiveListeningMotion,
)

# Startup animation: look around, then settle
STARTUP_ANIMATION = [
    (Pose(yaw=-15, pitch=5), 0.4),
    (Pose(yaw=15, pitch=-5), 0.4),
    (Pose(yaw=0, pitch=0, left_antenna=-30, right_antenna=-30), 0.3),
    (Pose(), 0.3),
]

# RMS threshold for speech detection (tuned to PersonaPlex output levels)
SPEECH_RMS_THRESHOLD = 0.001

# Normalization divisor: typical speech RMS ~0.05-0.2 for float32 PCM
LOUDNESS_NORMALIZER = 0.15


def setup_motion_manager(get_reachy, tick_rate=50.0):
    """Create and start a MotionManager with audio-reactive secondaries.

    Args:
        get_reachy: Callable returning the robot instance (real, MuJoCo, or stub).
                    Must support goto_target(head, antennas, duration).
        tick_rate: MotionManager update rate in Hz (default 50).

    Returns:
        Started MotionManager with breathing, listening, and audio_sway registered.
    """
    mm = MotionManager(get_reachy, tick_rate=tick_rate)
    mm.register_secondary("breathing", BreathingMotion())
    mm.register_secondary("listening", ReactiveListeningMotion())
    mm.register_secondary("audio_sway", AudioReactiveSway())
    mm.start()
    mm.submit_primary(MotionSequence(STARTUP_ANIMATION))
    return mm


def create_audio_bridge(motion_manager):
    """Create an on_audio_frame callback that drives MotionManager from PCM.

    The returned callback should be passed as on_audio_frame to
    PersonaPlex's ServerState. It computes RMS loudness from each decoded
    audio frame and drives the MotionManager's secondary motions:

      - Speaking (RMS > threshold): activate AudioReactiveSway, feed loudness
      - Listening (RMS <= threshold): deactivate sway, set listening pose

    Args:
        motion_manager: A started MotionManager instance.

    Returns:
        Callable[[np.ndarray], None] suitable for ServerState.on_audio_frame.
    """
    state = ["idle"]

    def on_audio_frame(pcm_np):
        rms = float(np.sqrt(np.mean(pcm_np ** 2)))
        loudness = min(1.0, rms / LOUDNESS_NORMALIZER)

        if rms > SPEECH_RMS_THRESHOLD:
            # Model is speaking — activate sway, feed loudness
            if state[0] != "speaking":
                listening = motion_manager.get_secondary("listening")
                if isinstance(listening, ReactiveListeningMotion):
                    listening.set_state("speaking")
                state[0] = "speaking"

            sway = motion_manager.get_secondary("audio_sway")
            if isinstance(sway, AudioReactiveSway):
                sway.set_active(True)
                sway.feed_loudness(loudness)
        else:
            # Silence — user speaking or pause
            if state[0] != "listening":
                listening = motion_manager.get_secondary("listening")
                if isinstance(listening, ReactiveListeningMotion):
                    listening.set_state("listening")
                sway = motion_manager.get_secondary("audio_sway")
                if isinstance(sway, AudioReactiveSway):
                    sway.set_active(False)
                state[0] = "listening"

    return on_audio_frame
