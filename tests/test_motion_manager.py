"""
Unit tests for bot.motion_manager — pure tests, no hardware or SDK dependencies.

Run with:
    cd /home/amar/baskd/reachy-mini && python -m pytest tests/test_motion_manager.py -v
"""

import math
import time
import unittest
from unittest.mock import MagicMock

from bot.motion_manager import (
    AudioReactiveSway,
    BreathingMotion,
    MotionManager,
    MotionSequence,
    Pose,
    ReactiveListeningMotion,
    single_pose,
)


# ---------------------------------------------------------------------------
# Pose tests
# ---------------------------------------------------------------------------


class TestPose(unittest.TestCase):
    """Tests for the Pose dataclass."""

    def test_default_pose_is_all_zeros(self):
        p = Pose()
        self.assertEqual(p.yaw, 0.0)
        self.assertEqual(p.pitch, 0.0)
        self.assertEqual(p.roll, 0.0)
        self.assertEqual(p.left_antenna, 0.0)
        self.assertEqual(p.right_antenna, 0.0)

    def test_add_sums_all_fields(self):
        a = Pose(yaw=1, pitch=2, roll=3, left_antenna=4, right_antenna=5)
        b = Pose(yaw=10, pitch=20, roll=30, left_antenna=40, right_antenna=50)
        result = a + b
        self.assertAlmostEqual(result.yaw, 11)
        self.assertAlmostEqual(result.pitch, 22)
        self.assertAlmostEqual(result.roll, 33)
        self.assertAlmostEqual(result.left_antenna, 44)
        self.assertAlmostEqual(result.right_antenna, 55)

    def test_add_with_zero_pose(self):
        a = Pose(yaw=5, pitch=-3)
        result = a + Pose()
        self.assertAlmostEqual(result.yaw, 5)
        self.assertAlmostEqual(result.pitch, -3)
        self.assertAlmostEqual(result.roll, 0)

    def test_add_with_negatives(self):
        a = Pose(yaw=10, pitch=-5)
        b = Pose(yaw=-3, pitch=8)
        result = a + b
        self.assertAlmostEqual(result.yaw, 7)
        self.assertAlmostEqual(result.pitch, 3)

    # -- lerp --

    def test_lerp_t0_returns_self(self):
        start = Pose(yaw=10, pitch=20, roll=30, left_antenna=40, right_antenna=50)
        target = Pose(yaw=100, pitch=200, roll=300, left_antenna=400, right_antenna=500)
        result = start.lerp(target, 0.0)
        self.assertAlmostEqual(result.yaw, 10)
        self.assertAlmostEqual(result.pitch, 20)
        self.assertAlmostEqual(result.roll, 30)
        self.assertAlmostEqual(result.left_antenna, 40)
        self.assertAlmostEqual(result.right_antenna, 50)

    def test_lerp_t1_returns_target(self):
        start = Pose(yaw=10, pitch=20, roll=30, left_antenna=40, right_antenna=50)
        target = Pose(yaw=100, pitch=200, roll=300, left_antenna=400, right_antenna=500)
        result = start.lerp(target, 1.0)
        self.assertAlmostEqual(result.yaw, 100)
        self.assertAlmostEqual(result.pitch, 200)
        self.assertAlmostEqual(result.roll, 300)
        self.assertAlmostEqual(result.left_antenna, 400)
        self.assertAlmostEqual(result.right_antenna, 500)

    def test_lerp_t05_returns_midpoint(self):
        start = Pose(yaw=0, pitch=0, roll=0, left_antenna=0, right_antenna=0)
        target = Pose(yaw=100, pitch=200, roll=300, left_antenna=400, right_antenna=500)
        result = start.lerp(target, 0.5)
        self.assertAlmostEqual(result.yaw, 50)
        self.assertAlmostEqual(result.pitch, 100)
        self.assertAlmostEqual(result.roll, 150)
        self.assertAlmostEqual(result.left_antenna, 200)
        self.assertAlmostEqual(result.right_antenna, 250)

    def test_lerp_clamps_t_below_zero(self):
        start = Pose(yaw=10)
        target = Pose(yaw=20)
        result = start.lerp(target, -5.0)
        self.assertAlmostEqual(result.yaw, 10.0)

    def test_lerp_clamps_t_above_one(self):
        start = Pose(yaw=10)
        target = Pose(yaw=20)
        result = start.lerp(target, 3.0)
        self.assertAlmostEqual(result.yaw, 20.0)

    def test_lerp_non_zero_start(self):
        start = Pose(yaw=20, pitch=40)
        target = Pose(yaw=40, pitch=80)
        result = start.lerp(target, 0.25)
        self.assertAlmostEqual(result.yaw, 25)
        self.assertAlmostEqual(result.pitch, 50)

    # -- clamp --

    def test_clamp_within_range_unchanged(self):
        p = Pose(yaw=10, pitch=-10, roll=5, left_antenna=30, right_antenna=-30)
        clamped = p.clamp()
        self.assertAlmostEqual(clamped.yaw, 10)
        self.assertAlmostEqual(clamped.pitch, -10)
        self.assertAlmostEqual(clamped.roll, 5)
        self.assertAlmostEqual(clamped.left_antenna, 30)
        self.assertAlmostEqual(clamped.right_antenna, -30)

    def test_clamp_yaw(self):
        p = Pose(yaw=100)
        clamped = p.clamp()
        self.assertAlmostEqual(clamped.yaw, 45)

        p2 = Pose(yaw=-100)
        clamped2 = p2.clamp()
        self.assertAlmostEqual(clamped2.yaw, -45)

    def test_clamp_pitch(self):
        p = Pose(pitch=100)
        clamped = p.clamp()
        self.assertAlmostEqual(clamped.pitch, 35)

        p2 = Pose(pitch=-100)
        clamped2 = p2.clamp()
        self.assertAlmostEqual(clamped2.pitch, -35)

    def test_clamp_roll(self):
        p = Pose(roll=100)
        clamped = p.clamp()
        self.assertAlmostEqual(clamped.roll, 25)

        p2 = Pose(roll=-100)
        clamped2 = p2.clamp()
        self.assertAlmostEqual(clamped2.roll, -25)

    def test_clamp_antenna(self):
        p = Pose(left_antenna=200, right_antenna=-200)
        clamped = p.clamp()
        self.assertAlmostEqual(clamped.left_antenna, 90)
        self.assertAlmostEqual(clamped.right_antenna, -90)

    def test_clamp_custom_ranges(self):
        p = Pose(yaw=50, pitch=50, roll=50, left_antenna=50, right_antenna=50)
        clamped = p.clamp(
            yaw_range=(-10, 10),
            pitch_range=(-5, 5),
            roll_range=(-2, 2),
            antenna_range=(-20, 20),
        )
        self.assertAlmostEqual(clamped.yaw, 10)
        self.assertAlmostEqual(clamped.pitch, 5)
        self.assertAlmostEqual(clamped.roll, 2)
        self.assertAlmostEqual(clamped.left_antenna, 20)
        self.assertAlmostEqual(clamped.right_antenna, 20)

    # -- close_to --

    def test_close_to_identical_poses(self):
        a = Pose(yaw=10, pitch=20)
        b = Pose(yaw=10, pitch=20)
        self.assertTrue(a.close_to(b))

    def test_close_to_within_threshold(self):
        a = Pose(yaw=10)
        b = Pose(yaw=10.2)
        self.assertTrue(a.close_to(b, threshold=0.3))

    def test_close_to_exceeds_threshold(self):
        a = Pose(yaw=10)
        b = Pose(yaw=11)
        self.assertFalse(a.close_to(b, threshold=0.3))

    def test_close_to_checks_all_fields(self):
        base = Pose()
        # Each field individually exceeding threshold should return False
        self.assertFalse(base.close_to(Pose(yaw=1), threshold=0.3))
        self.assertFalse(base.close_to(Pose(pitch=1), threshold=0.3))
        self.assertFalse(base.close_to(Pose(roll=1), threshold=0.3))
        self.assertFalse(base.close_to(Pose(left_antenna=1), threshold=0.3))
        self.assertFalse(base.close_to(Pose(right_antenna=1), threshold=0.3))

    def test_close_to_default_threshold(self):
        # Default threshold is 0.3
        a = Pose()
        b = Pose(yaw=0.29)
        self.assertTrue(a.close_to(b))

        c = Pose(yaw=0.31)
        self.assertFalse(a.close_to(c))


# ---------------------------------------------------------------------------
# MotionSequence tests
# ---------------------------------------------------------------------------


class TestMotionSequence(unittest.TestCase):
    """Tests for MotionSequence keyframe interpolation."""

    def test_get_pose_before_start_returns_zero(self):
        seq = MotionSequence([(Pose(yaw=30), 1.0)])
        pose = seq.get_pose(0.0)
        self.assertAlmostEqual(pose.yaw, 0.0)
        self.assertAlmostEqual(pose.pitch, 0.0)

    def test_get_pose_empty_keyframes_returns_zero(self):
        seq = MotionSequence([])
        seq.start(0.0)
        pose = seq.get_pose(0.5)
        self.assertAlmostEqual(pose.yaw, 0.0)

    def test_single_keyframe_interpolation(self):
        seq = MotionSequence([(Pose(yaw=30), 1.0)])
        seq.start(0.0)

        # At start: zero (lerp from zero to target at t=0)
        pose_start = seq.get_pose(0.0)
        self.assertAlmostEqual(pose_start.yaw, 0.0)

        # At midpoint: should be halfway
        pose_mid = seq.get_pose(0.5)
        self.assertAlmostEqual(pose_mid.yaw, 15.0)

        # At end: should be target
        pose_end = seq.get_pose(1.0)
        self.assertAlmostEqual(pose_end.yaw, 30.0)

    def test_multiple_keyframes_interpolation(self):
        seq = MotionSequence([
            (Pose(yaw=20), 1.0),   # 0-1s: zero -> 20
            (Pose(yaw=40), 1.0),   # 1-2s: 20 -> 40
        ])
        seq.start(0.0)

        # Halfway through first segment: 0 -> 20 at t=0.5
        p1 = seq.get_pose(0.5)
        self.assertAlmostEqual(p1.yaw, 10.0)

        # End of first segment / start of second: should be at 20
        p2 = seq.get_pose(1.0)
        self.assertAlmostEqual(p2.yaw, 20.0)

        # Halfway through second segment: 20 -> 40 at t=0.5
        p3 = seq.get_pose(1.5)
        self.assertAlmostEqual(p3.yaw, 30.0)

        # End of second segment
        p4 = seq.get_pose(2.0)
        self.assertAlmostEqual(p4.yaw, 40.0)

    def test_is_done_before_start_returns_false(self):
        seq = MotionSequence([(Pose(yaw=10), 1.0)])
        self.assertFalse(seq.is_done(100.0))

    def test_is_done_during_sequence(self):
        seq = MotionSequence([(Pose(yaw=10), 1.0)])
        seq.start(0.0)
        self.assertFalse(seq.is_done(0.5))

    def test_is_done_after_total_duration(self):
        seq = MotionSequence([(Pose(yaw=10), 1.0)])
        seq.start(0.0)
        self.assertTrue(seq.is_done(1.0))
        self.assertTrue(seq.is_done(5.0))

    def test_is_done_multiple_keyframes(self):
        seq = MotionSequence([
            (Pose(yaw=10), 0.5),
            (Pose(yaw=20), 0.5),
        ])
        seq.start(0.0)
        self.assertFalse(seq.is_done(0.5))
        self.assertTrue(seq.is_done(1.0))

    def test_hold_last_true_holds_final_pose(self):
        seq = MotionSequence([(Pose(yaw=30), 1.0)], hold_last=True)
        seq.start(0.0)
        pose = seq.get_pose(5.0)
        self.assertAlmostEqual(pose.yaw, 30.0)

    def test_hold_last_false_returns_zero_after_end(self):
        seq = MotionSequence([(Pose(yaw=30), 1.0)], hold_last=False)
        seq.start(0.0)
        pose = seq.get_pose(5.0)
        self.assertAlmostEqual(pose.yaw, 0.0)

    def test_start_with_nonzero_time(self):
        seq = MotionSequence([(Pose(yaw=20), 1.0)])
        seq.start(10.0)

        # At start time: should be zero (beginning of interpolation)
        pose_start = seq.get_pose(10.0)
        self.assertAlmostEqual(pose_start.yaw, 0.0)

        # Halfway: should be 10
        pose_mid = seq.get_pose(10.5)
        self.assertAlmostEqual(pose_mid.yaw, 10.0)

        # At end
        self.assertTrue(seq.is_done(11.0))
        pose_end = seq.get_pose(11.0)
        self.assertAlmostEqual(pose_end.yaw, 20.0)

    def test_zero_duration_keyframe(self):
        # A zero-duration keyframe should jump instantly (progress = 1.0)
        seq = MotionSequence([(Pose(yaw=30), 0.0)])
        seq.start(0.0)
        # is_done should be True immediately
        self.assertTrue(seq.is_done(0.0))

    def test_multi_field_keyframes(self):
        seq = MotionSequence([
            (Pose(yaw=10, pitch=20, roll=30), 1.0),
        ])
        seq.start(0.0)
        pose = seq.get_pose(0.5)
        self.assertAlmostEqual(pose.yaw, 5.0)
        self.assertAlmostEqual(pose.pitch, 10.0)
        self.assertAlmostEqual(pose.roll, 15.0)


# ---------------------------------------------------------------------------
# single_pose tests
# ---------------------------------------------------------------------------


class TestSinglePose(unittest.TestCase):
    """Tests for the single_pose() helper."""

    def test_creates_motion_sequence(self):
        ms = single_pose(Pose(yaw=15), duration=0.5)
        self.assertIsInstance(ms, MotionSequence)

    def test_single_pose_interpolation(self):
        ms = single_pose(Pose(yaw=20), duration=1.0)
        ms.start(0.0)
        pose = ms.get_pose(0.5)
        self.assertAlmostEqual(pose.yaw, 10.0)

    def test_single_pose_default_duration(self):
        ms = single_pose(Pose(yaw=10))
        ms.start(0.0)
        # Default duration is 0.5, so at t=0.5 it should be done
        self.assertTrue(ms.is_done(0.5))
        self.assertFalse(ms.is_done(0.25))

    def test_single_pose_holds_last(self):
        ms = single_pose(Pose(yaw=10), duration=0.5)
        ms.start(0.0)
        # After done, hold_last defaults to True
        pose = ms.get_pose(5.0)
        self.assertAlmostEqual(pose.yaw, 10.0)


# ---------------------------------------------------------------------------
# BreathingMotion tests
# ---------------------------------------------------------------------------


class TestBreathingMotion(unittest.TestCase):
    """Tests for BreathingMotion secondary motion."""

    def test_is_active_by_default(self):
        bm = BreathingMotion()
        self.assertTrue(bm.is_active())

    def test_set_active_false(self):
        bm = BreathingMotion()
        bm.set_active(False)
        self.assertFalse(bm.is_active())

    def test_set_active_true_after_false(self):
        bm = BreathingMotion()
        bm.set_active(False)
        bm.set_active(True)
        self.assertTrue(bm.is_active())

    def test_get_offset_returns_pose(self):
        bm = BreathingMotion()
        offset = bm.get_offset(1.0)
        self.assertIsInstance(offset, Pose)

    def test_get_offset_returns_nonzero(self):
        bm = BreathingMotion()
        # At t=0, sin(0) = 0 for all oscillators. Pick a time where sin is nonzero.
        # pitch_freq=0.25 Hz: sin(2*pi*0.25*1.0) = sin(pi/2) = 1.0
        offset = bm.get_offset(1.0)
        # pitch should be pitch_amplitude * sin(pi/2) = 0.8 * 1.0 = 0.8
        self.assertAlmostEqual(offset.pitch, 0.8, places=5)
        # yaw and roll should be 0 (breathing only affects pitch and antennas)
        self.assertAlmostEqual(offset.yaw, 0.0)
        self.assertAlmostEqual(offset.roll, 0.0)

    def test_get_offset_varies_over_time(self):
        bm = BreathingMotion()
        offset1 = bm.get_offset(0.5)
        offset2 = bm.get_offset(2.5)
        # These should differ because sinusoidal varies
        self.assertNotAlmostEqual(offset1.pitch, offset2.pitch, places=3)

    def test_antenna_offset_has_phase_difference(self):
        bm = BreathingMotion()
        offset = bm.get_offset(1.0)
        # Left and right antennas have different phases (0.4 offset)
        # so they should differ
        self.assertNotAlmostEqual(offset.left_antenna, offset.right_antenna, places=3)

    def test_custom_parameters(self):
        bm = BreathingMotion(pitch_amplitude=2.0, pitch_freq=0.5)
        # At t=0.5: sin(2*pi*0.5*0.5) = sin(pi/2) = 1.0
        offset = bm.get_offset(0.5)
        self.assertAlmostEqual(offset.pitch, 2.0, places=5)


# ---------------------------------------------------------------------------
# ReactiveListeningMotion tests
# ---------------------------------------------------------------------------


class TestReactiveListeningMotion(unittest.TestCase):
    """Tests for ReactiveListeningMotion state-driven secondary."""

    def test_default_state_is_idle(self):
        rlm = ReactiveListeningMotion()
        self.assertFalse(rlm.is_active())

    def test_listening_state_is_active(self):
        rlm = ReactiveListeningMotion()
        rlm.set_state("listening")
        self.assertTrue(rlm.is_active())

    def test_processing_state_is_active(self):
        rlm = ReactiveListeningMotion()
        rlm.set_state("processing")
        self.assertTrue(rlm.is_active())

    def test_speaking_state_is_not_active(self):
        rlm = ReactiveListeningMotion()
        rlm.set_state("speaking")
        self.assertFalse(rlm.is_active())

    def test_idle_state_is_not_active(self):
        rlm = ReactiveListeningMotion()
        rlm.set_state("listening")
        rlm.set_state("idle")
        self.assertFalse(rlm.is_active())

    def test_get_offset_idle_returns_zero(self):
        rlm = ReactiveListeningMotion()
        # Need a time well past transition to avoid blend
        offset = rlm.get_offset(time.monotonic() + 10)
        self.assertAlmostEqual(offset.yaw, 0.0)
        self.assertAlmostEqual(offset.pitch, 0.0)
        self.assertAlmostEqual(offset.roll, 0.0)

    def test_get_offset_listening_after_blend(self):
        rlm = ReactiveListeningMotion()
        rlm.set_state("listening")
        # Use a time far in the future so blend = 1.0
        # The _POSES["listening"] is Pose(pitch=-2, left_antenna=0, right_antenna=-25)
        far_future = time.monotonic() + 100
        offset = rlm.get_offset(far_future)
        self.assertAlmostEqual(offset.pitch, -2.0, places=1)
        self.assertAlmostEqual(offset.right_antenna, -25.0, places=1)

    def test_get_offset_processing_after_blend(self):
        rlm = ReactiveListeningMotion()
        rlm.set_state("processing")
        # _POSES["processing"] = Pose(pitch=-3, yaw=3, roll=-2, left_antenna=-15, right_antenna=5)
        far_future = time.monotonic() + 100
        offset = rlm.get_offset(far_future)
        self.assertAlmostEqual(offset.pitch, -3.0, places=1)
        self.assertAlmostEqual(offset.yaw, 3.0, places=1)
        self.assertAlmostEqual(offset.roll, -2.0, places=1)

    def test_blend_partial(self):
        rlm = ReactiveListeningMotion()
        # set_state records time.monotonic() internally
        rlm.set_state("listening")
        # Immediately after, blend should be partial (close to 0)
        t_now = rlm._transition_time
        offset = rlm.get_offset(t_now + 0.0001)
        # Should be very close to zero since blend is tiny
        self.assertAlmostEqual(offset.pitch, 0.0, places=0)

    def test_set_same_state_does_not_reset_transition(self):
        rlm = ReactiveListeningMotion()
        rlm.set_state("listening")
        t1 = rlm._transition_time
        rlm.set_state("listening")  # same state
        t2 = rlm._transition_time
        self.assertEqual(t1, t2)


# ---------------------------------------------------------------------------
# AudioReactiveSway tests
# ---------------------------------------------------------------------------


class TestAudioReactiveSway(unittest.TestCase):
    """Tests for AudioReactiveSway secondary motion."""

    def test_not_active_by_default(self):
        ars = AudioReactiveSway()
        self.assertFalse(ars.is_active())

    def test_set_active_true(self):
        ars = AudioReactiveSway()
        ars.set_active(True)
        self.assertTrue(ars.is_active())

    def test_set_active_false_resets_loudness(self):
        ars = AudioReactiveSway()
        ars.set_active(True)
        ars.feed_loudness(0.8)
        ars.set_active(False)
        self.assertFalse(ars.is_active())
        # Loudness should be reset to 0 when deactivated
        # get_offset with loudness < 0.01 returns zero Pose
        offset = ars.get_offset(1.0)
        self.assertAlmostEqual(offset.yaw, 0.0)
        self.assertAlmostEqual(offset.pitch, 0.0)

    def test_zero_loudness_returns_zero_pose(self):
        ars = AudioReactiveSway()
        ars.set_active(True)
        # No loudness fed, so loudness = 0
        offset = ars.get_offset(1.0)
        self.assertAlmostEqual(offset.yaw, 0.0)
        self.assertAlmostEqual(offset.pitch, 0.0)
        self.assertAlmostEqual(offset.roll, 0.0)
        self.assertAlmostEqual(offset.left_antenna, 0.0)
        self.assertAlmostEqual(offset.right_antenna, 0.0)

    def test_nonzero_loudness_returns_nonzero_pose(self):
        ars = AudioReactiveSway()
        ars.set_active(True)
        # Feed high loudness multiple times to get past EMA smoothing
        for _ in range(20):
            ars.feed_loudness(1.0)
        offset = ars.get_offset(0.3)
        # At least one field should be nonzero (unless we're at a sin zero crossing,
        # which is extremely unlikely given 4 oscillators with random phases)
        total = abs(offset.yaw) + abs(offset.pitch) + abs(offset.roll) + abs(offset.left_antenna)
        self.assertGreater(total, 0.01)

    def test_feed_loudness_updates_state(self):
        ars = AudioReactiveSway()
        ars.set_active(True)

        # Start with zero loudness
        offset_silent = ars.get_offset(0.5)

        # Feed loudness repeatedly
        for _ in range(20):
            ars.feed_loudness(1.0)

        offset_loud = ars.get_offset(0.5)

        total_silent = (
            abs(offset_silent.yaw) + abs(offset_silent.pitch) +
            abs(offset_silent.roll) + abs(offset_silent.left_antenna)
        )
        total_loud = (
            abs(offset_loud.yaw) + abs(offset_loud.pitch) +
            abs(offset_loud.roll) + abs(offset_loud.left_antenna)
        )
        self.assertGreater(total_loud, total_silent)

    def test_feed_loudness_clamps_input(self):
        ars = AudioReactiveSway()
        ars.set_active(True)
        # Feed out of range values
        ars.feed_loudness(5.0)
        # Internal loudness should use clamped value (1.0)
        with ars._lock:
            self.assertLessEqual(ars._loudness, 1.0)

    def test_feed_loudness_uses_ema_smoothing(self):
        ars = AudioReactiveSway()
        ars.set_active(True)
        # Feed 1.0 once: loudness = 0.7*0 + 0.3*1.0 = 0.3
        ars.feed_loudness(1.0)
        with ars._lock:
            self.assertAlmostEqual(ars._loudness, 0.3, places=5)

        # Feed 1.0 again: loudness = 0.7*0.3 + 0.3*1.0 = 0.51
        ars.feed_loudness(1.0)
        with ars._lock:
            self.assertAlmostEqual(ars._loudness, 0.51, places=5)

    def test_right_antenna_inverted(self):
        ars = AudioReactiveSway()
        ars.set_active(True)
        for _ in range(20):
            ars.feed_loudness(1.0)
        # Right antenna uses negative amplitude (-5.0), left uses positive (5.0)
        # At the same time with the same phase, they should be opposite in sign
        offset = ars.get_offset(0.3)
        # They use the same phase_offset["antenna"], so they should be negated
        if abs(offset.left_antenna) > 0.01:
            self.assertAlmostEqual(
                offset.left_antenna, -offset.right_antenna, places=5,
                msg="Left and right antenna should be opposite in sign"
            )


# ---------------------------------------------------------------------------
# MotionManager tests (mock the SDK)
# ---------------------------------------------------------------------------


class TestMotionManager(unittest.TestCase):
    """Tests for MotionManager composition and tick logic. SDK is mocked."""

    def _make_manager(self):
        """Create a MotionManager with a mock reachy and mock _send_pose."""
        mock_reachy = MagicMock()
        manager = MotionManager(get_reachy=lambda: mock_reachy, tick_rate=50.0)
        # Mock _send_pose to avoid SDK imports
        manager._send_pose = MagicMock()
        return manager

    def test_submit_primary_queues_motion(self):
        manager = self._make_manager()
        seq = MotionSequence([(Pose(yaw=10), 0.5)])
        manager.submit_primary(seq)
        self.assertFalse(manager._primary_queue.empty())

    def test_submit_pose_convenience(self):
        manager = self._make_manager()
        manager.submit_pose(Pose(yaw=15), duration=0.3)
        self.assertFalse(manager._primary_queue.empty())

    def test_register_secondary(self):
        manager = self._make_manager()
        bm = BreathingMotion()
        manager.register_secondary("breathing", bm)
        self.assertIs(manager.get_secondary("breathing"), bm)

    def test_unregister_secondary(self):
        manager = self._make_manager()
        bm = BreathingMotion()
        manager.register_secondary("breathing", bm)
        manager.unregister_secondary("breathing")
        self.assertIsNone(manager.get_secondary("breathing"))

    def test_unregister_secondary_nonexistent_is_safe(self):
        manager = self._make_manager()
        # Should not raise
        manager.unregister_secondary("nonexistent")

    def test_get_secondary_returns_none_for_unknown(self):
        manager = self._make_manager()
        self.assertIsNone(manager.get_secondary("unknown"))

    def test_tick_with_no_motion_sends_zero_pose(self):
        manager = self._make_manager()
        # First tick should send (last_sent_pose is None, so always sends)
        manager._tick(0.0)
        manager._send_pose.assert_called_once()
        sent = manager._send_pose.call_args[0][0]
        self.assertAlmostEqual(sent.yaw, 0.0)
        self.assertAlmostEqual(sent.pitch, 0.0)

    def test_tick_with_primary_motion(self):
        manager = self._make_manager()
        seq = MotionSequence([(Pose(yaw=20), 1.0)])
        manager.submit_primary(seq)

        # First tick: starts the sequence at t=0.0
        manager._tick(0.0)
        manager._send_pose.assert_called_once()
        sent_first = manager._send_pose.call_args[0][0]
        # At t=0.0: interpolation from zero to 20 at progress 0 = 0
        self.assertAlmostEqual(sent_first.yaw, 0.0)

        # Reset mock, tick at midpoint
        manager._send_pose.reset_mock()
        manager._last_sent_pose = None  # force send
        manager._tick(0.5)
        manager._send_pose.assert_called_once()
        sent_mid = manager._send_pose.call_args[0][0]
        self.assertAlmostEqual(sent_mid.yaw, 10.0)

    def test_tick_with_secondary_motion(self):
        manager = self._make_manager()
        bm = BreathingMotion()
        manager.register_secondary("breathing", bm)

        manager._tick(1.0)
        manager._send_pose.assert_called_once()
        sent = manager._send_pose.call_args[0][0]
        # Primary is zero; secondary breathing at t=1.0 contributes pitch ~0.8
        self.assertAlmostEqual(sent.pitch, 0.8, places=2)

    def test_tick_composes_primary_and_secondary(self):
        manager = self._make_manager()

        # Submit primary that moves yaw to 20 over 1 second
        seq = MotionSequence([(Pose(yaw=20), 1.0)])
        manager.submit_primary(seq)

        # Register breathing (affects pitch)
        bm = BreathingMotion(pitch_amplitude=1.0, pitch_freq=0.25)
        manager.register_secondary("breathing", bm)

        # Tick at t=0 to start the primary
        manager._tick(0.0)
        manager._send_pose.reset_mock()
        manager._last_sent_pose = None

        # Tick at t=0.5: primary yaw=10, breathing pitch = 1.0 * sin(2*pi*0.25*0.5) = sin(pi/4) ~ 0.707
        manager._tick(0.5)
        sent = manager._send_pose.call_args[0][0]
        self.assertAlmostEqual(sent.yaw, 10.0, places=1)
        expected_pitch = 1.0 * math.sin(2 * math.pi * 0.25 * 0.5)
        self.assertAlmostEqual(sent.pitch, expected_pitch, places=2)

    def test_tick_clamps_composed_result(self):
        manager = self._make_manager()

        # Submit primary with extreme yaw over 1 second
        seq = MotionSequence([(Pose(yaw=100), 1.0)])
        manager.submit_primary(seq)

        # Tick at t=0 to start it
        manager._tick(0.0)
        manager._send_pose.reset_mock()
        manager._last_sent_pose = None

        # Tick at t=1.0: end of sequence, get_pose returns Pose(yaw=100)
        # But is_done(1.0) is True ((1.0 - 0.0) >= 1.0), so _tick uses
        # _last_primary_pose. We need to tick just before 1.0 so the pose
        # is still active and approaches 100.
        manager._tick(0.999)
        sent = manager._send_pose.call_args[0][0]
        # At t=0.999: progress=0.999, yaw=99.9, clamped to 45
        self.assertAlmostEqual(sent.yaw, 45.0)

    def test_tick_skips_update_when_pose_unchanged(self):
        manager = self._make_manager()

        # First tick sends zero pose
        manager._tick(0.0)
        self.assertEqual(manager._send_pose.call_count, 1)

        # Second tick with same zero pose should skip (close_to threshold=0.5)
        manager._tick(0.001)
        self.assertEqual(manager._send_pose.call_count, 1)

    def test_tick_sends_when_pose_changes_enough(self):
        manager = self._make_manager()

        # First tick: zero pose
        manager._tick(0.0)
        self.assertEqual(manager._send_pose.call_count, 1)

        # Submit a primary that moves significantly
        seq = MotionSequence([(Pose(yaw=30), 1.0)])
        manager.submit_primary(seq)

        # Tick at 1.0: starts sequence at t=1.0, pose is still zero (elapsed=0)
        manager._tick(1.0)
        # close_to(Pose(), Pose(), 0.5) is True, so skip. Count stays 1.
        self.assertEqual(manager._send_pose.call_count, 1)

        # Tick at 1.5: elapsed=0.5, progress=0.5, yaw=15. Change from 0 > 0.5 threshold
        manager._tick(1.5)
        self.assertEqual(manager._send_pose.call_count, 2)

    def test_tick_primary_done_holds_last_primary_pose(self):
        manager = self._make_manager()
        seq = MotionSequence([(Pose(yaw=20), 1.0)], hold_last=True)
        manager.submit_primary(seq)

        # t=0: start sequence
        manager._tick(0.0)

        # t=0.5: halfway, _last_primary_pose = Pose(yaw=10)
        manager._tick(0.5)

        # t=0.9: near end, _last_primary_pose = Pose(yaw=18)
        manager._tick(0.9)
        # At t=0.9: progress=0.9, yaw=18
        near_end_pose = manager._last_primary_pose
        self.assertAlmostEqual(near_end_pose.yaw, 18.0, places=1)

        # t=5.0: sequence is done, _tick uses _last_primary_pose (holds it)
        manager._send_pose.reset_mock()
        manager._last_sent_pose = None
        manager._tick(5.0)
        sent = manager._send_pose.call_args[0][0]
        # Should hold the last pose we got from the sequence (yaw ~18)
        self.assertAlmostEqual(sent.yaw, 18.0, places=1)

    def test_tick_drains_queue_latest_wins(self):
        manager = self._make_manager()

        # Submit two sequences -- the second should replace the first
        seq1 = MotionSequence([(Pose(yaw=10), 1.0)])
        seq2 = MotionSequence([(Pose(yaw=40), 1.0)])
        manager.submit_primary(seq1)
        manager.submit_primary(seq2)

        # First tick drains both, starts the last one
        manager._tick(0.0)
        manager._send_pose.reset_mock()
        manager._last_sent_pose = None

        # Midpoint: should be interpolating toward yaw=40, not yaw=10
        manager._tick(0.5)
        sent = manager._send_pose.call_args[0][0]
        self.assertAlmostEqual(sent.yaw, 20.0, places=1)

    def test_tick_inactive_secondary_ignored(self):
        manager = self._make_manager()
        bm = BreathingMotion()
        bm.set_active(False)
        manager.register_secondary("breathing", bm)

        manager._tick(1.0)
        sent = manager._send_pose.call_args[0][0]
        # Breathing is inactive, so pitch should be 0
        self.assertAlmostEqual(sent.pitch, 0.0)

    def test_tick_multiple_secondaries_compose(self):
        manager = self._make_manager()

        bm = BreathingMotion(pitch_amplitude=1.0, pitch_freq=0.25)
        manager.register_secondary("breathing", bm)

        rlm = ReactiveListeningMotion()
        rlm.set_state("listening")
        manager.register_secondary("listening", rlm)

        # Use a time far in the future so the ReactiveListening blend = 1.0
        far_t = time.monotonic() + 100
        manager._tick(far_t)
        sent = manager._send_pose.call_args[0][0]

        # Breathing pitch at far_t + listening pitch(-2) should compose
        breathing_pitch = 1.0 * math.sin(2 * math.pi * 0.25 * far_t)
        listening_pitch = -2.0
        expected_pitch = breathing_pitch + listening_pitch
        # Clamp expected pitch
        expected_clamped = max(-35, min(35, expected_pitch))
        self.assertAlmostEqual(sent.pitch, expected_clamped, places=1)

    def test_start_and_stop(self):
        manager = self._make_manager()
        manager.start()
        self.assertTrue(manager._running)
        self.assertIsNotNone(manager._thread)
        manager.stop()
        self.assertFalse(manager._running)

    def test_start_idempotent(self):
        manager = self._make_manager()
        manager.start()
        thread1 = manager._thread
        manager.start()  # Should not create a new thread
        self.assertIs(manager._thread, thread1)
        manager.stop()


if __name__ == "__main__":
    unittest.main()
