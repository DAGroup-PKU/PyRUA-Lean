from __future__ import annotations

import unittest

from datafarm.simulation import (
    CameraFramePerturbation,
    CameraRandomizationPolicy,
    PosePerturbationBounds,
    RecoveryTrace,
    audit_recovery_trace,
    audit_camera_realization,
    default_camera_randomization_policy,
    sample_episode_randomization,
)


ORDER = (("cup-1", "left"), ("cup-2", "left"), ("cup-3", "right"))


class EpisodeRandomizationTests(unittest.TestCase):
    def test_same_seed_is_identical_and_different_seed_changes_digest(self) -> None:
        first = sample_episode_randomization(seed=41, object_arm_order=ORDER)
        repeat = sample_episode_randomization(seed=41, object_arm_order=ORDER)
        other = sample_episode_randomization(seed=42, object_arm_order=ORDER)
        self.assertEqual(first, repeat)
        self.assertNotEqual(first.digest, other.digest)

    def test_camera_residuals_are_episode_constant_and_bounded(self) -> None:
        policy = default_camera_randomization_policy()
        plan = sample_episode_randomization(
            seed=7, object_arm_order=ORDER, camera_policy=policy
        )
        self.assertFalse(plan.per_frame_camera_jitter)
        self.assertTrue(plan.intrinsics_fixed)
        for row in plan.cameras:
            self.assertTrue(row.episode_constant)
            bounds = policy.bounds[row.camera_id]
            for value, limit in zip(row.translation_m, bounds.translation_half_range_m):
                self.assertLessEqual(abs(value), limit)
            for value, limit in zip(row.rotation_deg, bounds.rotation_half_range_deg):
                self.assertLessEqual(abs(value), limit)

    def test_per_frame_camera_jitter_is_rejected(self) -> None:
        bounds = PosePerturbationBounds((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            CameraRandomizationPolicy(
                bounds={
                    "head_camera": bounds,
                    "left_wrist_camera": bounds,
                    "right_wrist_camera": bounds,
                },
                per_frame_jitter=True,
            )

    def test_recoverable_miss_has_nonzero_offset(self) -> None:
        from datafarm.simulation import GraspRandomizationPolicy

        plan = sample_episode_randomization(
            seed=9,
            object_arm_order=ORDER,
            grasp_policy=GraspRandomizationPolicy(
                recoverable_miss_fraction=1.0, lateral_realign_fraction=0.0
            ),
        )
        self.assertTrue(all(row.mode == "recoverable_miss" for row in plan.grasps))
        self.assertTrue(all(row.miss_offset_xy_m is not None for row in plan.grasps))

    def test_realized_camera_must_match_plan_on_every_frame(self) -> None:
        plan = sample_episode_randomization(seed=12, object_arm_order=ORDER)
        rows = [
            CameraFramePerturbation(
                camera_id=camera.camera_id,
                frame_index=frame,
                translation_m=camera.translation_m,
                rotation_deg=camera.rotation_deg,
            )
            for camera in plan.cameras
            for frame in range(3)
        ]
        self.assertTrue(
            audit_camera_realization(plan, rows, expected_frame_count=3).passed
        )
        rows[-1] = CameraFramePerturbation(
            camera_id=rows[-1].camera_id,
            frame_index=rows[-1].frame_index,
            translation_m=(0.2, 0.0, 0.0),
            rotation_deg=rows[-1].rotation_deg,
        )
        result = audit_camera_realization(plan, rows, expected_frame_count=3)
        self.assertFalse(result.passed)
        self.assertIn("realized_pose_matches_plan", result.failures)


class RecoveryTraceTests(unittest.TestCase):
    def test_robot_only_ordered_recovery_passes(self) -> None:
        result = audit_recovery_trace(
            RecoveryTrace(
                object_id="cup-3",
                arm="right",
                phases=(
                    "misaligned_approach",
                    "retract",
                    "visual_reacquire",
                    "recenter",
                    "verified_close",
                    "lift",
                    "place",
                ),
                close_count=1,
                open_count=1,
                object_attached_before_verified_close=False,
                object_pose_writes=0,
                final_grasp_verified=True,
                active_wrist_reacquired_object=True,
            )
        )
        self.assertTrue(result.passed)

    def test_chatter_or_hidden_object_motion_is_rejected(self) -> None:
        result = audit_recovery_trace(
            RecoveryTrace(
                object_id="cup-3",
                arm="right",
                phases=("misaligned_approach", "verified_close"),
                close_count=4,
                open_count=3,
                object_attached_before_verified_close=True,
                object_pose_writes=1,
                final_grasp_verified=False,
                active_wrist_reacquired_object=False,
            )
        )
        self.assertFalse(result.passed)
        self.assertIn("ordered_recovery_phases", result.failures)
        self.assertIn("single_close_open", result.failures)
        self.assertIn("robot_only_object_motion", result.failures)


if __name__ == "__main__":
    unittest.main()


class LateralRealignTests(unittest.TestCase):
    """The replacement for retract-and-return: slide over the cup, never step back."""

    def test_offsets_use_the_four_named_directions_and_cover_them(self) -> None:
        from datafarm.simulation import GraspRandomizationPolicy
        from datafarm.simulation.randomization import LATERAL_REALIGN_UNIT

        seen = set()
        for seed in range(8):
            plan = sample_episode_randomization(
                seed=seed, object_arm_order=ORDER,
                grasp_policy=GraspRandomizationPolicy(lateral_realign_fraction=1.0),
            )
            for row in plan.grasps:
                self.assertEqual(row.mode, "lateral_realign")
                self.assertIn(row.realign_direction, LATERAL_REALIGN_UNIT)
                unit = LATERAL_REALIGN_UNIT[row.realign_direction]
                offset = row.realign_offset_xy_m
                self.assertIsNotNone(offset)
                magnitude = max(abs(offset[0]), abs(offset[1]))
                self.assertGreaterEqual(magnitude, 0.018)
                self.assertLessEqual(magnitude, 0.030)
                # the offset points exactly along the named axis
                self.assertAlmostEqual(offset[0], magnitude * unit[0], places=9)
                self.assertAlmostEqual(offset[1], magnitude * unit[1], places=9)
                seen.add(row.realign_direction)
        self.assertEqual(seen, set(LATERAL_REALIGN_UNIT))

    def test_audit_requires_a_slide_above_the_cup_and_no_revisit(self) -> None:
        from datafarm.simulation import RecoveryTrace, audit_lateral_realign_trace

        def trace(**overrides):
            base = dict(
                object_id="paper_cup_2", arm="right",
                phases=("offset_approach", "lateral_realign", "verified_close"),
                close_count=1, open_count=1, object_attached_before_verified_close=False,
                object_pose_writes=0, final_grasp_verified=True, active_wrist_reacquired_object=True,
                realign_clearance_m=0.03, revisited_pose=False,
            )
            base.update(overrides)
            return RecoveryTrace(**base)

        self.assertTrue(audit_lateral_realign_trace(trace()).passed)
        # the old retract-and-return shape is refused
        legacy = audit_lateral_realign_trace(trace(
            phases=("offset_approach", "retract", "visual_reacquire", "recenter", "verified_close"),
            revisited_pose=True))
        self.assertFalse(legacy.passed)
        self.assertIn("no_retract_phase", legacy.failures)
        self.assertIn("no_revisited_pose", legacy.failures)
        # sliding at or below the rim is refused
        self.assertIn("realign_stays_above_object",
                      audit_lateral_realign_trace(trace(realign_clearance_m=0.0)).failures)
        self.assertIn("single_close_open", audit_lateral_realign_trace(trace(close_count=2)).failures)
