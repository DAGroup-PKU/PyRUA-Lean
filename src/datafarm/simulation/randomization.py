"""Deterministic domain-randomization plans for policy demonstrations.

The sampler only produces immutable episode plans.  A simulator adapter must
apply the realized camera pose once, before capture, and must execute the grasp
plan with robot commands.  Sampling a value is not evidence that an episode is
physically valid; the resulting rollout still needs the normal physics and
observation gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from typing import Mapping, Sequence
from datafarm.common import sha256_text


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class PosePerturbationBounds:
    """Independent symmetric SE(3) residual bounds around a calibrated pose."""

    translation_half_range_m: Vector3
    rotation_half_range_deg: Vector3

    def __post_init__(self) -> None:
        values = self.translation_half_range_m + self.rotation_half_range_deg
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("pose perturbation bounds must be finite and non-negative")


@dataclass(frozen=True)
class CameraPerturbation:
    camera_id: str
    translation_m: Vector3
    rotation_deg: Vector3
    episode_constant: bool = True


@dataclass(frozen=True)
class CameraRandomizationPolicy:
    """Per-camera episode-constant extrinsic randomization policy."""

    bounds: Mapping[str, PosePerturbationBounds]
    intrinsics_fixed: bool = True
    per_frame_jitter: bool = False

    def __post_init__(self) -> None:
        required = {"head_camera", "left_wrist_camera", "right_wrist_camera"}
        if set(self.bounds) != required:
            raise ValueError(f"camera bounds must be exactly {sorted(required)}")
        if self.per_frame_jitter:
            raise ValueError("policy demonstrations forbid per-frame camera jitter")


@dataclass(frozen=True)
class GraspRandomizationPolicy:
    """Bounds for ordinary grasp jitter and explicit recoverable misses."""

    nominal_xy_half_range_m: float = 0.004
    nominal_z_half_range_m: float = 0.003
    nominal_yaw_half_range_deg: float = 3.0
    recoverable_miss_fraction: float = 0.25
    recoverable_miss_radius_m: tuple[float, float] = (0.018, 0.030)
    recovery_lift_m: float = 0.035
    maximum_realignments: int = 1
    lateral_realign_fraction: float = 0.25
    lateral_realign_offset_m: tuple[float, float] = (0.018, 0.030)
    lateral_realign_directions: tuple[str, ...] = ("forward", "backward", "left", "right")

    def __post_init__(self) -> None:
        scalars = (
            self.nominal_xy_half_range_m,
            self.nominal_z_half_range_m,
            self.nominal_yaw_half_range_deg,
            self.recovery_lift_m,
        )
        if any(not math.isfinite(value) or value < 0 for value in scalars):
            raise ValueError("grasp randomization magnitudes must be finite and non-negative")
        if not 0.0 <= self.recoverable_miss_fraction <= 1.0:
            raise ValueError("recoverable_miss_fraction must be in [0, 1]")
        low, high = self.recoverable_miss_radius_m
        if not (0.0 < low <= high and math.isfinite(low) and math.isfinite(high)):
            raise ValueError("recoverable miss radius must be a positive ordered range")
        if self.maximum_realignments != 1:
            raise ValueError("the production teaching trace permits exactly one realignment")
        if not 0.0 <= self.lateral_realign_fraction <= 1.0:
            raise ValueError("lateral_realign_fraction must be in [0, 1]")
        low, high = self.lateral_realign_offset_m
        if not (0.0 < low <= high and math.isfinite(low) and math.isfinite(high)):
            raise ValueError("lateral realign offset must be a positive ordered range")
        if set(self.lateral_realign_directions) != set(LATERAL_REALIGN_DIRECTIONS):
            raise ValueError(f"lateral realign must cover {LATERAL_REALIGN_DIRECTIONS}")


LATERAL_REALIGN_DIRECTIONS = ("forward", "backward", "left", "right")
LATERAL_REALIGN_UNIT: dict[str, tuple[float, float]] = {
    "forward": (1.0, 0.0), "backward": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0),
}


@dataclass(frozen=True)
class GraspPerturbation:
    object_id: str
    arm: str
    mode: str
    nominal_offset_xyz_m: Vector3
    yaw_offset_deg: float
    miss_offset_xy_m: tuple[float, float] | None
    recovery_lift_m: float
    maximum_realignments: int
    realign_direction: str | None = None
    realign_offset_xy_m: tuple[float, float] | None = None


@dataclass(frozen=True)
class EpisodeRandomizationPlan:
    schema_version: str
    seed: int
    cameras: tuple[CameraPerturbation, ...]
    grasps: tuple[GraspPerturbation, ...]
    intrinsics_fixed: bool
    per_frame_camera_jitter: bool
    digest: str


@dataclass(frozen=True)
class RecoveryTrace:
    """Simulator-derived evidence for one planned recoverable miss."""

    object_id: str
    arm: str
    phases: tuple[str, ...]
    close_count: int
    open_count: int
    object_attached_before_verified_close: bool
    object_pose_writes: int
    final_grasp_verified: bool
    active_wrist_reacquired_object: bool
    realign_clearance_m: float | None = None
    revisited_pose: bool | None = None


@dataclass(frozen=True)
class RecoveryAudit:
    passed: bool
    checks: Mapping[str, bool]
    failures: tuple[str, ...]


@dataclass(frozen=True)
class CameraFramePerturbation:
    camera_id: str
    frame_index: int
    translation_m: Vector3
    rotation_deg: Vector3


@dataclass(frozen=True)
class CameraRealizationAudit:
    passed: bool
    checks: Mapping[str, bool]
    failures: tuple[str, ...]


def default_camera_randomization_policy() -> CameraRandomizationPolicy:
    """Conservative default profile around calibrated nominal poses.

    Bounds are episode-constant.  Head bounds are tighter than wrist bounds; a
    task can replace both after its own held-out sweep.
    """

    return CameraRandomizationPolicy(
        bounds={
            "head_camera": PosePerturbationBounds(
                translation_half_range_m=(0.003, 0.003, 0.003),
                rotation_half_range_deg=(0.5, 0.5, 0.5),
            ),
            "left_wrist_camera": PosePerturbationBounds(
                translation_half_range_m=(0.002, 0.002, 0.002),
                rotation_half_range_deg=(0.75, 0.75, 0.75),
            ),
            "right_wrist_camera": PosePerturbationBounds(
                translation_half_range_m=(0.002, 0.002, 0.002),
                rotation_half_range_deg=(0.75, 0.75, 0.75),
            ),
        }
    )


def _uniform_vector(rng: random.Random, half_range: Vector3) -> Vector3:
    return tuple(rng.uniform(-value, value) for value in half_range)  # type: ignore[return-value]


def _canonical_payload(
    seed: int,
    cameras: Sequence[CameraPerturbation],
    grasps: Sequence[GraspPerturbation],
    *,
    intrinsics_fixed: bool,
    per_frame_camera_jitter: bool,
) -> dict[str, object]:
    return {
        "schema_version": "datafarm.episode-randomization/1.0.0",
        "seed": seed,
        "cameras": [row.__dict__ for row in cameras],
        "grasps": [row.__dict__ for row in grasps],
        "intrinsics_fixed": intrinsics_fixed,
        "per_frame_camera_jitter": per_frame_camera_jitter,
    }


def sample_episode_randomization(
    *,
    seed: int,
    object_arm_order: Sequence[tuple[str, str]],
    camera_policy: CameraRandomizationPolicy | None = None,
    grasp_policy: GraspRandomizationPolicy | None = None,
) -> EpisodeRandomizationPlan:
    """Create a deterministic, content-addressed episode randomization plan."""

    camera_policy = camera_policy or default_camera_randomization_policy()
    grasp_policy = grasp_policy or GraspRandomizationPolicy()
    rng = random.Random(seed)
    cameras = tuple(
        CameraPerturbation(
            camera_id=camera_id,
            translation_m=_uniform_vector(rng, bounds.translation_half_range_m),
            rotation_deg=_uniform_vector(rng, bounds.rotation_half_range_deg),
        )
        for camera_id, bounds in sorted(camera_policy.bounds.items())
    )
    grasps: list[GraspPerturbation] = []
    realign_cursor = seed % len(LATERAL_REALIGN_DIRECTIONS)
    for position, (object_id, arm) in enumerate(object_arm_order):
        if arm not in {"left", "right"}:
            raise ValueError(f"unknown arm {arm!r}")
        nominal = (
            rng.uniform(-grasp_policy.nominal_xy_half_range_m, grasp_policy.nominal_xy_half_range_m),
            rng.uniform(-grasp_policy.nominal_xy_half_range_m, grasp_policy.nominal_xy_half_range_m),
            rng.uniform(-grasp_policy.nominal_z_half_range_m, grasp_policy.nominal_z_half_range_m),
        )
        # A lateral realign is a recovery without a retract: the arm arrives
        # above the object at an offset, slides horizontally onto the staging
        # point and descends.  Nothing is revisited, so one
        # observation never carries two different futures.
        realign = rng.random() < grasp_policy.lateral_realign_fraction
        realign_direction = None
        realign_offset = None
        if realign:
            realign_direction = grasp_policy.lateral_realign_directions[
                (realign_cursor + position) % len(grasp_policy.lateral_realign_directions)
            ]
            magnitude = rng.uniform(*grasp_policy.lateral_realign_offset_m)
            unit = LATERAL_REALIGN_UNIT[realign_direction]
            realign_offset = (magnitude * unit[0], magnitude * unit[1])
        recover = (not realign) and rng.random() < grasp_policy.recoverable_miss_fraction
        miss_offset = None
        if recover:
            radius = rng.uniform(*grasp_policy.recoverable_miss_radius_m)
            theta = rng.uniform(-math.pi, math.pi)
            miss_offset = (radius * math.cos(theta), radius * math.sin(theta))
        grasps.append(
            GraspPerturbation(
                object_id=object_id,
                arm=arm,
                mode="lateral_realign" if realign else ("recoverable_miss" if recover else "nominal_jitter"),
                nominal_offset_xyz_m=nominal,
                yaw_offset_deg=rng.uniform(
                    -grasp_policy.nominal_yaw_half_range_deg,
                    grasp_policy.nominal_yaw_half_range_deg,
                ),
                miss_offset_xy_m=miss_offset,
                recovery_lift_m=grasp_policy.recovery_lift_m if recover else 0.0,
                maximum_realignments=grasp_policy.maximum_realignments if (recover or realign) else 0,
                realign_direction=realign_direction,
                realign_offset_xy_m=realign_offset,
            )
        )
    payload = _canonical_payload(
        seed,
        cameras,
        grasps,
        intrinsics_fixed=camera_policy.intrinsics_fixed,
        per_frame_camera_jitter=camera_policy.per_frame_jitter,
    )
    digest = sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return EpisodeRandomizationPlan(
        schema_version=str(payload["schema_version"]),
        seed=seed,
        cameras=cameras,
        grasps=tuple(grasps),
        intrinsics_fixed=camera_policy.intrinsics_fixed,
        per_frame_camera_jitter=camera_policy.per_frame_jitter,
        digest=digest,
    )


def audit_lateral_realign_trace(trace: RecoveryTrace, *, min_clearance_m: float = 0.0) -> RecoveryAudit:
    """Fail closed unless the offset approach is fixed by one horizontal slide.

    Unlike a retract-and-return recovery this trace must never step back onto
    a pose it already held: the arm approaches above the object at an offset,
    translates horizontally onto the staging point at the same height
    and descends.  ``clearance_m`` is the lowest height of the realign segment
    above the object, and ``revisited_pose`` reports whether any later pose
    repeats an earlier one (one observation would then have two futures).
    """

    required = ("offset_approach", "lateral_realign", "verified_close")
    cursor = -1
    ordered = True
    for phase in required:
        try:
            cursor = trace.phases.index(phase, cursor + 1)
        except ValueError:
            ordered = False
            break
    checks = {
        "ordered_realign_phases": ordered,
        "no_retract_phase": not any(phase in {"retract", "visual_reacquire", "recenter"} for phase in trace.phases),
        "realign_stays_above_object": trace.realign_clearance_m is not None and trace.realign_clearance_m > min_clearance_m,
        "no_revisited_pose": trace.revisited_pose is False,
        "single_close_open": trace.close_count == 1 and trace.open_count == 1,
        "no_hidden_attachment": not trace.object_attached_before_verified_close,
        "robot_only_object_motion": trace.object_pose_writes == 0,
        "active_wrist_reacquired_object": trace.active_wrist_reacquired_object,
        "final_grasp_verified": trace.final_grasp_verified,
    }
    return RecoveryAudit(
        passed=all(checks.values()),
        checks=checks,
        failures=tuple(name for name, passed in checks.items() if not passed),
    )


def audit_recovery_trace(trace: RecoveryTrace) -> RecoveryAudit:
    """Fail closed unless a miss is recovered by a clean robot-only sequence."""

    required = (
        "misaligned_approach",
        "retract",
        "visual_reacquire",
        "recenter",
        "verified_close",
    )
    cursor = -1
    ordered = True
    for phase in required:
        try:
            cursor = trace.phases.index(phase, cursor + 1)
        except ValueError:
            ordered = False
            break
    checks = {
        "ordered_recovery_phases": ordered,
        "single_close_open": trace.close_count == 1 and trace.open_count == 1,
        "no_hidden_attachment": not trace.object_attached_before_verified_close,
        "robot_only_object_motion": trace.object_pose_writes == 0,
        "active_wrist_reacquired_object": trace.active_wrist_reacquired_object,
        "final_grasp_verified": trace.final_grasp_verified,
    }
    return RecoveryAudit(
        passed=all(checks.values()),
        checks=checks,
        failures=tuple(name for name, passed in checks.items() if not passed),
    )


def audit_camera_realization(
    plan: EpisodeRandomizationPlan,
    frames: Sequence[CameraFramePerturbation],
    *,
    expected_frame_count: int,
    tolerance: float = 1e-9,
) -> CameraRealizationAudit:
    """Verify that each realized camera stays at its planned episode pose."""

    planned = {row.camera_id: row for row in plan.cameras}
    grouped = {camera_id: [] for camera_id in planned}
    unknown = False
    for row in frames:
        if row.camera_id not in grouped:
            unknown = True
            continue
        grouped[row.camera_id].append(row)
    coverage = expected_frame_count > 0 and all(
        sorted(row.frame_index for row in rows) == list(range(expected_frame_count))
        for rows in grouped.values()
    )

    def close(a: Sequence[float], b: Sequence[float]) -> bool:
        return all(abs(x - y) <= tolerance for x, y in zip(a, b))

    matches_plan = coverage and all(
        close(row.translation_m, planned[camera_id].translation_m)
        and close(row.rotation_deg, planned[camera_id].rotation_deg)
        for camera_id, rows in grouped.items()
        for row in rows
    )
    constant = coverage and all(
        all(
            close(row.translation_m, rows[0].translation_m)
            and close(row.rotation_deg, rows[0].rotation_deg)
            for row in rows[1:]
        )
        for rows in grouped.values()
    )
    checks = {
        "known_cameras_only": not unknown,
        "complete_frame_coverage": coverage,
        "realized_pose_matches_plan": matches_plan,
        "episode_constant_no_jitter": constant,
        "intrinsics_fixed_by_plan": plan.intrinsics_fixed,
        "plan_forbids_per_frame_jitter": not plan.per_frame_camera_jitter,
    }
    return CameraRealizationAudit(
        passed=all(checks.values()),
        checks=checks,
        failures=tuple(name for name, passed in checks.items() if not passed),
    )
