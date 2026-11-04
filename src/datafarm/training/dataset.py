"""Generated-video → LeRobot inline-224 materialization for Piper PI0.5.

The three-view split is an explicit, frozen layout rather than an inferred
``split_y = 3 * H // 4``.  For the 1280x1080 Seedance carrier that inference
cuts at row 810, while the carrier's head panel ends at row 720, so head images
would absorb 90 rows of wrist view and both wrist images would be truncated.
"""

from __future__ import annotations

from datafarm.common import read_json, sha256_file
from concurrent.futures import Executor
from dataclasses import dataclass
import importlib.util
from io import BytesIO
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


@dataclass(frozen=True)
class TLayout:
    """Pixel rectangles (y0, y1, x0, x1) of each policy view in a carrier frame."""

    layout_id: str
    width: int
    height: int
    views: tuple[tuple[int, int, int, int], ...]

    def split(self, image: np.ndarray) -> tuple[np.ndarray, ...]:
        if image.shape != (self.height, self.width, 3):
            raise ValueError(
                f"{self.layout_id}: carrier frame {image.shape} is not "
                f"{self.height}x{self.width}x3"
            )
        return tuple(image[y0:y1, x0:x1] for y0, y1, x0, x1 in self.views)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout_id": self.layout_id,
            "width": self.width,
            "height": self.height,
            "views": dict(zip(CAMERA_KEYS, [list(view) for view in self.views])),
        }


# Seedance three-view carrier: 1280x720 head over two 640x360 wrists.
SEEDANCE_T_LAYOUT_1280X1080 = TLayout(
    layout_id="seedance-t-1280x1080-head720-v1",
    width=1280,
    height=1080,
    views=((0, 720, 0, 1280), (720, 1080, 0, 640), (720, 1080, 640, 1280)),
)


def load_piper_fk_factory(source: Path) -> Callable[[], Any]:
    """Load the official Piper FK class from an explicitly pinned source file."""

    if not source.is_file():
        raise FileNotFoundError(f"Piper FK source does not exist: {source}")
    module_name = f"_datafarm_piper_fk_{sha256_file(source)[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Piper FK from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "C_PiperForwardKinematics", None)
    if cls is None:
        raise AttributeError(f"{source} has no C_PiperForwardKinematics")
    return lambda: cls(dh_is_offset=1)


@dataclass(frozen=True)
class PiperEefEncoder:
    """Convert simulator Piper joint readback to the real dataset's 14-D EEF convention."""

    fk_factory: Callable[[], Any]
    real_gripper_open_m: float = 0.105

    def __post_init__(self) -> None:
        if self.real_gripper_open_m <= 0:
            raise ValueError("Gripper open distance must be positive")

    def _arm_pose(self, qpos: Sequence[float]) -> np.ndarray:
        values = np.asarray(qpos, dtype=np.float64)
        if values.shape != (8,):
            raise ValueError(f"Expected 8 simulator arm qpos values, got {values.shape}")
        fk = self.fk_factory()
        pose = np.asarray(fk.CalFK(values[:6].tolist())[-1], dtype=np.float64)
        if pose.shape != (6,):
            raise ValueError(f"Piper FK returned an unexpected pose shape: {pose.shape}")
        pose[:3] *= 1e-3  # official SDK: millimetres -> metres
        pose[3:] *= math.pi / 180.0  # official SDK: degrees -> radians
        return pose

    def _command_gripper(self, record: Mapping[str, Any], arm: str) -> float:
        command = record.get("command")
        if not isinstance(command, Mapping) or not isinstance(command.get(arm), Mapping):
            raise ValueError(f"Frame-aligned record has no {arm} expert command")
        normalized = float(command[arm]["gripper_command"])
        if not np.isfinite(normalized) or not -1e-6 <= normalized <= 1.0 + 1e-6:
            raise ValueError(f"Invalid normalized {arm} gripper command: {normalized}")
        return float(np.clip(normalized, 0.0, 1.0) * self.real_gripper_open_m)

    def record_to_state(self, record: Mapping[str, Any]) -> np.ndarray:
        actual = record.get("actual")
        if not isinstance(actual, Mapping):
            raise ValueError("Frame-aligned record has no actual robot readback")
        state = np.concatenate(
            [
                self._arm_pose(actual["left_qpos"]),
                [self._command_gripper(record, "left")],
                self._arm_pose(actual["right_qpos"]),
                [self._command_gripper(record, "right")],
            ]
        ).astype(np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("Encoded EEF state must be finite and 14-D")
        return state

    def records_to_state_action(
        self, records: Iterable[Mapping[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        all_state = np.stack([self.record_to_state(record) for record in records])
        if len(all_state) < 2:
            raise ValueError("EEF conversion needs at least two source frames")
        state = all_state[:-1].copy()
        action = all_state[1:].copy()
        # Action pose is the next puppet EEF pose; both state and action keep
        # the current master gripper command, as in the real data.
        action[:, 6] = state[:, 6]
        action[:, 13] = state[:, 13]
        return state, action


RECOVERY_EXCISION_PHASES = ("misaligned_approach", "visual_reacquire", "retract", "recenter_exact_joint_state")
RECOVERY_SPLICE_MAX_JOINT_GAP_RAD = 0.01


def recovery_excisions(
    payload: Mapping[str, Any], *, max_joint_gap_rad: float = RECOVERY_SPLICE_MAX_JOINT_GAP_RAD
) -> tuple[dict[str, Any], ...]:
    """Frame ranges of the deliberate misaligned-approach recovery demonstrations.

    The recovery drives the arm to a wrong pose, retracts and recenters on the
    *same* joint target before the qualified close, so one observation carries
    two different futures.  A state-free policy cannot tell the two visits apart
    and can oscillate on the real robot.  Each range
    covers the whole detour, from the first misaligned command through the last
    recenter frame; the frames on either side of the cut must hold the same
    joint target, which is what makes the splice exact.
    """

    records = payload["records"]
    spans: dict[str, dict[str, list[int]]] = {}
    for record in records:
        name = record.get("execution_phase")
        if not name or name == "None":
            continue
        for phase in RECOVERY_EXCISION_PHASES:
            if str(name).endswith("_" + phase):
                spans.setdefault(str(name)[: -len(phase) - 1], {}).setdefault(phase, []).append(
                    int(record["frame_index"])
                )
    excisions: list[dict[str, Any]] = []
    for prefix, phases in sorted(spans.items()):
        if "misaligned_approach" not in phases:
            continue
        missing = [phase for phase in RECOVERY_EXCISION_PHASES if phase not in phases]
        if missing:
            raise ValueError(f"{prefix}: incomplete recovery demonstration, missing {missing}")
        start = min(phases["misaligned_approach"])
        end = max(phases["recenter_exact_joint_state"])
        if start <= 0 or end >= len(records) - 1 or end <= start:
            raise ValueError(f"{prefix}: recovery range [{start}, {end}] is not splice-able")
        before, after = records[start - 1]["actual"], records[end]["actual"]
        gap = max(
            abs(float(a) - float(b))
            for key in ("left_qpos", "right_qpos")
            for a, b in zip(before[key], after[key])
        )
        if gap > max_joint_gap_rad:
            raise ValueError(
                f"{prefix}: recovery does not return to the pre-detour joint state "
                f"({gap:.4f} rad > {max_joint_gap_rad} rad); splicing would teleport the arm"
            )
        excisions.append({"phase_prefix": prefix, "start_frame": start, "end_frame": end,
                          "dropped_frames": end - start + 1, "splice_joint_gap_rad": round(gap, 6)})
    return tuple(excisions)


def retained_record_indices(
    payload: Mapping[str, Any], excisions: Sequence[Mapping[str, Any]]
) -> list[int]:
    dropped = {index for cut in excisions for index in range(int(cut["start_frame"]), int(cut["end_frame"]) + 1)}
    return [index for index in range(len(payload["records"])) if index not in dropped]


def validate_frame_aligned_state(payload: Mapping[str, Any], *, expected_frames: int) -> None:
    if payload.get("schema_version") != "frame_aligned_robot_state/1.0.0":
        raise ValueError("Unsupported frame-aligned robot-state schema")
    if float(payload.get("fps", -1)) != 30.0:
        raise ValueError(f"PI0.5 training contract requires 30 Hz state, got {payload.get('fps')}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != expected_frames:
        raise ValueError(
            f"Robot-state frame count {len(records) if isinstance(records, list) else None} "
            f"does not match video frame count {expected_frames}"
        )
    for index, record in enumerate(records):
        if record.get("frame_index") != index:
            raise ValueError(f"Non-contiguous frame_index at record {index}")


def layout_seam_report(
    image: np.ndarray | Sequence[np.ndarray], layout: TLayout, min_ratio: float = 8.0
) -> dict[str, Any]:
    """Measure that the declared panel boundaries are real seams of the frame.

    A boundary passes when the mean absolute step across it is at least ``min_ratio``
    times the median step over the frame: a split at a row/column with no
    discontinuity (a 3H/4 cut at row 810) scores about 1. Whether the
    declared seam is the *strongest* edge is reported but not gated, because a
    straight scene edge (the wall/table horizon in the head view, a stick on a dark
    table) can be sharper than a correct seam. With several frames the best frame
    is used per boundary (a generated render can make both sides of a seam the same
    colour in some frames); metrics are informational for renders, whose layout is
    gated on the carrier and on per-frame carrier alignment.
    """

    images = [image] if isinstance(image, np.ndarray) and image.ndim == 3 else list(image)
    head_bottom = layout.views[0][1]
    wrist_split = layout.views[1][3]
    row_ratios, col_ratios, strongest_rows, strongest_cols = [], [], [], []
    for frame in images:
        gray = frame.astype(np.float32).mean(axis=2)
        row = np.abs(np.diff(gray, axis=0)).mean(axis=1)
        col = np.abs(np.diff(gray[head_bottom:], axis=1)).mean(axis=0)
        row_ratios.append(float(row[head_bottom - 1] / (float(np.median(row)) or 1e-6)))
        col_ratios.append(float(col[wrist_split - 1] / (float(np.median(col)) or 1e-6)))
        strongest_rows.append(int(np.argmax(row)) + 1)
        strongest_cols.append(int(np.argmax(col)) + 1)
    report = {
        "frames": len(images),
        "declared_head_bottom": head_bottom,
        "declared_wrist_split": wrist_split,
        "row_seam_ratio": max(row_ratios),
        "col_seam_ratio": max(col_ratios),
        "strongest_row_seam": strongest_rows[int(np.argmax(row_ratios))],
        "strongest_bottom_col_seam": strongest_cols[int(np.argmax(col_ratios))],
        "min_ratio": min_ratio,
    }
    report["passed"] = report["row_seam_ratio"] >= min_ratio and report["col_seam_ratio"] >= min_ratio
    return report


def decode_carrier(
    path: Path, expected_frames: int, layout: TLayout
) -> Iterator[tuple[np.ndarray, ...]]:
    """Yield the first ``expected_frames`` split frames; the carrier holds one more."""

    import av  # noqa: PLC0415

    with av.open(str(path)) as container:
        streams = container.streams.video
        if len(streams) != 1:
            raise ValueError(f"Expected one video stream in {path}")
        count = 0
        for frame in container.decode(streams[0]):
            count += 1
            if count <= expected_frames:
                yield layout.split(frame.to_ndarray(format="rgb24"))
        if count != expected_frames + 1:
            raise ValueError(f"Carrier {path} has {count} frames; expected {expected_frames + 1}")


def sampled_carrier_frames(path: Path, count: int = 12, stride: int = 60) -> list[np.ndarray]:
    """Spread a few carrier frames for the seam report instead of just the first.

    layout_seam_report takes the best frame per boundary precisely because a
    generated render can paint both sides of a seam the same colour in some
    frames. Judging a whole episode on frame 0 therefore rejects sound carriers:
    an OOD sample scored 6.19 on its first frame (below the 8.0 bound) but 17.85
    across 19 sampled frames, with the strongest row and column seams landing
    exactly on the declared 720/640 boundaries.
    """

    import av  # noqa: PLC0415

    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index % stride == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= count:
                    break
    if not frames:
        raise ValueError(f"Carrier {path} has no frames")
    return frames


def png_bytes(image: np.ndarray) -> bytes:
    from PIL import Image  # noqa: PLC0415

    output = BytesIO()
    Image.fromarray(image, mode="RGB").save(output, format="PNG")
    return output.getvalue()


def sample_indices(length: int) -> set[int]:
    count = max(min(100, length), min(int(length**0.75), 10_000))
    return set(np.round(np.linspace(0, length - 1, count)).astype(int).tolist())


def feature_stats(array: np.ndarray, *, image: bool = False) -> dict[str, Any]:
    if image:
        axes: tuple[int, ...] | int = (0, 2, 3)
        keepdims = True
    else:
        axes = 0
        keepdims = array.ndim == 1
    stats = {
        "min": np.min(array, axis=axes, keepdims=keepdims),
        "max": np.max(array, axis=axes, keepdims=keepdims),
        "mean": np.mean(array, axis=axes, keepdims=keepdims),
        "std": np.std(array, axis=axes, keepdims=keepdims),
        "count": np.asarray([len(array)]),
    }
    if image:
        stats = {
            key: value if key == "count" else np.squeeze(value / 255.0, axis=0)
            for key, value in stats.items()
        }
    return {key: np.asarray(value).tolist() for key, value in stats.items()}


def materialize_episode(
    *,
    sample: Mapping[str, Any],
    generated_video: Path,
    expected_video_sha256: str,
    review: Mapping[str, Any],
    output_episode: int,
    global_index: int,
    schema: Any,
    target_root: Path,
    encoder: PiperEefEncoder,
    resize_with_pad: Callable[..., Any],
    executor: Executor,
    layout: TLayout = SEEDANCE_T_LAYOUT_1280X1080,
    batch_frames: int = 16,
    excise_recovery: bool = False,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Write one LeRobot v2.1 parquet episode with inline 224x224 PNG views."""

    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as parquet  # noqa: PLC0415

    sample_id = str(sample["sample_id"])
    source_frames = int(sample["frame_count"])
    expected_frames = source_frames - 1
    state_path = Path(sample["frame_aligned_robot_state"]["path"])
    state_payload = read_json(state_path)
    validate_frame_aligned_state(state_payload, expected_frames=source_frames)
    excisions = recovery_excisions(state_payload) if excise_recovery else ()
    retained = retained_record_indices(state_payload, excisions)
    records = [state_payload["records"][index] for index in retained]
    state, action = encoder.records_to_state_action(records)
    expected_frames = len(records) - 1
    if len(state) != expected_frames:
        raise ValueError(f"{sample_id}: state length mismatch")
    # Image i pairs with state i, so the kept media frames are the kept poses
    # except the last, which only supplies the final action target.
    retained_media_set = set(retained[:-1])

    generated_hash = sha256_file(generated_video)
    if generated_hash != expected_video_sha256:
        raise ValueError(f"{sample_id}: generated video SHA-256 drift")
    seam = layout_seam_report(sampled_carrier_frames(generated_video), layout)
    if not seam["passed"]:
        raise ValueError(f"{sample_id}: carrier seams do not match {layout.layout_id}: {seam}")

    encoded_views: list[list[bytes]] = [[], [], []]
    sampled_views: list[list[np.ndarray]] = [[], [], []]
    first_resized: list[np.ndarray] = []
    selected = sample_indices(expected_frames)
    frames = decode_carrier(generated_video, source_frames - 1, layout)
    kept = (frame for index, frame in enumerate(frames) if index in retained_media_set)
    processed = 0
    while processed < expected_frames:
        take = min(batch_frames, expected_frames - processed)
        batch = [next(kept) for _ in range(take)]
        for view_index in range(3):
            images = np.stack([frame[view_index] for frame in batch])
            if take < batch_frames:
                pad = np.repeat(images[-1:], batch_frames - take, axis=0)
                images = np.concatenate([images, pad], axis=0)
            resized = np.asarray(resize_with_pad(images, 224, 224))[:take]
            if resized.dtype != np.uint8 or resized.shape != (take, 224, 224, 3):
                raise ValueError(f"{sample_id}: unexpected resized batch {resized.shape}/{resized.dtype}")
            if processed == 0:
                first_resized.append(resized[0].copy())
            encoded_views[view_index].extend(executor.map(png_bytes, list(resized)))
            for local_index, image in enumerate(resized):
                if processed + local_index in selected:
                    sampled_views[view_index].append(np.moveaxis(image, -1, 0))
        processed += take
    for _ in kept:
        raise ValueError(f"{sample_id}: carrier decoder yielded an unexpected extra frame")

    timestamps = np.arange(expected_frames, dtype=np.float32) / np.float32(30.0)
    frame_indices = np.arange(expected_frames, dtype=np.int64)
    episode_indices = np.full(expected_frames, output_episode, dtype=np.int64)
    indices = np.arange(global_index, global_index + expected_frames, dtype=np.int64)
    task_indices = np.zeros(expected_frames, dtype=np.int64)

    def fixed_list(values: np.ndarray) -> Any:
        return pa.FixedSizeListArray.from_arrays(
            pa.array(values.reshape(-1), type=pa.float32()), values.shape[1]
        )

    def image_struct(values: list[bytes]) -> Any:
        return pa.StructArray.from_arrays(
            [
                pa.array(values, type=pa.binary()),
                pa.array([f"frame_{index:06d}.png" for index in range(expected_frames)]),
            ],
            names=["bytes", "path"],
        )

    arrays = [
        fixed_list(state.astype(np.float32)),
        fixed_list(action.astype(np.float32)),
        image_struct(encoded_views[0]),
        image_struct(encoded_views[1]),
        image_struct(encoded_views[2]),
        pa.array(timestamps, type=pa.float32()),
        pa.array(frame_indices, type=pa.int64()),
        pa.array(episode_indices, type=pa.int64()),
        pa.array(indices, type=pa.int64()),
        pa.array(task_indices, type=pa.int64()),
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    output_path = target_root / "data" / "chunk-000" / f"episode_{output_episode:06d}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, output_path, compression="snappy")

    # Round-trip gate: the stored first frame must decode to exactly the
    # declared-layout crop after resize_with_pad.
    stored = parquet.read_table(output_path, columns=list(CAMERA_KEYS)).slice(0, 1).to_pylist()[0]
    from PIL import Image  # noqa: PLC0415

    for key, expected in zip(CAMERA_KEYS, first_resized):
        decoded = np.asarray(Image.open(BytesIO(stored[key]["bytes"])).convert("RGB"))
        if not np.array_equal(decoded, expected):
            raise ValueError(f"{sample_id}: stored {key} frame 0 does not round-trip")

    stats = {
        "observation.state": feature_stats(state),
        "action": feature_stats(action),
        CAMERA_KEYS[0]: feature_stats(np.stack(sampled_views[0]), image=True),
        CAMERA_KEYS[1]: feature_stats(np.stack(sampled_views[1]), image=True),
        CAMERA_KEYS[2]: feature_stats(np.stack(sampled_views[2]), image=True),
        "timestamp": feature_stats(timestamps),
        "frame_index": feature_stats(frame_indices),
        "episode_index": feature_stats(episode_indices),
        "index": feature_stats(indices),
        "task_index": feature_stats(task_indices),
    }
    detail = {
        "output_episode_index": output_episode,
        "sample_id": sample_id,
        "frame_count": expected_frames,
        "source_frame_count": source_frames,
        "recovery_excisions": list(excisions),
        "recovery_excised_frames": sum(int(cut["dropped_frames"]) for cut in excisions),
        "state_path": str(state_path),
        "state_sha256": sha256_file(state_path),
        "image_source": "generated",
        "generated_video_path": str(generated_video),
        "generated_video_sha256": generated_hash,
        "carrier_layout": layout.layout_id,
        "carrier_seam_check": seam,
        "first_frame_round_trip": "exact",
        "semantic_checks": review["checks"],
        "semantic_waivers": review.get("waivers") or {},
        "source_reuse": "newly_materialized_from_canonical_generated_composite",
    }
    return expected_frames, stats, detail


def audit_action_shift(dataset_root: Path) -> dict[str, Any]:
    """Verify action pose = next state pose and action gripper = current gripper."""

    import pyarrow.parquet as parquet  # noqa: PLC0415

    files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot parquet episodes below {dataset_root}")
    maximum_pose_error = 0.0
    maximum_gripper_error = 0.0
    checked_pairs = 0
    pose_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    gripper_indices = [6, 13]
    for path in files:
        table = parquet.read_table(path, columns=["observation.state", "action"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != 14 or action.shape != state.shape:
            raise ValueError(f"Unexpected state/action arrays in {path}: {state.shape}, {action.shape}")
        if len(state) > 1:
            error = float(np.max(np.abs(action[:-1, pose_indices] - state[1:, pose_indices])))
            maximum_pose_error = max(maximum_pose_error, error)
            checked_pairs += len(state) - 1
        gripper_error = float(np.max(np.abs(action[:, gripper_indices] - state[:, gripper_indices])))
        maximum_gripper_error = max(maximum_gripper_error, gripper_error)
    if maximum_pose_error > 1e-6 or maximum_gripper_error > 1e-6:
        raise ValueError(
            "Dataset does not match the shifted EEF/current-gripper contract: "
            f"pose error {maximum_pose_error}, gripper error {maximum_gripper_error}"
        )
    return {
        "episodes": len(files),
        "checked_transition_pairs": checked_pairs,
        "max_pose_shift_abs_error": maximum_pose_error,
        "max_current_gripper_abs_error": maximum_gripper_error,
        "contract": "action pose is next EEF pose; action gripper equals current state gripper",
    }


def audit_joint_action_shift(dataset_root: Path, *, gripper_unit: str) -> dict[str, Any]:
    """Verify next-state absolute joint targets and current absolute grippers."""

    import pyarrow.parquet as parquet  # noqa: PLC0415

    from datafarm.training.joint_actions import GRIPPER_INDICES, GRIPPER_RANGES, JOINT_INDICES

    if gripper_unit not in GRIPPER_RANGES:
        raise ValueError(f"Unsupported joint gripper unit: {gripper_unit!r}")
    files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot parquet episodes below {dataset_root}")
    maximum_joint_error = 0.0
    maximum_gripper_error = 0.0
    checked_pairs = 0
    grip_min, grip_max = float("inf"), float("-inf")
    for path in files:
        table = parquet.read_table(path, columns=["observation.state", "action"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != 14 or action.shape != state.shape:
            raise ValueError(f"Unexpected state/action arrays in {path}: {state.shape}, {action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"Non-finite joint state/action in {path}")
        if len(state) > 1:
            error = float(np.max(np.abs(action[:-1, JOINT_INDICES] - state[1:, JOINT_INDICES])))
            maximum_joint_error = max(maximum_joint_error, error)
            checked_pairs += len(state) - 1
        gripper_error = float(np.max(np.abs(action[:, GRIPPER_INDICES] - state[:, GRIPPER_INDICES])))
        maximum_gripper_error = max(maximum_gripper_error, gripper_error)
        grips = np.concatenate((state[:, GRIPPER_INDICES].reshape(-1), action[:, GRIPPER_INDICES].reshape(-1)))
        grip_min, grip_max = min(grip_min, float(grips.min())), max(grip_max, float(grips.max()))
    allowed_min, allowed_max = GRIPPER_RANGES[gripper_unit]
    if grip_min < allowed_min - 1e-6 or grip_max > allowed_max + 1e-6:
        raise ValueError(
            f"Joint gripper values [{grip_min}, {grip_max}] violate declared {gripper_unit} "
            f"range [{allowed_min}, {allowed_max}]"
        )
    if maximum_joint_error > 1e-6 or maximum_gripper_error > 1e-6:
        raise ValueError(
            "Dataset does not match the next-joint/current-gripper contract: "
            f"joint error {maximum_joint_error}, gripper error {maximum_gripper_error}"
        )
    return {
        "episodes": len(files),
        "checked_transition_pairs": checked_pairs,
        "max_joint_shift_abs_error_rad": maximum_joint_error,
        "max_current_gripper_abs_error": maximum_gripper_error,
        "observed_gripper_range": [grip_min, grip_max],
        "gripper_unit": gripper_unit,
        "contract": "action joints are next absolute joint positions; action grippers equal current state grippers",
    }


def _rotation_matrices(rpy: np.ndarray) -> np.ndarray:
    """Piper SDK RPY (R = Rz(yaw) Ry(pitch) Rx(roll)) to rotation matrices."""

    roll, pitch, yaw = (np.asarray(rpy, dtype=np.float64)[..., index] for index in range(3))
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    matrix = np.empty(roll.shape + (3, 3))
    matrix[..., 0, 0] = cy * cp
    matrix[..., 0, 1] = cy * sp * sr - sy * cr
    matrix[..., 0, 2] = cy * sp * cr + sy * sr
    matrix[..., 1, 0] = sy * cp
    matrix[..., 1, 1] = sy * sp * sr + cy * cr
    matrix[..., 1, 2] = sy * sp * cr - cy * sr
    matrix[..., 2, 0] = -sp
    matrix[..., 2, 1] = cp * sr
    matrix[..., 2, 2] = cp * cr
    return matrix


def geodesic_angle(rpy: np.ndarray, reference_rpy: np.ndarray) -> np.ndarray:
    """Rotation angle between orientations; unlike RPY deltas it has no gimbal-lock gain."""

    relative = np.swapaxes(_rotation_matrices(reference_rpy), -1, -2) @ _rotation_matrices(rpy)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _angular_error(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(values - reference), np.cos(values - reference)))


def audit_terminal_home(dataset_root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Verify that every episode ends stationary at the same canonical home."""

    import pyarrow.parquet as parquet  # noqa: PLC0415

    schema = contract.get("schema_version")
    if schema != "eef_terminal_home_contract/1.1.0":
        raise ValueError("Missing or unsupported terminal-home contract")
    # The tail is bounded in translation (m), geodesic rotation (rad) and gripper (m),
    # because RPY amplifies tiny rotations near pitch 90 deg.
    tail_translation_limit = float(contract["max_stationary_tail_translation_m"])
    tail_rotation_limit = float(contract["max_stationary_tail_rotation_rad"])
    tail_gripper_limit = float(contract["max_stationary_tail_gripper_m"])
    if min(tail_translation_limit, tail_rotation_limit, tail_gripper_limit) < 0:
        raise ValueError("Invalid terminal-home tail tolerances")
    try:
        reference = np.asarray(contract.get("canonical_home_state_14d"), dtype=np.float64)
    except (TypeError, ValueError):
        reference = np.empty(0)
    if reference.shape != (14,):
        raise ValueError("terminal-home canonical_home_state_14d must contain 14 values")
    translation_limit = float(contract["max_translation_error_m"])
    rotation_limit = float(contract["max_rotation_error_rad"])
    gripper_limit = float(contract["max_gripper_error_m"])
    tail_frames = int(contract.get("stationary_tail_frames", 10))
    if min(translation_limit, rotation_limit, gripper_limit) < 0 or tail_frames < 2:
        raise ValueError("Invalid terminal-home tolerances")

    files = sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot parquet episodes below {dataset_root}")
    failures: list[dict[str, Any]] = []
    maxima: dict[str, float] = {
        "translation_error_m": 0.0,
        "rotation_error_rad": 0.0,
        "gripper_error_m": 0.0,
    }
    for path in files:
        table = parquet.read_table(path, columns=["observation.state", "action"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != 14 or action.shape != state.shape:
            raise ValueError(f"Unexpected state/action arrays in {path}: {state.shape}, {action.shape}")
        if len(state) < tail_frames:
            raise ValueError(f"Episode is shorter than the stationary tail: {path}")
        final = state[-1]
        translation_error = max(
            float(np.linalg.norm(final[0:3] - reference[0:3])),
            float(np.linalg.norm(final[7:10] - reference[7:10])),
        )
        rotation_error = max(
            float(np.max(_angular_error(final[3:6], reference[3:6]))),
            float(np.max(_angular_error(final[10:13], reference[10:13]))),
        )
        gripper_error = float(np.max(np.abs(final[[6, 13]] - reference[[6, 13]])))
        tail = state[-tail_frames:]
        tail_translation = max(
            float(np.max(np.linalg.norm(tail[:, 0:3] - final[0:3], axis=1))),
            float(np.max(np.linalg.norm(tail[:, 7:10] - final[7:10], axis=1))),
        )
        tail_rotation = max(
            float(np.max(geodesic_angle(tail[:, 3:6], final[3:6]))),
            float(np.max(geodesic_angle(tail[:, 10:13], final[10:13]))),
        )
        tail_gripper = float(np.max(np.abs(tail[:, [6, 13]] - final[[6, 13]])))
        tail_delta = max(tail_translation / tail_translation_limit if tail_translation_limit else tail_translation,
                         tail_rotation / tail_rotation_limit if tail_rotation_limit else tail_rotation,
                         tail_gripper / tail_gripper_limit if tail_gripper_limit else tail_gripper)
        tail_ok = (
            tail_translation <= tail_translation_limit
            and tail_rotation <= tail_rotation_limit
            and tail_gripper <= tail_gripper_limit
        )
        for key, value in (("stationary_tail_translation_m", tail_translation),
                           ("stationary_tail_rotation_rad", tail_rotation),
                           ("stationary_tail_gripper_m", tail_gripper)):
            maxima[key] = max(maxima.get(key, 0.0), value)
        maxima["translation_error_m"] = max(maxima["translation_error_m"], translation_error)
        maxima["rotation_error_rad"] = max(maxima["rotation_error_rad"], rotation_error)
        maxima["gripper_error_m"] = max(maxima["gripper_error_m"], gripper_error)
        reasons = []
        if translation_error > translation_limit:
            reasons.append("translation_not_home")
        if rotation_error > rotation_limit:
            reasons.append("rotation_not_home")
        if gripper_error > gripper_limit:
            reasons.append("gripper_not_canonical")
        if not tail_ok:
            reasons.append("terminal_tail_not_stationary")
        if np.all(np.abs(action[-1]) < 1e-8):
            reasons.append("invalid_all_zero_absolute_action")
        if reasons:
            failures.append({
                "episode": path.stem,
                "reasons": reasons,
                "translation_error_m": translation_error,
                "rotation_error_rad": rotation_error,
                "gripper_error_m": gripper_error,
                "stationary_tail_normalized": tail_delta,
            })

    result = {
        "schema_version": "eef_terminal_home_audit/1.0.0",
        "episodes": len(files),
        "passed": len(files) - len(failures),
        "failed": len(failures),
        "contract": dict(contract),
        "observed_maxima": maxima,
        "failures": failures,
        "scope": (
            "Dataset-level EEF/gripper/stationarity defence-in-depth; source rollout must also "
            "verify joint home, qvel, collision-free homing sweep, and contact clearance."
        ),
    }
    if failures:
        raise ValueError(
            "Terminal-home admission failed for "
            f"{len(failures)}/{len(files)} episodes; examples={failures[:5]}"
        )
    return result


# ---------------------------------------------------------------------------
# Canonical-HDF5 + uniformly scaled provider render
# ---------------------------------------------------------------------------

