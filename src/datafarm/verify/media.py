"""Decoding into a layout-normalised working representation.

Generated videos may arrive at a provider-specific resolution (for example a
uniform 1.225x upscale).  Every clip is therefore resized onto the carrier's
layout coordinates before any comparison, while the native size is kept for the
transport check.  Two copies are held:

* ``frames``: colour frames at ``work_scale`` of the layout, every ``stride``-th
  frame (sample ``j`` is source frame ``j * stride``);
* ``gray``: every frame, grayscale, at ``gray_scale`` of the layout, used for
  frame-rate-exact temporal, seam and event measurements.
"""

from __future__ import annotations

from datafarm.common import sha256_file
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Clip:
    frames: np.ndarray
    gray: np.ndarray
    stride: int
    fps: float
    frame_count: int
    native_size: tuple[int, int]
    layout_size: tuple[int, int]
    work_scale: float
    gray_scale: float
    sha256: str | None = None
    source: str | None = None

    @property
    def sample_frames(self) -> np.ndarray:
        return np.arange(len(self.frames)) * self.stride

    def sample_of(self, frame: int) -> int:
        return int(min(len(self.frames) - 1, max(0, round(frame / self.stride))))


def _sizes(layout_size: tuple[int, int], work_scale: float, gray_scale: float):
    lw, lh = layout_size
    work = (max(8, int(round(lw * work_scale))), max(8, int(round(lh * work_scale))))
    gray = (max(8, int(round(lw * gray_scale))), max(8, int(round(lh * gray_scale))))
    return work, gray


def load_clip(
    path: str | Path,
    *,
    layout_size: tuple[int, int],
    work_scale: float = 0.5,
    gray_scale: float = 0.25,
    stride: int = 2,
    hash_file: bool = True,
    max_frames: int | None = None,
) -> Clip:
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    native = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    work_size, gray_size = _sizes(layout_size, work_scale, gray_scale)
    frames: list[np.ndarray] = []
    grays: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if max_frames is not None and index >= max_frames:
            break
        if (frame.shape[1], frame.shape[0]) != tuple(layout_size):
            frame = cv2.resize(frame, tuple(layout_size), interpolation=cv2.INTER_AREA)
        if index % stride == 0:
            frames.append(cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA))
        small = cv2.resize(frame, gray_size, interpolation=cv2.INTER_AREA)
        grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        index += 1
    capture.release()
    empty_work = np.zeros((0, work_size[1], work_size[0], 3), np.uint8)
    empty_gray = np.zeros((0, gray_size[1], gray_size[0]), np.uint8)
    return Clip(
        frames=np.stack(frames) if frames else empty_work,
        gray=np.stack(grays) if grays else empty_gray,
        stride=stride,
        fps=fps,
        frame_count=index,
        native_size=native,
        layout_size=tuple(layout_size),
        work_scale=work_scale,
        gray_scale=gray_scale,
        sha256=sha256_file(path) if hash_file else None,
        source=str(path),
    )


def load_reference_clip(path: str | Path, *, layout_size: tuple[int, int], work_scale: float = 0.5,
                        target_samples: int = 120) -> Clip:
    """Real-world footage: sample about ``target_samples`` colour frames whatever its length."""

    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    stride = max(1, count // max(1, target_samples))
    return load_clip(path, layout_size=layout_size, work_scale=work_scale, stride=stride)


def derive_clip(base: Clip, *, frames: np.ndarray | None = None, gray: np.ndarray | None = None,
                **changes) -> Clip:
    """Copy a clip with replaced arrays (used by counterfactual generators)."""

    values = dict(base.__dict__)
    if frames is not None:
        values["frames"] = frames
    if gray is not None:
        values["gray"] = gray
        values["frame_count"] = len(gray)
    values.update(changes)
    values["sha256"] = None
    return Clip(**values)


def panel_slice(box, scale: float, shape: tuple[int, int]) -> tuple[slice, slice]:
    """Rows/cols of a layout-coordinate box at the given scale, clipped to shape."""

    height, width = shape
    x0 = int(round(box.x * scale))
    y0 = int(round(box.y * scale))
    x1 = int(round((box.x + box.w) * scale))
    y1 = int(round((box.y + box.h) * scale))
    return slice(max(0, y0), min(height, y1)), slice(max(0, x0), min(width, x1))
