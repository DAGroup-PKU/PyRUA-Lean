"""Project simulator object poses into every panel.

The Episode IR records each object's pose and every camera's intrinsics and
extrinsics per frame.  Projecting a simple bounding primitive (cylinder or box)
gives the full image footprint of an entity — including parts a colour key
cannot see in the carrier, such as the interior of a white container — so "is this rendered
blob backed by a real object?" is answered by simulator geometry, not by colour.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Mapping

import cv2
import numpy as np

from datafarm.verify.spec import EntitySpec, RenderVerifierSpec


def _rotation(q_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in q_wxyz)
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def primitive_points(geometry: Mapping[str, Any]) -> np.ndarray:
    kind = geometry.get("type", "cylinder")
    if kind in ("cylinder", "frustum"):
        if kind == "cylinder":
            r0 = r1 = float(geometry["radius"])
        else:
            r0, r1 = float(geometry["radius_bottom"]), float(geometry["radius_top"])
        height = float(geometry["height"])
        z0 = float(geometry.get("base_offset", 0.0))
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        unit = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return np.concatenate([np.c_[r0 * unit, np.full(24, z0)], np.c_[r1 * unit, np.full(24, z0 + height)]])
    if kind == "box":
        hx, hy, hz = (float(v) / 2 for v in geometry["size"])
        z0 = float(geometry.get("base_offset", -hz))
        corners = [(sx * hx, sy * hy, z0 + (hz * 2 if sz > 0 else 0.0))
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        return np.array(corners)
    raise ValueError(f"unknown geometry type {kind}")


def project_footprints(spec: RenderVerifierSpec, entity: EntitySpec, state: Mapping[str, Any],
                       sample_frames: np.ndarray, work_shape: tuple[int, int]) -> np.ndarray | None:
    """Boolean [n, H, W] work-scale masks of the projected entity footprint."""

    if not entity.state_objects or not entity.geometry:
        return None
    records = state["records"]
    points = primitive_points(entity.geometry)
    scale = spec.work_scale
    out = np.zeros((len(sample_frames),) + tuple(work_shape), bool)
    for index, frame in enumerate(sample_frames):
        record = records[min(int(frame), len(records) - 1)]
        objects = record.get("objects") or {}
        names = [n for n in objects if fnmatch.fnmatch(n, entity.state_objects)]
        for panel in spec.panels:
            if not panel.camera:
                continue
            camera = (record.get("cameras") or {}).get(panel.camera)
            if camera is None:
                continue
            K = np.asarray(camera["K"], float)
            T = np.asarray(camera["T_camera_world_cv"], float)
            nw, nh = panel.native_size
            sx, sy = panel.box.w / nw * scale, panel.box.h / nh * scale
            ox, oy = panel.box.x * scale, panel.box.y * scale
            clip = np.zeros(work_shape, np.uint8)
            for name in names:
                pose = objects[name]
                world = points @ _rotation(pose.get("quaternion_wxyz", (1, 0, 0, 0))).T + np.asarray(pose["position_m"])
                cam = (T[:3, :3] @ world.T + T[:3, 3:4]).T
                if (cam[:, 2] <= 0.02).any():
                    continue
                uv = (K @ cam.T).T
                uv = uv[:, :2] / uv[:, 2:3]
                px = np.c_[ox + uv[:, 0] * sx, oy + uv[:, 1] * sy]
                hull = cv2.convexHull(px.astype(np.float32)).astype(np.int32)
                cv2.fillConvexPoly(clip, hull, 1)
            # keep only this panel's region
            y0, y1 = int(round(panel.box.y * scale)), int(round((panel.box.y + panel.box.h) * scale))
            x0, x1 = int(round(panel.box.x * scale)), int(round((panel.box.x + panel.box.w) * scale))
            out[index, y0:y1, x0:x1] |= clip[y0:y1, x0:x1] > 0
    return out
