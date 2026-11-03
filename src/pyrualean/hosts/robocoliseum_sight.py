# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Measuring what a colour camera sees, through its depth image.

The benchmark's task objects are coloured blocks, so a region of one colour is the
handle a program has on them: :func:`segment` grows the region a pixel belongs to (or the
largest region in the image), and :func:`locate` projects every pixel of it into
``base_link`` through the depth image and reports its centre, top and extent.

The calibration the benchmark publishes with each frame is what this needs and all it
needs: ``fx, fy, cx, cy`` at the image resolution and ``optical_to_frame``, a 4x4 mapping
an optical-frame point (metres, +x right, +y down, +z forward) into ``base_link``.
Nothing here guesses a pose, and every measurement says when it has none.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import geniesim_bridge as bridge_mod

#: Pixels of one region must be within this RGB distance of the seed's colour ...
COLOUR_TOLERANCE = 60.0
#: ... and a region smaller than this is not an object.
MIN_REGION_PX = 60
#: ``locate`` with no pixel given takes the largest SMALL region: the head camera's
#: background, table and walls are a few large regions of one colour each, and a task
#: object is small.  A fraction of the image keeps this scale-free across resolutions.
MAX_OBJECT_FRACTION = 0.10
#: Points this close to the lowest one seen are the surface it stands on, not the object.
STAND_TOLERANCE_M = 0.008
#: The fewest valid points a 3D measurement needs.
MIN_POINTS = 20


def camera_name(camera: str, known: tuple[str, ...]) -> str:
    if camera not in known:
        raise ValueError(f"camera must be one of {known}, got {camera!r}")
    return camera


# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------


#: Two neighbouring pixels belong to one surface when their depths differ by less than this
#: (metres).  A same-coloured thing further back — a warehouse full of cardboard behind
#: cardboard packages — is then a different region however alike the colours are.
DEPTH_STEP_M = 0.03


def _label(image: np.ndarray, row: int, col: int, depth: np.ndarray | None = None) -> np.ndarray:
    """The connected region of ``image`` that contains pixel (``row``, ``col``).

    Pixels belong to it when their colour is close to that pixel's (the tolerance is on the
    seed's colour, not on each neighbour's, so a region does not drift across a gradient)
    AND they connect to it over the 4-neighbourhood.  Only the seed's own component is
    returned: another patch of a similar colour elsewhere in the image (a shadow along the
    table's edge for a dark block) is a different region however large it is.  With
    ``depth``, a step between two neighbours of more than :data:`DEPTH_STEP_M` also ends
    the region, so it does not run on into same-coloured things behind the object.
    """

    seed = image[row, col, :3].astype(np.float32)
    close = np.linalg.norm(image[..., :3].astype(np.float32) - seed, axis=2)
    allowed = close <= COLOUR_TOLERANCE
    if depth is None or depth.shape != allowed.shape:
        mask = np.zeros(allowed.shape, dtype=bool)
        return _flood(allowed, mask, np.array([row]), np.array([col]))
    return _flood_depth(allowed, np.asarray(depth, dtype=np.float64), row, col)


def _flood_depth(allowed: np.ndarray, depth: np.ndarray, row: int, col: int) -> np.ndarray:
    """Grow from (row, col) over allowed pixels whose depth is continuous with a neighbour."""

    from collections import deque

    height, width = allowed.shape
    mask = np.zeros(allowed.shape, dtype=bool)
    valid = np.isfinite(depth) & (depth > 0)
    mask[row, col] = True
    queue = deque([(row, col)])
    while queue:
        r, c = queue.popleft()
        here = depth[r, c] if valid[r, c] else None
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if not (0 <= nr < height and 0 <= nc < width) or mask[nr, nc] or not allowed[nr, nc]:
                continue
            if here is not None and valid[nr, nc] and abs(depth[nr, nc] - here) > DEPTH_STEP_M:
                continue
            mask[nr, nc] = True
            queue.append((nr, nc))
    return mask


def _flood(
    allowed: np.ndarray, mask: np.ndarray, seeds_rows: np.ndarray, seeds_cols: np.ndarray
) -> np.ndarray:
    """Grow ``mask`` from the allowed pixels, keeping only the largest connected part."""

    from collections import deque

    height, width = allowed.shape
    best = np.zeros_like(mask)
    best_size = 0
    for row, col in zip(seeds_rows.tolist(), seeds_cols.tolist(), strict=False):
        if mask[row, col]:
            continue
        queue = deque([(row, col)])
        mask[row, col] = True
        part = [(row, col)]
        while queue:
            r, c = queue.popleft()
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < height and 0 <= nc < width and allowed[nr, nc] and not mask[nr, nc]:
                    mask[nr, nc] = True
                    queue.append((nr, nc))
                    part.append((nr, nc))
        if len(part) > best_size:
            best_size = len(part)
            best = np.zeros_like(mask)
            for r, c in part:
                best[r, c] = True
    return best


def segment(
    frame: np.ndarray,
    row: int | None = None,
    col: int | None = None,
    depth: np.ndarray | None = None,
) -> dict[str, Any]:
    """The colour region of ``frame`` at a pixel, or the largest one when no pixel is given."""

    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("an image must be (H, W, 3)")
    height, width = image.shape[:2]
    if row is None and col is None:
        # The largest SMALL region is found by growing from a coarse grid of seeds: cheaper
        # than labelling every pixel, and every object is bigger than one grid cell.  The
        # cap matters: the background, the table and the walls are the largest regions of
        # the head camera, and ``locate()`` with no pixel means "the object", not "the
        # wall".
        step = max(8, min(height, width) // 40)
        rows = np.arange(0, height, step)
        cols = np.arange(0, width, step)
        mask = np.zeros((height, width), dtype=bool)
        cap = max(MIN_REGION_PX, int(MAX_OBJECT_FRACTION * height * width))
        best = np.zeros_like(mask)
        best_size = 0
        for r in rows:
            for c in cols:
                if mask[r, c]:
                    continue
                part = _label(image, int(r), int(c))
                mask |= part
                size = int(part.sum())
                if best_size < size <= cap:
                    best_size, best = size, part
        if best_size == 0:
            return {
                "found": False,
                "area_px": 0,
                "note": f"no region between {MIN_REGION_PX} and {cap} px",
            }
        chosen = best
    else:
        r, c = int(row), int(col)
        if not (0 <= r < height and 0 <= c < width):
            raise ValueError(f"pixel ({r}, {c}) is outside the {height}x{width} image")
        chosen = _label(image, r, c, depth)

    count = int(chosen.sum())
    if count == 0:
        return {"found": False, "area_px": 0}
    rows, cols = np.nonzero(chosen)
    colour = np.asarray(image)[chosen][:, :3].mean(axis=0)
    return {
        "found": True,
        "area_px": count,
        "centroid_rc": [int(rows.mean()), int(cols.mean())],
        "box_rc": [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())],
        "colour": "#{:02x}{:02x}{:02x}".format(*tuple(int(round(v)) for v in colour)),
        "mask": chosen,
    }


# ---------------------------------------------------------------------------
# three dimensions
# ---------------------------------------------------------------------------


def world_points(camera: bridge_mod.Camera, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """The base_link points of these pixels (N x 3); pixels without depth are dropped."""

    if camera.depth is None:
        raise KeyError("this frame has no depth image")
    if camera.fx is None or camera.optical_to_frame is None:
        raise KeyError("this frame has no calibration")
    depth = camera.depth
    rows = np.clip(np.asarray(rows), 0, camera.height - 1)
    cols = np.clip(np.asarray(cols), 0, camera.width - 1)
    d = depth[rows, cols]
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return np.zeros((0, 3))
    r, c, d = rows[valid], cols[valid], d[valid]
    optical = np.stack(
        [(c - camera.cx) / camera.fx * d, (r - camera.cy) / camera.fy * d, d, np.ones_like(d)]
    )
    return (camera.optical_to_frame @ optical)[:3].T


def top_plane(points: np.ndarray, rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    """The dominant plane among an object's upper points: (unit normal pointing up, inliers).

    RANSAC over the upper half of the points: that is the top face of a box, the lid of a
    tin, the side that faces up of a box lying on its side.
    """

    if len(points) < MIN_POINTS:
        return None
    z = points[:, 2]
    upper = points[z >= np.percentile(z, 50)]
    if len(upper) < MIN_POINTS:
        return None
    rng = np.random.default_rng(rng_seed)
    best_normal, best_inliers = None, np.zeros(len(upper), dtype=bool)
    for _ in range(150):
        a, b, c = upper[rng.choice(len(upper), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        inliers = np.abs((upper - a) @ normal) < 0.006
        if inliers.sum() > best_inliers.sum():
            best_normal, best_inliers = normal, inliers
    if best_normal is None or best_inliers.sum() < MIN_POINTS:
        return None
    # refine on the inliers
    face = upper[best_inliers]
    centred = face - face.mean(axis=0)
    normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
    if normal[2] < 0:
        normal = -normal
    return normal, face


#: An object whose horizontal extent is this many times longer one way than the other has a
#: direction to close across (``grasp_fingers``); a rounder one can be closed any way.
ELONGATED_RATIO = 1.4


def locate(camera: bridge_mod.Camera, region: dict[str, Any]) -> dict[str, Any]:
    """Centre, top and extent of a segmented region, in base_link."""

    if not region.get("found"):
        return {"found": False, "note": "no colour region found"}
    rows, cols = np.nonzero(region["mask"])
    try:
        points = world_points(camera, rows, cols)
    except KeyError as exc:
        return {
            "found": True,
            "area_px": region["area_px"],
            "colour": region["colour"],
            "pixel_uv": region["centroid_rc"],
            "note": str(exc),
        }
    if len(points) < MIN_POINTS:
        return {
            "found": True,
            "area_px": region["area_px"],
            "colour": region["colour"],
            "pixel_uv": region["centroid_rc"],
            "note": f"only {len(points)} pixels have a valid depth",
        }
    # The support the object stands on: measured around it when the region was found by
    # geometry, else the lowest thing the region covers (pixels just above it are floor).
    base_z = (
        float(region["support_z"])
        if region.get("support_z") is not None
        else float(np.percentile(points[:, 2], 1))
    )
    body = points[points[:, 2] > base_z + STAND_TOLERANCE_M]
    if len(body) < MIN_POINTS:
        body = points
    top = float(np.percentile(body[:, 2], 97))
    low = float(body[:, 2].min())
    extent = body.max(axis=0) - body.min(axis=0)
    centre = body.mean(axis=0)
    centre[2] = (top + low) / 2.0
    tilt = long_axis = None
    plane = top_plane(body)
    if plane is not None:
        normal, face = plane
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(normal[2])))))
        xy = face[:, :2] - face[:, :2].mean(axis=0)
        if len(xy) >= MIN_POINTS:
            axis = np.linalg.svd(xy, full_matrices=False)[2][0]
            long_axis = (math.degrees(math.atan2(axis[1], axis[0])) + 90.0) % 180.0 - 90.0
    # The finger line for a grasp across the object: the horizontal direction across its
    # longest extent, from ALL its points (the top face alone is a strip on a lying bottle).
    fingers = width = length = None
    xy_all = body[:, :2] - body[:, :2].mean(axis=0)
    if len(xy_all) >= MIN_POINTS:
        _, spread, axes = np.linalg.svd(xy_all, full_matrices=False)
        major = axes[0]
        along = xy_all @ major
        across = xy_all @ np.array([-major[1], major[0]])
        length = float(np.percentile(along, 98) - np.percentile(along, 2))
        width = float(np.percentile(across, 98) - np.percentile(across, 2))
        if spread[1] > 1e-9 and spread[0] / spread[1] >= ELONGATED_RATIO:
            fingers = [round(float(-major[1]), 3), round(float(major[0]), 3), 0.0]
            if long_axis is None:
                long_axis = (math.degrees(math.atan2(major[1], major[0])) + 90.0) % 180.0 - 90.0
    return {
        "found": True,
        "grasp_fingers": fingers,
        # the yaw that, pointing straight down, lays the fingers along ``fingers``
        # (robocoliseum_motion: down, the fingers close along yaw + 90 degrees from +x)
        "grasp_yaw": None
        if fingers is None
        else round((math.degrees(math.atan2(-fingers[0], fingers[1])) + 90.0) % 180.0 - 90.0, 1),
        "width_m": None if width is None else round(width, 4),
        "length_m": None if length is None else round(length, 4),
        "centre_xyz": [round(float(v), 4) for v in centre],
        "top_z": round(top, 4),
        "height_m": round(top - base_z, 4),
        "size_xyz": [round(float(v), 4) for v in extent],
        "pixel_uv": region["centroid_rc"],
        "area_px": region["area_px"],
        "colour": region["colour"],
        "support_z": round(base_z, 4),
        "tilt_deg": None if tilt is None else round(tilt, 1),
        "long_axis_deg": None if long_axis is None else round(long_axis, 1),
        "method": region.get("method", "colour"),
        "note": None if math.isfinite(top) else "no finite height",
    }


#: An object is what stands this far above the surface around it (metres) ...
ABOVE_SUPPORT_M = 0.012
#: ... measured in a window this many pixels wide around the pixel asked about.
SUPPORT_WINDOW_PX = 160


def _world_map(camera: bridge_mod.Camera) -> np.ndarray | None:
    """Every pixel's point in base_link (H x W x 3), NaN where there is no depth."""

    if camera.depth is None or camera.fx is None or camera.optical_to_frame is None:
        return None
    depth = np.asarray(camera.depth, dtype=np.float64)
    h, w = depth.shape
    rows, cols = np.mgrid[0:h, 0:w]
    valid = np.isfinite(depth) & (depth > 0)
    d = np.where(valid, depth, np.nan)
    optical = np.stack(
        [
            (cols - camera.cx) / camera.fx * d,
            (rows - camera.cy) / camera.fy * d,
            d,
            np.ones_like(d),
        ],
        axis=-1,
    )
    world = optical @ np.asarray(camera.optical_to_frame, dtype=np.float64).T
    return world[..., :3]


def support_height(world: np.ndarray, row: int, col: int) -> float | None:
    """The height of the surface the thing at (row, col) stands on.

    The most common height in a window around the pixel, among the points no higher than
    it: on a table that is the table top, whatever stands on it.
    """

    h, w = world.shape[:2]
    half = SUPPORT_WINDOW_PX // 2
    patch = world[
        max(0, row - half) : min(h, row + half), max(0, col - half) : min(w, col + half), 2
    ]
    seed_z = world[row, col, 2]
    z = patch[np.isfinite(patch)]
    if np.isfinite(seed_z):
        z = z[z <= seed_z + 0.005]
    if z.size < 50:
        return None
    bins = np.round(z / 0.005).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    return float(values[np.argmax(counts)] * 0.005)


def segment_geometric(
    frame: np.ndarray, camera: bridge_mod.Camera, row: int, col: int
) -> dict[str, Any] | None:
    """The object at (row, col) as what rises above its support surface, connected in depth.

    Colour does not matter, so a white package on a white table, in front of a warehouse
    of cardboard, is still one region.  None when the pixel is not on something standing
    above a surface (it is on the table itself): the caller falls back to colour.
    """

    world = _world_map(camera)
    if world is None:
        return None
    seed = world[row, col]
    if not np.all(np.isfinite(seed)):
        return None
    support = support_height(world, row, col)
    if support is None or seed[2] < support + ABOVE_SUPPORT_M:
        return None
    above = np.isfinite(world[..., 2]) & (world[..., 2] >= support + ABOVE_SUPPORT_M)
    chosen = _flood_depth(above, np.asarray(camera.depth, dtype=np.float64), row, col)
    count = int(chosen.sum())
    if count < MIN_REGION_PX // 2:
        return None
    rows, cols = np.nonzero(chosen)
    colour = np.asarray(frame)[chosen][:, :3].mean(axis=0)
    return {
        "found": True,
        "area_px": count,
        "centroid_rc": [int(rows.mean()), int(cols.mean())],
        "box_rc": [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())],
        "colour": "#{:02x}{:02x}{:02x}".format(*tuple(int(round(v)) for v in colour)),
        "mask": chosen,
        "support_z": round(support, 4),
        "method": "geometry",
    }


def region_from_mask(
    frame: np.ndarray, camera: bridge_mod.Camera, mask: np.ndarray, method: str
) -> dict[str, Any]:
    """A region, as ``locate`` reads one, from a mask made elsewhere (SAM3)."""

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != frame.shape[:2] or not mask.any():
        return {"found": False}
    rows, cols = np.nonzero(mask)
    centre_r, centre_c = float(rows.mean()), float(cols.mean())
    # the mask pixel nearest its centroid: a thin or curved mask's mean can fall outside it
    nearest = int(np.argmin((rows - centre_r) ** 2 + (cols - centre_c) ** 2))
    seed_r, seed_c = int(rows[nearest]), int(cols[nearest])
    world = _world_map(camera)
    support = support_height(world, seed_r, seed_c) if world is not None else None
    mean = frame[mask][:, :3].mean(axis=0)
    return {
        "found": True,
        "mask": mask,
        "area_px": int(mask.sum()),
        "centroid_rc": [seed_r, seed_c],
        "box_rc": [int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())],
        "colour": "#{:02x}{:02x}{:02x}".format(*tuple(int(v) for v in mean)),
        "support_z": support,
        "method": method,
    }


def back_project(camera: bridge_mod.Camera, row: int, col: int) -> dict[str, Any]:
    """The base_link point of one pixel."""

    if not (0 <= row < camera.height and 0 <= col < camera.width):
        raise ValueError(
            f"pixel ({row}, {col}) is outside the {camera.height}x{camera.width} image"
        )
    if camera.depth is None:
        raise KeyError("this frame has no depth image")
    if camera.fx is None or camera.optical_to_frame is None:
        raise KeyError("this frame has no calibration")
    value = float(camera.depth[row, col])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"no valid depth at pixel ({row}, {col})")
    points = world_points(camera, np.array([row]), np.array([col]))
    if len(points) == 0:
        raise ValueError(f"no valid depth at pixel ({row}, {col})")
    return {"xyz": [round(float(v), 4) for v in points[0]], "depth_m": round(value, 4)}


__all__ = [
    "COLOUR_TOLERANCE",
    "DEPTH_STEP_M",
    "MIN_POINTS",
    "MIN_REGION_PX",
    "back_project",
    "camera_name",
    "locate",
    "segment",
    "segment_geometric",
    "top_plane",
    "support_height",
    "world_points",
]
