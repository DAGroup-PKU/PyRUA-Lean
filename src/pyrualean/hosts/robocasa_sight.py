# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Objects measured in 3D from the robot's three colour cameras alone (RoboCasa365).

The robot ``robocasa_rgb`` reads no depth image.  What it has: three colour images
(``agentview_left`` / ``agentview_right``, 0.7 m apart on the robot's shoulders, and
``eye_in_hand`` on the gripper), where each camera is (from the robot's own state) and where
its own body is in each image (its model drawn at its joint angles, as RoboDojo's
``robot_pixels`` works it out).  Pixel matching between the views was measured first and does
not work in these kitchens: dense stereo, SIFT and plane-swept patches were right (within 3 cm
of the truth) for 20-45 % of pixels, and even at the true depth half the patches correlate below
0.4 across views.  Masks do: SAM3 finds the same object by its name in views 0.7 m apart.  So:

*   :func:`carve` - an object's VISUAL HULL: the points whose image falls inside its mask in
    every camera that sees that place.  A pixel on the robot's own body is ``unknown``: the
    arm in front of an object does not carve it away.  A mask that agrees with no point of the
    others is another object, and that camera is left out (``views`` counts the rest).
*   :func:`_explained` - the hull of something standing on a counter reaches below it (where no
    camera sees, the masks cannot cut) and above it (a ridge between the cameras' cones).  The
    object is the narrowest band of heights whose part of the hull still accounts for every
    camera's mask; its bottom is taken from there.  Nothing is matched pixel to pixel (like
    RoboDojo's ``locate``: masks and geometry only).
*   :func:`measure` - what ``locate`` reports, from the hull above its support.

Frames are the robot's (``hosts/robocasa_sim.py``): x right, y ahead, z up from the floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

CAMERAS = ("agentview_left", "agentview_right", "eye_in_hand")
#: Rays sampled from a mask, and the depths sampled along each (geometric, near .. far).
MAX_RAYS = 1500
NEAR_M, FAR_M = 0.08, 4.0
DEPTH_STEPS = 900
#: A mask is grown by this many pixels before carving (its edge is not exact).
GROW_PX = 2
#: A camera's mask belongs to the same object when at least this share of the reference rays
#: keep a point with it.
AGREE_SHARE = 0.3
#: The object's band of heights must account for this share of what the whole hull accounts
#: for in every camera.
EXPLAIN_SHARE = 0.95
#: A top that may be this much lower than reported is said so.
TOP_SLACK_M = 0.015
FOOTPRINT_CELL_M = 0.004


@dataclass
class View:
    """One camera: its image, intrinsics, pose in the robot frame and the robot's own pixels."""

    name: str
    rgb: np.ndarray
    K: np.ndarray
    T: np.ndarray  # camera -> robot, OpenCV axes (x right, y down, z ahead)
    robot: np.ndarray  # bool (H, W)
    robot_depth: np.ndarray  # (H, W) camera z of the robot's own surface, NaN elsewhere

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.rgb.shape[0]), int(self.rgb.shape[1])

    def rays(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Robot-frame directions of pixels, scaled so their camera-z component is 1."""

        d = np.stack(
            [
                (cols + 0.5 - self.K[0, 2]) / self.K[0, 0],
                (rows + 0.5 - self.K[1, 2]) / self.K[1, 1],
                np.ones(np.shape(rows)),
            ],
            -1,
        )
        return d @ self.T[:3, :3].T

    @property
    def origin(self) -> np.ndarray:
        return self.T[:3, 3]

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(col, row, camera z) of robot-frame points; z <= 0 is behind the camera."""

        pc = (pts - self.T[:3, 3]) @ self.T[:3, :3]
        z = pc[..., 2]
        zs = np.where(z > 1e-4, z, np.nan)
        return (
            self.K[0, 0] * pc[..., 0] / zs + self.K[0, 2] - 0.5,
            self.K[1, 1] * pc[..., 1] / zs + self.K[1, 2] - 0.5,
            z,
        )


def views_from_cameras(cameras: dict[str, Any]) -> dict[str, View]:
    out = {}
    for name, cam in cameras.items():
        rgb = np.asarray(cam["color"])
        robot = cam.get("robot_mask")
        robot = np.asarray(robot, bool) if robot is not None else np.zeros(rgb.shape[:2], bool)
        depth = cam.get("robot_depth")
        out[name] = View(
            name=name,
            rgb=rgb,
            K=np.asarray(cam["intrinsic"], float),
            T=np.asarray(cam["cam_to_robot"], float),
            robot=robot,
            robot_depth=(
                np.asarray(depth, np.float32) if depth is not None else np.where(robot, 0.0, np.nan)
            ),
        )
    return out


def candidates(sam: Any, image: np.ndarray, prompt: str, min_score: float, limit: int = 6):
    """Every mask SAM3 gives for ``prompt`` scoring at least ``min_score``: [(mask, score)],
    best first.  Several alike things (the knobs of a stove) each get theirs, so the one
    another camera shows can be picked by geometry rather than by score."""

    from .sam3_local import _array

    pixels = np.ascontiguousarray(np.asarray(image)[..., :3].astype(np.uint8))
    with sam._lock:
        state = sam._state_for(pixels)
        with sam._autocast():
            out = sam._processor.set_text_prompt(prompt=prompt.strip(), state=state)
    masks, scores = out.get("masks"), out.get("scores")
    if masks is None or scores is None:
        return []
    masks, scores = _array(masks), _array(scores).reshape(-1)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    found = []
    for i in np.argsort(-scores)[:limit]:
        m = np.asarray(masks[i]) > 0
        if float(scores[i]) >= min_score and m.any():
            found.append((m, float(scores[i])))
    return found


def _grow(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def _lookup(mask: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(inside the image, mask value) at float pixel coordinates."""

    h, w = mask.shape
    ui, vi = np.round(u), np.round(v)
    inside = np.isfinite(ui) & np.isfinite(vi) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    val = np.zeros(u.shape, bool)
    val[inside] = mask[vi[inside].astype(int), ui[inside].astype(int)]
    return inside, val


def _depths() -> np.ndarray:
    return np.geomspace(NEAR_M, FAR_M, DEPTH_STEPS)


def _robot_depth_at(view: View, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = view.robot_depth.shape
    ui = np.clip(np.nan_to_num(np.round(u), nan=0).astype(int), 0, w - 1)
    vi = np.clip(np.nan_to_num(np.round(v), nan=0).astype(int), 0, h - 1)
    return view.robot_depth[vi, ui]


def _accounted(view: View, pts: np.ndarray, mask: np.ndarray) -> float:
    """Share of a camera's mask (where the robot does not hide it) that points fall on."""

    visible = mask & ~view.robot
    total = int(visible.sum())
    if not total or not len(pts):
        return 0.0
    u, v, _ = view.project(pts)
    inside, _ = _lookup(visible, u, v)
    hit = np.zeros(visible.shape, bool)
    hit[np.round(v[inside]).astype(int), np.round(u[inside]).astype(int)] = True
    return float((_grow(hit, 3) & visible).sum()) / total


def carve(
    views: dict[str, View],
    ref: str,
    masks: dict[str, list[np.ndarray]],
    rays: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """The visual hull of an object: (points N x 4: x, y, z and the ray each lies on; the mask
    each agreeing camera showed it with; agreement per camera).

    Points are sampled along rays of the reference camera's mask (``masks[ref][0]``; or the
    pixels ``rays``, the first of them ray 0).  Of each other camera's candidate masks the one
    the object is, is the one whose hull with the reference mask accounts for most of BOTH
    masks: another thing's mask (a wall around a handle) keeps points on the reference rays
    as well, but the hull accounts for little of it.  Below AGREE_SHARE that camera is left
    out.  A camera cuts away what falls outside its mask unless the point is outside its image
    or BEHIND the robot's own body in it.  Points no second camera confirms are left out.
    """

    v0 = views[ref]
    if not masks.get(ref):
        return np.zeros((0, 4)), {}, {}
    m0 = masks[ref][0]
    if rays is None:
        rows, cols = np.nonzero(m0)
        if len(rows) > MAX_RAYS:
            pick = np.linspace(0, len(rows) - 1, MAX_RAYS).astype(int)
            rows, cols = rows[pick], cols[pick]
    else:
        rows, cols = rays
    depths = _depths()
    dirs = v0.rays(rows.astype(float), cols.astype(float))
    pts = v0.origin + dirs[:, None, :] * depths[None, :, None]
    keep = np.ones(pts.shape[:2], bool)
    confirmed = np.zeros(pts.shape[:2], bool)
    agree: dict[str, float] = {}
    used: dict[str, np.ndarray] = {ref: m0}
    for name, view in views.items():
        if name == ref or not masks.get(name):
            continue
        u, v, z = view.project(pts)
        inside = np.isfinite(u) & (u >= -0.5) & (u < view.shape[1] - 0.5)
        inside &= np.isfinite(v) & (v >= -0.5) & (v < view.shape[0] - 0.5) & (z > 0.02)
        hidden = z > _robot_depth_at(view, u, v) + 0.01  # behind the robot's own body there
        best = None
        for mask in masks[name]:
            _, hit = _lookup(_grow(mask, GROW_PX), u, v)
            ok = ~inside | hit | hidden
            seen = inside & hit
            both = ok & keep & seen
            mine = float(np.mean(both.any(axis=1)))  # of the reference rays
            theirs = _accounted(view, pts[both][:: max(1, int(both.sum()) // 20000)], mask)
            share = min(mine, theirs)
            if best is None or share > best[0]:
                best = (share, ok, seen, mask)
        if best is None:
            continue
        agree[name] = round(best[0], 3)
        if best[0] >= AGREE_SHARE:
            keep &= best[1]
            confirmed |= best[2]
            used[name] = best[3]
    if len(used) < 2:
        return np.zeros((0, 4)), used, agree
    good = keep & confirmed
    ray = np.broadcast_to(np.arange(len(rows))[:, None], good.shape)[good]
    return np.column_stack([pts[good], ray]), used, agree


def _explained(
    views: dict[str, View], ref: str, hull: np.ndarray, used: dict[str, np.ndarray]
) -> tuple[float, float]:
    """The narrowest band of heights whose hull points still account for every camera's mask.

    A hull from a few cameras is too tall: a ridge above the object and a wedge below it (into
    the counter) agree with every mask without being needed by any.  The object itself
    accounts for all of each silhouette, so the band from the lowest to the highest height it
    needs is the object's.  A camera's mask is accounted for where hull points project onto
    it; pixels where the robot hides the place do not count.
    """

    z = hull[:, 2]
    rays = hull[:, 3].astype(int)
    n_rays = int(rays.max()) + 1 if len(rays) else 0
    marks = {}
    for name, mask in used.items():
        if name == ref:
            continue
        view = views[name]
        u, v, depth = view.project(hull[:, :3])
        ui = np.clip(np.round(u).astype(int), 0, view.shape[1] - 1)
        vi = np.clip(np.round(v).astype(int), 0, view.shape[0] - 1)
        visible = mask & ~view.robot
        marks[name] = (ui, vi, visible, max(1, int(visible.sum())))

    def covered(lo: float, hi: float) -> float:
        sel = (z >= lo) & (z <= hi)
        worst = len(np.unique(rays[sel])) / max(1, n_rays)
        for ui, vi, visible, total in marks.values():
            hit = np.zeros(visible.shape, bool)
            hit[vi[sel], ui[sel]] = True
            hit = _grow(hit, 3) & visible
            worst = min(worst, float(hit.sum()) / total)
        return worst

    low, high = float(z.min()), float(z.max())
    need = EXPLAIN_SHARE * covered(low, high)
    a, b = low, high  # the top: the lowest height that still accounts for the masks
    while b - a > 0.002:
        mid = (a + b) / 2
        a, b = (a, mid) if covered(low, mid) >= need else (mid, b)
    top = b
    a, b = low, top  # the bottom: the highest that still does
    while b - a > 0.002:
        mid = (a + b) / 2
        a, b = (mid, b) if covered(mid, top) >= need else (a, mid)
    return a, top


def _prism_top(
    views: dict[str, View],
    used: dict[str, np.ndarray],
    xy: np.ndarray,
    bottom: float,
    tops: np.ndarray,
) -> tuple[float, float]:
    """(top, mean IoU) of the upright prism over footprint ``xy`` standing at ``bottom`` whose
    outline in every camera best matches the object's mask there (RoboDojo's ``prism``)."""

    cells = np.unique(np.floor(xy / FOOTPRINT_CELL_M).astype(np.int64), axis=0)
    cells = (cells + 0.5) * FOOTPRINT_CELL_M
    if len(cells) > 3000:
        cells = cells[np.linspace(0, len(cells) - 1, 3000).astype(int)]
    levels = np.arange(bottom, float(tops.max()) + 1e-9, 0.004)
    proj = {}
    for name, mask in used.items():
        view = views[name]
        pts = np.concatenate(
            [np.column_stack([cells, np.full(len(cells), z)]) for z in levels], axis=0
        )
        u, v, depth = view.project(pts)
        inside, _ = _lookup(mask, u, v)
        level = np.repeat(np.arange(len(levels)), len(cells))
        visible = mask & ~view.robot
        proj[name] = (
            np.round(v[inside]).astype(int),
            np.round(u[inside]).astype(int),
            level[inside],
            visible,
            ~view.robot,
        )
    best = (float(tops[0]), -1.0)
    for t in tops:
        k = int(np.searchsorted(levels, t, side="right"))
        ious = []
        for vi, ui, level, visible, free in proj.values():
            sel = level < k
            sil = np.zeros(visible.shape, bool)
            sil[vi[sel], ui[sel]] = True
            sil = _grow(sil, 2) & free
            union = int((sil | visible).sum())
            ious.append(float((sil & visible).sum()) / union if union else 0.0)
        score = float(np.mean(ious))
        if score > best[1]:
            best = (float(t), score)
    return best


def _footprint(xy: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """(heading of the long side in degrees from +x, -90..90; length; width; centre)."""

    cells = np.unique(np.floor(xy / FOOTPRINT_CELL_M).astype(np.int64), axis=0)
    cells = (cells + 0.5) * FOOTPRINT_CELL_M

    def extents(deg: float) -> tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
        a = math.radians(deg)
        u = np.array([math.cos(a), math.sin(a)])
        v = np.array([-u[1], u[0]])
        (ul, uh), (vl, vh) = np.percentile(cells @ u, [1, 99]), np.percentile(cells @ v, [1, 99])
        return float((uh - ul) * (vh - vl)), ul, uh, vl, vh, u, v

    spread, axes = np.linalg.eigh(np.cov((cells - cells.mean(axis=0)).T))
    if spread[1] >= 1.4 * max(float(spread[0]), 1e-12):
        heading = math.degrees(math.atan2(axes[1, 1], axes[0, 1]))
    else:
        coarse = min(np.arange(0.0, 90.0, 3.0), key=lambda d: extents(d)[0])
        heading = min(np.arange(coarse - 3.0, coarse + 3.01, 0.5), key=lambda d: extents(d)[0])
    _, ul, uh, vl, vh, u, v = extents(heading)
    centre = u * (ul + uh) / 2 + v * (vl + vh) / 2
    length, width = uh - ul, vh - vl
    if width > length:
        length, width, heading = width, length, heading + 90.0
    return (heading + 90.0) % 180.0 - 90.0, float(length), float(width), centre


def measure(
    views: dict[str, View],
    ref: str,
    masks: dict[str, list[np.ndarray]],
    known_height: float | None = None,
) -> dict[str, Any]:
    """``locate``'s measurement of one object from its candidate masks per camera (the
    reference camera's first one is the object)."""

    hull, used, agree = carve(views, ref, masks)
    out: dict[str, Any] = {"views": len(used) if len(hull) else 1, "agreement": agree}
    if not len(hull):
        out["geometry_error"] = (
            "only one camera shows it (the others do not see it, or show something else under "
            "these words): no 3D measurement; look from another place, e.g. bring the wrist "
            "camera nearer"
        )
        return out
    low, high = float(hull[:, 2].min()), float(hull[:, 2].max())
    bottom, top = _explained(views, ref, hull, used)
    note = None
    kind = "outlines"
    if known_height is not None and (high - low) > 0:
        kind = "remembered"
    # Seen from behind the robot and from above only, the top is ambiguous: every shape between
    # the upright prism that best matches the outlines (a lower bound: 1.8 cm low on a mug)
    # and the hull's ridge (an upper bound: 4.5 cm high) fits the masks.  The hull's top is
    # reported (safe to approach from above) and the note says how much lower it may be.
    top = float(np.percentile(hull[:, 2], 99.5))
    lower = hull[(hull[:, 2] > bottom + 0.004) & (hull[:, 2] < bottom + 0.5 * (top - bottom))]
    if len(lower) >= 20 and top > bottom + 0.01:
        low_top, _ = _prism_top(
            views, used, lower[:, :2], bottom, np.arange(bottom + 0.008, top + 0.0041, 0.004)
        )
        if top - low_top > TOP_SLACK_M:
            out["top_range"] = [round(low_top, 4), round(top, 4)]
    if kind == "remembered":
        bottom = top - known_height
        note = "the height is the one measured before with this prompt"
    body = hull[(hull[:, 2] > bottom + 0.003) & (hull[:, 2] <= top)]
    if len(body) < 20:
        body = hull
    support = bottom
    heading, length, width, mid = _footprint(body[:, :2])
    across = (heading + 180.0) % 180.0 - 90.0
    out.update(
        center_xyz=[round(float(mid[0]), 4), round(float(mid[1]), 4), round((top + support) / 2, 4)],
        top_z=round(top, 4),
        bottom_z=round(float(support), 4),
        height_m=round(top - float(support), 4),
        length_m=round(length, 4),
        width_m=round(width, 4),
        long_axis_deg=round(heading, 1),
        grasp_yaw=round(across, 1),
        bottom_from=kind,
        hull_points=int(len(hull)),
        hull_z=[round(low, 4), round(high, 4)],
    )
    if "top_range" in out:
        lo = out["top_range"][0]
        ridge = (
            f"seen from above and behind only, its top is between {lo:.3f} and {top:.3f} "
            "(the higher is reported): look at it from the side (the wrist camera about level "
            "with it) to measure its height"
        )
        note = f"{note}; {ridge}" if note else ridge
    if note:
        out["height_guess"] = note
    return out


#: ``probe``: depths tried along the pixel's ray to find the same thing in another camera, and
#: the neighbourhood (pixels) whose surface gives the normal.
PROBE_DEPTHS = 36
PROBE_PATCH_PX = 12


def probe(
    views: dict[str, View], sam: Any, camera: str, row: int, col: int, min_score: float = 0.2
) -> dict[str, Any]:
    """The point seen at one pixel, and the surface's normal there, from colour alone.

    SAM3 outlines whatever is under the pixel.  To find the same thing in the other cameras,
    points along the pixel's ray are projected into them and SAM3 outlines what is there; the
    outline whose hull with the first keeps the most of the rays around the pixel is taken.
    The point is where the pixel's ray first enters that hull; the normal is that of the
    hull's front surface over the neighbouring pixels (pointing towards the camera).
    """

    v0 = views[camera]
    first = sam.segment(v0.rgb, point=[int(row), int(col)], min_score=min_score)
    if not first.found:
        return {"found": False, "why": "nothing could be outlined under that pixel"}
    m0 = np.asarray(first.mask, bool)
    if not m0[int(row), int(col)]:
        m0 = m0.copy()
        m0[int(row), int(col)] = True
    # rays: the pixel's own (ray 0), its neighbourhood on the outlined thing (for the normal),
    # and a spread over the rest of the outline (so that the outlines can be compared whole)
    patch = np.zeros_like(m0)
    r0, r1 = max(0, row - PROBE_PATCH_PX), min(m0.shape[0], row + PROBE_PATCH_PX + 1)
    c0, c1 = max(0, col - PROBE_PATCH_PX), min(m0.shape[1], col + PROBE_PATCH_PX + 1)
    patch[r0:r1, c0:c1] = True
    near_r, near_c = np.nonzero(m0 & patch)
    order = np.argsort((near_r - row) ** 2 + (near_c - col) ** 2)[:150]
    rest_r, rest_c = np.nonzero(m0 & ~patch)
    spread = np.linspace(0, len(rest_r) - 1, min(len(rest_r), 350)).astype(int)
    rows = np.r_[row, near_r[order], rest_r[spread]]
    cols = np.r_[col, near_c[order], rest_c[spread]]
    n_near = 1 + len(order)
    # candidates in the other cameras: what SAM3 outlines where the pixel's ray could be
    centre = v0.origin + v0.rays(np.array(float(row)), np.array(float(col))) * _depths()[
        np.linspace(0, DEPTH_STEPS - 1, PROBE_DEPTHS).astype(int)
    ][:, None]
    masks: dict[str, list[np.ndarray]] = {camera: [m0]}
    for name, view in views.items():
        if name == camera:
            continue
        cu, cv, cz = view.project(centre)
        tried: list[np.ndarray] = []
        for x, y, d in zip(cu, cv, cz):
            if not (np.isfinite(x) and np.isfinite(y) and d > 0.02):
                continue
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= yi < view.shape[0] and 0 <= xi < view.shape[1]) or view.robot[yi, xi]:
                continue
            if any(m[yi, xi] for m in tried):
                continue
            seg = sam.segment(view.rgb, point=[yi, xi], min_score=min_score)
            if seg.found:
                tried.append(np.asarray(seg.mask, bool))
        masks[name] = tried
    hull, used, agree = carve(views, camera, masks, rays=(rows, cols))
    out: dict[str, Any] = {"found": True, "views": len(used), "score": first.score}
    on_pixel = hull[hull[:, 3] == 0] if len(hull) else hull
    if len(used) < 2 or not len(on_pixel):
        out["why"] = (
            "only one camera shows that place (or the others show something else there): "
            "no 3D point"
        )
        return out
    dist = np.linalg.norm(on_pixel[:, :3] - v0.origin, axis=1)
    xyz = on_pixel[int(np.argmin(dist)), :3]
    out["xyz"] = [round(float(c), 4) for c in xyz]
    out["depth_m"] = round(float((xyz - v0.origin) @ v0.T[:3, 2]), 4)
    fronts = []
    for i in range(n_near):
        mine = hull[hull[:, 3] == i]
        if len(mine):
            fronts.append(mine[int(np.argmin(np.linalg.norm(mine[:, :3] - v0.origin, axis=1))), :3])
    fronts = np.array(fronts) if fronts else np.zeros((0, 3))
    near = fronts[np.linalg.norm(fronts - xyz, axis=1) < 0.03] if len(fronts) else fronts
    if len(near) >= 8:
        centred = near - near.mean(axis=0)
        _, sing, vt = np.linalg.svd(centred, full_matrices=False)
        normal = vt[2]
        if normal @ (v0.origin - xyz) < 0:
            normal = -normal
        out["normal"] = [round(float(c), 3) for c in normal]
        out["flatness_m"] = round(float(sing[2] / math.sqrt(len(near))), 4)
    return out


__all__ = [
    "CAMERAS",
    "View",
    "candidates",
    "carve",
    "measure",
    "probe",
    "views_from_cameras",
]
