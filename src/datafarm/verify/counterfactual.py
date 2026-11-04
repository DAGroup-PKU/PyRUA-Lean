"""Counterfactual calibration: inject known defects into an accepted render.

A verifier is only trusted after it (a) passes a render that a reviewer has
accepted and (b) fails each synthetic defect in the family that owns it.  The
defects below cover common failure modes (extra or missing
objects, scaled objects, time lag, frozen or flickering frames, swapped panels,
aspect distortion, relighting, seam jumps, wrong material, a plain carrier copy).

All generators work on full-rate work-scale frames and return a new ``Clip``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from datafarm.verify.checks import _components, key_masks
from datafarm.verify.media import Clip, derive_clip
from datafarm.verify.spec import RenderVerifierSpec


def full_rate_frames(path: str, spec: RenderVerifierSpec) -> np.ndarray:
    capture = cv2.VideoCapture(path)
    work = (int(round(spec.layout_size[0] * spec.work_scale)), int(round(spec.layout_size[1] * spec.work_scale)))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame.shape[1], frame.shape[0]) != tuple(spec.layout_size):
            frame = cv2.resize(frame, tuple(spec.layout_size), interpolation=cv2.INTER_AREA)
        frames.append(cv2.resize(frame, work, interpolation=cv2.INTER_AREA))
    capture.release()
    return np.stack(frames)


def clip_from_full(base: Clip, full: np.ndarray) -> Clip:
    gray_size = (base.gray.shape[2], base.gray.shape[1])
    gray = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), gray_size, interpolation=cv2.INTER_AREA)
                     for f in full])
    return derive_clip(base, frames=full[:: base.stride].copy(), gray=gray)


@dataclass
class Counterfactual:
    name: str
    expected: tuple[str, ...]  # check names (or prefixes) that must fail
    make: Callable[..., np.ndarray | None]
    description: str


def _panel(spec, name):
    p = spec.panel(name).box
    s = spec.work_scale
    return slice(int(p.y * s), int((p.y + p.h) * s)), slice(int(p.x * s), int((p.x + p.w) * s))


def _pick_instance(full, carrier_full, spec, entity, panel="head", window=60, avoid=None):
    """Find a window where one carrier instance is clearly visible and isolated."""

    rows, cols = _panel(spec, panel)
    height, width = rows.stop - rows.start, cols.stop - cols.start
    masks = key_masks(carrier_full[:, rows, cols][::6], entity.carrier_keys)
    avoid_masks = key_masks(carrier_full[:, rows, cols][::6], avoid.carrier_keys) if avoid is not None else None
    min_area = 4 * entity.min_area_frac * height * width
    for start_i in range(0, len(masks) - window // 6):
        labels, stats, keep = _components(masks[start_i], min_area)
        for k in keep:
            x, y, w, h = stats[k, :4]
            if x < 4 or y < 4 or x + w > width - 4 or y + h > height - 4:
                continue
            box = (x, y, w, h)
            ok = True
            for j in range(start_i, start_i + window // 6):
                region = np.zeros((height, width), bool)
                region[max(0, y - 10):y + h + 10, max(0, x - 10):x + w + 10] = True
                if not (masks[j] & region).any() or (avoid_masks is not None and (avoid_masks[j] & region).any()):
                    ok = False
                    break
            if ok:
                return start_i * 6, box, (rows, cols)
    return None


def make_entity_delete(full, carrier_full, spec, entity, robot):
    found = _pick_instance(full, carrier_full, spec, entity, avoid=robot)
    if found is None:
        return None
    start, (x, y, w, h), (rows, cols) = found
    out = full.copy()
    for t in range(start, min(len(out), start + 60)):
        crop = out[t, rows, cols]
        mask = np.zeros(crop.shape[:2], np.uint8)
        mask[max(0, y - 8):y + h + 8, max(0, x - 8):x + w + 8] = 1
        out[t, rows, cols] = cv2.inpaint(crop, mask, 5, cv2.INPAINT_TELEA)
    return out


def make_entity_duplicate(full, carrier_full, spec, entity, robot):
    found = _pick_instance(full, carrier_full, spec, entity, avoid=robot)
    if found is None:
        return None
    start, (x, y, w, h), (rows, cols) = found
    out = full.copy()
    height, width = rows.stop - rows.start, cols.stop - cols.start
    # paste on the emptiest table area: lower-middle of the panel, away from the source
    ty = min(height - h - 4, int(height * 0.72))
    tx = (x + width // 2) % max(1, width - w - 4)
    for t in range(start, min(len(out), start + 60)):
        crop = out[t, rows, cols]
        patch = crop[y:y + h, x:x + w].copy()
        crop[ty:ty + h, tx:tx + w] = patch
        out[t, rows, cols] = crop
    return out


def make_entity_scale(full, carrier_full, spec, entity, robot, factor=1.5):
    """Render one whole object 1.5x larger (uniformly, anchored at its base)."""

    found = _pick_instance(full, carrier_full, spec, entity, avoid=robot)
    if found is None:
        return None
    start, (x, y, w, h), (rows, cols) = found
    lift = int(0.35 * h)  # include the rim/opening above the coloured body
    y0, y1 = max(0, y - lift), y + h
    x0, x1 = max(0, x - int(0.05 * w)), x + w + int(0.05 * w)
    out = full.copy()
    height, width = rows.stop - rows.start, cols.stop - cols.start
    for t in range(start, min(len(out), start + 60)):
        crop = out[t, rows, cols]
        patch = crop[y0:y1, x0:x1]
        ph, pw = patch.shape[:2]
        nh, nw = int(ph * factor), int(pw * factor)
        big = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LINEAR)
        cx = (x0 + x1) // 2
        ty0, tx0 = y1 - nh, cx - nw // 2          # unclipped placement
        ty, tx = max(0, ty0), max(0, tx0)
        by, bx = min(height, ty0 + nh), min(width, tx0 + nw)
        crop[ty:by, tx:bx] = big[ty - ty0:by - ty0, tx - tx0:bx - tx0]
        out[t, rows, cols] = crop
    return out


def make_time_shift(full, *_args, lag=15):
    return np.concatenate([np.repeat(full[:1], lag, axis=0), full[:-lag]])


def make_freeze(full, *_args, start_frac=0.35, length=30):
    out = full.copy()
    start = int(len(out) * start_frac)
    out[start:start + length] = out[start]
    return out


def make_flicker(full, *_args, frac=0.5, gain=1.3):
    out = full.copy()
    t = int(len(out) * frac)
    out[t:t + 2] = np.clip(out[t:t + 2].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out


def make_panel_swap(full, carrier_full, spec, *_args):
    out = full.copy()
    a, b = _panel(spec, "left_wrist"), _panel(spec, "right_wrist")
    left = out[:, a[0], a[1]].copy()
    out[:, a[0], a[1]] = out[:, b[0], b[1]]
    out[:, b[0], b[1]] = left
    return out


def make_squeeze(full, *_args, factor=0.89):
    out = np.empty_like(full)
    height, width = full.shape[1:3]
    new_w = int(width * factor)
    pad = (width - new_w) // 2
    for t, frame in enumerate(full):
        canvas = np.zeros_like(frame)
        canvas[:, pad:pad + new_w] = cv2.resize(frame, (new_w, height), interpolation=cv2.INTER_AREA)
        out[t] = canvas
    return out


def make_relight(full, carrier_full, spec, *_args, amount=28.0):
    """Gradual brightening of the head panel's upper wall, far from motion."""

    out = full.astype(np.float32)
    rows, cols = _panel(spec, "head")
    height = rows.stop - rows.start
    band = slice(rows.start, rows.start + int(height * 0.18))
    ramp = np.linspace(0, amount, len(out))[:, None, None, None]
    out[:, band, cols] += ramp
    return np.clip(out, 0, 255).astype(np.uint8)


def make_light_spot(full, carrier_full, spec, *_args, amount=45.0):
    out = full.astype(np.float32)
    rows, cols = _panel(spec, "head")
    height, width = rows.stop - rows.start, cols.stop - cols.start
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = int(height * 0.12), int(width * 0.5)
    blob = np.exp(-(((yy - cy) / (0.08 * height)) ** 2 + ((xx - cx) / (0.1 * width)) ** 2))[..., None]
    start = len(out) // 3
    for t in range(start, min(len(out), start + 90)):
        out[t, rows, cols] += amount * blob
    return np.clip(out, 0, 255).astype(np.uint8)


def make_seam_jump(full, carrier_full, spec, *_args, skip=12):
    if not spec.seams:
        return None
    seam = spec.seams[0]
    out = full.copy()
    out[seam:len(out) - skip] = full[seam + skip:]
    out[len(out) - skip:] = full[-1]
    return out


def make_recolor(full, carrier_full, spec, entity, *_args):
    """Paint the entity blue: presence and geometry intact, appearance wrong."""

    out = full.copy()
    masks = key_masks(carrier_full, entity.carrier_keys)
    for t in range(len(out)):
        m = masks[t]
        hsv = cv2.cvtColor(out[t], cv2.COLOR_BGR2HSV)
        hsv[..., 0][m] = 110
        hsv[..., 1][m] = np.maximum(hsv[..., 1][m], 150)
        out[t] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out


def make_robot_ghost(full, carrier_full, spec, entity, robot, window=45):
    """Show the arm where it was a while ago: a hallucinated arm on empty workcell."""

    if robot is None:
        return None
    rows, cols = _panel(spec, "head")
    height, width = rows.stop - rows.start, cols.stop - cols.start
    masks = key_masks(carrier_full[:, rows, cols][::3], robot.carrier_keys)
    for i in range(len(masks)):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(masks[i].astype(np.uint8))
        if count < 2:
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[k]
        if area < 0.03 * height * width:
            continue
        box = np.zeros((height, width), bool)
        box[y:y + h, x:x + w] = True
        # a later window in which the carrier shows no robot inside this box
        for j in range(i + 20, len(masks) - window // 3):
            if not any((masks[t] & box).any() for t in range(j, j + window // 3)):
                src_t, dst_t = i * 3, j * 3
                out = full.copy()
                patch_mask = cv2.dilate((labels == k).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                for t in range(dst_t, min(len(out), dst_t + window)):
                    crop = out[t, rows, cols]
                    crop[patch_mask] = full[src_t, rows, cols][patch_mask]
                    out[t, rows, cols] = crop
                return out
    return None


def make_white_copy(full, carrier_full, *_args):
    return carrier_full.copy()


SUITE = (
    Counterfactual("entity_delete", ("entity_",), make_entity_delete, "one object inpainted away for 2 s"),
    Counterfactual("entity_duplicate", ("entity_",), make_entity_duplicate, "extra object pasted on empty table for 2 s"),
    Counterfactual("entity_scale", ("entity_",), make_entity_scale, "one whole object rendered 1.5x larger for 2 s"),
    Counterfactual("time_shift", ("event_alignment",), make_time_shift, "whole render delayed 15 frames"),
    Counterfactual("freeze", ("temporal_consistency",), make_freeze, "30 frozen frames mid-motion"),
    Counterfactual("flicker", ("temporal_consistency",), make_flicker, "2-frame +30% brightness flash"),
    Counterfactual("panel_swap", ("source_structure",), make_panel_swap, "left and right wrist panels swapped"),
    Counterfactual("squeeze", ("source_structure",), make_squeeze, "horizontal 0.89x squeeze (aspect distortion)"),
    Counterfactual("relight_drift", ("background_stability",), make_relight, "upper wall slowly brightens by 28"),
    Counterfactual("light_spot", ("background_stability",), make_light_spot, "hallucinated light spot for 3 s"),
    Counterfactual("seam_jump", ("seam_continuity",), make_seam_jump, "12 frames skipped at the first seam"),
    Counterfactual("recolor", ("realism_vs_real_world",), make_recolor, "objects painted blue"),
    Counterfactual("white_copy", ("realism_vs_real_world",), make_white_copy, "carrier returned unchanged"),
    Counterfactual("robot_ghost", ("entity_robot_semantic",), make_robot_ghost,
                   "arm re-drawn for 1.5 s where the simulator arm has already left"),
)
