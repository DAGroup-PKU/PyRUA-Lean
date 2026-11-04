"""Realism: is the rendered appearance close to the real workcell?

The authority is real footage of the same embodiment/workcell (Grounding IR),
never the appearance reference picture or the prompt.  For each panel the frame
is split into regions — one per entity plus the background — and simple,
interpretable statistics are compared with the same region in real frames:

* ``color``      mean CIELAB colour (lighting dependent)
* ``contrast``   CIELAB lightness spread (lighting dependent)
* ``chroma``     region colour relative to the frame's mean colour (grey-world
                 normalised, so a global colour cast cancels)
* ``texture``    histogram of gradient magnitude
* ``sharpness``  log Laplacian variance (blur / over-smoothing)
* ``noise``      log high-frequency residual (sensor grain vs plastic look)

Each distance is divided by a fixed tolerance, so a score of 1 is roughly one
just-noticeable difference.  A region passes when every active feature is within
``realism_feature_score_max`` tolerances of the real distribution.

sim2real cohorts use every feature on every region.  OOD cohorts keep invariant
entities (robot, task objects) bound to reality but drop what the declared axes
perturb: lighting drops ``color``/``contrast`` (``chroma`` remains, so the robot
must still look like itself under the new light); background/table axes drop the
background region.  No rule prefers soft or hard light: if the real room has hard
shadows the render should too, and OOD lighting is free as long as invariant
entities survive it.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from datafarm.verify.checks import ABSTAIN, FAIL, NA, PASS, CheckResult, Context, Evidence, _disk

TOLERANCE = {"color": 8.0, "contrast": 5.0, "chroma": 6.0, "texture": 0.25, "sharpness": 0.35, "noise": 0.30}
LIGHTING_FEATURES = ("color", "contrast")
BACKGROUND_AXES = ("background", "table_material", "distractors")
TEXTURE_BINS = np.array([0, 2, 4, 8, 16, 32, 64, 1e9], np.float32)


def _region_features(panel_bgr: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray] | None:
    if mask.sum() < 64:
        return None
    lab = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    pixels = lab[mask]
    frame_mean = lab.reshape(-1, 3).mean(axis=0)
    lightness = lab[..., 0]
    gx = cv2.Sobel(lightness, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lightness, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)[mask]
    hist = np.histogram(grad, bins=TEXTURE_BINS)[0].astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    lap = cv2.Laplacian(lightness, cv2.CV_32F)[mask]
    residual = (lightness - cv2.GaussianBlur(lightness, (0, 0), 1.0))[mask]
    return {
        "color": pixels.mean(axis=0),
        "contrast": np.array([pixels[:, 0].std()]),
        "chroma": pixels.mean(axis=0) - frame_mean,
        "texture": hist,
        "sharpness": np.array([np.log10(float(lap.var()) + 1e-3)]),
        "noise": np.array([np.log10(float(np.median(np.abs(residual))) + 1e-3)]),
    }


def _distance(feature: str, a: np.ndarray, b: np.ndarray) -> float:
    if feature == "texture":
        return float(np.abs(a - b).sum())
    return float(np.linalg.norm(a - b))


def _collect(frames: np.ndarray, masks: dict[str, np.ndarray], rows, cols, indices,
             min_frames: int = 3) -> dict[str, dict[str, np.ndarray]]:
    """Median feature vector per region over the chosen frames.

    A region seen in fewer than ``min_frames`` frames is not characterised: a
    handful of close-ups is not a reliable picture of how the real object looks.
    """

    per_region: dict[str, dict[str, list[np.ndarray]]] = {}
    for j in indices:
        crop = frames[j][rows, cols]
        for region, mask in masks.items():
            feats = _region_features(crop, mask[j][rows, cols])
            if feats is None:
                continue
            slot = per_region.setdefault(region, {})
            for name, value in feats.items():
                slot.setdefault(name, []).append(value)
    return {region: {name: np.median(np.stack(values), axis=0) for name, values in feats.items()}
            for region, feats in per_region.items() if feats and len(next(iter(feats.values()))) >= min_frames}


def _region_masks(ctx: Context, which: str) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    union = None
    for entity in ctx.spec.entities:
        # Rendered regions come from the carrier (geometric authority); real
        # regions come from the target appearance keys.
        mask = ctx.mask("carrier" if which == "generated" else "real", entity)
        masks[entity.role] = mask
        union = mask if union is None else (union | mask)
    base = ctx.generated.frames if which == "generated" else ctx.real.frames  # type: ignore[union-attr]
    count = len(base) if which == "real" else ctx.samples
    if union is None:
        union = np.zeros((count,) + base.shape[1:3], bool)
    background = np.zeros_like(union)
    disk = _disk(max(2, int(0.02 * base.shape[2])))
    for j in range(len(union)):
        background[j] = ~(cv2.dilate(union[j].astype(np.uint8), disk) > 0)
    masks["background"] = background
    return masks


def active_features(ctx: Context, region: str) -> tuple[str, ...]:
    realism = ctx.spec.realism
    features = tuple(TOLERANCE)
    if realism.mode == "ood":
        axes = set(realism.perturbed_axes)
        if region == "background" and axes & set(BACKGROUND_AXES):
            return ()
        if "lighting" in axes or "camera_exposure" in axes:
            features = tuple(f for f in features if f not in LIGHTING_FEATURES)
        entity = next((e for e in ctx.spec.entities if e.role == region), None)
        if entity is not None and not entity.invariant:
            return ()
    return features


def score_against_real(ctx: Context, frames: np.ndarray, masks: dict[str, np.ndarray],
                       indices, real_feats: dict[str, dict[str, dict[str, np.ndarray]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for panel in ctx.spec.panels:
        rows, cols = ctx.work_slice(panel)
        feats = _collect(frames, masks, rows, cols, indices)
        panel_out = {}
        for region, reference in real_feats.get(panel.name, {}).items():
            active = active_features(ctx, region)
            if not active or region not in feats:
                continue
            scores = {f: round(_distance(f, feats[region][f], reference[f]) / TOLERANCE[f], 3) for f in active}
            panel_out[region] = scores
        out[panel.name] = panel_out
    return out


def check_realism(ctx: Context) -> CheckResult:
    spec = ctx.spec
    if ctx.real is None or not len(ctx.real.frames):
        return CheckResult("realism_vs_real_world", "realism", ABSTAIN,
                           reason="no real-world reference footage bound in Grounding")
    real_masks = _region_masks(ctx, "real")
    count_real = len(ctx.real.frames)
    real_idx = np.linspace(0, count_real - 1, min(spec.realism.real_frames, count_real)).astype(int)
    real_feats = {}
    real_rows = {}
    for panel in spec.panels:
        rows, cols = ctx.work_slice(panel, ctx.real)
        real_rows[panel.name] = (rows, cols)
        real_feats[panel.name] = _collect(ctx.real.frames, real_masks, rows, cols, real_idx,
                                          min_frames=max(3, len(real_idx) // 4))
    gen_masks = _region_masks(ctx, "generated")
    gen_idx = np.linspace(0, ctx.samples - 1, min(24, ctx.samples)).astype(int)
    generated = score_against_real(ctx, ctx.generated.frames, gen_masks, gen_idx, real_feats)
    carrier = score_against_real(ctx, ctx.carrier.frames, gen_masks, gen_idx, real_feats)
    # Real-vs-real baseline: first half against second half of the real footage.
    half = len(real_idx) // 2
    baseline = {}
    if half >= 3:
        first = {}
        for panel in spec.panels:
            rows, cols = real_rows[panel.name]
            first[panel.name] = _collect(ctx.real.frames, real_masks, rows, cols, real_idx[:half])
        baseline = score_against_real(ctx, ctx.real.frames, real_masks, real_idx[half:], first)
    limit = spec.threshold("realism_feature_score_max")
    advisory = spec.threshold("realism_feature_advisory")
    failures, evidence, warnings = [], [], []
    judged = 0
    for panel, regions in generated.items():
        for region, scores in regions.items():
            judged += 1
            # The real footage varies by itself (a moving wrist camera sees
            # different things over time); that natural spread is allowed on
            # top of the fixed tolerance.
            spread = baseline.get(panel, {}).get(region, {})
            allowed = {f: limit + 1.5 * float(spread.get(f, 0.0)) for f in scores}
            bad = {f: s for f, s in scores.items() if s > allowed[f]}
            soft = {f: s for f, s in scores.items() if advisory + 1.5 * float(spread.get(f, 0.0)) < s <= allowed[f]}
            if soft:
                warnings.append(f"{panel}:{region}:" + "/".join(f"{f}={v}" for f, v in sorted(soft.items())))
            if bad:
                failures.append(f"{panel}:{region}:" + "/".join(sorted(bad)))
                evidence.append(Evidence(panel, "unrealistic_appearance", 0,
                                         detail=f"{region} " + ", ".join(f"{f}={s}" for f, s in bad.items())))
    metrics = {"mode": spec.realism.mode, "perturbed_axes": list(spec.realism.perturbed_axes),
               "generated": generated, "carrier": carrier, "real_holdout": baseline,
               "real_source": ctx.real.source, "limit": limit, "advisory_limit": advisory,
               "advisory_warnings": warnings}
    if judged == 0:
        return CheckResult("realism_vs_real_world", "realism", ABSTAIN,
                           reason="no region could be compared with the real reference", metrics=metrics)
    return CheckResult("realism_vs_real_world", "realism", FAIL if failures else PASS,
                       reason=", ".join(failures), metrics=metrics, evidence=evidence)


__all__ = ["check_realism", "active_features", "NA"]
