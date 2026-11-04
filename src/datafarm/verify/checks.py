"""Fixed check families of the Render gate.

Every check compares the generated observation with an authority it does not
own: the simulator carrier for geometry, entities, events and time, and the real
reference footage for appearance.  A check returns ``pass``, ``fail``,
``abstain`` (the authority needed to judge is missing or unmeasurable, which
blocks admission) or ``not_applicable`` (an objective measurability condition,
e.g. a moving wrist camera has no static background; it does not block).

Defects must persist for ``persist_samples`` consecutive sampled frames before
they fail a check.  This removes single-frame colour-mask noise while still
catching every defect a viewer can notice (at stride 2 and 30 fps, three
samples are 0.2 s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from datafarm.verify.media import Clip, panel_slice
from datafarm.verify.spec import EntitySpec, PanelSpec, RenderVerifierSpec

PASS, FAIL, ABSTAIN, NA = "pass", "fail", "abstain", "not_applicable"


@dataclass
class Evidence:
    panel: str
    kind: str
    frame: int
    end_frame: int | None = None
    bbox: tuple[int, int, int, int] | None = None  # layout coordinates
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}


@dataclass
class CheckResult:
    name: str
    family: str
    verdict: str
    hard: bool = True
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "verdict": self.verdict,
            "hard": self.hard,
            "reason": self.reason,
            "metrics": self.metrics,
            "evidence": [item.to_dict() for item in self.evidence[:40]],
            "evidence_total": len(self.evidence),
        }


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True values."""

    out: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(np.asarray(flags, bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            out.append((start, index - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def persistent_runs(flags: np.ndarray, length: int) -> list[tuple[int, int]]:
    return [(a, b) for a, b in runs(flags) if b - a + 1 >= length]


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def key_masks(frames: np.ndarray, keys) -> np.ndarray:
    """Boolean masks [N,H,W] of the union of HSV keys, lightly cleaned."""

    out = np.zeros(frames.shape[:3], bool)
    small = _disk(1)
    close = _disk(2)
    for index, frame in enumerate(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame.shape[:2], np.uint8)
        for key in keys:
            mask |= cv2.inRange(hsv, np.array(key.lower, np.uint8), np.array(key.upper, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
        out[index] = mask > 0
    return out


def _lab(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def appearance_model(frames: np.ndarray, carrier: np.ndarray, broad: np.ndarray,
                     max_frames: int = 40) -> dict[str, Any] | None:
    """Robust CIELAB model of the entity as this video renders it.

    Pixels are taken where the carrier (geometric authority) has the entity,
    eroded so that edges and occluders are excluded, and where the broad target
    key agrees.  Median and a trimmed covariance with a floor give a Mahalanobis
    gate that adapts to the render's colour grading.
    """

    erode = _disk(2)
    samples = []
    for j in np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int):
        inner = (cv2.erode(carrier[j].astype(np.uint8), erode) > 0) & broad[j]
        if inner.sum() >= 30:
            samples.append(_lab(frames[j])[inner])
    if not samples:
        return None
    pixels = np.concatenate(samples)
    if len(pixels) > 200_000:
        pixels = pixels[np.random.default_rng(0).choice(len(pixels), 200_000, replace=False)]
    center = np.median(pixels, axis=0)
    spread = np.median(np.abs(pixels - center), axis=0) * 1.4826 + 1e-3
    inliers = pixels[(np.abs(pixels - center) <= 3.0 * spread).all(axis=1)]
    if len(inliers) < 50:
        inliers = pixels
    cov = np.cov(inliers.T) + np.diag([6.0 ** 2, 2.5 ** 2, 2.5 ** 2])
    chroma = float(np.hypot(center[1], center[2]))
    # A chromatic entity must stay chromatic: this separates a dark-rendered
    # object from black robot links and neutral cast shadows of similar lightness.
    chroma_min = 0.5 * chroma if chroma >= 6.0 else 0.0
    return {"center": center.tolist(), "inv_cov": np.linalg.inv(cov), "pixels": int(len(pixels)),
            "gate": 3.0, "chroma_min": chroma_min}


def model_masks(frames: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    center = np.asarray(model["center"], np.float32)
    inv = np.asarray(model["inv_cov"], np.float32)
    out = np.zeros(frames.shape[:3], bool)
    for index, frame in enumerate(frames):
        diff = _lab(frame) - center
        d2 = np.einsum("...i,ij,...j->...", diff, inv, diff)
        keep = d2 <= model["gate"] ** 2
        if model.get("chroma_min"):
            lab = diff + center
            keep &= np.hypot(lab[..., 1], lab[..., 2]) >= model["chroma_min"]
        out[index] = keep
    return out


class Context:
    def __init__(self, spec: RenderVerifierSpec, carrier: Clip, generated: Clip,
                 real: Clip | None = None, references: list[np.ndarray] | None = None,
                 state: dict[str, Any] | None = None):
        self.state = state
        self._footprints: dict[str, Any] = {}
        self.spec = spec
        self.carrier = carrier
        self.generated = generated
        self.real = real
        self.references = references or []
        self._masks: dict[tuple[str, str], np.ndarray] = {}
        self.models: dict[str, Any] = {}
        self.samples = min(len(carrier.frames), len(generated.frames))

    def mask(self, which: str, entity: EntitySpec) -> np.ndarray:
        key = (which, entity.role)
        if key not in self._masks:
            if which == "carrier":
                self._masks[key] = key_masks(self.carrier.frames[: self.samples], entity.carrier_keys)
            elif which == "generated":
                frames = self.generated.frames[: self.samples]
                if not entity.adaptive:
                    self._masks[key] = key_masks(frames, entity.target_keys)
                else:
                    # One appearance mode per colour part: carrier_keys[i] and
                    # target_keys[i] describe the same part of one object.
                    paired = len(entity.carrier_keys) == len(entity.target_keys)
                    parts = (list(zip(entity.carrier_keys, entity.target_keys)) if paired
                             else [(None, None)])
                    out = np.zeros(frames.shape[:3], bool)
                    models = []
                    relit = (self.spec.realism.mode == "ood"
                             and {"lighting", "camera_exposure"} & set(self.spec.realism.perturbed_axes))
                    for c_key, t_key in parts:
                        c_mask = (key_masks(self.carrier.frames[: self.samples], (c_key,)) if c_key
                                  else self.mask("carrier", entity))
                        if relit:
                            # Deliberate relighting moves every colour; a fixed key
                            # would read that as a missing object.  Learn the
                            # entity's look from carrier-authorised pixels only.
                            broad = np.ones(frames.shape[:3], bool)
                        else:
                            broad = key_masks(frames, (t_key,) if t_key else entity.target_keys)
                        model = appearance_model(frames, c_mask, broad)
                        models.append(model)
                        out |= broad & model_masks(frames, model) if model is not None else broad
                    self.models[entity.role] = models
                    self._masks[key] = out
            elif which == "real":
                assert self.real is not None
                self._masks[key] = key_masks(self.real.frames, entity.real_keys or entity.target_keys)
            else:
                raise KeyError(which)
        return self._masks[key]

    def carrier_foreground(self) -> np.ndarray:
        """Where the carrier departs from its own static background (per sample).

        The per-pixel temporal median of the carrier is the empty workcell;
        anything that ever differs from it — every robot link whatever its
        shade, moved objects, simulator shadows — is simulator content.
        """

        if "__fg__" not in self._masks:
            frames = self.carrier.frames[: self.samples]
            step = max(1, len(frames) // 60)
            background = np.median(frames[::step].astype(np.int16), axis=0)
            fg = np.zeros(frames.shape[:3], bool)
            n = len(frames)
            # Simulator renders are noise-free, so small differences are real
            # content: a light-grey link against a grey wall differs by ~10-20.
            for j, frame in enumerate(frames):
                f = frame.astype(np.int16)
                mask = np.abs(f - background).max(axis=2) > 12
                for k in (-8, -4, 4, 8):  # samples; stride 2 -> +-8 / +-16 frames
                    if 0 <= j + k < n:
                        mask |= np.abs(f - frames[j + k].astype(np.int16)).max(axis=2) > 6
                fg[j] = cv2.dilate(mask.astype(np.uint8), _disk(3)) > 0
            self._masks["__fg__"] = fg
        return self._masks["__fg__"]

    def camera_profile(self, panel: PanelSpec) -> dict[str, float] | None:
        """How the real camera itself behaves in this panel (Grounding authority).

        ``exposure_shift_max``: largest global brightness shift of the real
        footage relative to its first frames; ``luma_step_p999``: 99.9th
        percentile of frame-to-frame mean-brightness steps.  A render may do
        what the real camera does, so these bound the exposure-drift and
        flicker thresholds from below.
        """

        if self.real is None or len(self.real.gray) < 10:
            return None
        key = f"camera:{panel.name}"
        if key not in self._footprints:
            rows, cols = panel_slice(panel.box, self.real.gray_scale, self.real.gray.shape[1:3])
            gray = self.real.gray[:, rows, cols].astype(np.float32)
            ref = cv2.GaussianBlur(np.median(gray[: min(5, len(gray))], axis=0), (5, 5), 0)
            shifts = [abs(float(np.median(cv2.GaussianBlur(f, (5, 5), 0) - ref))) for f in gray[::3]]
            steps = np.abs(np.diff(gray.mean(axis=(1, 2))))
            self._footprints[key] = {"exposure_shift_max": float(max(shifts)),
                                     "luma_step_p999": float(np.percentile(steps, 99.9)),
                                     "frames": int(len(gray))}
        return self._footprints[key]

    def footprint(self, entity: EntitySpec) -> np.ndarray | None:
        """Projected simulator footprint of the entity per sample, if available."""

        if entity.role not in self._footprints:
            from datafarm.verify.geometry import project_footprints

            value = None
            if self.state is not None:
                value = project_footprints(self.spec, entity, self.state,
                                           self.carrier.sample_frames[: self.samples],
                                           self.carrier.frames.shape[1:3])
            self._footprints[entity.role] = value
        return self._footprints[entity.role]

    def work_slice(self, panel: PanelSpec, clip: Clip | None = None):
        clip = clip or self.carrier
        return panel_slice(panel.box, clip.work_scale, clip.frames.shape[1:3])

    def gray_slice(self, panel: PanelSpec, clip: Clip | None = None):
        clip = clip or self.carrier
        return panel_slice(panel.box, clip.gray_scale, clip.gray.shape[1:3])

    def to_layout(self, bbox, panel: PanelSpec, scale: float) -> tuple[int, int, int, int]:
        x, y, w, h = bbox
        return (int(panel.box.x + x / scale), int(panel.box.y + y / scale), int(w / scale), int(h / scale))


# ---------------------------------------------------------------------------
# transport


def check_transport(ctx: Context) -> CheckResult:
    spec, gen, car = ctx.spec, ctx.generated, ctx.carrier
    expected = spec.expected_frames or car.frame_count
    lw, lh = spec.layout_size
    gw, gh = gen.native_size
    aspect_error = abs((gw / max(gh, 1)) / (lw / lh) - 1.0) if gh else float("inf")
    luma = gen.gray.reshape(len(gen.gray), -1).mean(axis=1) if len(gen.gray) else np.zeros(0)
    dark = np.flatnonzero(luma <= spec.threshold("black_luma_max"))
    metrics = {
        "generated_frames": gen.frame_count,
        "carrier_frames": car.frame_count,
        "expected_frames": expected,
        "generated_fps": round(gen.fps, 4),
        "expected_fps": spec.fps,
        "generated_native_size": list(gen.native_size),
        "aspect_relative_error": round(aspect_error, 6),
        "dark_frames": int(len(dark)),
        "generated_sha256": gen.sha256,
        "carrier_sha256": car.sha256,
    }
    problems = []
    if gen.frame_count != expected:
        problems.append(f"frame count {gen.frame_count} != {expected}")
    if car.frame_count != expected:
        problems.append(f"carrier frame count {car.frame_count} != {expected}")
    if abs(gen.fps - spec.fps) > spec.threshold("fps_tolerance"):
        problems.append(f"fps {gen.fps:.3f} != {spec.fps}")
    if aspect_error > spec.threshold("aspect_tolerance"):
        problems.append(f"aspect error {aspect_error:.4f}")
    evidence = [Evidence("layout", "dark_frame", int(i)) for i in dark[:10]]
    if len(dark):
        problems.append(f"{len(dark)} black frames")
    return CheckResult("transport_integrity", "transport", FAIL if problems else PASS,
                       reason="; ".join(problems), metrics=metrics, evidence=evidence)


# ---------------------------------------------------------------------------
# structure


def _geometry_descriptor(panel_bgr: np.ndarray) -> np.ndarray:
    height, width = panel_bgr.shape[:2]
    size = (96, max(16, int(round(96 * height / max(width, 1)))))
    gray = cv2.cvtColor(cv2.resize(panel_bgr, size, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    blocks = []
    for value in (blurred, gradient):
        value = value - float(value.mean())
        value = value / max(float(value.std()), 1.0)
        blocks.append(value.reshape(-1))
    descriptor = np.concatenate(blocks)
    return descriptor / max(float(np.linalg.norm(descriptor)), 1e-6)


def _edge_image(panel_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return magnitude / max(float(magnitude.max()), 1e-6)


def _dark_edge_depths(gray_panel: np.ndarray, level: float = 20.0) -> tuple[int, int, int, int]:
    """Depth (pixels) of near-black stripes along the top, bottom, left, right edges."""

    # A crop/offset stripe is uniformly black along the whole edge; a black
    # robot link reaching the edge is dark but textured, so require low spread.
    rows = (gray_panel.mean(axis=1) < level) & (gray_panel.std(axis=1) < 8.0)
    cols = (gray_panel.mean(axis=0) < level) & (gray_panel.std(axis=0) < 8.0)

    def depth(flags):
        n = 0
        for flag in flags:
            if not flag:
                break
            n += 1
        return n

    return depth(rows), depth(rows[::-1]), depth(cols), depth(cols[::-1])


def dark_border_runs(ctx: Context, panel: PanelSpec) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Samples where the render shows a black edge stripe the carrier does not have.

    Panel assembly errors (a provider output cropped or offset by a few pixels)
    show up as a black band at a panel edge — between head and wrists, or down
    one side.  Depth is measured at work scale; >= 2 px there is >= 4 layout px.
    """

    rows, cols = ctx.work_slice(panel)
    extra = np.zeros(ctx.samples, np.int32)
    near_zero_min = ctx.spec.threshold("black_border_near_zero_fraction_min")
    for j in range(ctx.samples):
        generated = ctx.generated.frames[j][rows, cols]
        g = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY).astype(np.float32)
        c = cv2.cvtColor(ctx.carrier.frames[j][rows, cols], cv2.COLOR_BGR2GRAY).astype(np.float32)
        for edge, (a, b) in enumerate(zip(_dark_edge_depths(g), _dark_edge_depths(c))):
            if a <= b:
                continue
            if near_zero_min:
                band = (generated[:a] if edge == 0 else generated[-a:] if edge == 1
                        else generated[:, :a] if edge == 2 else generated[:, -a:])
                if float(np.all(band < 3, axis=2).mean()) < near_zero_min:
                    continue
                # Black interior material may sit behind a visible rim at the
                # panel edge. Padding must reach that edge, not merely occupy
                # part of a larger dark band. Use two work pixels so one row
                # of codec ringing at a panel seam does not hide real padding.
                edge_band = (generated[:2] if edge == 0 else generated[-2:] if edge == 1
                             else generated[:, :2] if edge == 2 else generated[:, -2:])
                if float(np.all(edge_band < 3, axis=2).mean()) < near_zero_min:
                    continue
            extra[j] = max(extra[j], a - b)
    bad = extra >= 2
    return persistent_runs(bad, int(ctx.spec.threshold("persist_samples"))), extra


def check_structure(ctx: Context) -> CheckResult:
    spec = ctx.spec
    n = ctx.samples
    persist = int(spec.threshold("persist_samples"))
    cosine_min = spec.threshold("structure_cosine_min")
    metrics: dict[str, Any] = {}
    evidence: list[Evidence] = []
    failed_panels = []
    window = None
    for panel in spec.panels:
        rows, cols = ctx.work_slice(panel)
        cosines = np.array([
            float(np.dot(_geometry_descriptor(ctx.generated.frames[j][rows, cols]),
                         _geometry_descriptor(ctx.carrier.frames[j][rows, cols])))
            for j in range(n)
        ])
        shifts = []
        for j in range(0, n, max(1, n // 24)):
            a = _edge_image(ctx.carrier.frames[j][rows, cols])
            b = _edge_image(ctx.generated.frames[j][rows, cols])
            if window is None or window.shape != a.shape:
                window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(a, b, window)
            if response > 0.05:
                shifts.append(float(np.hypot(dx, dy)))
        width = cols.stop - cols.start
        median_shift = float(np.median(shifts)) / width if shifts else 0.0
        bad = cosines < cosine_min
        sustained = persistent_runs(bad, 4 * persist)
        bad_fraction = float(bad.mean()) if n else 1.0
        metrics[panel.name] = {
            "cosine_median": round(float(np.median(cosines)), 4) if n else None,
            "cosine_p05": round(float(np.percentile(cosines, 5)), 4) if n else None,
            "bad_fraction": round(bad_fraction, 4),
            "longest_bad_run_samples": max((b - a + 1 for a, b in runs(bad)), default=0),
            "median_shift_frac": round(median_shift, 4),
        }
        panel_failed = False
        if bad_fraction > spec.threshold("structure_bad_fraction_max") or sustained:
            panel_failed = True
            for a, b in (sustained or runs(bad))[:5]:
                evidence.append(Evidence(panel.name, "geometry_drift", int(a * ctx.carrier.stride),
                                         int(b * ctx.carrier.stride),
                                         detail=f"min cosine {cosines[a:b + 1].min():.3f}"))
        if median_shift > spec.threshold("structure_shift_frac_max"):
            panel_failed = True
            evidence.append(Evidence(panel.name, "global_misalignment", 0,
                                     detail=f"median shift {median_shift:.3%} of width"))
        border_runs, extra = dark_border_runs(ctx, panel)
        metrics[panel.name]["black_border_samples"] = int((extra >= 2).sum())
        metrics[panel.name]["black_border_max_px_layout"] = int(extra.max() / ctx.carrier.work_scale) if n else 0
        if border_runs:
            panel_failed = True
            for a, b in border_runs[:3]:
                evidence.append(Evidence(panel.name, "black_border", int(a * ctx.carrier.stride),
                                         int(b * ctx.carrier.stride),
                                         detail=f"up to {int(extra[a:b + 1].max() / ctx.carrier.work_scale)} px black edge "
                                                f"not in the carrier (panel crop/offset error)"))
        if panel_failed:
            failed_panels.append(panel.name)
    return CheckResult("source_structure", "structure", FAIL if failed_panels else PASS,
                       reason=("panels " + ", ".join(failed_panels)) if failed_panels else "",
                       metrics=metrics, evidence=evidence)


# ---------------------------------------------------------------------------
# entities


def _components(mask: np.ndarray, min_area: float):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [k for k in range(1, count) if stats[k, cv2.CC_STAT_AREA] >= min_area]
    return labels, stats, keep


def _bbox_edge_density(frame: np.ndarray, box: np.ndarray) -> float:
    x, y, w, h = (int(v) for v in box[:4])
    crop = frame[y:y + h, x:x + w]
    if min(crop.shape[:2]) < 3:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float((cv2.Canny(gray, 50, 100) > 0).mean())


def check_entity(ctx: Context, entity: EntitySpec) -> CheckResult:
    spec = ctx.spec
    n = ctx.samples
    persist = int(spec.threshold("persist_samples"))
    recall_min = spec.threshold("entity_recall_min")
    support_min = spec.threshold("entity_support_min")
    ratio_max = spec.threshold("entity_extent_ratio_max")
    carrier_mask = ctx.mask("carrier", entity)
    generated_mask = ctx.mask("generated", entity)
    presence_mask = (key_masks(ctx.generated.frames[:n], entity.presence_keys)
                     if entity.presence_keys else generated_mask)
    occluders = None
    for other in spec.entities:
        if other.role != entity.role:
            mask = ctx.mask("carrier", other)
            occluders = mask if occluders is None else (occluders | mask)
    footprint = ctx.footprint(entity)
    unmeasurable: list[str] = []
    scale = ctx.carrier.work_scale
    metrics: dict[str, Any] = {"footprint_authority": footprint is not None}
    evidence: list[Evidence] = []
    failures: list[str] = []
    for panel in spec.entity_panels(entity):
        active = entity.active_checks(panel.name)
        rows, cols = ctx.work_slice(panel)
        height, width = rows.stop - rows.start, cols.stop - cols.start
        min_area = entity.min_area_frac * height * width
        areas = []
        reference_edges = []
        edge_fraction = entity.unsupported_edge_reference_fraction.get(panel.name, 0.0)
        for j in range(0, n, max(1, n // 40)):
            _, stats, keep = _components(carrier_mask[j][rows, cols], min_area)
            areas.extend(int(stats[k, cv2.CC_STAT_AREA]) for k in keep)
            if edge_fraction:
                target_frame = ctx.generated.frames[j][rows, cols]
                reference_edges.extend(_bbox_edge_density(target_frame, stats[k]) for k in keep)
        if not areas:
            metrics[panel.name] = {"visible_samples": 0}
            continue
        typical_area = float(np.median(areas))
        edge_min = (edge_fraction * float(np.percentile(reference_edges, 10))
                    if reference_edges else 0.0)
        low_edge_discarded = 0
        radius = int(np.clip(spec.threshold("entity_tolerance_frac") * np.sqrt(typical_area),
                             2, 0.05 * min(height, width)))
        disk = _disk(radius)
        # Can colour tell this entity apart from this render's background at all?
        # (a wooden table vs a wooden object cannot).  Measure how much of the
        # background far from every simulated object is keyed as the entity.
        leak_values = []
        diffuse_leak_values = []
        for j in range(0, n, max(1, n // 30)):
            anything = carrier_mask[j][rows, cols].copy()
            if occluders is not None:
                anything |= occluders[j][rows, cols]
            if footprint is not None:
                anything |= footprint[j][rows, cols]
            far = ~(cv2.dilate(anything.astype(np.uint8), _disk(3 * radius)) > 0)
            if far.mean() > 0.05:
                target = generated_mask[j][rows, cols].astype(np.uint8)
                if entity.instance_open_px_work:
                    size = entity.instance_open_px_work
                    target = cv2.morphologyEx(target, cv2.MORPH_OPEN,
                                              np.ones((size, size), np.uint8))
                leak_values.append(float(target[far].mean()))
                # A detached entity-sized blob is the very signal the
                # unsupported-instance test is meant to catch. It cannot also
                # make that test unmeasurable by raising background leakage.
                # Keep broad, thin and border-touching colour regions in the
                # background estimate; only compact interior blobs are exempt.
                diffuse = target.copy()
                leak_labels, leak_stats, leak_keep = _components(target, 1)
                for k in leak_keep:
                    x, y, w, h, area = (int(v) for v in leak_stats[k])
                    if (0.25 * typical_area <= area <= 4 * typical_area
                            and x > 1 and y > 1
                            and x + w < width - 1 and y + h < height - 1
                            and max(w, h) <= 3 * max(1, min(w, h))
                            and area >= 0.35 * w * h):
                        diffuse[leak_labels == k] = 0
                diffuse_leak_values.append(float(diffuse[far].mean()))
        leak = float(np.median(leak_values)) if leak_values else 0.0
        diffuse_leak = float(np.median(diffuse_leak_values)) if diffuse_leak_values else leak
        if leak > spec.threshold("entity_key_leak_max") and active & {"unsupported", "extent"}:
            # Where the background itself reads as the entity, neither new
            # blobs nor growth can be attributed to the entity.
            lost = sorted(active & {"unsupported", "extent"})
            if diffuse_leak <= spec.threshold("entity_key_leak_max"):
                lost = [name for name in lost if name != "unsupported"]
            active = active - set(lost)
            if lost:
                unmeasurable.append(f"{panel.name}:{'/'.join(lost)} (background keyed as {entity.role}: {leak:.1%})")
        missing = np.zeros(n, bool)
        unsupported = np.zeros(n, bool)
        extent = np.zeros(n, bool)
        visible = np.zeros(n, bool)
        detail: dict[str, list] = {"missing": [None] * n, "unsupported": [None] * n, "extent": [None] * n}
        worst = {"recall": 1.0, "support": 1.0, "ratio": 1.0}
        grow_values: list[float] = []
        fp_cache: dict[int, np.ndarray] = {}
        for j in range(n):
            cm = carrier_mask[j][rows, cols]
            gm = generated_mask[j][rows, cols]
            if entity.instance_open_px_work:
                size = entity.instance_open_px_work
                gm_instance = cv2.morphologyEx(gm.astype(np.uint8), cv2.MORPH_OPEN,
                                               np.ones((size, size), np.uint8)) > 0
            else:
                gm_instance = gm
            labels, stats, keep = _components(cm, min_area)
            visible[j] = bool(keep)
            cm8 = cm.astype(np.uint8)
            # Presence can use the lower-chroma look of a real reference, but
            # unsupported/extent must keep the stricter new-instance key.
            gen_near = cv2.dilate(presence_mask[j][rows, cols].astype(np.uint8), disk) > 0
            car_near = cv2.dilate(cm8, disk) > 0
            if footprint is not None:
                # Anything inside a simulated object's projected footprint is
                # backed by that object (its interior, its shaded side).
                car_near |= cv2.dilate(footprint[j][rows, cols].astype(np.uint8), disk) > 0
            if occluders is not None:
                # "Unsupported" means a new object on empty background.  A blob
                # on top of another simulated entity (one object keyed as another
                # object's colour) is an identity/appearance question, not a
                # hallucinated object.
                car_near |= cv2.dilate(occluders[j][rows, cols].astype(np.uint8), disk) > 0
            if "missing" in active:
                observed_near = None
                for k in keep:
                    comp = labels == k
                    recall = float(gen_near[comp].mean())
                    if recall < recall_min and entity.presence_keys:
                        # Explicit presence keys already allow a material to
                        # differ from its strict instance colour. Check actual
                        # source-aligned boundaries and local contrast before
                        # declaring a desaturated or shaded part absent.
                        if observed_near is None:
                            from datafarm.verify.appearance_presence import source_boundary_presence
                            observed, _ = source_boundary_presence(
                                cm8 * 255, presence_mask[j][rows, cols].astype(np.uint8) * 255,
                                ctx.generated.frames[j][rows, cols], min_area)
                            observed_near = cv2.dilate(observed, disk) > 0
                        recall = max(recall, float(observed_near[comp].mean()))
                    worst["recall"] = min(worst["recall"], recall)
                    if recall < recall_min:
                        missing[j] = True
                        x, y, w, h = stats[k, :4]
                        detail["missing"][j] = (int(x), int(y), int(w), int(h), recall)
            if "unsupported" in active:
                # A hallucinated entity is entity-sized; slivers are mask noise.
                g_labels, g_stats, g_keep = _components(gm_instance, max(min_area, 0.25 * typical_area))
                for g in g_keep:
                    comp = g_labels == g
                    support = float(car_near[comp].mean())
                    # A blob that covers most of a carrier instance *is* that
                    # instance, even if it also spreads beyond it (a darker
                    # rendered object interior, a merged shadow).  Oversize is the
                    # extent sub-check's job; unsupported means a new instance.
                    touched = np.unique(labels[comp & cm])
                    touched = [k for k in touched if k in keep]
                    if touched:
                        instance = np.isin(labels, touched)
                        covered = float((comp & instance).sum()) / max(1.0, float(instance.sum()))
                        if covered >= 0.5:
                            support = max(support, 1.0)
                    if support < support_min:
                        if edge_min and _bbox_edge_density(ctx.generated.frames[j][rows, cols],
                                                           g_stats[g]) < edge_min:
                            low_edge_discarded += 1
                            continue
                        unsupported[j] = True
                        x, y, w, h = g_stats[g, :4]
                        detail["unsupported"][j] = (int(x), int(y), int(w), int(h), support)
                    worst["support"] = min(worst["support"], support)
            if "extent" in active and keep:
                # A nearby, disconnected target-colour sliver is not growth of
                # this carrier instance. Keep only rendered components that
                # touch its tolerance halo; the unsupported check owns new
                # components elsewhere. This also avoids panel-edge codec
                # colour from enlarging the instance's bounding box.
                extent_labels, _, _ = _components(gm_instance, 1)
                others_all = np.zeros_like(cm8)
                for k in keep:
                    others_all[labels == k] = 1
                occluder = occluders[j][rows, cols] if occluders is not None else None
                for k in keep:
                    x, y, w, h = stats[k, :4]
                    if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
                        continue  # cut by the panel border; extent is not observable
                    comp8 = (labels == k).astype(np.uint8)
                    if occluder is not None and (occluder & (cv2.dilate(comp8, disk) > 0)).any():
                        continue  # touches another entity: visible extent depends on occlusion
                    # Search far enough to see a grown object (up to ~2x), but
                    # never into a neighbouring instance.
                    reach = max(2 * radius, int(0.6 * max(w, h)))
                    region = cv2.dilate(comp8, _disk(reach)) > 0
                    others = others_all.copy()
                    others[labels == k] = 0
                    region &= ~(cv2.dilate(others, disk) > 0)
                    if occluder is not None:
                        # never attribute another entity's pixels to this one
                        region &= ~(cv2.dilate(occluder.astype(np.uint8), disk) > 0)
                    halo = cv2.dilate(comp8, disk) > 0
                    linked_ids = np.unique(extent_labels[region & halo & gm_instance])
                    linked_ids = linked_ids[linked_ids > 0]
                    if not len(linked_ids):
                        continue
                    linked = np.isin(extent_labels, linked_ids) & region
                    points = cv2.findNonZero(linked.astype(np.uint8))
                    if points is None:
                        continue
                    gx, gy, gw, gh = cv2.boundingRect(points)
                    shrink = min(gw / max(w, 1), gh / max(h, 1))
                    if footprint is not None:
                        # Growth = rendered entity pixels outside the simulated
                        # object's projected silhouette (an interior the carrier
                        # key cannot see is inside it and therefore allowed).
                        fp_labels = fp_cache.get(j)
                        if fp_labels is None:
                            _, fp_labels = cv2.connectedComponents(footprint[j][rows, cols].astype(np.uint8))
                            fp_cache[j] = fp_labels
                        ids = np.unique(fp_labels[comp8 > 0])
                        ids = ids[ids > 0]
                        if not len(ids):
                            continue  # object not projectable here (e.g. against a wrist lens)
                        own = np.isin(fp_labels, ids)
                        silhouette = cv2.dilate(own.astype(np.uint8), _disk(max(1, radius // 2))) > 0
                        outside = float((linked & ~silhouette & ~halo).sum())
                        grow = 1.0 + outside / max(1.0, float(comp8.sum()))
                    else:
                        grow = max(gw / max(w, 1), gh / max(h, 1))
                    hi, lo = grow, shrink
                    grow_values.append(hi)
                    if hi > worst["ratio"]:
                        worst["ratio"] = hi
                    # Only growth is judged: a shaded part dropping out of the
                    # colour mask looks like shrinking but is not a defect, and a
                    # truly vanished object is the "missing" sub-check's job.
                    if hi > ratio_max:
                        extent[j] = True
                        detail["extent"][j] = (int(gx), int(gy), int(gw), int(gh), round(hi, 3))
        panel_metrics = {"visible_samples": int(visible.sum()), "tolerance_px_work": radius,
                         "background_key_leak": round(leak, 4)}
        if edge_fraction:
            panel_metrics["unsupported_edge_reference_p10"] = round(
                float(np.percentile(reference_edges, 10)), 4) if reference_edges else None
            panel_metrics["unsupported_edge_min"] = round(edge_min, 4)
            panel_metrics["unsupported_low_edge_discarded"] = low_edge_discarded
        if grow_values:
            panel_metrics["extent_growth_p50_p99_max"] = [round(float(np.percentile(grow_values, q)), 3)
                                                         for q in (50, 99, 100)]
        for kind, flags in (("missing", missing), ("unsupported", unsupported), ("extent", extent)):
            if kind not in active:
                continue
            bad = persistent_runs(flags, persist)
            panel_metrics[f"{kind}_samples"] = int(flags.sum())
            panel_metrics[f"{kind}_persistent_runs"] = len(bad)
            if bad:
                failures.append(f"{panel.name}:{kind}")
            for a, b in bad[:6]:
                mid = (a + b) // 2
                if detail[kind][mid] is None:
                    mid = next(j for j in range(a, b + 1) if detail[kind][j] is not None)
                x, y, w, h, value = detail[kind][mid]
                evidence.append(Evidence(
                    panel.name, f"{entity.role}_{kind}", int(mid * ctx.carrier.stride),
                    bbox=ctx.to_layout((x, y, w, h), panel, scale),
                    detail=f"value {value:.3f}; persists frames {a * ctx.carrier.stride}-{b * ctx.carrier.stride}"))
        metrics[panel.name] = panel_metrics
    metrics["worst"] = {k: round(v, 4) for k, v in worst.items()} if n else {}
    models = ctx.models.get(entity.role) or []
    metrics["appearance_model_lab"] = [[round(float(v), 2) for v in m["center"]] if m else None
                                       for m in models]
    metrics["unmeasurable"] = unmeasurable
    if failures:
        verdict, reason = FAIL, ", ".join(failures)
    elif unmeasurable:
        verdict, reason = ABSTAIN, "; ".join(unmeasurable)
    else:
        verdict, reason = PASS, ""
    return CheckResult(f"entity_{entity.role}", "entity", verdict, reason=reason,
                       metrics=metrics, evidence=evidence)


# ---------------------------------------------------------------------------
# events


def _roi_motion(gray: np.ndarray, rows, cols, roi: np.ndarray, lo: int, hi: int) -> np.ndarray:
    lo, hi = max(1, lo), min(len(gray) - 1, hi)
    out = np.zeros(max(0, hi - lo + 1), np.float32)
    if not roi.any():
        return out
    for t in range(lo, hi + 1):
        a = gray[t][rows, cols].astype(np.float32)
        b = gray[t - 1][rows, cols].astype(np.float32)
        out[t - lo] = float(np.abs(a - b)[roi].mean())
    return out


def check_events(ctx: Context) -> CheckResult:
    spec = ctx.spec
    if not spec.events:
        return CheckResult("event_alignment", "event", NA, hard=True,
                           reason="spec declares no events for this episode")
    entities = {e.role: e for e in spec.entities}
    half, search = 30, 15
    lag_max, corr_min = spec.threshold("event_lag_max"), spec.threshold("event_corr_min")
    metrics: dict[str, Any] = {}
    evidence: list[Evidence] = []
    failed, abstained = [], []
    for event in spec.events:
        panels = [spec.panel(p) for p in event.panels] if event.panels else list(spec.panels)
        results = []
        skipped = []
        for panel in panels:
            wr, wc = ctx.work_slice(panel)
            roi = np.zeros((wr.stop - wr.start, wc.stop - wc.start), bool)
            lo_s = ctx.carrier.sample_of(event.frame - half)
            hi_s = ctx.carrier.sample_of(event.frame + half)
            for role in event.roles:
                mask = ctx.mask("carrier", entities[role])
                roi |= mask[lo_s:hi_s + 1, wr, wc].any(axis=0)
            if roi.mean() < 0.002:
                continue
            roi = cv2.dilate(roi.astype(np.uint8), _disk(max(2, int(0.04 * roi.shape[1])))) > 0
            gr, gc = ctx.gray_slice(panel)
            roi_gray = cv2.resize(roi.astype(np.uint8), (gc.stop - gc.start, gr.stop - gr.start),
                                  interpolation=cv2.INTER_NEAREST) > 0
            lo, hi = event.frame - half - search, event.frame + half + search
            mc = _roi_motion(ctx.carrier.gray, gr, gc, roi_gray, lo, hi)
            mg = _roi_motion(ctx.generated.gray, gr, gc, roi_gray, lo, hi)
            if len(mc) < 2 * half or float(mc.std()) < 0.15:
                skipped.append({"panel": panel.name, "reason": "insufficient_source_motion_in_window"})
                continue  # carrier shows too little motion here to time the event
            # Motion elsewhere in the 90-frame correlation window does not
            # establish that this particular event is visible.  A task may
            # require local source motion before interpreting a correlation lag.
            event_index = event.frame - max(1, lo)
            local = mc[max(0, event_index - 4):min(len(mc), event_index + 5)]
            local_peak = float(local.max()) if len(local) else 0.0
            if local_peak < spec.threshold("event_local_motion_min"):
                skipped.append({"panel": panel.name, "reason": "insufficient_source_motion_at_event",
                                "source_motion_peak": round(local_peak, 4)})
                continue
            core = slice(search, len(mc) - search)
            ref = mc[core] - mc[core].mean()
            best = (-2.0, 0)
            for lag in range(-search, search + 1):
                seg = mg[core.start + lag: core.stop + lag]
                if len(seg) != len(ref):
                    continue
                seg = seg - seg.mean()
                denom = float(np.linalg.norm(ref) * np.linalg.norm(seg))
                corr = float(np.dot(ref, seg) / denom) if denom > 1e-6 else -1.0
                if corr > best[0]:
                    best = (corr, lag)
            results.append({"panel": panel.name, "corr": round(best[0], 3), "lag": best[1]})
        metrics[event.event_id] = {"frame": event.frame, "panels": results, "skipped_panels": skipped}
        if not results:
            abstained.append(event.event_id)
            continue
        good = [r for r in results if r["corr"] >= corr_min and abs(r["lag"]) <= lag_max]
        contradict = [r for r in results if r["corr"] >= 0.6 and abs(r["lag"]) > lag_max]
        if not good or contradict:
            failed.append(event.event_id)
            worst = max(results, key=lambda r: (abs(r["lag"]), -r["corr"]))
            evidence.append(Evidence(worst["panel"], "event_misaligned", event.frame,
                                     detail=f"{event.event_id}: lag {worst['lag']} corr {worst['corr']}"))
    if failed:
        verdict, reason = FAIL, "misaligned " + ", ".join(failed)
    elif abstained and len(abstained) == len(spec.events):
        verdict, reason = ABSTAIN, "no event measurable in any panel"
    else:
        verdict, reason = PASS, (f"unmeasurable: {', '.join(abstained)}" if abstained else "")
    metrics["unmeasurable"] = abstained
    return CheckResult("event_alignment", "event", verdict, reason=reason, metrics=metrics, evidence=evidence)


# ---------------------------------------------------------------------------
# temporal and seams


def _panel_diffs(gray: np.ndarray, rows, cols) -> tuple[np.ndarray, np.ndarray]:
    crop = gray[:, rows, cols].astype(np.float32)
    diff = np.abs(np.diff(crop, axis=0)).mean(axis=(1, 2))
    luma = np.abs(np.diff(crop.mean(axis=(1, 2))))
    return diff, luma


def _source_supported_temporal_steps(generated: np.ndarray, carrier: np.ndarray,
                                     flagged: np.ndarray, *, min_share: float,
                                     gray_scale: float) -> np.ndarray:
    """Identify a gradual, source-backed occlusion rather than a camera flash.

    A dark real object front can change most of a wrist panel in one frame while
    its light-grey simulator counterpart changes the panel mean only slightly.
    Require the changed render pixels to follow the source's moving footprint,
    leave measurable static background, and continue in the same direction in
    a neighbouring frame. A whole-panel flash fails the spatial condition and
    a one-frame flash fails the temporal condition.
    """

    explained = np.zeros(len(flagged), bool)
    if min_share <= 0:
        return explained
    gmean = np.diff(generated.mean(axis=(1, 2)))
    cmean = np.diff(carrier.mean(axis=(1, 2)))
    radius = max(1, round(8 * gray_scale))
    for i in np.flatnonzero(flagged):
        if not cmean[i] or np.sign(gmean[i]) != np.sign(cmean[i]):
            continue
        # A moving surface enters over several frames. A single-frame flash
        # returns in the opposite direction; a persistent exposure step has
        # no adjacent source-backed luminance transition.
        neighbours = [j for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(gmean)]
        if not any(np.sign(gmean[j]) == np.sign(gmean[i])
                   and abs(gmean[j]) >= 0.25 * abs(gmean[i]) for j in neighbours):
            continue
        # The carrier and edit may place the same edge transition one frame
        # apart. Use adjacent source transitions, within existing event lag
        # tolerance, rather than demanding identical contrast at one instant.
        source_delta = np.max(np.abs(np.diff(
            carrier[max(0, i - 1):min(len(carrier), i + 3)].astype(np.int16), axis=0)), axis=0)
        source_motion = cv2.dilate((source_delta > 5).astype(np.uint8), _disk(radius)) > 0
        footprint_fraction = float(source_motion.mean())
        if footprint_fraction < 0.20:
            continue
        if footprint_fraction > 0.70:
            # A withdrawing wrist camera can move nearly the whole image.
            # Such support cannot alone excuse a flash: require real optical
            # flow and photometric agreement after compensating that motion.
            previous, current = generated[i], generated[i + 1]
            flow = cv2.calcOpticalFlowFarneback(current, previous, None, .5, 3, 15, 3, 5, 1.2, 0)
            yy, xx = np.mgrid[:current.shape[0], :current.shape[1]].astype(np.float32)
            predicted = cv2.remap(previous, xx + flow[:, :, 0], yy + flow[:, :, 1],
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            residual = np.abs(predicted.astype(np.float32) - current)
            if (float(np.median(np.linalg.norm(flow, axis=2))) < 4 * gray_scale
                    or float(np.median(residual)) > 15.0):
                continue
        render_delta = np.abs(generated[i + 1].astype(np.int16) - generated[i].astype(np.int16))
        changed = render_delta > 20
        changed_fraction = float(changed.mean())
        if changed_fraction < 0.15:
            continue
        supported = changed & source_motion
        if (float(supported.sum()) / max(1, int(changed.sum())) >= min_share
                and float((changed & ~source_motion).mean()) <= 0.15):
            explained[i] = True
    return explained


def check_temporal(ctx: Context) -> CheckResult:
    spec = ctx.spec
    metrics: dict[str, Any] = {}
    evidence: list[Evidence] = []
    failed = []
    seams = set(spec.seams)
    for panel in spec.panels:
        rows, cols = ctx.gray_slice(panel)
        diff, luma = _panel_diffs(ctx.generated.gray, rows, cols)
        c_diff, c_luma = _panel_diffs(ctx.carrier.gray, rows, cols)
        count = min(len(diff), len(c_diff))
        if count < 3:
            failed.append(panel.name)
            continue
        diff, luma, c_diff, c_luma = diff[:count], luma[:count], c_diff[:count], c_luma[:count]
        # A higher-contrast render turns the same carrier event into a larger
        # pixel change; compare changes in units of each clip's own contrast.
        g_con = float(np.median(ctx.generated.gray[:count + 1, rows, cols].reshape(count + 1, -1).std(axis=1))) or 1.0
        c_con = float(np.median(ctx.carrier.gray[:count + 1, rows, cols].reshape(count + 1, -1).std(axis=1))) or 1.0
        c_diff_scaled = c_diff * (g_con / c_con)
        local = np.array([np.median(diff[max(0, i - 7): i + 8]) for i in range(count)])
        # diff[i] is the change into frame i + 1; seams are judged by the continuity check.
        not_seam = np.array([(i + 1) not in seams for i in range(count)])
        jumps = (diff > 4 * np.maximum(local, 0.3)) & (diff > 3 * np.maximum(c_diff_scaled, 0.3)) & not_seam
        freezes = (diff < 0.05) & (c_diff > 0.5)
        step_max = spec.threshold("temporal_luma_step_max")
        profile = ctx.camera_profile(panel)
        if profile is not None:
            step_max = max(step_max, 1.25 * profile["luma_step_p999"])
        unexplained = (luma > np.maximum(step_max, 2.5 * c_luma * (g_con / c_con) + 1.0)) & not_seam
        candidate = jumps | unexplained
        explained = _source_supported_temporal_steps(
            ctx.generated.gray[:count + 1, rows, cols],
            ctx.carrier.gray[:count + 1, rows, cols], candidate,
            min_share=spec.threshold("temporal_source_motion_share_min"),
            gray_scale=ctx.generated.gray_scale,
        )
        jumps &= ~explained
        unexplained &= ~explained
        metrics[panel.name] = {
            "jumps": int(jumps.sum()), "freezes": int(freezes.sum()),
            "unexplained_luma_steps": int(unexplained.sum()),
            "source_supported_steps": int(explained.sum()),
            "source_supported_frames": [int(i + 1) for i in np.flatnonzero(explained)[:8]],
            "max_luma_step": round(float(luma.max()), 3),
            "luma_step_limit": round(float(step_max), 2),
        }
        bad = (unexplained.sum() > 0 or jumps.sum() > spec.threshold("temporal_jumps_max")
               or freezes.sum() > spec.threshold("temporal_freezes_max"))
        if bad:
            failed.append(panel.name)
            for kind, flags in (("luma_step", unexplained), ("jump", jumps), ("freeze", freezes)):
                for a, b in runs(flags)[:4]:
                    evidence.append(Evidence(panel.name, kind, int(a + 1), int(b + 1)))
    return CheckResult("temporal_consistency", "temporal", FAIL if failed else PASS,
                       reason=("panels " + ", ".join(failed)) if failed else "",
                       metrics=metrics, evidence=evidence)


def check_continuity(ctx: Context) -> CheckResult:
    """Chunk seams must look like any other frame transition.

    Two things can jump at a seam: brightness/white balance (photometric) and
    content (structural).  The photometric step is judged with the same limit as
    flicker anywhere else; the structural jump is measured after matching the
    two frames' global gain and bias, relative to local motion and the carrier.
    """

    spec = ctx.spec
    if not spec.seams:
        if spec.chunked:
            return CheckResult("seam_continuity", "continuity", ABSTAIN,
                               reason="chunked render without declared seams")
        return CheckResult("seam_continuity", "continuity", NA, reason="single-chunk render")
    multiple = spec.threshold("seam_jump_multiple_max")
    metrics: dict[str, Any] = {}
    evidence: list[Evidence] = []
    failed = []
    for panel in spec.panels:
        rows, cols = ctx.gray_slice(panel)
        gen = ctx.generated.gray[:, rows, cols].astype(np.float32)
        car = ctx.carrier.gray[:, rows, cols].astype(np.float32)
        count = min(len(gen), len(car))
        step_max = spec.threshold("temporal_luma_step_max")
        profile = ctx.camera_profile(panel)
        if profile is not None:
            step_max = max(step_max, 1.25 * profile["luma_step_p999"])

        def structural(a: np.ndarray, b: np.ndarray) -> float:
            # match b's gain/bias to a before differencing
            sa, sb = float(a.std()) or 1.0, float(b.std()) or 1.0
            b2 = (b - b.mean()) * (sa / sb) + a.mean()
            return float(np.abs(a - b2).mean())

        rows_out = []
        for seam in spec.seams:
            if not 2 <= seam < count - 9:
                rows_out.append({"seam": seam, "status": "out_of_range"})
                failed.append(f"{panel.name}@{seam}")
                continue
            neighbours = [structural(gen[t - 1], gen[t]) for t in list(range(seam - 8, seam)) + list(range(seam + 1, seam + 9))]
            local = float(np.median(neighbours))
            jump = structural(gen[seam - 1], gen[seam])
            carrier_jump = float(np.abs(car[seam] - car[seam - 1]).mean())
            luma_step = abs(float(gen[seam].mean() - gen[seam - 1].mean()))
            contrast = (float(np.median(gen[seam - 8:seam + 8].reshape(16, -1).std(axis=1)))
                        / max(float(np.median(car[seam - 8:seam + 8].reshape(16, -1).std(axis=1))), 1.0))
            carrier_luma = abs(float(car[seam].mean() - car[seam - 1].mean())) * contrast
            ratio = jump / max(local, 0.3)
            # Tiny texture re-rolls in a static view give huge ratios over a
            # near-zero local motion; require a visible absolute change too.
            structural_bad = (ratio > multiple and jump > 3.0 * max(carrier_jump, 0.3)
                              and jump > spec.threshold("seam_structural_min"))
            photometric_bad = luma_step > max(step_max, 2.5 * carrier_luma + 1.0)
            rows_out.append({"seam": seam, "structural_ratio": round(ratio, 2), "structural_jump": round(jump, 2),
                             "luma_step": round(luma_step, 2),
                             "luma_limit": round(step_max, 2)})
            if structural_bad or photometric_bad:
                failed.append(f"{panel.name}@{seam}")
                kind = "seam_structural_jump" if structural_bad else "seam_photometric_step"
                evidence.append(Evidence(panel.name, kind, seam,
                                         detail=f"structural {ratio:.1f}x local, luma step {luma_step:.1f}"))
        metrics[panel.name] = rows_out
    return CheckResult("seam_continuity", "continuity", FAIL if failed else PASS,
                       reason=", ".join(failed), metrics=metrics, evidence=evidence)


# ---------------------------------------------------------------------------
# background


def _background_exposure_shift(delta: np.ndarray, judged: np.ndarray,
                               far: np.ndarray) -> tuple[float, bool]:
    """Avoid fitting camera exposure to a moving object's broad dark shadow.

    A shadow can cover most of the static carrier pixels while leaving an
    unshadowed upper quartile near the first-frame exposure. Correct only in
    that observable case. A true global exposure drop has no near-zero upper
    quartile; bright blobs are still measured against the median.
    """

    values = delta[judged]
    median = float(np.median(values))
    if median < -20.0 and float(far.mean()) < 0.03:
        upper = float(np.percentile(values, 75))
        if abs(upper) <= 5.0 and upper - median >= 20.0:
            return upper, True
    return median, False


def check_background(ctx: Context) -> CheckResult:
    """Where the simulator background is unchanged, the render must be unchanged too.

    For every frame t the judged pixels are those whose carrier value still
    equals the carrier's first frames, excluding a shadow margin around anything
    that has moved by t (real rooms have shadows that follow the arm).  On those
    pixels the generated frame is compared with its own first frames: the signed
    mean change measures exposure / white-balance drift, the p95 of the blurred
    absolute change measures local relighting or texture crawl.  A defect must
    persist for ``persist_samples`` frames.
    """

    spec = ctx.spec
    persist = int(spec.threshold("persist_samples"))
    base_mean_max = spec.threshold("background_mean_drift_max")
    local_max = spec.threshold("background_local_drift_p95_max")
    min_static = spec.threshold("background_min_static_fraction")
    metrics: dict[str, Any] = {}
    evidence: list[Evidence] = []
    failed, judged = [], 0
    for panel in spec.panels:
        if panel.motion != "fixed":
            metrics[panel.name] = {"status": "moving_camera"}
            continue
        rows, cols = ctx.gray_slice(panel)
        profile = ctx.camera_profile(panel)
        mean_max = base_mean_max
        if profile is not None:
            mean_max = max(base_mean_max, 1.25 * profile["exposure_shift_max"])
        count = min(len(ctx.carrier.gray), len(ctx.generated.gray))
        carrier = ctx.carrier.gray[:count, rows, cols].astype(np.float32)
        gen = ctx.generated.gray[:count, rows, cols].astype(np.float32)
        c_ref = np.median(carrier[: min(5, count)], axis=0)
        g_ref = cv2.GaussianBlur(np.median(gen[: min(5, count)], axis=0), (5, 5), 0)
        margin = _disk(max(2, int(spec.threshold("background_shadow_margin_frac") * carrier.shape[2])))
        # Real light can throw a moving body's shadow well away from it (a lifted
        # object, an arm overhead). A departing initial shadow can also reveal a
        # brighter surface. Its return is bounded by nearby opening appearance,
        # observed source motion, and the configured coarse exposure allowance.
        shadow_reach = _disk(max(3, int(spec.threshold("background_shadow_reach_frac") * carrier.shape[2])))
        opening_surface_ceiling = cv2.dilate(g_ref, shadow_reach)
        moved = np.zeros(c_ref.shape, np.uint8)
        step = 3
        frames = list(range(0, count, step))
        mean_drift = np.zeros(len(frames), np.float32)
        local_p95 = np.zeros(len(frames), np.float32)
        blob_frac = np.zeros(len(frames), np.float32)
        blob_box: list = [None] * len(frames)
        shadow_shift_corrections = 0
        blob_level = spec.threshold("background_blob_level")
        panel_area = float(carrier.shape[1] * carrier.shape[2])
        fractions = np.zeros(len(frames), np.float32)
        for i, t in enumerate(frames):
            changed = (np.abs(carrier[t] - c_ref) >= 4).astype(np.uint8)
            moved |= changed
            judged_px = ~(cv2.dilate(changed, margin) > 0)
            far_px = ~(cv2.dilate(changed, shadow_reach) > 0)
            fractions[i] = judged_px.mean()
            if fractions[i] < min_static:
                continue
            delta = cv2.GaussianBlur(gen[t], (5, 5), 0) - g_ref
            shift, shadow_corrected = _background_exposure_shift(delta, judged_px, far_px)
            shadow_shift_corrections += int(shadow_corrected)
            # Exposure drift is a camera property, judged against the real
            # camera's own drift; local drift is what remains after removing it.
            mean_drift[i] = abs(shift)
            residual = delta - shift
            restored_shadow = ((cv2.dilate(moved, shadow_reach) > 0)
                               & (opening_surface_ceiling - g_ref > blob_level)
                               & (residual > 0) & (residual <= base_mean_max)
                               & (g_ref + residual <= opening_surface_ceiling + 3))
            local = np.concatenate([np.maximum(residual[judged_px & ~restored_shadow], 0.0),
                                    np.maximum(-residual[far_px], 0.0)])        # darkening far from motion
            local_p95[i] = float(np.percentile(local, 95)) if local.size else 0.0
            # A compact hallucinated light is small relative to the background,
            # so also measure the largest coherent brightened blob.  Darkening is
            # left to the p95 statistic: soft arm shadows can land anywhere.
            odd = (residual > blob_level) & judged_px & ~restored_shadow
            if odd.any():
                count, _, stats, _ = cv2.connectedComponentsWithStats(odd.astype(np.uint8), connectivity=8)
                if count > 1:
                    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    blob_frac[i] = float(stats[biggest, cv2.CC_STAT_AREA]) / panel_area
                    blob_box[i] = tuple(int(v) for v in stats[biggest, :4])
        usable = fractions >= min_static
        if usable.mean() < 0.3:
            metrics[panel.name] = {"status": "insufficient_static_background",
                                   "usable_fraction": round(float(usable.mean()), 3),
                                   "min_static_fraction": round(min_static, 3)}
            continue
        judged += 1
        blob_max = spec.threshold("background_blob_area_frac_max")
        bad = usable & ((mean_drift > mean_max) | (local_p95 > local_max) | (blob_frac > blob_max))
        sustained = persistent_runs(bad, persist)
        metrics[panel.name] = {
            "exposure_drift_limit": round(mean_max, 2),
            "exposure_limit_source": "real-camera x1.25" if profile is not None and mean_max > base_mean_max else "default",
            "usable_fraction": round(float(usable.mean()), 3),
            "min_static_fraction": round(min_static, 3),
            "skipped_low_static_frames": int((~usable).sum()) * step,
            "judged_pixel_fraction_median": round(float(np.median(fractions[usable])), 3),
            "mean_drift_max": round(float(mean_drift[usable].max()), 3),
            "local_drift_p95_max": round(float(local_p95[usable].max()), 3),
            "odd_blob_area_frac_max": round(float(blob_frac[usable].max()), 4),
            "bad_frames": int(bad.sum()) * step,
            "shadow_shift_corrections": shadow_shift_corrections,
        }
        if sustained:
            failed.append(panel.name)
            for a, b in sustained[:4]:
                worst = a + int(np.argmax(local_p95[a:b + 1] + 1000 * blob_frac[a:b + 1]))
                box = blob_box[worst]
                layout_box = ctx.to_layout(box, panel, ctx.carrier.gray_scale) if box else None
                evidence.append(Evidence(panel.name, "background_drift", frames[worst], bbox=layout_box,
                                         detail=f"mean {mean_drift[worst]:.1f}, local p95 {local_p95[worst]:.1f}, "
                                                f"blob {blob_frac[worst]:.3%}; persists frames {frames[a]}-{frames[b]}"))
    if failed:
        return CheckResult("background_stability", "background", FAIL, reason="panels " + ", ".join(failed),
                           metrics=metrics, evidence=evidence)
    if judged == 0:
        return CheckResult("background_stability", "background", NA,
                           reason="no fixed panel has measurable static background", metrics=metrics)
    return CheckResult("background_stability", "background", PASS, metrics=metrics)


# ---------------------------------------------------------------------------
# reference leak


def check_reference_leak(ctx: Context) -> CheckResult:
    spec = ctx.spec
    if not ctx.references:
        return CheckResult("reference_layout_leak", "reference_leak", NA, reason="no appearance reference image")
    fill_max = spec.threshold("reference_leak_fill_max")
    head = min(ctx.samples, 5)
    metrics: dict[str, Any] = {}
    evidence: list[Evidence] = []
    leaks = []
    scale = ctx.carrier.work_scale
    for ref_index, reference in enumerate(ctx.references):
        work_ref = cv2.resize(reference, (ctx.carrier.frames.shape[2], ctx.carrier.frames.shape[1]),
                              interpolation=cv2.INTER_AREA)
        for entity in spec.entities:
            if "unsupported" not in entity.checks:
                continue
            ref_mask = key_masks(work_ref[None], entity.target_keys)[0]
            car = ctx.mask("carrier", entity)[:head].any(axis=0)
            gen = ctx.mask("generated", entity)[:head]
            for panel in spec.entity_panels(entity):
                rows, cols = ctx.work_slice(panel)
                height, width = rows.stop - rows.start, cols.stop - cols.start
                labels, stats, keep = _components(ref_mask[rows, cols], entity.min_area_frac * height * width)
                car_near = cv2.dilate(car[rows, cols].astype(np.uint8), _disk(6)) > 0
                for k in keep:
                    comp = labels == k
                    if float(car_near[comp].mean()) >= 0.3:
                        continue  # the simulator also has the entity there
                    fill = float(max(g[rows, cols][comp].mean() for g in gen))
                    key = f"ref{ref_index}:{panel.name}:{entity.role}:{k}"
                    metrics[key] = round(fill, 3)
                    if fill >= fill_max:
                        leaks.append(key)
                        x, y, w, h = stats[k, :4]
                        evidence.append(Evidence(panel.name, "reference_layout_copied", 0,
                                                 bbox=ctx.to_layout((x, y, w, h), panel, scale),
                                                 detail=f"{entity.role} fill {fill:.2f}"))
    return CheckResult("reference_layout_leak", "reference_leak", FAIL if leaks else PASS,
                       reason=", ".join(leaks), metrics=metrics, evidence=evidence)
