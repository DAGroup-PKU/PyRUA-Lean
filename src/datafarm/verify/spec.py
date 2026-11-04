"""Declarative render-verifier specification.

The fixed part of the Render gate lives in this package: the check families,
their decision logic, persistence rules, fail-closed behaviour and the report
format.  What changes from task to task is only *data*: panel layout, which
entities the Task IR names and how they look, which events matter, which chunk
seams exist, and which real-world footage defines the target domain.  An agent
authoring a new task writes a spec file; it does not write a new verifier.

Every threshold carries a provenance string so a receipt can say whether a value
came from the package default, from a counterfactual calibration run, or from an
explicit task-level override.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

SPEC_SCHEMA = "datafarm.render-verifier-spec/1.0.0"

# Check families are fixed.  A spec may tune or disable a family only through an
# explicit, reasoned waiver; it may never invent a family or rename one.
FAMILIES = (
    "transport",
    "structure",
    "entity",
    "event",
    "temporal",
    "continuity",
    "background",
    "realism",
    "reference_leak",
)

REALISM_MODES = ("sim2real", "ood")
# Nuisance axes an OOD cohort may perturb on purpose.  Anything not listed stays
# bound to the real world.
OOD_AXES = ("lighting", "background", "table_material", "distractors", "camera_exposure")


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def parse(cls, value: Sequence[int]) -> "Box":
        x, y, w, h = (int(item) for item in value)
        if w <= 0 or h <= 0:
            raise ValueError(f"panel box must have positive size: {value}")
        return cls(x, y, w, h)


@dataclass(frozen=True)
class PanelSpec:
    name: str
    box: Box
    # "fixed" cameras see a static background; "moving" cameras (wrists) do not,
    # so background-stability is not applicable to them.
    motion: str = "fixed"
    # Simulator camera feeding this panel and its native image size, used to
    # project object poses from the Episode IR.
    camera: str | None = None
    native_size: tuple[int, int] = (640, 360)


@dataclass(frozen=True)
class ColorKey:
    """HSV range in OpenCV units (H 0-179, S/V 0-255)."""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    @classmethod
    def parse(cls, value: Mapping[str, Any]) -> "ColorKey":
        lower = tuple(int(v) for v in value["lower"])
        upper = tuple(int(v) for v in value["upper"])
        if len(lower) != 3 or len(upper) != 3:
            raise ValueError("color key bounds must have three channels")
        return cls(lower, upper)  # type: ignore[arg-type]


@dataclass(frozen=True)
class EntitySpec:
    """One Task IR role as it must appear in every rendered panel.

    ``carrier_keys`` segment the entity in the simulator carrier, which is the
    geometric authority.  ``target_keys`` segment the entity as it should look in
    the target domain (real robot footage); they are used on the generated video
    and on the real reference, never on the carrier.
    """

    role: str
    carrier_keys: tuple[ColorKey, ...]
    target_keys: tuple[ColorKey, ...]
    # Sub-checks: missing (carrier shows it, render does not), unsupported (render
    # shows it where the carrier has none), extent (projected size/shape drift).
    checks: tuple[str, ...] = ("missing", "unsupported", "extent")
    min_area_frac: float = 0.0015
    panels: tuple[str, ...] = ()
    # Entities that must stay bound to the real world even in OOD cohorts.
    invariant: bool = True
    # Conservative keys for segmenting the entity in real footage (realism
    # regions).  Defaults to target_keys.
    real_keys: tuple[ColorKey, ...] = ()
    # Optional broader appearance key used only to test whether an expected
    # carrier instance remains visible. New-instance and extent checks retain
    # target_keys, so a low-saturation real object does not turn background
    # highlights into hallucinated entities.
    presence_keys: tuple[ColorKey, ...] = ()
    # Learn this video's own entity appearance (CIELAB) inside carrier-authorised
    # pixels and keep only target-key pixels close to it.  Presence is then
    # judged independently of how well the colour matches reality, which is the
    # realism family's job.
    adaptive: bool = True
    # Optional target-colour instance core. A thick object survives opening;
    # thin attached cast-colour trails do not enlarge it or glue two objects
    # together. Applied to unsupported and extent, never to presence.
    instance_open_px_work: int = 0
    # In specified panels, compare candidate-instance edge density with this
    # video's carrier-backed instance appearance. This can reject broad flat
    # colour-keyed backgrounds without lowering extra-instance sensitivity in
    # other panels. Zero/absent keeps the original colour-only behaviour.
    unsupported_edge_reference_fraction: Mapping[str, float] = field(default_factory=dict)

    def active_checks(self, panel: str) -> set[str]:
        """Sub-checks for a panel; ``kind:panelA|panelB`` restricts a kind."""

        out = set()
        for item in self.checks:
            kind, _, where = item.partition(":")
            if not where or panel in where.split("|"):
                out.add(kind)
        return out

    # Episode-state object names (glob) and a bounding primitive, e.g.
    # {"type": "cylinder", "radius": 0.045, "height": 0.09}.  When given, the
    # projected footprint is the authority for whether a rendered blob is backed
    # by a simulated object.
    state_objects: str | None = None
    geometry: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EventSpec:
    event_id: str
    frame: int
    roles: tuple[str, ...]
    panels: tuple[str, ...] = ()
    kind: str = "contact"


@dataclass(frozen=True)
class RealismSpec:
    mode: str = "sim2real"
    perturbed_axes: tuple[str, ...] = ()
    real_reference: str | None = None
    real_layout: str | None = None
    real_frames: int = 24


@dataclass(frozen=True)
class Threshold:
    value: float
    provenance: str = "default-v1"


DEFAULT_THRESHOLDS: dict[str, float] = {
    # transport
    "fps_tolerance": 0.25,
    "aspect_tolerance": 0.002,
    "black_luma_max": 8.0,
    # structure: projected-geometry cosine per panel and sampled frame
    "structure_cosine_min": 0.30,
    "structure_bad_fraction_max": 0.10,
    "structure_shift_frac_max": 0.03,
    # A task may require evidence of digital near-zero padding before an edge
    # stripe is called a crop error. Zero preserves the original dark-band rule.
    "black_border_near_zero_fraction_min": 0.0,
    # entity
    "entity_recall_min": 0.50,
    "entity_support_min": 0.30,
    # bbox growth without a footprint; with a footprint: 1 + outside-area / area
    "entity_extent_ratio_max": 1.30,
    "entity_tolerance_frac": 0.15,
    "entity_key_leak_max": 0.02,
    # event
    "event_lag_max": 6.0,
    "event_corr_min": 0.30,
    "event_local_motion_min": 0.0,
    # temporal (per panel, full frame rate)
    # 10/255 (~4%): side-by-side review found 6-8 level global steps invisible;
    # the real camera's own step statistic can only raise this.
    "temporal_luma_step_max": 10.0,
    "temporal_jumps_max": 3.0,
    "temporal_freezes_max": 2.0,
    # Disabled by default. A task may allow a large photometric step only
    # when a matching, sustained source-panel occlusion explains its location.
    "temporal_source_motion_share_min": 0.0,
    # continuity
    "seam_jump_multiple_max": 5.0,
    "seam_structural_min": 4.0,
    # background
    # Global exposure drift: a small human-labelled sample found no visible
    # exposure problem up to ~38 levels; kept only as a gross guard.
    "background_mean_drift_max": 60.0,
    "background_local_drift_p95_max": 12.0,
    "background_shadow_margin_frac": 0.06,
    "background_shadow_reach_frac": 0.20,
    "background_min_static_fraction": 0.03,
    "background_blob_level": 15.0,
    "background_blob_area_frac_max": 0.006,
    # realism
    # Defaults; recalibrate per deployment.
    "realism_feature_score_max": 6.0,
    "realism_feature_advisory": 3.0,
    # reference leak
    "reference_leak_fill_max": 0.50,
    # shared persistence: a defect must last this many consecutive sampled frames
    "persist_samples": 3.0,
}


@dataclass(frozen=True)
class RenderVerifierSpec:
    spec_id: str
    task_family: str
    layout_size: tuple[int, int]
    panels: tuple[PanelSpec, ...]
    entities: tuple[EntitySpec, ...]
    fps: float
    events: tuple[EventSpec, ...] = ()
    seams: tuple[int, ...] = ()
    chunked: bool = False
    realism: RealismSpec = field(default_factory=RealismSpec)
    reference_images: tuple[str, ...] = ()
    thresholds: Mapping[str, Threshold] = field(default_factory=dict)
    waivers: Mapping[str, str] = field(default_factory=dict)
    work_scale: float = 0.5
    stride: int = 2
    expected_frames: int | None = None
    task_ir_roles: tuple[str, ...] = ()

    def threshold(self, name: str) -> float:
        if name in self.thresholds:
            return float(self.thresholds[name].value)
        return float(DEFAULT_THRESHOLDS[name])

    def threshold_provenance(self, name: str) -> str:
        if name in self.thresholds:
            return self.thresholds[name].provenance
        return "default-v1"

    def panel(self, name: str) -> PanelSpec:
        for panel in self.panels:
            if panel.name == name:
                return panel
        raise KeyError(name)

    def entity_panels(self, entity: EntitySpec) -> tuple[PanelSpec, ...]:
        if not entity.panels:
            return self.panels
        return tuple(self.panel(name) for name in entity.panels)

    def with_episode(
        self,
        *,
        events: Sequence[EventSpec] | None = None,
        seams: Sequence[int] | None = None,
        expected_frames: int | None = None,
        chunked: bool | None = None,
    ) -> "RenderVerifierSpec":
        return replace(
            self,
            events=tuple(events) if events is not None else self.events,
            seams=tuple(int(s) for s in seams) if seams is not None else self.seams,
            expected_frames=expected_frames if expected_frames is not None else self.expected_frames,
            chunked=self.chunked if chunked is None else chunked,
        )


def _keys(value: Any) -> tuple[ColorKey, ...]:
    if isinstance(value, Mapping):
        value = [value]
    return tuple(ColorKey.parse(item) for item in value)


def spec_from_dict(data: Mapping[str, Any]) -> RenderVerifierSpec:
    if data.get("schema_version") != SPEC_SCHEMA:
        raise ValueError(f"unsupported spec schema {data.get('schema_version')!r}")
    panels = tuple(
        PanelSpec(str(p["name"]), Box.parse(p["box"]), str(p.get("motion", "fixed")),
                  p.get("camera"), tuple(int(v) for v in p.get("native_size", (640, 360))))
        for p in data["panels"]
    )
    entities = tuple(
        EntitySpec(
            role=str(e["role"]),
            carrier_keys=_keys(e["carrier_keys"]),
            target_keys=_keys(e["target_keys"]),
            checks=tuple(e.get("checks", ("missing", "unsupported", "extent"))),
            min_area_frac=float(e.get("min_area_frac", 0.0015)),
            panels=tuple(e.get("panels", ())),
            invariant=bool(e.get("invariant", True)),
            real_keys=_keys(e["real_keys"]) if e.get("real_keys") else (),
            presence_keys=_keys(e["presence_keys"]) if e.get("presence_keys") else (),
            adaptive=bool(e.get("adaptive", True)),
            instance_open_px_work=int(e.get("instance_open_px_work", 0)),
            unsupported_edge_reference_fraction={str(k): float(v) for k, v in
                                                 e.get("unsupported_edge_reference_fraction", {}).items()},
            state_objects=e.get("state_objects"),
            geometry=e.get("geometry"),
        )
        for e in data.get("entities", ())
    )
    realism_data = data.get("realism", {})
    realism = RealismSpec(
        mode=str(realism_data.get("mode", "sim2real")),
        perturbed_axes=tuple(realism_data.get("perturbed_axes", ())),
        real_reference=realism_data.get("real_reference"),
        real_layout=realism_data.get("real_layout"),
        real_frames=int(realism_data.get("real_frames", 24)),
    )
    thresholds = {
        name: Threshold(float(item["value"]), str(item.get("provenance", "task-override")))
        if isinstance(item, Mapping)
        else Threshold(float(item), "task-override")
        for name, item in data.get("thresholds", {}).items()
    }
    events = tuple(
        EventSpec(str(e["event_id"]), int(e["frame"]), tuple(e.get("roles", ())),
                  tuple(e.get("panels", ())), str(e.get("kind", "contact")))
        for e in data.get("events", ())
    )
    spec = RenderVerifierSpec(
        spec_id=str(data["spec_id"]),
        task_family=str(data["task_family"]),
        layout_size=(int(data["layout_size"][0]), int(data["layout_size"][1])),
        panels=panels,
        entities=entities,
        fps=float(data["fps"]),
        events=events,
        seams=tuple(int(s) for s in data.get("seams", ())),
        chunked=bool(data.get("chunked", False)),
        realism=realism,
        reference_images=tuple(data.get("reference_images", ())),
        thresholds=thresholds,
        waivers=dict(data.get("waivers", {})),
        work_scale=float(data.get("work_scale", 0.5)),
        stride=int(data.get("stride", 2)),
        expected_frames=data.get("expected_frames"),
        task_ir_roles=tuple(data.get("task_ir_roles", ())),
    )
    return spec


def load_spec(path: str | Path) -> RenderVerifierSpec:
    return spec_from_dict(json.loads(Path(path).read_text()))


def validate_spec(spec: RenderVerifierSpec) -> list[str]:
    """Return violations of the fixed authoring principles (empty list = valid).

    These are the rules an agent-authored spec must satisfy before any sample is
    judged with it.  They are deliberately about coverage and fail-closed
    behaviour, not about task semantics.
    """

    problems: list[str] = []
    names = [p.name for p in spec.panels]
    if len(set(names)) != len(names):
        problems.append("P0: panel names must be unique")
    width, height = spec.layout_size
    for panel in spec.panels:
        b = panel.box
        if b.x < 0 or b.y < 0 or b.x + b.w > width or b.y + b.h > height:
            problems.append(f"P0: panel {panel.name} lies outside the layout")
        if panel.motion not in ("fixed", "moving"):
            problems.append(f"P0: panel {panel.name} motion must be fixed|moving")
    roles = {e.role for e in spec.entities}
    # P1: every Task IR role that is visible must be an entity with a presence check.
    for role in spec.task_ir_roles:
        if role not in roles:
            problems.append(f"P1: Task IR role {role!r} has no entity check")
    for entity in spec.entities:
        if entity.instance_open_px_work not in (0,) and (
            entity.instance_open_px_work < 3 or entity.instance_open_px_work % 2 == 0
        ):
            problems.append(f"P1: entity {entity.role} instance_open_px_work must be 0 or odd >= 3")
        for panel, fraction in entity.unsupported_edge_reference_fraction.items():
            if panel not in names or not 0 < fraction <= 1:
                problems.append(f"P1: entity {entity.role} unsupported edge reference for {panel} must name a panel and be in (0, 1]")
        if not entity.carrier_keys or not entity.target_keys:
            problems.append(f"P1: entity {entity.role} needs carrier and target keys")
        if not any(c.partition(":")[0] == "missing" and not c.partition(":")[2] for c in entity.checks):
            problems.append(f"P1: entity {entity.role} must at least check 'missing'")
        for panel in entity.panels:
            if panel not in names:
                problems.append(f"P1: entity {entity.role} names unknown panel {panel}")
    # P2: every event must reference known roles.
    for event in spec.events:
        for role in event.roles:
            if role not in roles:
                problems.append(f"P2: event {event.event_id} names unknown role {role}")
    # P3: multi-chunk renders must declare their seams.
    if spec.chunked and not spec.seams:
        problems.append("P3: chunked render declares no seams; continuity cannot be judged")
    # P4: realism target must be explicit.
    if spec.realism.mode not in REALISM_MODES:
        problems.append(f"P4: realism mode must be one of {REALISM_MODES}")
    for axis in spec.realism.perturbed_axes:
        if axis not in OOD_AXES:
            problems.append(f"P4: unknown OOD axis {axis!r}")
    if spec.realism.mode == "sim2real" and spec.realism.perturbed_axes:
        problems.append("P4: sim2real cohorts may not declare perturbed axes")
    if spec.realism.mode == "ood" and not spec.realism.perturbed_axes:
        problems.append("P4: OOD cohorts must declare which axes are perturbed")
    # P5: waivers must name a known family and give a reason.
    for family, reason in spec.waivers.items():
        if family not in FAMILIES:
            problems.append(f"P5: waiver for unknown family {family!r}")
        if not str(reason).strip():
            problems.append(f"P5: waiver for {family} has no reason")
    for name in spec.thresholds:
        if name not in DEFAULT_THRESHOLDS:
            problems.append(f"P5: unknown threshold {name!r}")
    return problems
