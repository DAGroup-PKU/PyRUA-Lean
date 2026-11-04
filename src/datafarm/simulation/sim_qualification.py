"""Simulator qualification: make the simulator support the task's reasonable actions.

A simulator adapter is the signing authority for Episode IR, but its physics can make a reasonable
action impossible: a closed parallel gripper may be unable to hold a hanging cloth, so the real robot's
motion fails for every expert.  Before episodes are generated for a task, the
pipeline therefore runs scripted *skill probes* (canonical, real-robot-like actions for the task's
skills) over candidate *simulator variants* (parameter patches and optional contact models), judges
each probe against declared criteria, and selects the smallest change under which every required
probe passes.  The choice is issued as a certificate that binds the exact variant files and the probe
evidence; ``verify_certificate`` recomputes it independently.

This module holds the decision logic only.  Running probes and extracting metrics from simulator
state is the job of task code.
"""

from __future__ import annotations

from datafarm.common import is_finite_number, sha256_json
import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "datafarm.sim-qualification.certificate/1.0.0"


@dataclass(frozen=True)
class SimulatorVariant:
    """One candidate simulator configuration relative to the task's base configuration."""

    name: str
    material_patch: Mapping[str, Any] = field(default_factory=dict)
    scene_patch: Mapping[str, Any] = field(default_factory=dict)
    runtime_flags: tuple[str, ...] = ()
    note: str = ""

    @property
    def change_magnitude(self) -> int:
        """Number of independent changes; the selector prefers the smallest."""

        return len(self.material_patch) + len(self.scene_patch) + len(self.runtime_flags)


def apply_patch(document: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``document`` with dotted-key values replaced.

    A string value ``"*k"`` multiplies the existing number by ``k``.  A path that does not already
    exist is an error: a qualification must never silently add an unconsumed field.
    """

    out = copy.deepcopy(dict(document))
    for dotted, value in patch.items():
        keys = dotted.split(".")
        node: Any = out
        for key in keys[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise ValueError(f"patch path does not exist: {dotted}")
            node = node[key]
        last = keys[-1]
        if not isinstance(node, dict) or last not in node:
            raise ValueError(f"patch path does not exist: {dotted}")
        if isinstance(value, str) and value.startswith("*"):
            base = node[last]
            if not isinstance(base, (int, float)) or isinstance(base, bool):
                raise ValueError(f"multiplier needs a numeric field: {dotted}")
            value = base * float(value[1:])
        node[last] = value
    return out


@dataclass(frozen=True)
class ProbeOutcome:
    probe: str
    kind: str
    passed: bool
    failures: tuple[str, ...]
    advisories: tuple[str, ...]
    metrics: Mapping[str, Any]


def evaluate_hold_probe(
    probe: str,
    metrics: Mapping[str, Any],
    *,
    max_slip_m: float = 0.03,
    min_lift_m: float = 0.10,
    max_strain_p95: float = 1.15,
    max_strain_p999: float | None = None,
    min_flat_area: float = 0.85,
    max_flat_height_m: float = 0.02,
) -> ProbeOutcome:
    """A grasp-and-hang probe: the closed jaws must carry the hanging cloth.

    Required: the probe reached the hang (``hang_z_max_m`` present), the cloth is lifted at least
    ``min_lift_m``, every gripped patch stays within ``max_slip_m`` of its pad, and the hanging cloth
    is not stretched beyond ``max_strain_p95``.  When ``max_strain_p999`` is set, the 99.9th percentile
    of the spring strain (the few springs next to a gripped patch, where a contact model concentrates
    the load) is required to stay below it too.  The final flatness depends on the scripted lay-down
    and is advisory.
    """

    failures: list[str] = []
    advisories: list[str] = []
    z_max = metrics.get("hang_z_max_m")
    if not is_finite_number(z_max):
        failures.append("hang_not_reached")
    elif z_max < min_lift_m:
        failures.append("cloth_not_lifted")
    slips = metrics.get("held_dz_m") or {}
    if not slips:
        failures.append("grip_slip_unmeasured")
    elif any(not is_finite_number(v) or v < -max_slip_m for v in slips.values()):
        failures.append("grip_slipped")
    strain = metrics.get("hang_strain_p95")
    if not is_finite_number(strain):
        failures.append("hang_strain_unmeasured")
    elif strain > max_strain_p95:
        failures.append("cloth_overstretched")
    if max_strain_p999 is not None:
        local = metrics.get("hang_strain_p999")
        if not is_finite_number(local):
            failures.append("local_strain_unmeasured")
        elif local > max_strain_p999:
            failures.append("cloth_locally_overstretched")
    area, height = metrics.get("final_area_frac"), metrics.get("final_height_mm")
    if not (is_finite_number(area) and area >= min_flat_area and is_finite_number(height) and height / 1000.0 < max_flat_height_m):
        advisories.append("laid_down_not_flat")
    return ProbeOutcome(probe, "hold", not failures, tuple(failures), tuple(advisories), dict(metrics))


def evaluate_fold_probe(
    probe: str,
    metrics: Mapping[str, Any],
    *,
    min_bottom_layer_capture: float = 0.5,
    require_settled: bool = False,
    max_creep_m: float | None = None,
    min_rest_window_s: float = 8.0,
) -> ProbeOutcome:
    """A scripted fold probe judged by the task verifier.

    Required: both folds are in place at the last release (``m1_at_release``).  Optional requirements, used
    when a spec sets them: the folds are still in place in the settled final state (``settled_passed``), and
    a folded cloth left alone stays put: the largest landmark drift between 2 s after the release and the end
    (``creep_mm``, over a window of at least ``min_rest_window_s``) is at most ``max_creep_m``.  A bottom
    layer that is rarely lifted and robot-gate failures are advisory.
    """

    failures: list[str] = []
    advisories: list[str] = []
    if metrics.get("m1_at_release") is not True:
        failures.append("folds_not_in_place_at_release")
    if require_settled:
        settled = metrics.get("settled_passed")
        if settled is None:
            failures.append("settled_unmeasured")
        elif settled is not True:
            failures.append("folds_not_in_place_when_settled")
    if max_creep_m is not None:
        creep, window = metrics.get("creep_mm"), metrics.get("rest_window_s")
        if not (is_finite_number(creep) and is_finite_number(window)):
            failures.append("creep_unmeasured")
        elif window < min_rest_window_s:
            failures.append("rest_window_short")
        elif creep / 1000.0 > max_creep_m:
            failures.append("fold_creeps")
    capture = metrics.get("bottom_layer_capture_frac")
    if not is_finite_number(capture) or capture < min_bottom_layer_capture:
        advisories.append("bottom_layer_rarely_captured")
    if metrics.get("robot_failures"):
        advisories.append("robot_gate_failed")
    return ProbeOutcome(probe, "fold", not failures, tuple(failures), tuple(advisories), dict(metrics))


@dataclass(frozen=True)
class VariantResult:
    variant: SimulatorVariant
    outcomes: tuple[ProbeOutcome, ...]

    def outcome(self, probe: str) -> ProbeOutcome | None:
        return next((o for o in self.outcomes if o.probe == probe), None)

    def passes(self, required: Iterable[str]) -> bool:
        return all((o := self.outcome(p)) is not None and o.passed for p in required)

    @property
    def score(self) -> float:
        """Passed probes count 2, each advisory costs 0.25; used only to break ties."""

        return sum(2.0 * o.passed - 0.25 * len(o.advisories) for o in self.outcomes)


@dataclass(frozen=True)
class QualificationDecision:
    selected: str | None
    ranking: tuple[tuple[str, bool, int, float], ...]   # (variant, passes required, change magnitude, score)
    reason: str


def select_variant(results: Sequence[VariantResult], required_probes: Sequence[str]) -> QualificationDecision:
    """Pick the smallest change under which every required probe passes (ties: higher score, name)."""

    if not required_probes:
        raise ValueError("at least one required probe is needed")
    ranking = sorted(
        ((r.variant.name, r.passes(required_probes), r.variant.change_magnitude, r.score) for r in results),
        key=lambda row: (not row[1], row[2], -row[3], row[0]),
    )
    passing = [row for row in ranking if row[1]]
    if not passing:
        return QualificationDecision(None, tuple(ranking), "no variant passes every required probe")
    best = passing[0]
    return QualificationDecision(best[0], tuple(ranking),
                                 f"smallest change ({best[2]}) passing {len(required_probes)} required probes")


def build_certificate(
    *,
    task_family: str,
    decision: QualificationDecision,
    results: Sequence[VariantResult],
    required_probes: Sequence[str],
    file_digests: Mapping[str, Mapping[str, str]],
    criteria: Mapping[str, Any],
    canonical_digests: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Bind the decision to the evidence and to the exact files of the selected variant.

    ``file_digests`` are SHA-256 digests of the bytes; ``canonical_digests`` (for JSON files) are
    :func:`canonical_json_sha256` digests, so a run can prove it used the same content even when the file
    was re-serialised.  The evidence keeps every probe's metrics.
    """

    if decision.selected is None:
        raise ValueError("cannot certify: " + decision.reason)
    chosen = next(r for r in results if r.variant.name == decision.selected)
    if decision.selected not in file_digests:
        raise ValueError("file digests of the selected variant are required")
    return {
        "schema": SCHEMA,
        "task_family": task_family,
        "selected_variant": {
            "name": chosen.variant.name, "note": chosen.variant.note,
            "material_patch": dict(chosen.variant.material_patch), "scene_patch": dict(chosen.variant.scene_patch),
            "runtime_flags": list(chosen.variant.runtime_flags), "files_sha256": dict(file_digests[decision.selected]),
            "files_canonical_sha256": dict((canonical_digests or {}).get(decision.selected, {})),
        },
        "required_probes": list(required_probes),
        "criteria": dict(criteria),
        "ranking": [list(row) for row in decision.ranking],
        "evidence": {
            r.variant.name: {o.probe: {"passed": o.passed, "failures": list(o.failures),
                                       "advisories": list(o.advisories), "metrics": dict(o.metrics)}
                             for o in r.outcomes}
            for r in results
        },
        "reason": decision.reason,
    }


def verify_certificate(certificate: Mapping[str, Any], results: Sequence[VariantResult]) -> tuple[str, ...]:
    """Independent check: recompute the selection and the evidence from the probe outcomes."""

    failures: list[str] = []
    if certificate.get("schema") != SCHEMA:
        failures.append("schema")
    required = certificate.get("required_probes") or []
    try:
        decision = select_variant(results, required)
    except ValueError:
        return tuple(failures + ["required_probes"])
    selected = (certificate.get("selected_variant") or {}).get("name")
    if decision.selected != selected:
        failures.append("selection_mismatch")
    for r in results:
        claimed = (certificate.get("evidence") or {}).get(r.variant.name, {})
        for o in r.outcomes:
            if (claimed.get(o.probe) or {}).get("passed") != o.passed:
                failures.append(f"evidence_mismatch:{r.variant.name}:{o.probe}")
    if not (certificate.get("selected_variant") or {}).get("files_sha256"):
        failures.append("files_unbound")
    return tuple(failures)


def canonical_json_sha256(document: Any) -> str:
    """SHA-256 of a JSON document that ignores key order and whitespace."""

    return sha256_json(document, allow_nan=True)


def verify_run_binding(certificate: Mapping[str, Any], used: Mapping[str, str]) -> tuple[str, ...]:
    """Check that a run used exactly the certified simulator files.

    ``used`` maps each certified file name (``material.json``, ``scene.json`` and the runtime-flag files) to
    the digest of the file the run actually used: :func:`canonical_json_sha256` for ``.json`` files, the
    SHA-256 of the bytes otherwise.  Every certified file must be reported and match; a reported file that
    the certificate does not list is an error too.
    """

    selected = certificate.get("selected_variant") or {}
    raw = selected.get("files_sha256") or {}
    canonical = selected.get("files_canonical_sha256") or {}
    if not raw:
        return ("files_unbound",)
    problems: list[str] = []
    for name, digest in raw.items():
        expected = canonical.get(name) if name.endswith(".json") else digest
        if expected is None:
            problems.append(f"canonical_digest_missing:{name}")
        elif name not in used:
            problems.append(f"not_reported:{name}")
        elif used[name] != expected:
            problems.append(f"mismatch:{name}")
    problems.extend(f"not_certified:{name}" for name in used if name not in raw)
    return tuple(problems)

