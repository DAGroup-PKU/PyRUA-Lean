"""Run every fixed check family and aggregate a fail-closed verdict."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from datafarm.verify import checks as C
from datafarm.verify.media import Clip, load_clip, load_reference_clip
from datafarm.verify.realism import check_realism
from datafarm.verify.spec import DEFAULT_THRESHOLDS, FAMILIES, RenderVerifierSpec, validate_spec

VERIFIER_ID = "datafarm-render-verifier-v2"
VERIFIER_VERSION = "2.0.0"


def run_checks(spec: RenderVerifierSpec, carrier: Clip, generated: Clip, real: Clip | None = None,
               references: Sequence[np.ndarray] = (), families: Sequence[str] | None = None,
               state: dict[str, Any] | None = None) -> dict[str, Any]:
    problems = validate_spec(spec)
    if problems:
        return {"verifier": VERIFIER_ID, "version": VERIFIER_VERSION, "spec_id": spec.spec_id,
                "verdict": C.ABSTAIN, "reason": "spec violates authoring principles",
                "spec_problems": problems, "checks": []}
    ctx = C.Context(spec, carrier, generated, real, list(references), state)
    wanted = set(families or FAMILIES)
    results: list[C.CheckResult] = []
    timings: dict[str, float] = {}

    def run(family: str, fn, *args):
        if family not in wanted:
            return
        start = time.time()
        if family in spec.waivers:
            results.append(C.CheckResult(fn.__name__.replace("check_", ""), family, C.NA,
                                         reason=f"waived: {spec.waivers[family]}"))
        else:
            results.append(fn(*args))
        timings[results[-1].name] = round(time.time() - start, 2)

    run("transport", C.check_transport, ctx)
    if ctx.samples > 0:
        run("structure", C.check_structure, ctx)
        if "entity" in wanted:
            for entity in spec.entities:
                run("entity", C.check_entity, ctx, entity)
                if any(c.partition(":")[0] == "semantic_unsupported" for c in entity.checks):
                    from datafarm.verify.semantic import check_semantic_unsupported
                    run("entity", check_semantic_unsupported, ctx, entity)
        run("event", C.check_events, ctx)
        run("temporal", C.check_temporal, ctx)
        run("continuity", C.check_continuity, ctx)
        run("background", C.check_background, ctx)
        run("realism", check_realism, ctx)
        run("reference_leak", C.check_reference_leak, ctx)
    hard = [r for r in results if r.hard]
    if any(r.verdict == C.FAIL for r in hard):
        verdict = C.FAIL
    elif any(r.verdict == C.ABSTAIN for r in hard):
        verdict = C.ABSTAIN
    else:
        verdict = C.PASS
    return {
        "verifier": VERIFIER_ID,
        "version": VERIFIER_VERSION,
        "spec_id": spec.spec_id,
        "task_family": spec.task_family,
        "realism_mode": spec.realism.mode,
        "perturbed_axes": list(spec.realism.perturbed_axes),
        "verdict": verdict,
        "failed": [r.name for r in hard if r.verdict == C.FAIL],
        "abstained": [r.name for r in hard if r.verdict == C.ABSTAIN],
        "thresholds": {name: {"value": spec.threshold(name), "provenance": spec.threshold_provenance(name)}
                       for name in sorted(DEFAULT_THRESHOLDS)},
        "inputs": {"carrier": carrier.source, "carrier_sha256": carrier.sha256,
                   "generated": generated.source, "generated_sha256": generated.sha256,
                   "real": real.source if real else None, "real_sha256": real.sha256 if real else None},
        "timings_s": timings,
        "checks": [r.to_dict() for r in results],
    }


def verify_paths(spec: RenderVerifierSpec, carrier: str | Path, generated: str | Path,
                 real: str | Path | None = None, references: Sequence[str | Path] = (),
                 families: Sequence[str] | None = None, state: dict[str, Any] | None = None) -> dict[str, Any]:
    kwargs = dict(layout_size=spec.layout_size, work_scale=spec.work_scale, stride=spec.stride)
    carrier_clip = load_clip(carrier, **kwargs)
    generated_clip = load_clip(generated, **kwargs)
    real_clip = None
    if real:
        real_clip = load_reference_clip(real, layout_size=spec.layout_size, work_scale=spec.work_scale)
    refs = []
    for path in references:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read reference image {path}")
        refs.append(cv2.resize(image, spec.layout_size, interpolation=cv2.INTER_AREA))
    return run_checks(spec, carrier_clip, generated_clip, real_clip, refs, families, state)
