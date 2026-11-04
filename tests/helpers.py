from __future__ import annotations


from datafarm.core.model import (
    Authority,
    IRKind,
    StageName,
)

NOW = "2026-09-19T08:00:00Z"


def payload_for(kind: IRKind) -> dict:
    if kind is IRKind.GROUNDING:
        return {
            "source": {},
            "transfer_scope": {},
            "camera_clock": {},
            "reference_roles": [],
            "system_identification": {"physical_quantities": []},
            "rights": {},
        }
    if kind is IRKind.TASK:
        return {
            "demand": {},
            "roles": [],
            "stage_graph": {},
            "predicates": {},
            "forbidden_shortcuts": [],
            "skill_evidence": [],
        }
    if kind is IRKind.EPISODE:
        return {
            "setup": {},
            "actions": {},
            "state_trace": {},
            "contacts": {},
            "events": [],
            "frame_map": {},
            "replay": {},
        }
    if kind is IRKind.RENDER:
        return {
            "carrier": {},
            "immutable_invariants": [],
            "allowed_changes": [],
            "frame_map": {},
            "references": [],
            "backend": {},
        }
    if kind is IRKind.ADMISSION:
        return {
            "episode": {},
            "rendered_observation": {},
            "checks": {},
            "verdict": "candidate",
            "cost": {},
            "training_bundle": {},
            "repair_predecessor": None,
        }
    raise AssertionError(kind)


STAGE_KIND = {
    StageName.GROUNDING: IRKind.GROUNDING,
    StageName.TASK: IRKind.TASK,
    StageName.EPISODE: IRKind.EPISODE,
    StageName.RENDER: IRKind.RENDER,
    StageName.ADMISSION: IRKind.ADMISSION,
}

STAGE_AUTHORITY = {
    StageName.GROUNDING: Authority.REFERENCE_AUDITOR,
    StageName.TASK: Authority.TASK_VERIFIER,
    StageName.EPISODE: Authority.SIMULATOR,
    StageName.RENDER: Authority.RENDER_CONTRACT_VERIFIER,
    StageName.ADMISSION: Authority.ADMISSION_ROOT,
}
