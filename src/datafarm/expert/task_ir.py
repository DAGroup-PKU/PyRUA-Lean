"""Stage-graph profile of the Task IR (``datafarm.task-ir`` payload, profile ``stage-graph/1``).

The v1 Task IR schema only requires six payload keys. This profile gives them typed content so that the
same document is (a) the expert's skeleton, (b) the runtime monitor list (stage markers and predicate
checks that feed the rollout digest), and (c) the map from terminal predicates to judge criteria:

    demand               str
    roles                {role: {"binding": str, "kind": str}}
    stage_graph          {"stages": [stage...], "sync": [{"wait_for": id, "before": id, "rule": str}]}
        stage            {"id", "skill", "arm" (left|right|both|either), "role", "frame",
                          "exit": [predicate ids], "guard": [predicate ids], "recovery": str|[str]}
    predicates           {id: {"kind": exit|guard|transient|terminal, "expr": str,
                               "tolerance": number|str, "judge_criterion": str (terminal/transient only)}}
    forbidden_shortcuts  [str]
    skill_evidence       [{"skill": str, "evidence": str}]
    strategy_space       optional [{"param": str, "values": [..], "reason": str}]

``validate_stage_graph`` returns human-readable problems (empty list = valid). It checks structure and
cross-references only; whether predicates are physically adequate is decided by rollouts and the judge.
"""
from __future__ import annotations

from typing import Any, Mapping

PROFILE = "stage-graph/1"
ARMS = {"left", "right", "both", "either"}
PRED_KINDS = {"exit", "guard", "transient", "terminal"}
REQUIRED = ("demand", "roles", "stage_graph", "predicates", "forbidden_shortcuts", "skill_evidence")


def _is_num_or_str(v: Any) -> bool:
    return isinstance(v, (int, float, str)) and not isinstance(v, bool)


def validate_stage_graph(payload: Mapping[str, Any], *, judge_criteria: list[str] | None = None) -> list[str]:
    """Validate a Task IR payload against the stage-graph profile.

    judge_criteria: optional list of criterion ids from the frozen judge; when given, every criterion must be
    covered by at least one terminal/transient predicate and no predicate may cite an unknown criterion.
    """
    p: list[str] = []
    if not isinstance(payload, Mapping):
        return ["payload must be an object"]
    for k in REQUIRED:
        if k not in payload:
            p.append(f"missing required key '{k}'")
    if p:
        return p
    if not isinstance(payload["demand"], str) or not payload["demand"].strip():
        p.append("demand must be a non-empty string")
    roles = payload["roles"]
    if not isinstance(roles, Mapping) or not roles:
        p.append("roles must be a non-empty object")
        roles = {}
    for name, r in roles.items():
        if not isinstance(r, Mapping) or not isinstance(r.get("binding"), str) or not isinstance(r.get("kind"), str):
            p.append(f"role '{name}' needs string 'binding' and 'kind'")
    preds = payload["predicates"]
    if not isinstance(preds, Mapping) or not preds:
        p.append("predicates must be a non-empty object")
        preds = {}
    for pid, pr in preds.items():
        if not isinstance(pr, Mapping):
            p.append(f"predicate '{pid}' must be an object")
            continue
        if pr.get("kind") not in PRED_KINDS:
            p.append(f"predicate '{pid}': kind must be one of {sorted(PRED_KINDS)}")
        if not isinstance(pr.get("expr"), str) or not pr["expr"].strip():
            p.append(f"predicate '{pid}': expr must state what is measured from live state")
        if "tolerance" not in pr or not _is_num_or_str(pr.get("tolerance")):
            p.append(f"predicate '{pid}': explicit numeric tolerance (or a stated threshold) is required")
        if pr.get("kind") in ("terminal", "transient"):
            jc = pr.get("judge_criterion")
            if not isinstance(jc, str) or not jc:
                p.append(f"predicate '{pid}': terminal/transient predicates must name the judge criterion they cover")
            elif judge_criteria is not None and jc not in judge_criteria:
                p.append(f"predicate '{pid}': unknown judge criterion '{jc}'")
    sg = payload["stage_graph"]
    stages = sg.get("stages") if isinstance(sg, Mapping) else None
    if not isinstance(stages, list) or not stages:
        p.append("stage_graph.stages must be a non-empty list")
        stages = []
    ids: list[str] = []
    for i, st in enumerate(stages):
        if not isinstance(st, Mapping):
            p.append(f"stage #{i} must be an object")
            continue
        sid = st.get("id")
        if not isinstance(sid, str) or not sid:
            p.append(f"stage #{i}: id required")
            continue
        if sid in ids:
            p.append(f"stage '{sid}': duplicate id")
        ids.append(sid)
        if not isinstance(st.get("skill"), str) or not st["skill"]:
            p.append(f"stage '{sid}': skill required")
        if st.get("arm") not in ARMS:
            p.append(f"stage '{sid}': arm must be one of {sorted(ARMS)}")
        if st.get("role") is not None and st.get("role") not in roles:
            p.append(f"stage '{sid}': unknown role '{st.get('role')}'")
        exits = st.get("exit") or []
        if not exits:
            p.append(f"stage '{sid}': at least one exit predicate is required")
        for key in ("exit", "guard"):
            for ref in st.get(key) or []:
                if ref not in preds:
                    p.append(f"stage '{sid}': {key} references unknown predicate '{ref}'")
        if not st.get("recovery"):
            p.append(f"stage '{sid}': recovery (what to do when the guard/exit fails) is required")
    for s in (sg.get("sync") or []) if isinstance(sg, Mapping) else []:
        if not isinstance(s, Mapping) or s.get("wait_for") not in ids or s.get("before") not in ids:
            p.append(f"sync point {s!r} must reference two known stage ids")
        elif ids.index(s["wait_for"]) >= ids.index(s["before"]):
            p.append(f"sync point {s['wait_for']}->{s['before']} contradicts stage order")
    terminal = [pid for pid, pr in preds.items() if isinstance(pr, Mapping) and pr.get("kind") in ("terminal", "transient")]
    if not terminal:
        p.append("at least one terminal or transient predicate (mapped to the judge) is required")
    if judge_criteria is not None:
        covered = {preds[t].get("judge_criterion") for t in terminal}
        for c in judge_criteria:
            if c not in covered:
                p.append(f"judge criterion '{c}' is not covered by any terminal/transient predicate")
    if not isinstance(payload["forbidden_shortcuts"], list) or not payload["forbidden_shortcuts"]:
        p.append("forbidden_shortcuts must list at least one shortcut (e.g. state writes, judge access)")
    ev = payload["skill_evidence"]
    if not isinstance(ev, list):
        p.append("skill_evidence must be a list")
    else:
        for e in ev:
            if not isinstance(e, Mapping) or not e.get("skill") or not e.get("evidence"):
                p.append(f"skill_evidence entry {e!r} needs 'skill' and 'evidence'")
    for s in payload.get("strategy_space") or []:
        if not isinstance(s, Mapping) or not s.get("param") or not isinstance(s.get("values"), list) or len(s["values"]) < 2:
            p.append(f"strategy_space entry {s!r} needs 'param' and >= 2 'values'")
    return p

