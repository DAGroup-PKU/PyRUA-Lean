"""The digest of a run directory and the gate between two libraries."""

from __future__ import annotations

import json

import pytest

from pyrualean.evolve.digest import digest, render
from pyrualean.evolve.gate import IncompleteComparison, compare


def _run(
    root,
    *,
    task="plug_in_charger",
    layout=0,
    eval_seed=1,
    success=False,
    stages=None,
    calls=(),
    cells=1,
):
    root.mkdir(parents=True)
    verdict = {"success": success, "eval_seed": eval_seed, "step": 120, "step_limit": 400}
    if stages is not None:
        verdict["stages_passed"], verdict["stages_total"] = stages
    result = {
        "task": task,
        "seed": layout,
        "environment_success": success,
        "status": "completed",
        "verdict": verdict,
        "library": {"sha256": "abc"},
        "generation": {"runtime": "openai", "total_tokens": 1000},
    }
    (root / "result.json").write_text(json.dumps(result))
    (root / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    (root / "workspace").mkdir()
    for i in range(1, cells + 1):
        (root / "workspace" / f"cell_{i:03d}.py").write_text("pass")
    return root


CALLS = [
    {"index": 0, "tool": "locate", "ok": True, "stateful": False, "summary": {"found": True}},
    {"index": 1, "tool": "move_to", "ok": True, "stateful": True, "env_steps": 40,
     "summary": {"reached": False, "dist_to_target_m": 0.031}},
    {"index": 2, "tool": "set_gripper", "ok": True, "stateful": True, "env_steps": 10,
     "summary": {"holding": False, "width_m": 0.0}},
    {"index": 3, "tool": "locate", "ok": True, "stateful": False, "summary": {"found": False}},
    {"index": 4, "tool": "move_to", "ok": False, "stateful": True, "env_steps": 5,
     "error": "ToolError: out of reach", "summary": {}},
    {"index": 5, "tool": "move_to", "ok": True, "stateful": True, "env_steps": 30,
     "summary": {"reached": True}},
]  # fmt: skip


def test_digest_flags_the_calls_that_did_not_do_what_was_asked(tmp_path):
    facts = digest(_run(tmp_path / "r", stages=(1, 4), calls=CALLS, cells=3))
    assert facts["success"] is False and (facts["stages_passed"], facts["stages_total"]) == (1, 4)
    assert [f["index"] for f in facts["failed_calls"]] == [1, 2, 3, 4]
    whys = [f["why"] for f in facts["failed_calls"]]
    assert whys[0] == "not reached (0.031 m off)" and whys[1] == "holds nothing"
    assert whys[2] == "not found" and whys[3].startswith("error: ToolError")
    assert facts["steps_by_tool"] == {"move_to": 75, "set_gripper": 10}
    assert facts["cells"] == ["cell_001.py", "cell_002.py", "cell_003.py"]
    text = render(facts)
    assert "stages passed: 1/4" in text and "#1 move_to: not reached" in text


def _client_result(root, name, score, success=False):
    trial = root / "bench" / "eval_result" / "RoboDojo" / "plug_in_charger" / name
    trial.mkdir(parents=True)
    detail = {"layout_id": 0, "success": success, "score": score}
    (trial / "_result.json").write_text(json.dumps({"score": score * 100, "details": {"0": detail}}))


def test_digest_reads_robodojo_own_score_from_its_result_file(tmp_path):
    root = _run(tmp_path / "r", stages=(0, 1))
    assert digest(root)["score"] is None
    _client_result(root, "trial-a", 0.4)
    facts = digest(root)
    assert facts["score"] == 0.4
    assert "score: 0.4" in render(facts)
    _client_result(root, "trial-b", 0.15)
    with pytest.raises(ValueError, match="2 RoboDojo results"):
        digest(root)


def test_blocking_stage_folds_repeated_checks(tmp_path):
    facts = digest(_run(tmp_path / "r", stages=(0, 1)))
    pair = "is_pointA_in_B_functional_bbox+is_A_point_above_B_point_by_z_range"
    facts["blocking_stage"] = [
        {"check": "+".join([pair] * 4), "labels": {}, "holds": True},
        {"check": "+".join([pair] * 4), "labels": {}, "holds": True},
        {"check": "is_axis_up+is_axis_up", "labels": {}, "holds": True},
        {"check": "is_joint_position_above_ratio", "labels": {"label": "toaster"}, "holds": False},
        {"check": "a+b+a", "labels": {}, "holds": None},
    ]
    line = next(x for x in render(facts).splitlines() if x.startswith("blocking stage"))
    assert line == (
        f"blocking stage: ({pair})*4: holds (x2); is_axis_up*2: holds; "
        "is_joint_position_above_ratio(toaster): FAILS; a+b+a: not evaluated"
    )


def test_truncated_is_the_end_of_the_episode_not_a_failed_call(tmp_path):
    ended = [{"index": 0, "tool": "view_env_state", "ok": True, "stateful": False,
              "summary": {"truncated": True, "step": 88}}]  # fmt: skip
    won = digest(_run(tmp_path / "won", success=True, calls=ended))
    assert won["failed_calls"] == [] and won["ended_at_call"] == 0
    assert "ended the episode (truncated) by call #0: all of the task's checks" in render(won)
    lost = digest(_run(tmp_path / "lost", calls=ended))
    assert "failed with budget left" in render(lost)
    assert digest(_run(tmp_path / "running", calls=CALLS))["ended_at_call"] is None


def test_a_claimed_success_the_benchmark_failed_is_an_overclaim(tmp_path):
    def run(name, success, finish):
        root = _run(tmp_path / name, success=success)
        result = json.loads((root / "result.json").read_text())
        result["finish"] = finish
        (root / "result.json").write_text(json.dumps(result))
        return digest(root)

    claimed = {"status": "success", "summary": "two slices in, lever down"}
    over = run("over", False, claimed)
    assert over["claim"] == "overclaim"
    assert "OVERCLAIM: the policy finished with success" in render(over)
    assert "OVERCLAIM" in render(over, scoring=False)  # success is shown either way
    assert run("agreed", True, claimed)["claim"] == "agreed"
    assert run("failed", False, {"status": "failure", "summary": "x"})["claim"] == "agreed"
    quiet = run("quiet", True, {"status": "stuck", "summary": "x"})
    assert quiet["claim"] == "unclaimed_success" and "unclaimed success" in render(quiet)
    # a success that ended the episode before any finish: nothing to compare
    ended = run("ended", True, None)
    assert ended["claim"] is None and "OVERCLAIM" not in render(ended)
    # a replay's finish is the replay runtime's, not a claim
    root = _run(tmp_path / "replay", success=True)
    result = json.loads((root / "result.json").read_text())
    result["finish"] = {"status": "failure", "summary": "replayed 4 of 4 cells"}
    result["generation"]["runtime"] = "replay"
    (root / "result.json").write_text(json.dumps(result))
    assert digest(root)["claim"] is None and "unclaimed" not in render(digest(root))


def test_a_run_without_a_verdict_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        digest(tmp_path / "empty")


def _side(tmp_path, name, outcomes):
    return [
        digest(_run(tmp_path / name / f"{task}_{layout}", task=task, layout=layout, success=s,
                    stages=st))
        for (task, layout), (s, st) in outcomes.items()
    ]  # fmt: skip


def test_more_stages_passed_wins_when_successes_tie(tmp_path):
    parent = _side(tmp_path, "p", {("plug", 0): (False, (0, 4)), ("plug", 1): (False, (1, 4))})
    child = _side(tmp_path, "c", {("plug", 0): (False, (2, 4)), ("plug", 1): (False, (1, 4))})
    verdict = compare(parent, child)
    assert verdict["accepted"] is True
    assert verdict["per_task"]["plug"]["decided_by"] == "mean stages passed"


def test_a_tie_keeps_the_parent_and_a_regression_on_any_task_rejects(tmp_path):
    same = {("plug", 0): (False, (1, 4)), ("mugs", 0): (False, (2, 5))}
    assert compare(_side(tmp_path, "p", same), _side(tmp_path, "c", same))["accepted"] is False
    parent = _side(tmp_path, "p2", {("plug", 0): (False, (1, 4)), ("mugs", 0): (True, (5, 5))})
    child = _side(tmp_path, "c2", {("plug", 0): (True, (4, 4)), ("mugs", 0): (False, (3, 5))})
    verdict = compare(parent, child)
    assert verdict["accepted"] is False and verdict["reason"] == "regresses on mugs"


def test_sides_on_different_cells_are_not_compared(tmp_path):
    parent = _side(tmp_path, "p", {("plug", 0): (False, (1, 4)), ("plug", 1): (False, (1, 4))})
    child = _side(tmp_path, "c", {("plug", 0): (True, (4, 4))})
    with pytest.raises(IncompleteComparison, match="parent only"):
        compare(parent, child)


def _result(root, **changes):
    path = root / "result.json"
    result = json.loads(path.read_text())
    result.update(changes)
    path.write_text(json.dumps(result))
    return root


def test_an_infrastructure_run_is_reported_and_never_compared(tmp_path):
    broken = _result(
        _run(tmp_path / "boot"), status="boot_error", error={"type": "PermissionError"}
    )
    facts = digest(broken)
    assert facts["kind"] == "infra" and "INFRASTRUCTURE" in render(facts)
    good = digest(_run(tmp_path / "ok", stages=(0, 1)))
    assert good["kind"] == "policy"
    with pytest.raises(IncompleteComparison, match="infrastructure"):
        compare([facts], [good])


def test_runs_made_with_different_settings_are_not_compared(tmp_path):
    parent = digest(_result(_run(tmp_path / "p", stages=(0, 1)), model="model-a"))
    child = digest(_result(_run(tmp_path / "c", stages=(1, 1), success=True), model="model-b"))
    with pytest.raises(IncompleteComparison, match="execution settings"):
        compare([parent], [child])


def test_the_effort_compared_is_the_one_the_openai_runtime_sent(tmp_path):
    common = {"reasoning_effort": "xhigh", "host": {"runtime": "openai"}}
    high = digest(_result(_run(tmp_path / "h"), **common, generation={
        "options": {"extra_body": {"reasoning_effort": "high"}}}))  # fmt: skip
    low = digest(_result(_run(tmp_path / "l"), **common, generation={
        "options": {"extra_body": {"reasoning_effort": "low"}}}))  # fmt: skip
    assert high["identity"]["reasoning_effort"] == "high"
    assert high["identity"] != low["identity"]


def test_a_run_without_an_eval_seed_is_refused(tmp_path):
    no_seed = digest(_run(tmp_path / "n", eval_seed=None, stages=(0, 1)))
    with pytest.raises(IncompleteComparison, match="eval seed"):
        compare([no_seed], [no_seed])


def test_score_breaks_a_tie_only_when_every_episode_carries_one(tmp_path):
    def side(name, scores):
        runs = []
        for layout, score in enumerate(scores):
            root = _run(tmp_path / name / str(layout), layout=layout, stages=(0, 1))
            if score is not None:
                result = json.loads((root / "result.json").read_text())
                result["verdict"]["score"] = score
                (root / "result.json").write_text(json.dumps(result))
            runs.append(digest(root))
        return runs

    verdict = compare(side("p", [0.15, 0.15]), side("c", [0.40, 0.15]))
    assert verdict["accepted"] is True
    assert verdict["per_task"]["plug_in_charger"]["decided_by"] == "mean score"
    partial = compare(side("p2", [0.15, None]), side("c2", [0.40, 0.15]))
    assert partial["accepted"] is False and partial["reason"].startswith("tie")


def test_repeats_of_a_cell_are_compared_when_numbered(tmp_path):
    def run(name, repeat, success):
        facts = digest(_run(tmp_path / name, success=success, stages=(int(success), 1)))
        facts["repeat"] = repeat
        return facts

    parent = [run("p0", 0, False), run("p1", 1, False)]
    child = [run("c0", 0, True), run("c1", 1, False)]
    assert compare(parent, child)["accepted"] is True
    with pytest.raises(IncompleteComparison, match="repeat number"):
        compare([run("d0", 0, False), run("d1", 0, False)], child)


def test_a_failed_call_points_at_its_cell_and_the_frames_around_it(tmp_path):
    import os

    root = _run(tmp_path / "r", stages=(0, 1), calls=CALLS, cells=2)
    turns = [
        {"index": 0, "tool": "python", "calls": [0, 2], "started_at": 1000.0, "elapsed_s": 5.0},
        {"index": 1, "tool": "python", "calls": [2, 6], "started_at": 1010.0, "elapsed_s": 5.0},
    ]
    (root / "tool_turns.json").write_text(json.dumps(turns))
    for name, stamp in (("0000", 999.0), ("0001", 1003.0), ("0002", 1012.0)):
        (root / "steps" / name).mkdir(parents=True)
        os.utime(root / "steps" / name, (stamp, stamp))
    facts = digest(root)
    by_index = {f["index"]: f for f in facts["failed_calls"]}
    assert by_index[1]["cell"] == "cell_001.py"
    assert by_index[1]["frames"] == {"before": ["0000"], "during": ["0001"]}
    assert by_index[3]["cell"] == "cell_002.py"
    assert by_index[3]["frames"] == {"before": ["0001"], "during": ["0002"]}
    assert "[cell_001.py; frames 0000..0001]" in render(facts)
