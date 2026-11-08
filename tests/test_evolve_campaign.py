"""The campaign runner's rules, with a fake launcher in place of RoboDojo."""

from __future__ import annotations

import json

import pytest

from pyrualean.evolve.campaign import (
    BudgetExhausted,
    Campaign,
    CampaignError,
    summary,
)
from pyrualean.evolve.library import Library

LESSON = """## grasp-001
stage: grasp
symptom: s
applies: a
do: d
status: hypothesis
evidence: e
"""


def _campaign(tmp_path, **budgets):
    library = Library(tmp_path / "lib")
    source = tmp_path / "src" / "shared" / "lessons"
    source.mkdir(parents=True)
    (source / "grasp.md").write_text(LESSON)
    revision = library.freeze(tmp_path / "src", parent=None)
    library.set_active(revision)
    root = tmp_path / "camp"
    root.mkdir()
    config = {
        "launch_enabled": True,
        "library": str(library.root),
        "launcher": "/bin/true",
        "runs_root": str(tmp_path / "runs"),
        "gpu": 6,
        "tasks": ["plug_in_charger", "hang_mugs_random"],
        "splits": {
            "dev": {"eval_seed": 1, "layouts": [0, 1]},
            "val": {"eval_seed": 1, "layouts": [5]},
            "heldout": {"eval_seed": 0, "layouts": [0]},
        },
        "budgets": {"max_episodes": 50, "max_hours": 10, "max_infra_retries": 1, **budgets},
    }
    (root / "campaign.json").write_text(json.dumps(config))
    return Campaign(root), revision


class FakeLauncher:
    """Writes the run directory a launcher call would, with a scripted outcome per call."""

    def __init__(self, runs_root, outcomes=None):
        self.runs_root = runs_root
        self.outcomes = list(outcomes or [])
        self.calls = []

    def __call__(self, argv):
        task, eval_seed, layout, _gpu, tag, view = argv[1:7]
        self.calls.append(argv)
        outcome = self.outcomes.pop(0) if self.outcomes else "fail"
        run = self.runs_root / tag / "results" / f"{task}_l{layout}_s{eval_seed}-1"
        run.mkdir(parents=True)
        if outcome == "nothing":
            return 1  # the launcher died before an episode existed
        status = "boot_error" if outcome == "infra" else "completed"
        verdict = None if outcome == "infra" else {
            "success": outcome == "success", "eval_seed": int(eval_seed), "step": 10,
            "step_limit": 400, "stages_passed": int(outcome == "success"), "stages_total": 1,
        }  # fmt: skip
        result = {
            "task": task,
            "seed": int(layout),
            "status": status,
            "environment_success": outcome == "success",
            "verdict": verdict,
            "library": {"source": view},
            "host": {"runtime": "openai"},
        }
        (run / "result.json").write_text(json.dumps(result))
        return 0


def test_nothing_starts_unless_launch_is_enabled(tmp_path):
    campaign, revision = _campaign(tmp_path)
    campaign.config["launch_enabled"] = False
    with pytest.raises(CampaignError, match="launch_enabled"):
        campaign.run(revision, "dev", campaign.cells("dev"), launch=FakeLauncher(tmp_path))


def test_each_cell_runs_against_its_task_view_and_is_recorded(tmp_path):
    campaign, revision = _campaign(tmp_path)
    launcher = FakeLauncher(tmp_path / "runs", ["success", "fail", "fail", "fail"])
    results = campaign.run(revision, "dev", campaign.cells("dev"), launch=launcher)
    assert [r["success"] for r in results] == [True, False, False, False]
    views = {c[1]: c[6] for c in launcher.calls}
    assert views["plug_in_charger"].endswith(f"views/{revision}/plug_in_charger")
    assert campaign.state["episodes"] == 4 and campaign.state["infra_runs"] == 0
    assert len(campaign.episodes()) == 4


def test_a_restart_resumes_and_does_not_run_a_cell_twice(tmp_path):
    campaign, revision = _campaign(tmp_path)
    launcher = FakeLauncher(tmp_path / "runs")
    campaign.run(revision, "dev", campaign.cells("dev", ["plug_in_charger"]), launch=launcher)
    again = Campaign(campaign.root)  # a new process reading state.json
    results = again.run(revision, "dev", again.cells("dev"), launch=launcher)
    assert len(launcher.calls) == 4  # two plug cells once, then only the two mug cells
    assert len(results) == 4 and again.state["episodes"] == 4


def test_an_infrastructure_run_is_retried_and_never_counted_as_a_policy_failure(tmp_path):
    campaign, revision = _campaign(tmp_path)
    launcher = FakeLauncher(tmp_path / "runs", ["infra", "success"])
    cells = campaign.cells("dev", ["plug_in_charger"])[:1]
    (result,) = campaign.run(revision, "dev", cells, launch=launcher)
    assert result["success"] is True
    assert campaign.state["episodes"] == 1 and campaign.state["infra_runs"] == 1
    assert [e["kind"] for e in campaign.episodes()] == ["infra", "policy"]


def test_exhausted_infra_retries_stop_with_the_record_kept(tmp_path):
    campaign, revision = _campaign(tmp_path)
    launcher = FakeLauncher(tmp_path / "runs", ["infra", "nothing"])
    cells = campaign.cells("dev", ["plug_in_charger"])[:1]
    with pytest.raises(CampaignError, match="infrastructure"):
        campaign.run(revision, "dev", cells, launch=launcher)
    assert campaign.state["episodes"] == 0 and campaign.state["infra_runs"] == 2
    assert [e["launcher_exit"] for e in campaign.episodes()] == [0, 1]


def test_the_episode_budget_is_a_hard_stop(tmp_path):
    campaign, revision = _campaign(tmp_path, max_episodes=3)
    launcher = FakeLauncher(tmp_path / "runs")
    with pytest.raises(BudgetExhausted, match="episodes: 3 of 3"):
        campaign.run(revision, "dev", campaign.cells("dev"), launch=launcher)
    assert len(launcher.calls) == 3 and campaign.state["episodes"] == 3


def test_the_dollar_budget_counts_the_keys_spend_delta(tmp_path):
    campaign, revision = _campaign(tmp_path, max_usd=5.0)
    readings = iter([100.0, 103.0, 103.0, 106.5])
    launcher = FakeLauncher(tmp_path / "runs")
    with pytest.raises(BudgetExhausted, match="dollars"):
        campaign.run(
            revision, "dev", campaign.cells("dev"), launch=launcher, spend=lambda: next(readings)
        )
    assert campaign.state["usd"] == pytest.approx(6.5) and len(launcher.calls) == 2


def test_held_out_runs_only_when_asked_and_once_per_revision(tmp_path):
    campaign, revision = _campaign(tmp_path)
    launcher = FakeLauncher(tmp_path / "runs")
    with pytest.raises(CampaignError, match="allow_heldout"):
        campaign.run(revision, "heldout", campaign.cells("heldout"), launch=launcher)
    campaign.run(
        revision, "heldout", campaign.cells("heldout"), launch=launcher, allow_heldout=True
    )
    with pytest.raises(CampaignError, match="already run"):
        campaign.run(
            revision, "heldout", campaign.cells("heldout"), launch=launcher, allow_heldout=True
        )


def test_a_tampered_revision_is_never_run(tmp_path):
    campaign, revision = _campaign(tmp_path)
    lesson = campaign.library.revisions / revision / "shared/lessons/grasp.md"
    lesson.chmod(0o644)
    lesson.write_text(LESSON.replace("hypothesis", "accepted"))
    with pytest.raises(ValueError, match="modified"):
        campaign.run(revision, "dev", campaign.cells("dev"), launch=FakeLauncher(tmp_path))


def test_summary_separates_policy_episodes_from_infrastructure_runs(tmp_path):
    campaign, revision = _campaign(tmp_path)
    launcher = FakeLauncher(tmp_path / "runs", ["success", "infra", "fail"])
    campaign.run(revision, "dev", campaign.cells("dev", ["plug_in_charger"]), launch=launcher)
    row = summary(campaign)["by_revision"][f"{revision[:8]} dev plug_in_charger"]
    assert row == {"episodes": 2, "successes": 1, "infra": 1}


def test_the_real_launcher_is_told_where_the_runner_looks_for_its_run(tmp_path):
    from pyrualean.evolve.campaign import _find_run, _subprocess_launch

    script = tmp_path / "launcher.sh"
    script.write_text(
        "#!/bin/bash\n"
        'run="${EVOLVE_RUNS_ROOT}/$5/results/$1_l$3_s$2-1"\n'
        'mkdir -p "$run" && echo \'{"task": "t"}\' > "$run/result.json"\n'
    )
    script.chmod(0o755)
    runs = tmp_path / "runs"
    code = _subprocess_launch(runs)([str(script), "plug", "1", "3", "6", "tagX", "/view"])
    assert code == 0
    assert _find_run(runs / "tagX") == runs / "tagX" / "results" / "plug_l3_s1-1"
    assert (runs / "tagX.log").is_file()
