"""Offline contract tests; never import Isaac Sim, load SAM3 or contact a model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.evolve.campaign import BudgetExhausted, CampaignError
from pyrualean.evolve.evidence import DevEvidence
from pyrualean.evolve.gate import IncompleteComparison, compare
from pyrualean.evolve.library import Library, LibraryError, validate
from pyrualean.evolve.memory import EpisodeMemory, compact_messages
from pyrualean.evolve.online import OnlineLoop, initialize, save
from pyrualean.evolve.safety import check_source
from pyrualean.evolve.search import (
    create_candidate,
    evidence_bundle,
    parse_response,
    proposal_prompt,
    should_search,
)
from pyrualean.evolve.splits import LayoutRegistry, json_hash, private_assets
from pyrualean.openai_runtime import run_openai_agent

TASK = "plug_in_charger"
LESSON = """## retry
stage: grasp
symptom: lost grip
applies: uncertain grip
do: observe before retry
status: hypothesis
evidence: dev observation, not a proven fix
"""


def test_memory_is_task_scoped_fresh_and_records_evidence(tmp_path):
    source = tmp_path / "source"
    for task in (TASK, "make_kong"):
        folder = source / "tasks" / task / "lessons"
        folder.mkdir(parents=True)
        (folder / f"{task}.md").write_text(LESSON)
    library = Library(tmp_path / "library")
    skills = source / "tasks" / TASK / "skills"
    skills.mkdir()
    (skills / "grip.py").write_text(
        'def check(robo, arm="left"):\n    return {"status": "needs_observation"}\n'
    )
    revision = library.freeze(source, parent=None)
    view = library.view(revision, TASK)
    memory = EpisodeMemory(view, tmp_path / "episode", candidate=True)
    cards = memory.recall("grasp")["lessons"]
    assert len(cards) == 1 and cards[0]["file"] == TASK + ".md"
    assert "arm='left'" in memory.recall("grasp")["skills"][0]["signature"]
    assert not memory.recall("place")["lessons"]
    assert not EpisodeMemory(view, tmp_path / "normal").recall("grasp")["lessons"]
    memory.begin_cell("print('ok')", 1)
    with pytest.raises(ValueError, match="evidence"):
        memory.progress("grasp", confirmed=["holding"])
    memory.progress("grasp", confirmed=["holding"], evidence=["cell:1"])
    assert EpisodeMemory(view, tmp_path / "new").state["confirmed"] == []
    assert "model_report" in (tmp_path / "episode/progress.json").read_text()
    (view / "lessons" / (TASK + ".md")).chmod(0o644)
    (view / "lessons" / (TASK + ".md")).write_text("tampered")
    with pytest.raises(LibraryError, match="view"):
        library.view(revision, TASK)


def test_short_cells_use_memory_and_preserve_primitive_results(robo, tmp_path):
    memory = EpisodeMemory(None, tmp_path)
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace", harness=memory)
    reply = toolkit.execute_tool(
        "python", {"code": "harness.progress('locate', pending=['find bowl'])"}
    )
    assert "exception" not in reply.result
    assert memory.state["stage"] == "locate"
    rejected = toolkit.execute_tool("python", {"code": "robo.foo = 4"})
    assert "error" in rejected.result
    reply = toolkit.execute_tool("python", {"code": "harness.failures.clear()"})
    assert "AttributeError" in reply.result["exception"]


def test_repeated_failures_warn_without_blocking_or_merging_numeric_adjustments(tmp_path):
    class Record:
        index = 0
        stateful = True

        def to_dict(self):
            return {"ok": True, "summary": {"reached": False}}

    class Ledger:
        def __init__(self):
            self.records = []

        def __len__(self):
            return len(self.records)

    class Robot:
        def __init__(self):
            self._ledger = Ledger()

        def move_to(self, position):
            self._ledger.records.append(Record())
            return {"reached": False}

    memory = EpisodeMemory(None, tmp_path, max_attempts=2)
    view = memory.robot_view(Robot())
    view.move_to([0.101, 0.0, 0.0])
    view.move_to([0.102, 0.0, 0.0])
    assert view.move_to([0.103, 0.0, 0.0]) == {"reached": False}
    assert len(memory.failures) == 3
    view.move_to([0.101, 0.0, 0.0])
    assert view.move_to([0.101, 0.0, 0.0]) == {"reached": False}
    assert "retry_warning" in (tmp_path / "events.jsonl").read_text()
    assert "Action remains allowed" in memory.context()


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef f(robo): return os.getcwd()",
        "def f(robo, value=print('side effect')): pass",
        "@print('side effect')\ndef f(robo): pass",
        "def f(robo): return robo._backend",
        "def f(robo): return f(robo)",
        "def f(robo):\n import numpy as np\n return np.load('/private')",
    ],
)
def test_generated_skills_refuse_unsafe_paths_and_import_effects(source):
    assert check_source(source, skill=True)


def test_skill_dependencies_and_symlinks_are_checked(tmp_path):
    skills = tmp_path / "shared/skills"
    skills.mkdir(parents=True)
    (skills / "good.py").write_text("from library.bad import run\ndef f(robo): return run(robo)")
    (skills / "bad.py").write_text("import socket\ndef run(robo): return socket.socket()")
    assert validate(tmp_path)
    (skills / "bad.py").unlink()
    (skills / "bad.py").symlink_to(skills / "good.py")
    assert any("symbolic" in problem for problem in validate(tmp_path))


def test_context_compaction_keeps_tool_pairs_and_full_history():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "API"},
                {"type": "image_url", "image_url": {"url": "old"}},
            ],
        }
    ]
    for index in range(9):
        messages += [
            {"role": "assistant", "tool_calls": [{"id": str(index)}, {"id": f"b{index}"}]},
            {"role": "tool", "tool_call_id": str(index), "content": "ok"},
            {"role": "tool", "tool_call_id": f"b{index}", "content": "ok"},
        ]
    compacted = compact_messages(messages, "progress", 6)
    assert len(messages) == 28
    assert len(compacted) == 20
    assert compacted[2]["tool_calls"][0]["id"] == "3"
    assert len(compacted[0]["content"]) == 1


def test_request_cap_counts_transport_retries(robo, tmp_path):
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace")
    calls = []

    def transport(*args):
        calls.append(args)
        return 500, {"error": "offline fake"}

    result = run_openai_agent(
        "test",
        [],
        toolkit,
        model="fake",
        base_url="http://unused",
        api_key=None,
        max_turns=10,
        budget_s=30,
        out_dir=tmp_path,
        max_requests=2,
        retries=20,
        transport=transport,
        sleep=lambda _: None,
    )
    assert len(calls) == result["requests"] == 2


def _registry(root):
    rows = []
    for split, count in (("dev", 2), ("val", 1)):
        for index in range(count):
            value = {"fixture": f"{split}-{index}"}
            path = root / split / f"{index}.json"
            save(path, value)
            receipt = path.with_suffix(".physics.json")
            save(
                receipt,
                {
                    "content_sha256": json_hash(value),
                    "env_config_sha256": "env",
                    "validator": "offline fixture only",
                    "evidence": ["fake"],
                    "checks": {
                        name: True
                        for name in ("startup", "stable", "objects_valid", "instruction_matches")
                    },
                },
            )
            rows.append(
                {
                    "task": TASK,
                    "split": split,
                    "layout": index,
                    "file": str(path.relative_to(root)),
                    "content_sha256": json_hash(value),
                    "env_config_sha256": "env",
                    "physics_receipt": str(receipt.relative_to(root)),
                }
            )
    save(root / "manifest.json", {"instances": rows})
    return root / "manifest.json"


def test_split_content_duplicates_tampering_and_physics(tmp_path):
    path = _registry(tmp_path)
    registry = LayoutRegistry(path)
    row = registry.select("dev", TASK, 0)
    receipt = tmp_path / row["physics_receipt"]
    receipt.unlink()
    with pytest.raises(FileNotFoundError):
        registry.select("dev", TASK, 0)
    data = json.loads(path.read_text())
    duplicate = {**data["instances"][0], "split": "heldout"}
    data["instances"].append(duplicate)
    save(path, data)
    with pytest.raises(ValueError, match="duplicate"):
        LayoutRegistry(path)


def test_private_assets_excludes_official_layouts_and_trajectories(tmp_path):
    assets = tmp_path / "assets"
    for name in ("Object", "Robots", "Eval_Layout", "Train_Layout", "Traj"):
        (assets / name).mkdir(parents=True)
    layout = tmp_path / "private.json"
    layout.write_text('{"private": true}')
    target = tmp_path / "client/Assets"
    private_assets(target, assets, layout, task=TASK, suite="arx_x5", eval_seed=1, layout_index=0)
    assert not (target / "Traj").exists()
    assert not (target / "Eval_Layout").is_symlink()
    assert (
        target / f"Eval_Layout/RoboDojo/arx_x5/1/{TASK}_0.json"
    ).read_text() == layout.read_text()


def _proposal(ids, count=1):
    return {
        "observed": "primitive failed",
        "hypothesis": "observation insufficient",
        "validation": "paired online layouts",
        "proposals": [
            {
                "strategy": strategy,
                "difference": strategy + " before moving",
                "expected": "new RGB evidence",
                "evidence": ids,
                "files": {
                    f"tasks/{TASK}/lessons/{strategy}.md": LESSON.replace(
                        "observe before retry", strategy + " before retry"
                    )
                },
            }
            for strategy in ["observation", "approach"][:count]
        ],
    }


def test_reflection_never_accepts_val_or_edits_to_primitives(tmp_path):
    save(tmp_path / "result.json", {"split": "val", "task": TASK})
    with pytest.raises(ValueError, match="dev"):
        evidence_bundle([tmp_path], TASK)
    proposal = _proposal(["dev1"])
    proposal["proposals"][0]["files"] = {"src/pyrualean/arms.py": "pass"}
    with pytest.raises(LibraryError, match="path"):
        parse_response(json.dumps(proposal), task=TASK, count=1, evidence_ids={"dev1"})


def test_search_needs_two_consecutive_non_improving_round_transitions():
    unchanged = {"successes": 0, "mean_stages": 1, "mean_score": 0.1}
    improved = {**unchanged, "mean_stages": 2}
    assert not should_search([], unchanged)
    assert not should_search([unchanged], unchanged)
    assert should_search([unchanged, unchanged], unchanged)
    assert not should_search([unchanged, unchanged], improved)
    assert not should_search([unchanged, improved], improved)


def test_normal_dynamic_loops_are_allowed_but_io_is_not(robo, tmp_path):
    source = (
        "values = list(range(30))\nresult = [value * 2 for value in values]\n"
        "while values:\n values.pop()\nprint(len(result))"
    )
    assert check_source(source) == []
    memory = EpisodeMemory(None, tmp_path)
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace", harness=memory)
    reply = toolkit.execute_tool("python", {"code": source})
    assert "exception" not in reply.result and "error" not in reply.result
    assert check_source("for path in paths:\n open(path)")


def test_default_context_and_recall_do_not_silently_truncate(tmp_path):
    messages = [{"role": "user", "content": "initial image and API"}]
    messages += [{"role": "assistant", "content": str(index)} for index in range(12)]
    assert compact_messages(messages, "progress") == [
        messages[0],
        {"role": "user", "content": "progress"},
        *messages[1:],
    ]
    memory = EpisodeMemory(None, tmp_path)
    memory.cards = [
        {"id": str(index), "stage": "grasp", "scope": "task", "file": "all.md"}
        for index in range(6)
    ]
    memory.skills = [{"call": f"skill_{index}"} for index in range(5)]
    assert len(memory.recall("grasp")["lessons"]) == 6
    assert len(memory.recall("grasp")["skills"]) == 5
    memory.lesson_limit, memory.skill_limit = 2, 1
    assert len(memory.recall("grasp")["lessons"]) == 2
    assert len(memory.recall("grasp")["skills"]) == 1


def test_no_change_round_is_complete_and_next_round_can_continue(tmp_path, monkeypatch):
    loop, calls, _ = _loop(tmp_path, monkeypatch)
    parent = loop.library.active()
    original = loop.reflector

    def no_change(prompt, **kwargs):
        raw, usage = original(prompt, **kwargs)
        response = json.loads(raw)
        response.update(proposals=[], no_change="No supported improvement yet")
        response.pop("proposals")
        return json.dumps(response), usage

    loop.reflector = no_change
    loop.iterate(TASK, 0)
    assert len(calls) == 2
    assert loop.library.active() == parent
    assert loop.library.promotions()[0]["skipped"]
    loop.iterate(TASK, 0)
    assert len(calls) == 2
    loop.reflector = original
    loop.iterate(TASK, 1)
    assert len(calls) == 6
    assert loop.library.active() != parent


def test_identical_proposal_skips_val_without_aborting(tmp_path, monkeypatch):
    loop, calls, _ = _loop(tmp_path, monkeypatch)
    loop.iterate(TASK, 0)
    parent = loop.library.active()
    loop.iterate(TASK, 1)
    assert len(calls) == 6
    assert loop.library.active() == parent
    assert loop.library.promotions()[-1]["skipped"]


def test_one_dev_layout_is_a_valid_explicit_configuration(tmp_path, monkeypatch):
    loop, calls, _ = _loop(tmp_path, monkeypatch)
    loop.config["splits"]["dev"]["layouts"] = [0]
    loop.iterate(TASK, 0)
    assert len(calls) == 3


def test_search_can_recur_after_stagnation_without_a_once_per_task_default(tmp_path, monkeypatch):
    loop, _, _ = _loop(tmp_path, monkeypatch)
    loop.config["learning"]["search_rounds_per_task"] = None
    loop.config["budgets"]["max_reflection_requests"] = 5
    requested = []

    def no_change(prompt, **kwargs):
        requested.append("exactly 2 proposal(s)" in prompt)
        return json.dumps(
            {
                "observed": "same failure",
                "hypothesis": "uncertain",
                "validation": "need evidence",
                "proposals": [],
                "no_change": "No justified candidate",
            }
        ), {}

    loop.reflector = no_change
    for round_index in range(5):
        loop.iterate(TASK, round_index)
    assert requested == [False, False, True, True, True]
    assert loop.budget["search_rounds"][TASK] == 3


def test_same_category_and_numeric_adjustments_are_not_rejected():
    response = _proposal(["dev1"], 2)
    for index, proposal in enumerate(response["proposals"]):
        proposal["strategy"] = "grasp"
        proposal["files"] = {
            f"tasks/{TASK}/lessons/offset.md": LESSON.replace(
                "observe before retry", f"adjust relative offset by {0.01 + index * 0.01}"
            )
        }
    assert (
        len(
            parse_response(json.dumps(response), task=TASK, count=2, evidence_ids={"dev1"})[
                "proposals"
            ]
        )
        == 2
    )


def test_old_cards_cannot_disappear_or_silently_gain_verified_status(tmp_path):
    library = Library(tmp_path / "lib")
    relative = f"tasks/{TASK}/lessons/observation.md"
    path = tmp_path / "source" / relative
    path.parent.mkdir(parents=True)
    original = LESSON.replace("status: hypothesis", "status: accepted")
    original += "\n" + original.replace("## retry", "## retained")
    path.write_text(original)
    parent = library.freeze(path.parents[3], parent=None)
    proposal = _proposal(["dev1"])["proposals"][0]
    with pytest.raises(LibraryError, match="keep or explicitly retire"):
        create_candidate(library, parent, "dropped", proposal)
    proposal["files"][relative] = original.replace("observe before retry", "different advice", 1)
    with pytest.raises(LibraryError, match="cannot claim to be verified"):
        create_candidate(library, parent, "false-status", proposal)
    proposal["files"][relative] = original.replace("status: accepted", "status: retired", 1)
    _, revision = create_candidate(library, parent, "retired", proposal)
    assert "## retry" in (library.revisions / revision / relative).read_text()
    prompt = proposal_prompt({"task": TASK}, library, revision, 1)
    assert "status: retired" in prompt and "## retained" in prompt


def test_rejected_history_is_available_without_val_details(tmp_path, monkeypatch):
    loop, _, reflections = _loop(tmp_path, monkeypatch)
    original = loop.worker

    def fail_candidate(config, request, timeout):
        code = original(config, request, timeout)
        root = Path(request["output"]) / "results/episode"
        (root / "calls.jsonl").write_text("")
        if request["split"] == "val":
            result = json.loads((root / "result.json").read_text())
            result["environment_success"] = False
            result["verdict"]["stages_passed"] = 0
            save(root / "result.json", result)
            (root / "messages.json").write_text('"SECRET_VAL_DETAILS"')
        return code

    loop.worker = fail_candidate
    loop.iterate(TASK, 0)
    loop.iterate(TASK, 1)
    payload = json.loads(reflections[-1][reflections[-1].index('{"evidence":') :])
    assert payload["history"][0]["accepted"] is False
    assert payload["history"][0]["proposal"]["proposals"][0]["strategy"] == "observation"
    assert "metrics" not in payload["history"][0]
    assert len(payload["artifacts"]) == 4
    assert all(item["artifact"].startswith("dev-") for item in payload["artifacts"])
    assert "SECRET_VAL_DETAILS" not in reflections[-1]


def test_dev_reader_is_paged_and_cannot_read_hidden_results_or_other_splits(tmp_path):
    run = tmp_path / "dev"
    save(run / "result.json", {"task": TASK, "split": "dev", "episode_id": "episode"})
    (run / "calls.jsonl").write_text("abcdefgh")
    evidence = DevEvidence([run], TASK)
    result, _ = evidence.read("episode/calls.jsonl", length=3)
    assert result["text"] == "abc" and result["next_offset"] == 3
    result, _ = evidence.read("episode/calls.jsonl", offset=3)
    assert result["text"] == "defgh" and result["next_offset"] is None
    for artifact in ("episode/result.json", "episode/../result.json", "/etc/passwd"):
        with pytest.raises(ValueError, match="unknown"):
            evidence.read(artifact)
    (run / "calls.jsonl").unlink()
    (run / "calls.jsonl").symlink_to(run / "result.json")
    with pytest.raises(ValueError, match="escaped"):
        evidence.read("episode/calls.jsonl")
    for split in ("val", "heldout"):
        save(run / "result.json", {"task": TASK, "split": split})
        with pytest.raises(ValueError, match="dev"):
            DevEvidence([run], TASK)


def test_reflection_reads_text_and_images_with_budgeted_followups(tmp_path):
    from pyrualean.evolve.search import reflect

    run = tmp_path / "dev"
    save(run / "result.json", {"task": TASK, "split": "dev", "episode_id": "episode"})
    (run / "calls.jsonl").write_text("original dev observation")
    (run / "images").mkdir()
    (run / "images/frame.png").write_bytes(b"offline image")
    evidence = DevEvidence([run], TASK)
    requests, reserved, events = [], [], []

    def transport(url, body, timeout):
        requests.append(json.loads(json.dumps(body)))
        if len(requests) <= 2:
            artifact = "episode/calls.jsonl" if len(requests) == 1 else "episode/images/frame.png"
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": str(len(requests)),
                        "type": "function",
                        "function": {
                            "name": "read_dev_evidence",
                            "arguments": json.dumps({"artifact": artifact}),
                        },
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "{}"}
        return 200, {"choices": [{"message": message}]}

    raw, usage = reflect(
        "dev evidence",
        model="fake",
        base_url="http://unused",
        api_key=None,
        timeout_s=2,
        max_tokens=100,
        evidence=evidence,
        transport=transport,
        reserve_next=lambda: reserved.append(1),
        record=events.append,
    )
    assert raw == "{}" and usage["requests"] == 3 and len(reserved) == 2
    assert "original dev observation" in requests[1]["messages"][-1]["content"]
    assert requests[2]["messages"][-1]["content"][-1]["type"] == "image_url"
    assert len([event for event in events if event["type"] == "evidence_read"]) == 2


def test_reflection_tool_followups_cannot_exceed_global_budget(tmp_path, monkeypatch):
    loop, _, _ = _loop(tmp_path, monkeypatch)
    loop.config["budgets"]["max_reflection_requests"] = 2

    def excessive_followups(prompt, **kwargs):
        kwargs["reserve_next"]()
        kwargs["reserve_next"]()
        raise AssertionError("unbudgeted request")

    loop.reflector = excessive_followups
    with pytest.raises(BudgetExhausted):
        loop.iterate(TASK, 0)
    assert loop.budget["reflection_requests"] == 2
    with pytest.raises(CampaignError, match="interrupted reflection"):
        loop.iterate(TASK, 0)
    assert loop.budget["reflection_requests"] == 2


def _loop(tmp_path, monkeypatch):
    config = {
        "launch_enabled": False,
        "library": str(tmp_path / "lib"),
        "layout_manifest": str(_registry(tmp_path / "layouts")),
        "tasks": [TASK],
        "gpu": 0,
        "splits": {
            "dev": {"eval_seed": 0, "layouts": [0, 1]},
            "val": {"eval_seed": 1, "layouts": [0]},
            "heldout": {"eval_seed": 2, "layouts": []},
        },
        "execution": {
            "max_model_requests": 3,
            "model": "fake",
            "base_url": "unused",
            "api_key_env": "NO_KEY",
            "max_tokens": 200,
        },
        "learning": {
            "format_retries": 1,
            "reflection_timeout_seconds": 3,
            "search_rounds_per_task": 1,
        },
        "budgets": {
            "max_episodes": 30,
            "max_online_attempts": 30,
            "max_hours": 1,
            "episode_wall_seconds": 10,
            "max_infra_retries": 0,
            "max_model_requests": 100,
            "max_reflection_requests": 4,
        },
    }
    save(tmp_path / "config.json", config)
    root = tmp_path / "campaign"
    initialize(tmp_path / "config.json", root)
    config = json.loads((root / "campaign.json").read_text())
    config["launch_enabled"] = True
    save(root / "campaign.json", config)
    save(root / "manifest.json", {"execution_sha256": "fixed"})
    monkeypatch.setattr("pyrualean.evolve.online.execution_identity", lambda config: "fixed")
    calls = []
    reflection_calls = []

    def worker(config_path, request, timeout):
        from pyrualean.play import copy_library

        calls.append(request)
        output = Path(request["output"]) / "results/episode"
        view = Path(request["view"])
        improved = any((view / "lessons").glob("observation.md"))
        instance = LayoutRegistry(config["layout_manifest"]).select(
            request["split"], TASK, request["layout"]
        )
        result = {
            "task": TASK,
            "seed": request["layout"],
            "split": request["split"],
            "status": "completed",
            "environment_success": improved,
            "episode_id": Path(request["output"]).name,
            "instance": {
                key: instance[key] for key in ("task", "content_sha256", "env_config_sha256")
            },
            "revision": request["revision"],
            "execution_sha256": "fixed",
            "host": {"runtime": "openai"},
            "verdict": {
                "eval_seed": request["eval_seed"],
                "stages_passed": int(improved),
                "stages_total": 1,
            },
            "generation": {"requests": 1},
            "library": copy_library(view, output / "workspace/library"),
        }
        save(output / "result.json", result)
        return 0

    def reflector(prompt, **kwargs):
        reflection_calls.append(prompt)
        bundle = json.loads(prompt[prompt.index('{"evidence":') :])["evidence"]
        ids = [episode["id"] for episode in bundle["episodes"]]
        return json.dumps(_proposal(ids)), {"tokens": 1}

    return OnlineLoop(root, worker=worker, reflector=reflector), calls, reflection_calls


def test_complete_online_iteration_and_resume_are_idempotent(tmp_path, monkeypatch):
    loop, calls, reflections = _loop(tmp_path, monkeypatch)
    parent = loop.library.active()
    loop.iterate(TASK, 0)
    assert loop.library.active() != parent
    assert len(calls) == 4 and len(reflections) == 1
    assert len(loop.library.promotions()) == 1
    assert all(request["split"] in {"dev", "val"} for request in calls)
    assert all(request["revision"] == parent for request in calls if request["split"] == "dev")
    assert loop.library.promotions()[0]["candidate_dev"] == {}
    assert "validation" not in reflections[0].split('"episodes":')[1].split('"current_library":')[0]
    loop.iterate(TASK, 0)
    assert len(calls) == 4 and len(reflections) == 1
    assert len(loop.library.promotions()) == 1
    assert loop.budget["model_requests"] == 5
    report = loop.report()
    assert report["active"] == loop.library.active()


def test_crash_reserves_attempts_and_requests_and_blocks_silent_restart(tmp_path, monkeypatch):
    loop, _, _ = _loop(tmp_path, monkeypatch)

    def crash(*args):
        raise RuntimeError("process disappeared")

    loop.worker = crash
    with pytest.raises(RuntimeError, match="disappeared"):
        loop.run_cells(loop.library.active(), "dev", loop.campaign.cells("dev")[:1])
    assert loop.campaign.state["launch_attempts"] == 1
    assert loop.budget["model_requests"] == 3
    with pytest.raises(CampaignError, match="interrupted"):
        loop.run_cells(loop.library.active(), "dev", loop.campaign.cells("dev")[:1])
    loop.recover()
    assert loop.campaign.state["seconds"] >= 10
    assert loop.budget["model_requests"] == 3


def test_global_budget_and_launch_switch_prevent_model_calls(tmp_path, monkeypatch):
    loop, calls, _ = _loop(tmp_path, monkeypatch)
    loop.config["launch_enabled"] = False
    with pytest.raises(CampaignError, match="launch_enabled"):
        loop.run_cells(loop.library.active(), "dev", loop.campaign.cells("dev"))
    loop.config["budgets"]["max_model_requests"] = 0
    with pytest.raises(BudgetExhausted):
        loop._reserve(1)
    assert calls == []


def test_val_regression_keeps_parent(tmp_path, monkeypatch):
    loop, calls, _ = _loop(tmp_path, monkeypatch)
    original = loop.worker
    parent = loop.library.active()

    def regress(config, request, timeout):
        code = original(config, request, timeout)
        if request["split"] == "val":
            path = Path(request["output"]) / "results/episode/result.json"
            result = json.loads(path.read_text())
            result["environment_success"] = request["revision"] == parent
            result["verdict"]["stages_passed"] = int(result["environment_success"])
            save(path, result)
        return code

    loop.worker = regress
    loop.iterate(TASK, 0)
    assert len(calls) == 4
    assert loop.library.active() == parent
    assert not loop.library.promotions()[0]["accepted"]
    loop.iterate(TASK, 1)
    assert len(calls) == 6
    assert loop.library.active() == parent
    assert len(loop.library.promotions()) == 2
    assert sum(request["split"] == "val" for request in calls) == 2


def test_next_round_reuses_promoted_revision_val_and_checks_new_candidate(tmp_path, monkeypatch):
    loop, calls, _ = _loop(tmp_path, monkeypatch)
    loop.iterate(TASK, 0)
    parent = loop.library.active()
    original = loop.reflector

    def revised_proposal(prompt, **kwargs):
        raw, usage = original(prompt, **kwargs)
        proposal = json.loads(raw)
        files = proposal["proposals"][0]["files"]
        for name in files:
            files[name] = files[name].replace("observation before retry", "verify before retry")
        return json.dumps(proposal), usage

    loop.reflector = revised_proposal
    loop.iterate(TASK, 1)
    assert len(calls) == 7
    assert [request["split"] for request in calls[4:]] == ["dev", "dev", "val"]
    assert calls[-1]["revision"] != parent
    assert loop.library.active() == parent
    assert not loop.library.promotions()[-1]["accepted"]


@pytest.mark.parametrize("dev_advantage", ["better", "tie", "worse"])
def test_repeated_failure_compares_two_different_online_strategies(
    tmp_path, monkeypatch, dev_advantage
):
    loop, calls, _ = _loop(tmp_path, monkeypatch)
    monkeypatch.setattr("pyrualean.evolve.online.should_search", lambda history, current: True)
    original = loop.worker
    parent = loop.library.active()

    def failed_evidence(config, request, timeout):
        code = original(config, request, timeout)
        if request["split"] == "dev" and dev_advantage != "better":
            result_path = Path(request["output"]) / "results/episode/result.json"
            result = json.loads(result_path.read_text())
            result["environment_success"] = (
                dev_advantage == "worse" and request["revision"] == parent
            )
            result["verdict"]["stages_passed"] = int(result["environment_success"])
            save(result_path, result)
        path = Path(request["output"]) / "results/episode/calls.jsonl"
        path.write_text(
            json.dumps({"index": 0, "tool": "move_to", "ok": True, "summary": {"reached": False}})
            + "\n"
        )
        return code

    def reflect_two(prompt, **kwargs):
        assert "exactly 2 proposal(s)" in prompt
        bundle = json.loads(prompt[prompt.index('{"evidence":') :])["evidence"]
        return json.dumps(_proposal([row["id"] for row in bundle["episodes"]], 2)), {}

    loop.worker, loop.reflector = failed_evidence, reflect_two
    loop.iterate(TASK, 0)
    assert len(calls) == 8
    assert len({request["revision"] for request in calls}) == 3
    assert loop.budget["search_rounds"][TASK] == 1
    assert loop.library.active() != parent
    assert len(loop.library.promotions()[0]["candidate_dev"]) == 2


def test_service_failure_is_not_robot_failure(tmp_path):
    from pyrualean.evolve.digest import classify

    assert (
        classify(
            {
                "status": "completed",
                "verdict": {"success": False},
                "generation": {"status": "error"},
            }
        )
        == "infra"
    )


def test_reflection_images_are_explicit_dev_evidence_not_tools(tmp_path):
    from pyrualean.evolve.search import reflect

    image = tmp_path / "recorded.png"
    image.write_bytes(b"offline fake image")
    requests = []

    def transport(url, body, timeout):
        requests.append(body)
        return 200, {"choices": [{"message": {"content": "{}"}}]}

    reflect(
        "dev facts",
        model="fake",
        base_url="http://unused",
        api_key=None,
        timeout_s=1,
        max_tokens=100,
        transport=transport,
        images=[str(image)],
    )
    assert "tools" not in requests[0]
    content = requests[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "dev facts"}
    assert content[-1]["type"] == "image_url"


def test_identity_tampering_cannot_publish_or_recover_as_success(tmp_path, monkeypatch):
    loop, _, _ = _loop(tmp_path, monkeypatch)
    original = loop.worker

    def tamper(config, request, timeout):
        code = original(config, request, timeout)
        path = Path(request["output"]) / "results/episode/result.json"
        result = json.loads(path.read_text())
        result["library"]["sha256"] = "not-the-evaluated-library"
        save(path, result)
        return code

    loop.worker = tamper
    with pytest.raises(CampaignError, match="identity"):
        loop.run_cells(loop.library.active(), "dev", loop.campaign.cells("dev")[:1])
    loop.recover()
    assert loop.campaign.state["episodes"] == 0
    assert loop.campaign.state["infra_runs"] == 1
    assert not loop.library.promotions()


def test_recovery_after_journal_append_does_not_count_the_run_twice(tmp_path, monkeypatch):
    loop, _, _ = _loop(tmp_path, monkeypatch)
    original_save = loop.campaign._save

    def interrupted_save():
        if loop.campaign.state["done"]:
            raise RuntimeError("crash after journal, before state save")
        original_save()

    loop.campaign._save = interrupted_save
    with pytest.raises(RuntimeError, match="journal"):
        loop.run_cells(loop.library.active(), "dev", loop.campaign.cells("dev")[:1])
    resumed = OnlineLoop(loop.root)
    resumed.recover()
    assert resumed.campaign.state["episodes"] == 1
    assert len(resumed.campaign.episodes()) == 1
    resumed.recover()
    assert len(resumed.campaign.episodes()) == 1


def test_gate_refuses_identity_mismatch_and_empty_comparison():
    with pytest.raises(IncompleteComparison):
        compare([], [])
    facts = {
        "task": TASK,
        "eval_seed": 0,
        "layout": 0,
        "success": False,
        "kind": "policy",
        "run_dir": "fake",
        "instance": {"content_sha256": "a"},
    }
    with pytest.raises(IncompleteComparison, match="identity"):
        compare([facts], [{**facts, "instance": {"content_sha256": "b"}}])
