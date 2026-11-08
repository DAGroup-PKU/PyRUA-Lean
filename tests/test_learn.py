"""The learn campaign offline: fake episodes and a fake worker; no simulator, model or CLI."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pyrualean.evolve.library import Library
from pyrualean.learn import config as configuration
from pyrualean.learn import worker as workers
from pyrualean.learn.campaign import Campaign, init, paired, sign_test, summarize
from pyrualean.learn.episode import end_reason
from pyrualean.learn.replay import replay
from pyrualean.learn.runner import BudgetExhausted, Runner

TASK = "swap_blocks"
SKILL = '''"""Press the button."""


def press(robo, arm="left"):
    """Press once and report."""
    return {"ok": True, "why": "pressed"}
'''
CARD = """## swap_blocks-manipulate-001
stage: manipulate
symptom: the button was not pressed after a move
applies: after each placement
do: call library.press.press
status: hypothesis
evidence: dev l0 cell_009
"""


def make_config(tmp_path: Path, **overrides) -> Path:
    source = tmp_path / "bench"
    for part in ("env_cfg", "task", "src"):
        (source / part).mkdir(parents=True, exist_ok=True)
        (source / part / "x.txt").write_text(part)
    config = {
        "tasks": [TASK],
        "dev": {"eval_seed": 1, "layouts": [0, 1]},
        "test": {"eval_seed": 2, "layouts": [0, 1, 2]},
        "rounds": 3,
        "source_root": str(source),
        "sim_python": "/bin/true",
        "sam3_checkpoint": "/dev/null",
        "slots": {"gpus": [0], "per_gpu": 4},
        "execution": {"model": "m", "base_url": "https://x/v1", "api_key_env": "KEY_FOR_TEST"},
        "worker": {"max_replays": 2, "max_replays_per_layout": 1},
        "budgets": {
            "max_launches": 100,
            "max_hours": 1,
            "max_infra_retries": 1,
            "episode_watchdog_seconds": 6000,
        },
    }
    config.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


class FakeEpisodes:
    """Writes the result an episode process would; success when the library has a skill."""

    def __init__(self, infra_once: set[str] | None = None):
        self.calls: list[dict] = []
        self.infra_once = set(infra_once or ())

    def __call__(self, output: Path, env: dict, timeout_s: float) -> int:
        request = json.loads((output / "request.json").read_text())
        self.calls.append(request)
        tag = f"{request['split']}-{request['layout']}"
        if tag in self.infra_once:
            self.infra_once.discard(tag)
            return 1  # crashed before writing a result
        view = Path(request["view"])
        skilled = any((view / "skills").glob("*.py")) if (view / "skills").is_dir() else False
        success = skilled and request["layout"] != 1
        run = output / "episode"
        (run / "workspace").mkdir(parents=True)
        (run / "workspace" / "cell_001.py").write_text("print(1)")
        result = {
            "task": request["task"],
            "seed": request["layout"],
            "status": "completed",
            "environment_success": success,
            "verdict": {
                "eval_seed": request["eval_seed"],
                "stages_passed": 12 if success else 2,
                "stages_total": 12,
            },
            "generation": {"runtime": "openai", "requests": 10, "input_tokens": 1000},
            "identity": request["identity"],
            "end": "success" if success else "the policy called finish",
            # a failed episode the policy took for a success (an overclaim)
            "finish": None if success else {"status": "success", "summary": "done"},
        }
        (run / "result.json").write_text(json.dumps(result))
        return 0


def improving_worker(session: Path, env: dict) -> dict:
    skills = session / "candidate" / "tasks" / TASK / "skills"
    lessons = session / "candidate" / "tasks" / TASK / "lessons"
    (skills / "press.py").write_text(SKILL)
    (lessons / "buttons.md").write_text(CARD)
    (session / "FINDINGS.md").write_text("## Episodes\nbutton never pressed\n")
    assert env["PYRUALEAN_LEARN_SESSION"] == str(session.resolve())
    assert "KEY_FOR_TEST" not in env
    return {"exit": 0, "total_cost_usd": 1.5}


def test_learns_on_dev_then_tests_against_v0_on_unseen_layouts(tmp_path, monkeypatch):
    monkeypatch.setenv("KEY_FOR_TEST", "secret")
    root = tmp_path / "campaign"
    init(root, make_config(tmp_path))
    episodes = FakeEpisodes()
    campaign = Campaign(root, launch=episodes, worker_launch=improving_worker)
    campaign.run()

    learn = json.loads((root / "tasks" / TASK / "learn.json").read_text())
    rounds = learn["rounds"]
    assert [r["dev"]["successes"] for r in rounds] == [0, 1, 1]
    v0 = rounds[0]["revision"]
    assert rounds[1]["parent"] == v0 and rounds[1]["revision"] != v0
    assert rounds[1]["changes"]["added"] == [
        f"tasks/{TASK}/lessons/buttons.md",
        f"tasks/{TASK}/skills/press.py",
    ]
    # the worker made the same change again in round 2: content-addressed, same revision
    assert rounds[2]["revision"] == rounds[1]["revision"]
    # every failed dev episode claimed success: counted per batch and shown to the worker
    assert [r["dev"]["overclaims"] for r in rounds] == [2, 1, 1]
    episodes_md = (root / "tasks" / TASK / "worker-r1" / "episodes.md").read_text()
    assert "successes 0/2" in episodes_md and ", 2 overclaimed" in episodes_md
    assert episodes_md.count("OVERCLAIM: the policy finished with success") == 2
    history_md = (root / "tasks" / TASK / "worker-r2" / "history.md").read_text()
    assert "dev: 0/2 successes, 2 overclaimed" in history_md
    assert "dev: 1/2 successes, 1 overclaimed" in history_md

    # dev used only eval seed 1, test only eval seed 2; test ran v0 and the final revision
    dev = [c for c in episodes.calls if c["split"] == "dev"]
    test = [c for c in episodes.calls if c["split"] == "test"]
    assert {c["eval_seed"] for c in dev} == {1} and {c["eval_seed"] for c in test} == {2}
    assert len(dev) == 2 * 3 and len(test) == 3 * 2
    assert {c["revision"] for c in test} == {v0, rounds[2]["revision"]}

    report = campaign.report()
    assert report["tasks"][TASK]["test"] == {
        "pairs": 3,
        "missing": 0,
        "v0_successes": 0,
        "final_successes": 2,
        "final_only": 2,
        "v0_only": 0,
        "sign_test_p": 0.5,
    }
    assert report["spend"]["worker_usd"] == 3.0

    # resuming runs nothing again
    again = FakeEpisodes()
    Campaign(root, launch=again, worker_launch=improving_worker).run()
    assert again.calls == []


def test_a_batch_short_of_episodes_neither_rolls_back_nor_counts_as_best(tmp_path):
    root = tmp_path / "campaign"
    init(root, make_config(tmp_path, rounds=2, run_test=False,
                           learning={"rollback_if_worse_by": 1}))  # fmt: skip

    def episodes(output, env, timeout_s):
        request = json.loads((output / "request.json").read_text())
        skilled = (Path(request["view"]) / "skills").is_dir()
        if skilled and request["layout"] == 0:
            return 1  # infrastructure, every attempt: layout 0 is missing from round 1
        FakeEpisodes()(output, env, timeout_s)
        result_path = output / "episode" / "result.json"
        result = json.loads(result_path.read_text())
        result["environment_success"] = not skilled  # v0 solves both, round 1 none
        result_path.write_text(json.dumps(result))
        return 0

    campaign = Campaign(root, launch=episodes, worker_launch=improving_worker)
    campaign.learn(TASK)
    rounds = json.loads((root / "tasks" / TASK / "learn.json").read_text())["rounds"]
    assert rounds[0]["dev"]["episodes"] == 2 and rounds[1]["dev"]["episodes"] == 1
    assert "rolled_back_to" not in rounds[1] and rounds[1]["active"] == rounds[1]["revision"]


def test_rollback_when_a_revision_does_worse_on_dev(tmp_path):
    root = tmp_path / "campaign"
    init(root, make_config(tmp_path, rounds=3, learning={"rollback_if_worse_by": 1}))

    def episodes(output, env, timeout_s):
        request = json.loads((output / "request.json").read_text())
        view = Path(request["view"])
        skilled = (view / "skills").is_dir()
        # v0 solves layout 0; any learned revision solves nothing
        FakeEpisodes()(output, env, timeout_s)
        result_path = output / "episode" / "result.json"
        result = json.loads(result_path.read_text())
        result["environment_success"] = not skilled and request["layout"] == 0
        result_path.write_text(json.dumps(result))
        return 0

    campaign = Campaign(root, launch=episodes, worker_launch=improving_worker)
    final = campaign.learn(TASK)
    rounds = json.loads((root / "tasks" / TASK / "learn.json").read_text())["rounds"]
    v0 = rounds[0]["revision"]
    assert rounds[1]["rolled_back_to"] == v0 and final == v0
    # round 2's worker started again from v0 and read v0's dev batch
    assert rounds[2]["parent"] == v0
    brief = (root / "tasks" / TASK / "worker-r2" / "episodes.md").read_text()
    assert f"revision {v0}" in brief
    history = (root / "tasks" / TASK / "worker-r2" / "history.md").read_text()
    assert "ROLLED BACK" in history


def test_infra_failures_are_retried_and_never_scored(tmp_path):
    root = tmp_path / "campaign"
    init(root, make_config(tmp_path, rounds=1))
    episodes = FakeEpisodes(infra_once={"dev-0"})
    campaign = Campaign(root, launch=episodes)
    campaign.learn(TASK)
    entries = campaign.runner.entries()
    assert [e["kind"] for e in entries].count("infra") == 1
    rounds = json.loads((root / "tasks" / TASK / "learn.json").read_text())["rounds"]
    assert rounds[0]["dev"]["episodes"] == 2


def test_launch_budget_is_hard(tmp_path):
    root = tmp_path / "campaign"
    path = make_config(tmp_path)
    config = json.loads(path.read_text())
    config["budgets"]["max_launches"] = 1
    path.write_text(json.dumps(config))
    init(root, path)
    runner = Runner(root, configuration.load(root / "config.json"), launch=FakeEpisodes())
    library = Library(root / "library" / TASK)
    (root / "empty").mkdir()
    revision = library.freeze(root / "empty", parent=None)
    view = library.view(revision, TASK)
    runner.episode(split="dev", task=TASK, revision=revision, layout=0, view=view)
    with pytest.raises(BudgetExhausted):
        runner.episode(split="dev", task=TASK, revision=revision, layout=1, view=view)


def _session(tmp_path: Path) -> tuple[Path, Library, Path]:
    root = tmp_path / "campaign"
    init(root, make_config(tmp_path))
    config = configuration.load(root / "config.json")
    campaign = Campaign(root, launch=FakeEpisodes())
    library = campaign.library(TASK)
    session = root / "tasks" / TASK / "worker-r1"
    workers.prepare(
        session,
        task=TASK,
        round_index=1,
        parent=library.active(),
        library=library,
        campaign=root,
        config_path=root / "config.json",
        config=config,
        episodes="",
        history="",
    )
    return session, library, root


def test_candidate_rules(tmp_path):
    session, library, _ = _session(tmp_path)
    candidate = session / "candidate"
    (candidate / "tasks" / TASK / "skills" / "press.py").write_text(SKILL)
    assert workers.problems(session, library) == []

    (candidate / "tasks" / "make_kong" / "lessons").mkdir(parents=True)
    (candidate / "tasks" / "make_kong" / "lessons" / "x.md").write_text(CARD)
    assert any("only tasks/swap_blocks/" in p for p in workers.problems(session, library))
    (candidate / "tasks" / "make_kong" / "lessons" / "x.md").unlink()

    verified = CARD.replace("status: hypothesis", "status: dev_verified").replace(
        "evidence: dev l0 cell_009", "evidence: looked right"
    )
    (candidate / "tasks" / TASK / "lessons" / "b.md").write_text(verified)
    assert any("dev_verified without" in p for p in workers.problems(session, library))
    # "dev" inside another word, a replay that never ran, or this round's own batch: refused
    for evidence in ("no deviation seen", "replays/00 showed it", "dev round 1"):
        (candidate / "tasks" / TASK / "lessons" / "b.md").write_text(
            verified.replace("evidence: looked right", f"evidence: {evidence}")
        )
        assert any("dev_verified without" in p for p in workers.problems(session, library))
    for evidence in ("dev round 0, layouts 0/1", "parent dev: l0 cell 5"):
        (candidate / "tasks" / TASK / "lessons" / "b.md").write_text(
            verified.replace("evidence: looked right", f"evidence: {evidence}")
        )
        assert workers.problems(session, library) == []
    (candidate / "tasks" / TASK / "lessons" / "b.md").write_text(CARD)

    (candidate / "tasks" / TASK / "skills" / "bad.py").write_text(
        "import os\ndef f(robo): return os.listdir('/')"
    )
    assert workers.problems(session, library)
    (candidate / "tasks" / TASK / "skills" / "bad.py").unlink()
    outcome = workers.accept(session, library)
    assert outcome["accepted"] and outcome["revision"] != library.active()


def test_old_cards_cannot_be_dropped(tmp_path):
    session, library, root = _session(tmp_path)
    lessons = session / "candidate" / "tasks" / TASK / "lessons"
    (lessons / "b.md").write_text(CARD)
    revision = workers.accept(session, library)["revision"]
    session2 = root / "tasks" / TASK / "worker-r2"
    workers.prepare(
        session2,
        task=TASK,
        round_index=2,
        parent=revision,
        library=library,
        campaign=root,
        config_path=root / "config.json",
        config=configuration.load(root / "config.json"),
        episodes="",
        history="",
    )
    (session2 / "candidate" / "tasks" / TASK / "lessons" / "b.md").write_text(
        CARD.replace("swap_blocks-manipulate-001", "swap_blocks-manipulate-002")
    )
    assert any("card removed" in p for p in workers.problems(session2, library))


def test_replay_is_dev_only_budgeted_and_uses_the_candidate(tmp_path):
    session, library, root = _session(tmp_path)
    (session / "candidate" / "tasks" / TASK / "skills" / "press.py").write_text(SKILL)
    cells = session / "replays" / "cells-a"
    cells.mkdir(parents=True)
    (cells / "cell_001.py").write_text("from library.press import press\nprint(press(robo))")
    episodes = FakeEpisodes()
    runner = Runner(root, configuration.load(root / "config.json"), launch=episodes)
    text = replay(session, cells, 0, runner=runner)
    assert "success: True" in text
    call = episodes.calls[-1]
    assert call["replay_dir"].endswith("replays/00/cells") and call["eval_seed"] == 1
    assert (Path(call["view"]) / "skills" / "press.py").exists()
    with pytest.raises(ValueError, match="not a dev layout"):
        replay(session, cells, 2, runner=runner)
    with pytest.raises(BudgetExhausted, match="layout 0"):
        replay(session, cells, 0, runner=runner)
    replay(session, cells, 1, runner=runner)
    with pytest.raises(BudgetExhausted, match="this session"):
        replay(session, cells, 1, runner=runner)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(ValueError, match="under the session"):
        replay(session, outside, 0, runner=runner)


def test_worker_command_is_scoped(tmp_path):
    session, _, root = _session(tmp_path)
    argv = workers.command(configuration.load(root / "config.json"), session)
    allowed = argv[argv.index("--allowedTools") + 1 : argv.index("--disallowedTools")]
    denied = argv[argv.index("--disallowedTools") + 1 : argv.index("--no-session-persistence")]
    # RoboDojo's success definition of this task is readable, never writable
    bench = tmp_path / "bench"
    assert f"Read(/{bench / 'task' / 'RoboDojo' / 'tasks' / (TASK + '.py')})" in allowed
    assert f"Read(/{bench / 'env' / 'reward_manager' / 'reward_manager.py'})" in allowed
    assert f"Edit(/{bench / 'task' / 'RoboDojo' / 'tasks' / (TASK + '.py')})" in denied
    assert not any("make_kong" in a for a in allowed)
    # reads go through the path-scoped tools: shell readers are denied, but not python - a deny
    # rule outranks the allow rules of the replay and check commands
    assert "Bash(cat:*)" in denied
    assert not any(a.startswith(("Bash(python", "Bash(python3")) for a in denied)
    assert "Bash(python -m pyrualean.learn.replay:*)" in allowed
    assert "Bash(python -m pyrualean.learn.worker check)" in allowed
    # the brief says how RoboDojo ends and judges an episode
    assert "evaluated after EVERY native action" in argv[2]
    # transcripts live outside everything the worker may read
    assert not str(workers.transcripts(session)).startswith(str(session.resolve()))
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert all(not a.startswith("Bash(") or "pyrualean.learn" in a or a == "Bash(ls:*)"
               for a in allowed)  # fmt: skip
    writes = [a for a in allowed if a.startswith(("Write", "Edit"))]
    assert all(str(session.resolve()) in a for a in writes)
    reads = [a for a in allowed if a.startswith("Read")]
    assert any(f"runs/dev/{TASK}" in a for a in reads)
    # the brief names the tools that work, so the worker does not burn turns on refusals
    assert "Bash runs exactly three commands" in argv[2]
    assert not any("runs/test" in a for a in reads)


def test_end_reason_names_harness_cutoffs():
    nudged = {"generation": {"finish_reasons": ["tool_calls", "stop"]}, "status": "completed"}
    assert "nudges exhausted" in end_reason(nudged)
    assert end_reason({"environment_success": True}) == "success"
    assert end_reason({"status": "call_budget"}) == "cell budget exhausted"


def test_config_rejects_overlapping_splits(tmp_path):
    path = make_config(tmp_path, test={"eval_seed": 1, "layouts": [5]})
    with pytest.raises(configuration.ConfigError, match="different eval seeds"):
        configuration.load(path)


def test_sign_test_and_pairs():
    assert sign_test(0, 0) is None
    assert sign_test(5, 0) == pytest.approx(0.0625)
    pairs = [
        {"v0": False, "final": True},
        {"v0": True, "final": True},
        {"v0": None, "final": True},
    ]
    assert paired(pairs)["pairs"] == 2 and paired(pairs)["missing"] == 1


def test_cut_off_worker_is_continued_not_restarted(tmp_path):
    session, library, root = _session(tmp_path)
    config = configuration.load(root / "config.json")
    briefs = []

    def slow_then_done(session_dir, env):
        briefs.append(workers.command(config, session_dir)[2])
        skill = session_dir / "candidate" / "tasks" / TASK / "skills" / "press.py"
        if len(briefs) == 1:
            skill.write_text(SKILL)  # wrote the fix, then hit the wall clock
            return {"exit": 124, "stalled": "wall clock"}
        assert skill.exists()  # the second attempt finds the first one's work
        (session_dir / "FINDINGS.md").write_text("done")
        return {"exit": 0}

    record = workers.run(config, session, launch=slow_then_done)
    assert workers.succeeded(record) and record["attempt"] == 1
    assert "Resuming" not in briefs[0] and "Resuming" in briefs[1]
    assert workers.accept(session, library)["changed"] is True


def test_cut_off_candidate_is_used_and_an_empty_failure_holds_the_task(tmp_path):
    from pyrualean.learn.campaign import WorkerFailed

    session, library, root = _session(tmp_path)
    config = configuration.load(root / "config.json")

    def cut_off(session_dir, env):
        (session_dir / "candidate" / "tasks" / TASK / "skills" / "press.py").write_text(SKILL)
        return {"exit": 124, "stalled": "wall clock"}

    workers.run(config, session, launch=cut_off)
    outcome = workers.accept(session, library)
    assert outcome["accepted"] and outcome["changed"]  # no FINDINGS, but a valid change

    root2 = tmp_path / "second" / "campaign"
    init(root2, make_config(tmp_path / "second", rounds=2, run_test=False))
    episodes = FakeEpisodes()
    campaign = Campaign(
        root2, launch=episodes, worker_launch=lambda s, e: {"exit": 1, "is_error": True,
                                                           "result": "refused: Usage Policy"}
    )  # fmt: skip
    with pytest.raises(WorkerFailed):
        campaign.learn(TASK)
    # round 1 never ran the parent again
    assert {c["repeat"] if "repeat" in c else c["output"] for c in episodes.calls}
    assert all("-r1-" not in c["output"] for c in episodes.calls)


def test_worker_is_retried_from_the_parent_after_a_failure(tmp_path):
    session, library, root = _session(tmp_path)
    config = configuration.load(root / "config.json")
    calls = []

    def flaky(session_dir, env):
        calls.append(1)
        if len(calls) == 1:
            junk = session_dir / "candidate" / "tasks" / TASK / "skills" / "half.py"
            junk.write_text("def half(robo):\n    return {")  # left broken by a stalled worker
            return {"exit": 124, "stalled": "no output for 901 s"}  # stuck: its work is dropped
        improving_worker(session_dir, env)
        return {"exit": 0}

    record = workers.run(config, session, launch=flaky)
    assert len(calls) == 2 and record["attempt"] == 1 and workers.succeeded(record)
    assert (session / "worker-a0.json").exists()
    assert not (session / "candidate" / "tasks" / TASK / "skills" / "half.py").exists()
    assert workers.accept(session, library)["changed"] is True


def test_no_change_is_accepted_and_a_failed_worker_is_not(tmp_path):
    session, library, root = _session(tmp_path)
    config = configuration.load(root / "config.json")

    def nothing(session_dir, env):
        (session_dir / "FINDINGS.md").write_text("nothing worth changing")
        return {"exit": 0}

    workers.run(config, session, launch=nothing)
    assert workers.accept(session, library) == {
        "accepted": True,
        "revision": library.active(),
        "changed": False,
    }

    session2, library2, root2 = _session(tmp_path / "second")
    workers.run(config, session2, launch=lambda s, e: {"exit": 124, "stalled": "wall clock"})
    outcome = workers.accept(session2, library2)
    assert not outcome["accepted"] and "worker failed" in outcome["problems"][0]


def test_learning_code_is_not_part_of_the_execution_identity(tmp_path):
    from pyrualean.learn.episode import _policy_hash

    package = tmp_path / "pkg"
    (package / "learn").mkdir(parents=True)
    (package / "play.py").write_text("x = 1")
    (package / "learn" / "worker.py").write_text("a = 1")
    before = _policy_hash(package)
    (package / "learn" / "worker.py").write_text("a = 2")
    assert _policy_hash(package) == before
    (package / "play.py").write_text("x = 2")
    assert _policy_hash(package) != before


def test_replay_skips_a_snapshot_left_by_an_interrupted_attempt(tmp_path):
    session, library, root = _session(tmp_path)
    cells = session / "replays" / "cells-a"
    cells.mkdir(parents=True)
    (cells / "cell_001.py").write_text("print(1)")
    (session / "replays" / "00" / "cells").mkdir(parents=True)  # no replays.json entry
    runner = Runner(root, configuration.load(root / "config.json"), launch=FakeEpisodes())
    replay(session, cells, 0, runner=runner)
    assert (session / "replays" / "01" / "cells" / "cell_001.py").exists()


def _ok_services(config):
    return []


def test_provider_outage_pauses_without_using_up_retries(tmp_path):
    from pyrualean.learn.runner import ServiceDown, paused

    root = tmp_path / "campaign"
    init(root, make_config(tmp_path, rounds=1))

    def outage(output, env, timeout_s):
        FakeEpisodes()(output, env, timeout_s)
        path = output / "episode" / "result.json"
        result = json.loads(path.read_text())
        result["status"] = "planner_error"
        path.write_text(json.dumps(result))
        return 0

    campaign = Campaign(root, launch=outage)
    library = campaign.library(TASK)
    view = library.view(library.active(), TASK)
    with pytest.raises(ServiceDown):
        for _ in range(5):
            campaign.runner.episode(
                split="dev", task=TASK, revision=library.active(), layout=0, view=view
            )
    assert paused(root)
    state = json.loads((root / "state.json").read_text())
    assert all(n == 0 for n in state["attempts"].values())  # nothing charged to the job

    campaign = Campaign(root, launch=FakeEpisodes())
    campaign.resume(check=_ok_services)
    assert not paused(root)
    campaign.run()
    rounds = json.loads((root / "tasks" / TASK / "learn.json").read_text())["rounds"]
    assert rounds[0]["dev"]["episodes"] == 2


def test_worker_service_failure_pauses_and_the_round_is_redone(tmp_path):
    from pyrualean.learn.runner import paused

    root = tmp_path / "campaign"
    init(root, make_config(tmp_path, rounds=2, run_test=False))

    def refused(session, env):
        junk = session / "candidate" / "tasks" / TASK / "skills" / "half.py"
        junk.write_text("def half(robo):\n    return {")
        return {
            "exit": 1,
            "is_error": True,
            "result": "API Error: Unable to connect to API (ConnectionRefused)",
        }

    campaign = Campaign(root, launch=FakeEpisodes(), worker_launch=refused)
    campaign.run()
    assert paused(root)
    rounds = json.loads((root / "tasks" / TASK / "learn.json").read_text())["rounds"]
    assert len(rounds) == 1  # round 1 never ran its dev batch on the parent
    session = root / "tasks" / TASK / "worker-r1"
    assert not (session / "worker.json").exists()
    assert not (session / "candidate" / "tasks" / TASK / "skills" / "half.py").exists()

    campaign = Campaign(root, launch=FakeEpisodes(), worker_launch=improving_worker)
    campaign.resume(check=_ok_services)
    campaign.run()
    rounds = json.loads((root / "tasks" / TASK / "learn.json").read_text())["rounds"]
    assert rounds[1]["revision"] != rounds[0]["revision"] and rounds[1]["dev"]["successes"] == 1


def test_rounds_consumed_by_a_dead_worker_are_requeued(tmp_path):
    root = tmp_path / "campaign"
    init(root, make_config(tmp_path, rounds=3, run_test=False))
    # the old behaviour: a worker whose service failed was taken as "keep the parent"
    campaign = Campaign(root, launch=FakeEpisodes(), worker_launch=improving_worker)
    campaign.learn(TASK)
    session = root / "tasks" / TASK / "worker-r1"
    (session / "worker.json").write_text(
        json.dumps({"exit": 1, "is_error": True, "result": "API Error: Unable to connect to API"})
    )
    campaign.requeue(TASK)
    rounds = json.loads((root / "tasks" / TASK / "learn.json").read_text())["rounds"]
    assert len(rounds) == 1
    assert Library(root / "library" / TASK).active() == rounds[0]["revision"]
    assert list((root / "tasks" / TASK).glob("worker-r1.void-*"))
    assert list((root / "tasks" / TASK).glob("voided-r1-*.json"))


def test_environment_null_unsets_a_variable(tmp_path, monkeypatch):
    from pyrualean.learn.runner import environment

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:1")
    assert "HTTPS_PROXY" not in environment({"environment": {"HTTPS_PROXY": None}})


def test_a_round_held_by_a_refused_candidate_is_requeued_on_resume(tmp_path):
    from pyrualean.learn.campaign import WorkerFailed

    root = tmp_path / "campaign"
    init(root, make_config(tmp_path, rounds=2, run_test=False))
    failing = Campaign(root, launch=FakeEpisodes(), worker_launch=lambda s, e: {
        "exit": 1, "is_error": True, "result": "permission denied"})  # fmt: skip
    with pytest.raises(WorkerFailed):
        failing.learn(TASK)
    session = root / "tasks" / TASK / "worker-r1"
    assert (session / "refused.json").exists() and not (session / "accepted.json").exists()

    campaign = Campaign(root, launch=FakeEpisodes(), worker_launch=improving_worker)
    campaign.resume(check=_ok_services)
    assert list((root / "tasks" / TASK).glob("worker-r1.void-*"))
    campaign.learn(TASK)  # a new worker ran round 1
    rounds = json.loads((root / "tasks" / TASK / "learn.json").read_text())["rounds"]
    assert rounds[1]["revision"] != rounds[0]["revision"]


def test_a_refusal_is_not_an_outage_and_an_unrecorded_round_is_requeued(tmp_path):
    refusal = {"exit": 1, "is_error": True, "result": "API Error: Claude Code is unable to respond "
               "to this request, which appears to violate our Usage Policy"}  # fmt: skip
    outage = {
        "exit": 1,
        "is_error": True,
        "result": "API Error: Unable to connect to API (ConnectionRefused)",
    }
    assert not workers.service_error(refusal) and workers.service_error(outage)

    root = tmp_path / "campaign"
    init(root, make_config(tmp_path, rounds=2, run_test=False))
    def nothing(session_dir, env):
        (session_dir / "FINDINGS.md").write_text("nothing worth changing")
        return {"exit": 0}

    campaign = Campaign(root, launch=FakeEpisodes(), worker_launch=nothing)
    campaign.learn(TASK)  # r0 and r1 with no worker change
    state = json.loads((root / "tasks" / TASK / "learn.json").read_text())
    state["rounds"] = state["rounds"][:1]  # stopped before r1's dev batch was recorded
    (root / "tasks" / TASK / "learn.json").write_text(json.dumps(state))
    session = root / "tasks" / TASK / "worker-r1"
    (session / "worker-a0.json").write_text(json.dumps(outage))
    (session / "worker.json").write_text(json.dumps(refusal))
    campaign.requeue(TASK)
    assert not (root / "tasks" / TASK / "worker-r1").exists()
    assert list((root / "tasks" / TASK).glob("worker-r1.void-*"))


def test_a_round_that_ran_the_parent_after_a_failed_worker_is_redone(tmp_path):
    root = tmp_path / "campaign"
    init(root, make_config(tmp_path, rounds=3, run_test=False))
    campaign = Campaign(root, launch=FakeEpisodes(), worker_launch=improving_worker)
    campaign.learn(TASK)
    session = root / "tasks" / TASK / "worker-r1"  # old behaviour: refused, parent ran r1
    (session / "worker.json").write_text(json.dumps({"exit": 124, "stalled": "wall clock"}))
    (session / "accepted.json").write_text(json.dumps({"accepted": False, "revision": "x"}))
    campaign.requeue(TASK)
    rounds = json.loads((root / "tasks" / TASK / "learn.json").read_text())["rounds"]
    assert len(rounds) == 1


def test_an_attempt_interrupted_by_a_stop_is_resumed(tmp_path):
    session, library, root = _session(tmp_path)
    (session / "worker.stream.jsonl").write_text('{"type":"system","subtype":"init"}\n')
    config = configuration.load(root / "config.json")
    seen = []
    workers.run(
        config,
        session,
        launch=lambda s, e: seen.append(workers.command(config, s)[2]) or {"exit": 0},
    )
    assert "interrupted: campaign stopped" in seen[0]
    assert not (session / "worker.stream.jsonl").exists()


def test_worker_may_read_earlier_rounds_findings(tmp_path):
    session, library, root = _session(tmp_path)
    later = root / "tasks" / TASK / "worker-r3"
    spec = json.loads((session / "session.json").read_text()) | {"round": 3}
    later.mkdir()
    (later / "session.json").write_text(json.dumps(spec))
    (later / "BRIEF.md").write_text((session / "BRIEF.md").read_text())
    argv = workers.command(configuration.load(root / "config.json"), later)
    allowed = argv[argv.index("--allowedTools") + 1 : argv.index("--disallowedTools")]
    tasks = (root / "tasks" / TASK).resolve()
    assert f"Read(/{tasks / 'worker-r1' / 'FINDINGS.md'})" in allowed
    assert f"Read(/{tasks / 'worker-r2' / 'FINDINGS.md'})" in allowed
    assert not any("worker-r3/FINDINGS" in a and a.startswith("Read") for a in allowed)
    assert not any("make_kong" in a for a in allowed)


def test_brief_states_the_goal_in_robodojo_measures(tmp_path):
    session, _, _ = _session(tmp_path)
    brief = (session / "BRIEF.md").read_text()
    assert "## Goal" in brief and brief.index("## Goal") < brief.index("## Inputs")
    assert "same\n2 dev layouts" in brief and "2 or more" in brief
    assert "Expected effect" in brief
    assert "{n_dev}" not in brief and "{rollback}" not in brief


def test_mean_score_only_when_every_episode_carries_one():
    entries = [
        {"success": True, "stages": [12, 12], "score": 1.0},
        {"success": False, "stages": [2, 12], "score": 0.4},
    ]
    assert summarize(entries)["mean_score"] == 0.7
    assert (
        summarize([*entries, {"success": False, "stages": [0, 12], "score": None}])["mean_score"]
        is None
    )


def test_new_tasks_keys_and_shared_slots(tmp_path, monkeypatch):
    from pyrualean.learn.campaign import use_key
    from pyrualean.learn.slots import slot

    for task in ("organize_table", "pick_from_conveyor_by_image", "play_Xylophone"):
        assert task in configuration.TASKS
    config = configuration.load(make_config(tmp_path))
    monkeypatch.setenv("KEY_FOR_TEST", "old")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "own")
    use_key(config, "bbb")  # no worker key: the worker CLI keeps its own login
    import os

    assert os.environ["KEY_FOR_TEST"] == "bbb" and os.environ["ANTHROPIC_API_KEY"] == "own"
    env = workers.worker_environment(config, dict(os.environ))
    assert env["ANTHROPIC_API_KEY"] == "own" and "KEY_FOR_TEST" not in env
    flags = workers.isolation(config)
    assert flags[flags.index("--setting-sources") + 1] == "project,local"  # no user settings

    shared = tmp_path / "shared"
    settings = {"gpus": [0], "per_gpu": 1, "root": str(shared)}
    with slot(tmp_path / "a", settings) as gpu:
        assert gpu == 0 and (shared / "gpu0-0.lock").exists()
        assert not (tmp_path / "a" / "slots").exists()


def test_policy_and_worker_keys_from_their_own_variables(tmp_path, monkeypatch):
    import os

    from pyrualean.learn.campaign import main

    for task in ("store_laptop_and_headphones", "deposit_coin"):
        assert task in configuration.TASKS
    path = make_config(
        tmp_path,
        worker={
            "max_replays": 2, "max_replays_per_layout": 1, "api_key_env": "WORKER_KEY_FOR_TEST"
        },  # fmt: skip
    )
    root = tmp_path / "campaign"
    init(root, path)
    checked = []
    monkeypatch.setattr(
        "pyrualean.learn.campaign.check_services",
        lambda config, verbose=False: checked.append(dict(os.environ)) or [],
    )
    monkeypatch.setenv("KEY_FOR_TEST", "policy-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "the user's own")
    monkeypatch.delenv("WORKER_KEY_FOR_TEST", raising=False)
    with pytest.raises(ValueError, match="WORKER_KEY_FOR_TEST"):
        main([str(root), "check-services"])
    monkeypatch.setenv("WORKER_KEY_FOR_TEST", "worker-key")
    assert main([str(root), "check-services"]) == 0
    env = checked[-1]
    assert env["KEY_FOR_TEST"] == "policy-key" and env["ANTHROPIC_API_KEY"] == "worker-key"
    assert "WORKER_KEY_FOR_TEST" not in env  # reaches the worker only as its token
    assert "ANTHROPIC_AUTH_TOKEN" not in env  # nor does another credential of the CLI
    config = configuration.load(root / "config.json")
    worker = workers.worker_environment(config, env)
    assert worker["ANTHROPIC_API_KEY"] == "worker-key" and "KEY_FOR_TEST" not in worker

    # a gateway: the worker's key as ANTHROPIC_AUTH_TOKEN
    gateway = tmp_path / "gateway"
    worker = {
        "max_replays": 2, "max_replays_per_layout": 1,
        "api_key_env": "WORKER_KEY_FOR_TEST", "token_env": "ANTHROPIC_AUTH_TOKEN",
    }  # fmt: skip
    init(gateway, make_config(tmp_path / "g", worker=worker))
    monkeypatch.setenv("WORKER_KEY_FOR_TEST", "worker-key")
    assert main([str(gateway), "check-services"]) == 0
    assert checked[-1]["ANTHROPIC_AUTH_TOKEN"] == "worker-key"
    assert "ANTHROPIC_API_KEY" not in checked[-1]
    with pytest.raises(configuration.ConfigError, match="api_key_env"):
        configuration.load(make_config(tmp_path / "bad", worker={"api_key_env": ""}))
    with pytest.raises(configuration.ConfigError, match="token_env"):
        configuration.load(make_config(tmp_path / "bad2", worker={"token_env": "OTHER"}))


class SightedEpisodes(FakeEpisodes):
    """Also writes what a real run dir holds that tells how RoboDojo scores the task."""

    def __call__(self, output: Path, env: dict, timeout_s: float) -> int:
        code = super().__call__(output, env, timeout_s)
        run = output / "episode"
        if (run / "result.json").exists():
            result = json.loads((run / "result.json").read_text())
            result["verdict"]["blocking_stage"] = [
                {"check": "is_lift", "labels": {"label": "block0"}, "holds": False}
            ]
            result["task_language"] = "Swap the two blocks."
            (run / "result.json").write_text(json.dumps(result))
            (run / "bench" / "task").mkdir(parents=True)
            (run / "bench" / "task" / f"{TASK}.py").write_text("def run_reward(self): ...")
            (run / "steps" / "0000").mkdir(parents=True)
            (run / "steps" / "0000" / "cam_head.png").write_bytes(b"png")
        return code


#: What tells a worker how RoboDojo checks and scores a task.
SCORING_WORDS = (
    "run_reward", "get_score", "reward_manager", "func_parser", "blocking stage", "is_lift",
    "stages passed", "mean stages", "score:", "mean score", "EVERY native action",
)  # fmt: skip


def test_worker_scoring_off_shows_only_the_instruction_and_success(tmp_path, monkeypatch):
    monkeypatch.setenv("KEY_FOR_TEST", "secret")
    root = tmp_path / "campaign"
    init(root, make_config(tmp_path, rounds=2, run_test=False))
    Campaign(root, launch=SightedEpisodes(), worker_launch=improving_worker, scoring="off").run()
    session = (root / "tasks" / TASK / "worker-r1").resolve()
    assert json.loads((session / "session.json").read_text())["scoring"] == "off"
    brief, episodes = (session / "BRIEF.md").read_text(), (session / "episodes.md").read_text()
    history = (session / "history.md").read_text()
    for word in SCORING_WORDS:
        assert word not in brief and word not in episodes and word not in history, word
    assert "success: False" in episodes and "instruction: Swap the two blocks." in episodes
    assert "successes 0/2" in episodes and "Do not look for the benchmark's code" in brief
    # every run dir it names is the filtered copy in the session: frames, no benchmark tree,
    # no stages in the verdict
    named = [Path(line[len("run dir: ") :]) for line in episodes.splitlines()
             if line.startswith("run dir: ")]  # fmt: skip
    assert len(named) == 2
    for run in named:
        assert run.is_relative_to(session / "evidence")
        assert (run / "steps" / "0000" / "cam_head.png").exists()
        assert (run / "workspace" / "cell_001.py").exists() and not (run / "bench").exists()
        verdict = json.loads((run / "result.json").read_text())["verdict"]
        assert "stages_passed" not in verdict and "blocking_stage" not in verdict
    assert not list((session / "evidence").rglob("bench"))
    sources = session / "sources" / "pyrualean"
    assert (sources / "robodojo.py").exists() and (sources / "hosts").is_dir()
    assert not (sources / "learn").exists() and not (sources / "evolve").exists()
    assert str(sources / "robodojo.py") in brief

    argv = workers.command(configuration.load(root / "config.json"), session)
    allowed = argv[argv.index("--allowedTools") + 1 : argv.index("--disallowedTools")]
    denied = argv[argv.index("--disallowedTools") + 1 : argv.index("--no-session-persistence")]
    reads = [a for a in allowed if a.startswith(("Read", "Glob", "Grep"))]
    # nothing outside the session but earlier findings; RoboDojo and the raw runs denied
    assert all(str(session) in a or a.endswith("/FINDINGS.md)") for a in reads)
    assert "--add-dir" not in argv
    bench, raw = (tmp_path / "bench").resolve(), (root / "runs").resolve()
    assert f"Read(/{bench}/**)" in denied and f"Grep(/{raw}/**)" in denied
    assert not any(str(bench) in a for a in allowed)


def test_worker_scoring_off_replay_prints_and_keeps_only_success(tmp_path, monkeypatch):
    monkeypatch.setenv("KEY_FOR_TEST", "secret")
    root = tmp_path / "campaign"
    init(root, make_config(tmp_path, rounds=2, run_test=False))
    Campaign(root, launch=SightedEpisodes(), worker_launch=improving_worker, scoring="off").run()
    session = root / "tasks" / TASK / "worker-r1"
    cells = session / "replays" / "cells-a"
    cells.mkdir(parents=True)
    (cells / "cell_001.py").write_text("print(1)")
    runner = Runner(root, configuration.load(root / "config.json"), launch=SightedEpisodes())
    text = replay(session, cells, 0, runner=runner)
    assert "success: True" in text
    for word in SCORING_WORDS:
        assert word not in text, word
    used = json.loads((session / "replays.json").read_text())
    assert used[0]["entry"]["kind"] == "policy" and "stages" not in used[0]["entry"]
    shown = Path(used[0]["entry"]["run_dir"])
    assert shown.is_relative_to(session / "replays" / "00") and not (shown / "bench").exists()
    # the run itself, with its benchmark tree, is outside the session
    assert not list(session.rglob("bench"))
    assert list((root / ".replays").rglob("bench"))
    # a judged replay still counts as evidence for dev_verified
    assert workers._cites_evidence("replays/00", {"00"}, 1)
    assert re.search(r"/replays/(\d{2})/", used[0]["entry"]["run_dir"]).group(1) == "00"


def test_worker_scoring_is_fixed_at_the_first_run(tmp_path, monkeypatch):
    from pyrualean.learn.campaign import scoring_mode

    monkeypatch.setenv("KEY_FOR_TEST", "secret")
    root = tmp_path / "campaign"
    init(root, make_config(tmp_path))
    assert scoring_mode(root, None) == "on" and not (root / "worker_scoring").exists()
    assert scoring_mode(root, "off") == "off" and scoring_mode(root, None) == "off"
    with pytest.raises(ValueError, match="fixed at its first run"):
        scoring_mode(root, "on")
    # a campaign whose workers ran sighted cannot turn blind
    other = tmp_path / "other"
    init(other, make_config(tmp_path))
    Campaign(other, launch=FakeEpisodes(), worker_launch=improving_worker).run()
    with pytest.raises(ValueError, match="already ran with scoring on"):
        scoring_mode(other, "off")
    assert scoring_mode(other, "on") == "on"
