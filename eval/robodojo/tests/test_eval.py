"""eval/ offline: a hand-made finished learn campaign, fake episodes and fake GPUs."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode import EXPANSION_HEADING, append_expansion, identity, library_sha256  # noqa: E402
from export import ExportError, export  # noqa: E402
from run import Evaluation, Gpus  # noqa: E402

TASK = "swap_blocks"
FINAL = "f" * 16
EXECUTION = {"model": "m", "base_url": "https://x/v1", "api_key_env": "KEY_FOR_TEST",
             "suite": "arx_x5", "policy_seconds": 600}  # fmt: skip


def bench(tmp_path: Path) -> Path:
    source = tmp_path / "bench"
    for part in ("env_cfg", "task", "src"):
        (source / part).mkdir(parents=True, exist_ok=True)
        (source / part / "x.txt").write_text(part)
    layouts = source / "Assets" / "Eval_Layout" / "RoboDojo" / "arx_x5" / "0"
    layouts.mkdir(parents=True)
    for n in range(7):
        (layouts / f"{TASK}_{n}.json").write_text("{}")
    return source


def campaign(tmp_path: Path) -> Path:
    """What a finished learn campaign leaves for one task (its test of ``FINAL``)."""

    root = tmp_path / "learn"
    settings = {
        "source_root": str(bench(tmp_path)),
        "sim_python": "/bin/true",
        "sam3_checkpoint": "/dev/null",
        "sam3_sha256": "s",
        "execution": EXECUTION,
        "environment": {"HTTP_PROXY": None},
    }
    config = {"tasks": [TASK], "dev": {"eval_seed": 1}, "test": {"eval_seed": 2}, **settings}
    (root / "tasks" / TASK).mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(config))
    view = root / "library" / TASK / "views" / FINAL / TASK
    (view / "skills").mkdir(parents=True)
    (view / "lessons").mkdir()
    (view / "skills" / "press.py").write_text('"""Press."""\n')
    (view / "lessons" / "press.md").write_text("## card\n")
    (root / "library" / TASK / "active.json").write_text(json.dumps({"revision": FINAL}))
    pairs = [{"layout": n, "v0": False, "final": n != 1} for n in range(3)]
    (root / "tasks" / TASK / "test.json").write_text(json.dumps({"final": FINAL, "pairs": pairs}))
    tested = identity(settings)
    ledger = []
    for n in range(3):
        run = root / "runs" / "test" / TASK / f"{FINAL}-l{n}-r0-a0"
        (run / "episode").mkdir(parents=True)
        (run / "request.json").write_text(json.dumps({"identity": tested, "config": config}))
        library = {"sha256": library_sha256(view)}
        (run / "episode" / "result.json").write_text(json.dumps({"library": library}))
        ledger.append({"kind": "policy", "split": "test", "revision": FINAL, "seconds": 60})
    (root / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ledger))
    return root


def plan(tmp_path: Path, source: Path, /, **overrides) -> Path:
    config = {
        "eval_seed": 0,
        "tasks": {TASK: {"campaign": str(source), "layouts": 4}},
        "concurrency": 2,
        "gpus": [0, 1],
        "per_gpu": 3,
        "need_gib": 40,
        "budgets": {
            "max_launches": 20,
            "max_hours": 1,
            "max_infra_retries": 1,
            "episode_watchdog_seconds": 6000,
        },
    }
    config.update(overrides)
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(config))
    return path


def campaign_without_test(tmp_path: Path) -> Path:
    """A campaign run without test: its last round left ``FINAL`` active, run on dev."""

    root = campaign(tmp_path)
    config = json.loads((root / "config.json").read_text())
    config["rounds"] = 2
    (root / "config.json").write_text(json.dumps(config))
    (root / "tasks" / TASK / "test.json").unlink()
    (root / "runs" / "test").rename(root / "runs" / "dev")
    dev = {"episodes": 3, "successes": 1}
    rounds = [
        {"round": 0, "revision": "e" * 16, "active": "e" * 16, "done": True, "dev": dev},
        {"round": 1, "revision": FINAL, "active": FINAL, "done": True, "dev": dev},
    ]
    (root / "tasks" / TASK / "learn.json").write_text(json.dumps({"rounds": rounds}))
    ledger = [{"kind": "policy", "split": "dev", "revision": FINAL, "seconds": 60}]
    (root / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ledger))
    return root


class FakeEpisodes:
    """Writes the result an episode process would; success except on layout 1."""

    def __init__(
        self,
        infra_once: set[int] | None = None,
        hold_s: float = 0.0,
        why=None,
        unstable: set[int] | None = None,
    ):
        self.calls: list[dict] = []
        self.infra_once = set(infra_once or ())
        self.unstable = set(unstable or ())
        self.hold_s, self.why = hold_s, why
        self.running = self.most = 0
        self.lock = threading.Lock()

    def __call__(self, output: Path, env: dict, timeout_s: float, started) -> int:
        request = json.loads((output / "request.json").read_text())
        started(4242)
        with self.lock:
            self.calls.append(request)
            self.running += 1
            self.most = max(self.most, self.running)
        time.sleep(self.hold_s)
        with self.lock:
            self.running -= 1
        if request["layout"] in self.infra_once:
            self.infra_once.discard(request["layout"])
            return 1  # crashed before writing a result
        success = request["layout"] != 1 and self.why is None
        view = Path(request["view"])
        run = output / "episode"
        run.mkdir(parents=True)
        if request["layout"] in self.unstable:  # what the client leaves for an unstable scene
            (run / "robodojo_eval_client.log").write_text(
                "Unstable envs: [0]\nSuccess nums: 0, Fail nums: 0, Unstable nums: 1\n"
            )
            status = {"status": "boot_error", "identity": request["identity"]}
            (run / "result.json").write_text(json.dumps(status))
            return 0
        (run / "robodojo_eval_client.log").write_text(
            "Success nums: 0, Fail nums: 1, Unstable nums: 0\n"
        )
        result = {
            "status": self.why or "completed",
            "environment_success": success,
            "verdict": {"stages_passed": 12 if success else 2, "stages_total": 12},
            "generation": {"requests": 10, "input_tokens": 1000},
            "prompt": {"sha256": "p"},
            "library": {"sha256": library_sha256(view)},
            "identity": request["identity"],
        }
        (run / "result.json").write_text(json.dumps(result))
        return 0


def gpus(config_root: Path, free=None) -> Gpus:
    config = json.loads((config_root / "config.json").read_text())
    return Gpus(config, free=free or (lambda: {0: 100.0, 1: 100.0}), held=lambda: {}, poll_s=0.01)


def test_exports_the_tested_view_and_runs_it_on_the_new_seed(tmp_path):
    source = campaign(tmp_path)
    before = {p: p.stat().st_mtime for p in source.rglob("*")}
    root = tmp_path / "seed0"
    assert export(root, plan(tmp_path, source))["episodes"] == 4
    config = json.loads((root / "config.json").read_text())
    record = config["libraries"][TASK]
    assert record["revision"] == FINAL
    assert record["source_test"] == {"eval_seed": 2, "episodes": 3, "successes": 2}
    assert config["environment"] == {"HTTP_PROXY": None}
    assert library_sha256(root / "libraries" / TASK) == record["sha256"]

    episodes = FakeEpisodes(infra_once={2}, hold_s=0.2)
    report = Evaluation(root, launch=episodes, gpus=gpus(root)).run()
    assert sorted(c["layout"] for c in episodes.calls) == [0, 1, 2, 2, 3]
    assert {c["eval_seed"] for c in episodes.calls} == {0}
    assert {c["view"] for c in episodes.calls} == {str(root / "libraries" / TASK)}
    assert episodes.most <= 2  # the concurrency limit, though six GPU places are free
    task = report["tasks"][TASK]
    assert (task["episodes"], task["missing"], task["successes"]) == (4, 0, 3)
    assert report["spend"]["infra"] == 1
    assert report["distinct_prompts"] == 1
    # the source campaign is read, never written
    assert {p: p.stat().st_mtime for p in source.rglob("*")} == before

    again = FakeEpisodes()
    assert Evaluation(root, launch=again, gpus=gpus(root)).run()["tasks"][TASK]["successes"] == 3
    assert again.calls == []


def test_infra_beyond_the_retries_is_missing_not_failed(tmp_path):
    root = tmp_path / "seed0"
    export(root, plan(tmp_path, campaign(tmp_path), concurrency=4))
    episodes = FakeEpisodes()

    def crash_l3(output, env, timeout_s, started):
        if json.loads((output / "request.json").read_text())["layout"] == 3:
            return 1
        return episodes(output, env, timeout_s, started)

    task = Evaluation(root, launch=crash_l3, gpus=gpus(root)).run()["tasks"][TASK]
    assert (task["episodes"], task["missing"], task["successes"]) == (3, 1, 2)


def test_provider_outage_pauses_without_using_up_retries(tmp_path):
    root = tmp_path / "seed0"
    export(root, plan(tmp_path, campaign(tmp_path), concurrency=1, pause_after_provider_failures=2))
    outage = FakeEpisodes(why="planner_error")
    report = Evaluation(root, launch=outage, gpus=gpus(root)).run()
    assert (root / "PAUSED.json").exists()
    assert report["overall"]["episodes"] == 0
    assert len(outage.calls) == 2  # then nothing new started
    evaluation = Evaluation(root, launch=FakeEpisodes(), gpus=gpus(root))
    evaluation.resume(check=lambda config: [])
    assert not (root / "PAUSED.json").exists()
    assert evaluation.run()["tasks"][TASK]["successes"] == 3


def test_a_gpu_takes_an_episode_only_with_room_for_what_ours_will_still_take(tmp_path):
    root = tmp_path / "seed0"
    export(root, plan(tmp_path, campaign(tmp_path)))
    free = {0: 90.0, 1: 30.0}  # GPU 1 is too full for one; GPU 0 has room for two
    pool = gpus(root, free=lambda: dict(free))
    assert pool.acquire("a") == 0
    assert pool.acquire("b") == 0  # 90 - 40 pending from a: room for one more
    took = []
    third = threading.Thread(target=lambda: took.append(pool.acquire("c")))
    third.start()
    time.sleep(0.2)
    assert took == []  # 90 - 80 pending: no room anywhere
    pool.release("a")
    third.join(timeout=2)
    assert took == [0]


def test_export_refuses_what_would_not_compare(tmp_path):
    source = campaign(tmp_path)
    with pytest.raises(ExportError, match="no official layout"):
        export(
            tmp_path / "a",
            plan(tmp_path, source, tasks={TASK: {"campaign": str(source), "layouts": 8}}),
        )
    with pytest.raises(ExportError, match="what it learned on"):
        export(tmp_path / "b", plan(tmp_path, source, eval_seed=1))
    active = source / "library" / TASK / "active.json"
    active.write_text(json.dumps({"revision": "0" * 16}))
    with pytest.raises(ExportError, match="not the one tested"):
        export(tmp_path / "c", plan(tmp_path, source))
    active.write_text(json.dumps({"revision": FINAL}))
    skill = source / "library" / TASK / "views" / FINAL / TASK / "skills" / "press.py"
    skill.write_text('"""Edited."""\n')
    with pytest.raises(ExportError, match="not what its test episodes read"):
        export(tmp_path / "d", plan(tmp_path, source))


def test_changed_settings_refuse_to_run(tmp_path):
    root = tmp_path / "seed0"
    export(root, plan(tmp_path, campaign(tmp_path)))
    config = json.loads((root / "config.json").read_text())
    config["execution"]["nudges"] = 2
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="changed since export"):
        Evaluation(root, launch=FakeEpisodes(), gpus=gpus(root))


def test_unstable_and_crashed_layouts_are_replaced_until_enough_are_judged(tmp_path):
    root = tmp_path / "seed0"
    source = campaign(tmp_path)
    tasks = {TASK: {"campaign": str(source), "layouts": 4, "pool": 7}}
    export(root, plan(tmp_path, source, tasks=tasks, concurrency=3))
    # l2 is unstable (never retried); l3 crashes on both attempts
    episodes = FakeEpisodes(unstable={2})
    crashes = []

    def launch(output, env, timeout_s, started):
        if json.loads((output / "request.json").read_text())["layout"] == 3:
            crashes.append(1)
            return 1
        return episodes(output, env, timeout_s, started)

    report = Evaluation(root, launch=launch, gpus=gpus(root)).run()
    task = report["tasks"][TASK]
    assert sorted(c["layout"] for c in episodes.calls) == [0, 1, 2, 4, 5]
    assert len(crashes) == 2  # the one retry, then the next layout
    assert task["judged_layouts"] == [0, 1, 4, 5]
    assert task["skipped_layouts"] == {"2": "unstable", "3": "no result"}
    assert (task["episodes"], task["missing"], task["successes"]) == (4, 0, 3)

    # resumed: nothing judged or skipped runs again, and no further layout is taken
    again = FakeEpisodes()
    evaluation = Evaluation(root, launch=again, gpus=gpus(root))
    evaluation.resume(check=lambda config: [])
    assert evaluation.run()["tasks"][TASK]["judged_layouts"] == [0, 1, 4, 5]
    assert again.calls == []


def test_only_runs_the_named_layouts_and_replaces_none(tmp_path):
    source = campaign(tmp_path)
    refused = (([1, 1], 7, "distinct"), ([1], 7, "exactly"), ([1, 6], 6, "pool above"))
    for n, (only, pool, match) in enumerate(refused):
        tasks = {TASK: {"campaign": str(source), "layouts": 2, "pool": pool, "only": only}}
        with pytest.raises(ExportError, match=match):
            export(tmp_path / f"refused{n}", plan(tmp_path, source, tasks=tasks))
    root = tmp_path / "seed0"
    tasks = {TASK: {"campaign": str(source), "layouts": 3, "pool": 7, "only": [5, 2, 6]}}
    export(root, plan(tmp_path, source, tasks=tasks, concurrency=1))
    # l2 is unstable: it stays missing, no other layout takes its place
    episodes = FakeEpisodes(unstable={2})
    task = Evaluation(root, launch=episodes, gpus=gpus(root)).run()["tasks"][TASK]
    assert [c["layout"] for c in episodes.calls] == [5, 2, 6]
    assert task["judged_layouts"] == [5, 6] and task["skipped_layouts"] == {"2": "unstable"}
    assert (task["episodes"], task["missing"]) == (2, 1)


def test_a_pool_too_small_leaves_the_rest_missing(tmp_path):
    root = tmp_path / "seed0"
    source = campaign(tmp_path)
    tasks = {TASK: {"campaign": str(source), "layouts": 4, "pool": 5}}
    export(root, plan(tmp_path, source, tasks=tasks))
    report = Evaluation(root, launch=FakeEpisodes(unstable={0, 2}), gpus=gpus(root)).run()
    task = report["tasks"][TASK]
    assert (task["episodes"], task["missing"]) == (3, 1)
    assert task["judged_layouts"] == [1, 3, 4]


def test_exports_the_active_revision_of_a_campaign_run_without_test(tmp_path):
    source = campaign_without_test(tmp_path)
    root = tmp_path / "seed0"
    with pytest.raises(ExportError, match="has no test"):
        export(tmp_path / "a", plan(tmp_path, source))
    export(root, plan(tmp_path, source, **{"source": "active"}))
    record = json.loads((root / "config.json").read_text())["libraries"][TASK]
    assert record["revision"] == FINAL
    assert record["source_test"] == {
        "split": "dev", "eval_seed": 1, "round": 1, "episodes": 3, "successes": 1,
    }  # fmt: skip
    assert library_sha256(root / "libraries" / TASK) == record["sha256"]
    rounds = json.loads((source / "tasks" / TASK / "learn.json").read_text())
    rounds["rounds"][-1]["active"] = "e" * 16
    (source / "tasks" / TASK / "learn.json").write_text(json.dumps(rounds))
    with pytest.raises(ExportError, match="not the last round's"):
        export(tmp_path / "b", plan(tmp_path, source, **{"source": "active"}))


def test_a_library_learned_with_other_code_runs_only_when_named(tmp_path):
    source = campaign(tmp_path)
    for run in (source / "runs" / "test" / TASK).glob("*/request.json"):
        request = json.loads(run.read_text())
        request["identity"] = "old-prompt"
        run.write_text(json.dumps(request))
    with pytest.raises(ExportError, match="identity differs"):
        export(tmp_path / "a", plan(tmp_path, source))
    with pytest.raises(ExportError, match="names no task"):
        export(tmp_path / "b", plan(tmp_path, source, allow_identity_mismatch=["other"]))
    root = tmp_path / "c"
    export(root, plan(tmp_path, source, allow_identity_mismatch=[TASK]))
    config = json.loads((root / "config.json").read_text())
    assert config["libraries"][TASK]["learned_identity"] == "old-prompt"
    assert config["identity"] != "old-prompt"
    assert Evaluation(root, launch=FakeEpisodes(), gpus=gpus(root)).run()["tasks"][TASK]["episodes"] == 4


def test_a_task_source_and_a_plan_environment_override(tmp_path):
    source = campaign_without_test(tmp_path)
    tasks = {TASK: {"campaign": str(source), "layouts": 4, "source": "active"}}
    root = tmp_path / "seed0"
    direct = {"HTTP_PROXY": None, "HTTPS_PROXY": None}
    export(root, plan(tmp_path, source, tasks=tasks, environment=direct))
    config = json.loads((root / "config.json").read_text())
    assert config["libraries"][TASK]["source_test"]["split"] == "dev"
    assert config["environment"] == direct
    with pytest.raises(ExportError, match="source is test or active"):
        bad = {TASK: {"campaign": str(source), "layouts": 4, "source": "best"}}
        export(tmp_path / "b", plan(tmp_path, source, tasks=bad))


def test_a_given_library_directory_runs_with_the_campaigns_settings(tmp_path):
    source = campaign_without_test(tmp_path)
    # round 0 ran the empty library: its dev runs carry the settings and identity
    rounds = json.loads((source / "tasks" / TASK / "learn.json").read_text())
    rounds["rounds"][0]["revision"] = FINAL
    (source / "tasks" / TASK / "learn.json").write_text(json.dumps(rounds))
    given = tmp_path / "transfer"
    (given / "lessons").mkdir(parents=True)
    (given / "lessons" / "general.md").write_text("## card\n")
    tasks = {TASK: {"campaign": str(source), "layouts": 4, "library": str(given)}}
    root = tmp_path / "seed0"
    export(root, plan(tmp_path, source, tasks=tasks))
    record = json.loads((root / "config.json").read_text())["libraries"][TASK]
    assert record["revision"] == "given" and record["library_dir"] == str(given)
    assert record["sha256"] == library_sha256(given) == library_sha256(root / "libraries" / TASK)
    episodes = FakeEpisodes()
    report = Evaluation(root, launch=episodes, gpus=gpus(root)).run()
    assert report["tasks"][TASK]["episodes"] == 4
    assert {c["view"] for c in episodes.calls} == {str(root / "libraries" / TASK)}
    with pytest.raises(ExportError, match="no library directory"):
        bad = {TASK: {"campaign": str(source), "layouts": 4, "library": str(tmp_path / "none")}}
        export(tmp_path / "b", plan(tmp_path, source, tasks=bad))


def test_a_task_no_campaign_learned_runs_with_another_task_s_settings(tmp_path):
    source = campaign_without_test(tmp_path)
    rounds = json.loads((source / "tasks" / TASK / "learn.json").read_text())
    rounds["rounds"][0]["revision"] = FINAL
    (source / "tasks" / TASK / "learn.json").write_text(json.dumps(rounds))
    other = "press_by_number"
    config = json.loads((source / "config.json").read_text())
    layouts = Path(config["source_root"], "Assets", "Eval_Layout", "RoboDojo", "arx_x5", "0")
    for n in range(4):
        (layouts / f"{other}_{n}.json").write_text("{}")
    given = tmp_path / "transfer"
    (given / "lessons").mkdir(parents=True)
    (given / "lessons" / "general.md").write_text("## card\n")
    spec = {"campaign": str(source), "layouts": 4, "library": str(given)}
    with pytest.raises(ExportError, match="is not a task of"):
        export(tmp_path / "a", plan(tmp_path, source, tasks={other: spec}))
    with pytest.raises(ExportError, match="goes with a given library"):
        bad = {other: {"campaign": str(source), "layouts": 4, "settings_task": TASK}}
        export(tmp_path / "b", plan(tmp_path, source, tasks=bad))
    root = tmp_path / "seed0"
    export(root, plan(tmp_path, source, tasks={other: {**spec, "settings_task": TASK}}))
    exported = json.loads((root / "config.json").read_text())
    assert exported["libraries"][other]["settings_task"] == TASK
    report = Evaluation(root, launch=FakeEpisodes(), gpus=gpus(root)).run()
    assert report["tasks"][other]["episodes"] == 4


def test_each_layout_s_expansion_is_frozen_and_ends_the_prompt(tmp_path, monkeypatch):
    import pyrualean.play as play

    source = campaign(tmp_path)
    notes = tmp_path / "notes"
    for n in range(4):
        (notes / f"l{n}").mkdir(parents=True)
        (notes / f"l{n}" / "expanded.txt").write_text(f"layout {n}\n")
    pattern = str(notes / "l{layout}" / "expanded.txt")
    root = tmp_path / "seed0"
    export(root, plan(tmp_path, source, tasks={TASK: {"campaign": str(source), "layouts": 4,
                                                      "expansion": pattern}}))  # fmt: skip
    frozen = json.loads((root / "config.json").read_text())["expansions"][TASK]
    assert sorted(frozen) == ["0", "1", "2", "3"] and frozen["2"]["text"] == "layout 2"
    monkeypatch.setattr(play, "build_prompt", lambda *a, **k: "contract\n\n# Task card")
    assert play.build_prompt() == "contract\n\n# Task card"
    append_expansion(frozen["2"]["text"])
    assert play.build_prompt() == f"contract\n\n# Task card\n\n{EXPANSION_HEADING}\n\nlayout 2"
    (notes / "l3" / "expanded.txt").write_text("\n")
    with pytest.raises(ExportError, match="no expansion"):
        export(tmp_path / "b", plan(tmp_path, source, tasks={TASK: {
            "campaign": str(source), "layouts": 4, "expansion": pattern}}))  # fmt: skip
    with pytest.raises(ExportError, match="with {layout}"):
        export(tmp_path / "c", plan(tmp_path, source, tasks={TASK: {
            "campaign": str(source), "layouts": 4, "expansion": str(notes)}}))  # fmt: skip


def test_the_addendum_is_frozen_and_ends_every_prompt(tmp_path, monkeypatch):
    import pyrualean.play as play

    source = campaign(tmp_path)
    text = tmp_path / "style.md"
    text.write_text("write programs\n")
    root = tmp_path / "seed0"
    export(root, plan(tmp_path, source, addendum={"path": str(text), "heading": "# How to write code"}))
    frozen = json.loads((root / "config.json").read_text())["addendum"]
    assert frozen["text"] == "write programs" and frozen["heading"] == "# How to write code"
    monkeypatch.setattr(play, "build_prompt", lambda *a, **k: "contract\n\n# Task card")
    append_expansion(frozen["text"], frozen["heading"])
    assert play.build_prompt() == "contract\n\n# Task card\n\n# How to write code\n\nwrite programs"
    with pytest.raises(ExportError, match="addendum"):
        export(tmp_path / "b", plan(tmp_path, source, addendum={"path": str(text)}))
    text.write_text("\n")
    with pytest.raises(ExportError, match="no text"):
        export(tmp_path / "c", plan(tmp_path, source, addendum={"path": str(text), "heading": "# H"}))


def test_a_learned_seed_needs_a_given_library(tmp_path):
    source = campaign(tmp_path)
    with pytest.raises(ExportError, match="goes with a given library"):
        export(tmp_path / "a", plan(tmp_path, source, eval_seed=1, allow_learned_seed=True))
