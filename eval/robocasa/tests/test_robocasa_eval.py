"""eval/robocasa/ offline: a fake lean tree, kitchen packages, paper runs, episodes and GPUs."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robocasa_episode import identity, library_sha256, seeds_of, split_task  # noqa: E402
from run import ConfigError, RobocasaEvaluation, base, init, sign_test  # noqa: E402

TASKS = ["OpenDrawer", "CoffeeSetupMug"]


def tree(tmp_path: Path) -> dict[str, str]:
    """The files the config names: code, the kitchen's packages, assets, SAM3."""

    files = {
        "repo/src/pyrualean/play.py": "play",
        "site/robocasa/__init__.py": "",
        "site/robocasa/models/big.xml": "<not hashed/>",
        "site/robosuite/__init__.py": "",
        "assets/objects/a.xml": "<x/>",
        "assets/fixtures/f.xml": "<x/>",
        "sam3.pt": "weights",
        "bin/sim-python": "",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return {
        "python": "/bin/true",
        "lean": str(tmp_path / "repo"),
        "sim_python": str(tmp_path / "bin/sim-python"),
        "sim_packages": str(tmp_path / "site"),
        "assets": str(tmp_path / "assets"),
        "sam3_checkpoint": str(tmp_path / "sam3.pt"),
    }


def paper(tmp_path: Path) -> Path:
    """The paper's runs: OpenDrawer succeeded on seeds 1-2, CoffeeSetupMug on seed 6 only."""

    root = tmp_path / "paper"
    if root.is_dir():
        return root
    outcomes = {("OpenDrawer", s): s <= 2 for s in (1, 2, 3)}
    outcomes.update({("CoffeeSetupMug", s): s == 6 for s in (1, 2, 4, 6)})
    for (task, seed), success in outcomes.items():
        run = root / f"rcnv1-cells-target-{task}-s{seed}"
        run.mkdir(parents=True)
        (run / "result.json").write_text(
            json.dumps({"environment_success": success, "tool_turns": 20})
        )
    (root / "rcnv1-cells-target-OpenDrawer-s9").mkdir()  # no result.json: ignored
    return root


def plan(tmp_path: Path, /, **overrides) -> Path:
    config = {
        "label": "test",
        "task_names": TASKS,
        "seeds": [1, 2, 3],
        "seed_overrides": {"CoffeeSetupMug": [1, 2, 6]},
        "concurrency": 3,
        "gpus": [0, 1],
        "per_gpu": 2,
        "need_gib": 10,
        "pause_after_provider_failures": 3,
        "budgets": {
            "max_launches": 40,
            "max_hours": 1,
            "max_infra_retries": 1,
            "episode_watchdog_seconds": 8400,
        },
        "environment": {"HTTP_PROXY": None, "CUDA_VISIBLE_DEVICES": "3"},
        "paths": tree(tmp_path),
        "sam3_sha256": "s" * 64,
        "reference": {"label": "paper no-vla", "dirs": [str(paper(tmp_path))]},
        "execution": {
            "robot": "robocasa_rgb",
            "primitives": "no-vla",
            "split": "target",
            "runtime": "codex",
            "codex_bin": "/bin/codex",
            "codex_home": "/codex-home",
            "model": "gpt-6-astra",
            "base_url": None,
            "api_key_env": None,
            "extra_body": {"reasoning_effort": "high"},
            "images": "on-demand",
            "feedback": "pure",
            "max_cells": 40,
            "max_model_requests": 80,
            "max_tokens": 8192,
            "nudges": 20,
            "provider_retries": 2,
            "policy_seconds": 7200,
            "max_episode_steps": 10000,
        },
    }
    config.update(overrides)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(config))
    return path


def fake_sam3(_path: Path) -> str:
    return "s" * 64


class FakeEpisodes:
    """Writes the result an episode process would: every seed-1 episode and CoffeeSetupMug s6
    succeed."""

    def __init__(
        self,
        provider_error: set[str] | None = None,
        why: str | None = None,
        over_then_error: set[str] | None = None,
    ):
        self.over_then_error = set(over_then_error or ())
        self.calls: list[dict] = []
        self.envs: list[dict] = []
        self.provider_error = set(provider_error or ())
        self.why = why
        self.extra_success: set[str] = set()
        self.wrong_library = False
        self.lock = threading.Lock()

    def __call__(self, output: Path, env: dict, timeout_s: float, started) -> int:
        request = json.loads((output / "request.json").read_text())
        started(4242)
        with self.lock:
            self.calls.append(request)
            self.envs.append(env)
        task = request["task"]
        seed = seeds_of(request["config"], task)[request["layout"]]
        run = output / "episode"
        run.mkdir(parents=True)
        key = f"{split_task(task)[1]}/s{seed}"  # the RoboCasa task, whatever the arm
        if key in self.provider_error:
            self.provider_error.discard(key)  # a retry gets through
            over = key in self.over_then_error
            (run / "result.json").write_text(
                json.dumps(
                    {
                        "status": "planner_error",
                        "environment_success": False,
                        "verdict": {"solved": False, "over": over, "step": 900, "step_limit": 900}
                        if over
                        else None,
                        "library": None,
                        "identity": request["identity"],
                    }
                )
            )
            return 0
        success = seed == 1 or key == "CoffeeSetupMug/s6" or key in self.extra_success
        result = {
            # RoboCasa: no verdict record; a failed episode still ran its predicate to the end
            "status": self.why or ("completed" if success else "call_budget"),
            "environment_success": success,
            "verdict": None,
            "generation": {"requests": 12, "input_tokens": 1000},
            "prompt": {"sha256": "p"},
            # an arm's episode records the library it copied (play.copy_library's hash)
            "library": {
                "sha256": "x" * 64 if self.wrong_library else library_sha256(Path(request["view"]))
            }
            if "/" in task
            else None,
            "tool_turns": 12,
            "env_steps": 300,
            "identity": request["identity"],
        }
        (run / "result.json").write_text(json.dumps(result))
        return 0


def gpus(root: Path) -> base.Gpus:
    config = json.loads((root / "config.json").read_text())
    return base.Gpus(config, free=lambda: {0: 100.0, 1: 100.0}, held=lambda: {}, poll_s=0.01)


def evaluation(root: Path, episodes: FakeEpisodes) -> RobocasaEvaluation:
    return RobocasaEvaluation(root, launch=episodes, gpus=gpus(root))


def test_init_freezes_the_tasks_seeds_and_identity(tmp_path):
    root = tmp_path / "root"
    config = init(root, plan(tmp_path), sam3_hash=fake_sam3)
    assert config["order"] == TASKS
    assert config["tasks"] == {"OpenDrawer": {"layouts": 3}, "CoffeeSetupMug": {"layouts": 3}}
    assert seeds_of(config, "CoffeeSetupMug") == [1, 2, 6] and seeds_of(config, "OpenDrawer") == [
        1, 2, 3,
    ]  # fmt: skip
    assert config["libraries"]["OpenDrawer"]["sha256"] is None
    assert config["identity"] == identity(config)
    with pytest.raises(ConfigError, match="initialised"):
        init(root, plan(tmp_path), sam3_hash=fake_sam3)


def test_the_identity_covers_kitchen_sources_but_not_its_assets(tmp_path):
    config = json.loads(plan(tmp_path).read_text())
    before = identity(config)
    (tmp_path / "site/robocasa/models/big.xml").write_text("<changed/>")
    assert identity(config) == before
    (tmp_path / "site/robocasa/__init__.py").write_text("# changed")
    assert identity(config) != before


def test_init_refuses_what_would_not_run(tmp_path):
    with pytest.raises(ConfigError, match="sha256"):
        init(tmp_path / "a", plan(tmp_path), sam3_hash=lambda _p: "x" * 64)
    path = plan(tmp_path)
    config = json.loads(path.read_text())
    config["execution"]["robot"] = "robocasa"
    path.write_text(json.dumps(config))
    with pytest.raises(ConfigError, match="robocasa_rgb"):
        init(tmp_path / "b", path, sam3_hash=fake_sam3)
    with pytest.raises(ConfigError, match="seeds"):
        init(tmp_path / "c", plan(tmp_path, seeds=[1, 1]), sam3_hash=fake_sam3)
    with pytest.raises(ConfigError, match="seed_overrides"):
        init(
            tmp_path / "d",
            plan(tmp_path, seed_overrides={"CoffeeSetupMug": [1, 6]}),
            sam3_hash=fake_sam3,
        )


def test_runs_every_task_and_seed_and_pairs_them_with_the_paper(tmp_path):
    root = tmp_path / "root"
    init(root, plan(tmp_path), sam3_hash=fake_sam3)
    episodes = FakeEpisodes()
    report = evaluation(root, episodes).run()
    assert len(episodes.calls) == 6
    seen = {(c["task"], seeds_of(c["config"], c["task"])[c["layout"]]) for c in episodes.calls}
    assert seen == {("OpenDrawer", s) for s in (1, 2, 3)} | {
        ("CoffeeSetupMug", s) for s in (1, 2, 6)
    }
    env = episodes.envs[0]
    assert "CUDA_VISIBLE_DEVICES" not in env and "HTTP_PROXY" not in env
    assert env["PYTHONPATH"] == str(tmp_path / "repo/src")
    assert env["SAM3_CHECKPOINT_PATH"] == str(tmp_path / "sam3.pt")
    assert env["HF_HOME"] == str(tmp_path / "repo/.cache/huggingface")
    report = evaluation(root, FakeEpisodes()).report()
    assert report["ours"] == {"episodes": 6, "successes": 3, "planned": 6}
    assert report["paper"]["cells"] == 6 and report["paper"]["successes"] == 3
    # ours: OD s1, CSM s1, CSM s6; paper: OD s1, OD s2, CSM s6
    assert report["paired"] == {
        "pairs": 6,
        "ours_only": 1,
        "paper_only": 1,
        "both": 2,
        "neither": 2,
        "judged_without_paper": 0,
        "sign_test_p": 1.0,
    }
    assert report["per_task"]["CoffeeSetupMug"]["paper"] == [0, 0, 1]
    entries = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
    assert {e["kind"] for e in entries} == {"policy"}
    again = FakeEpisodes()
    evaluation(root, again).run()
    assert again.calls == []


def test_only_seeds_runs_just_those_cells_and_pairs_them(tmp_path):
    root = tmp_path / "root"
    only = {"OpenDrawer": [3, 1], "CoffeeSetupMug": [6]}
    config = init(root, plan(tmp_path, only_seeds=only), sam3_hash=fake_sam3)
    assert config["tasks"] == {"OpenDrawer": {"layouts": 2}, "CoffeeSetupMug": {"layouts": 1}}
    assert seeds_of(config, "OpenDrawer") == [1, 3] and seeds_of(config, "CoffeeSetupMug") == [6]
    episodes = FakeEpisodes()
    evaluation(root, episodes).run()
    seen = {(c["task"], seeds_of(c["config"], c["task"])[c["layout"]]) for c in episodes.calls}
    assert seen == {("OpenDrawer", 1), ("OpenDrawer", 3), ("CoffeeSetupMug", 6)}
    report = evaluation(root, FakeEpisodes()).report()
    assert report["per_task"]["OpenDrawer"] == {
        **report["per_task"]["OpenDrawer"],
        "seeds": [1, 3],
        "ours": [1, 0],
        "paper": [1, 0],
    }
    assert report["ours"]["planned"] == 3 and report["paired"]["pairs"] == 3
    for bad in ({"Unknown": [1]}, {"OpenDrawer": [4]}, {"CoffeeSetupMug": [3]}, {"OpenDrawer": []}):
        with pytest.raises(ConfigError, match="only_seeds"):
            init(tmp_path / "bad", plan(tmp_path, only_seeds=bad), sam3_hash=fake_sam3)


def test_report_with_other_roots_adds_their_tasks(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    init(a, plan(tmp_path, task_names=["OpenDrawer"], seed_overrides={}), sam3_hash=fake_sam3)
    init(b, plan(tmp_path, task_names=["CoffeeSetupMug"]), sam3_hash=fake_sam3)
    evaluation(a, FakeEpisodes()).run()
    evaluation(b, FakeEpisodes()).run()
    report = evaluation(a, FakeEpisodes()).report([evaluation(b, FakeEpisodes())])
    assert report["ours"]["episodes"] == 6 and set(report["per_task"]) == set(TASKS)
    assert report["paired"]["pairs"] == 6
    with pytest.raises(RuntimeError, match="share tasks"):
        evaluation(a, FakeEpisodes()).report([evaluation(a, FakeEpisodes())])


def test_a_provider_failure_before_the_kitchen_ended_the_episode_runs_it_again(tmp_path):
    root = tmp_path / "root"
    init(root, plan(tmp_path), sam3_hash=fake_sam3)
    episodes = FakeEpisodes(provider_error={"OpenDrawer/s2"})
    evaluation(root, episodes).run()
    assert len(episodes.calls) == 7  # the failed launch, then the same seed again
    report = evaluation(root, episodes).report()
    assert report["per_task"]["OpenDrawer"]["ours"] == [1, 0, 0]
    assert report["ours"]["episodes"] == 6


def test_a_provider_failure_after_the_budget_is_spent_is_a_judged_failure(tmp_path):
    """The kitchen ended the episode (900 of 900 native actions) before the provider failed:
    the outcome is final, not a reason to run the episode again."""
    root = tmp_path / "root"
    init(root, plan(tmp_path), sam3_hash=fake_sam3)
    episodes = FakeEpisodes(provider_error={"OpenDrawer/s2"}, over_then_error={"OpenDrawer/s2"})
    evaluation(root, episodes).run()
    assert len(episodes.calls) == 6  # judged: not run again
    report = evaluation(root, episodes).report()
    assert report["per_task"]["OpenDrawer"]["ours"] == [1, 0, 0]
    assert report["ours"]["episodes"] == 6 and report["paired"]["pairs"] == 6


def test_provider_failures_pause_the_run(tmp_path):
    root = tmp_path / "root"
    init(root, plan(tmp_path, concurrency=1), sam3_hash=fake_sam3)
    episodes = FakeEpisodes(why="planner_error")
    evaluation(root, episodes).run()
    assert len(episodes.calls) == 3
    assert "provider" in json.loads((root / "PAUSED.json").read_text())["reason"]


def test_sign_test():
    assert sign_test(0, 0) is None and sign_test(2, 0) == 0.5 and sign_test(3, 3) == 1.0


def libraries(tmp_path: Path) -> dict[str, dict[str, str]]:
    """Two libraries: off10 with a lesson and a skill, on10 with a lesson only."""

    for name, files in (
        ("off10", {"lessons/general.md": "## card", "skills/moves.py": "def f():\n    pass\n"}),
        ("on10", {"lessons/cross_task.md": "## other card"}),
    ):
        for rel, text in files.items():
            path = tmp_path / "libs" / name / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    return {name: {"library": str(tmp_path / "libs" / name)} for name in ("off10", "on10")}


def test_split_task_and_seeds_of_an_arm_task(tmp_path):
    config = json.loads(plan(tmp_path).read_text())
    assert split_task("OpenDrawer") == (None, "OpenDrawer")
    assert split_task("off10/CoffeeSetupMug") == ("off10", "CoffeeSetupMug")
    assert seeds_of(config, "off10/CoffeeSetupMug") == [1, 2, 6]


def test_arms_run_every_task_once_per_arm_with_its_own_library(tmp_path):
    root = tmp_path / "root"
    config = init(root, plan(tmp_path, arms=libraries(tmp_path)), sam3_hash=fake_sam3)
    assert config["order"] == [
        "off10/OpenDrawer", "on10/OpenDrawer", "off10/CoffeeSetupMug", "on10/CoffeeSetupMug",
    ]  # fmt: skip
    assert config["tasks"]["on10/CoffeeSetupMug"] == {"layouts": 3}
    assert (root / "libraries/off10/OpenDrawer/skills/moves.py").is_file()
    assert not (root / "libraries/on10/OpenDrawer/skills").exists()
    assert config["libraries"]["on10/OpenDrawer"]["sha256"] == library_sha256(
        tmp_path / "libs/on10"
    )
    # the library is not part of the identity: a library-free root of the same settings shares it
    assert (
        config["identity"] == init(tmp_path / "v0", plan(tmp_path), sam3_hash=fake_sam3)["identity"]
    )
    episodes = FakeEpisodes()
    evaluation(root, episodes).run()
    assert len(episodes.calls) == 12
    views = {c["task"]: c["view"] for c in episodes.calls}
    assert views["on10/CoffeeSetupMug"] == str(root / "libraries/on10/CoffeeSetupMug")
    entries = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
    assert {e["arm"] for e in entries} == {"off10", "on10"}
    assert {e["seed"] for e in entries if e["task"].endswith("CoffeeSetupMug")} == {1, 2, 6}
    report = evaluation(root, FakeEpisodes()).report()
    assert report["arms"]["off10"] == {"episodes": 6, "successes": 3}
    assert report["paired_with_v0"] == {}  # no v0 root given
    assert report["per_task"]["on10/CoffeeSetupMug"]["paper"] == [0, 0, 1]


def test_arms_pair_with_the_episodes_of_a_library_free_root(tmp_path):
    a, v0 = tmp_path / "arms", tmp_path / "v0"
    init(a, plan(tmp_path, arms=libraries(tmp_path)), sam3_hash=fake_sam3)
    init(v0, plan(tmp_path), sam3_hash=fake_sam3)
    better = FakeEpisodes()
    better.extra_success = {"OpenDrawer/s2", "OpenDrawer/s3"}  # both arms win where v0 failed
    evaluation(a, better).run()
    evaluation(v0, FakeEpisodes()).run()
    report = evaluation(a, FakeEpisodes()).report([evaluation(v0, FakeEpisodes())])
    assert report["arms"]["v0"]["successes"] == 3 and report["arms"]["on10"]["successes"] == 5
    assert report["paired_with_v0"]["on10_vs_v0"] == {
        "pairs": 6, "on10_only": 2, "v0_only": 0, "both": 3, "neither": 1, "sign_test_p": 0.5,
    }  # fmt: skip
    # the arms and the library-free root share RoboCasa tasks but no evaluation task
    assert set(report["per_task"]) >= {"OpenDrawer", "off10/OpenDrawer"}


def test_an_arm_episode_that_read_another_library_stops_the_run(tmp_path):
    root = tmp_path / "root"
    init(root, plan(tmp_path, arms=libraries(tmp_path), concurrency=1), sam3_hash=fake_sam3)
    episodes = FakeEpisodes()
    episodes.wrong_library = True
    with pytest.raises(RuntimeError, match="another library"):
        evaluation(root, episodes).run()


def test_init_refuses_bad_arms(tmp_path):
    libs = libraries(tmp_path)
    for bad in (
        {},
        {"v0": libs["off10"]},
        {"off/10": libs["off10"]},
        {"x": {"library": str(tmp_path / "none")}},
    ):
        with pytest.raises(ConfigError, match="arms"):
            init(tmp_path / "bad", plan(tmp_path, arms=bad), sam3_hash=fake_sam3)
