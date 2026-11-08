"""eval/libero/ offline: a fake RPent/LIBERO tree, fake episodes and fake GPUs."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libero_episode import identity, library_sha256, parse_task  # noqa: E402
from run import ConfigError, LiberoEvaluation, base, init, sign_test  # noqa: E402

SUITE = "libero_10_swap"


def tree(tmp_path: Path) -> dict[str, str]:
    """The files the config names: code, LIBERO's suite, assets, SAM3."""

    files = {
        "lean/src/pyrualean/play.py": "play",
        "rpent/robots/libero/tools.py": "tools",
        "rpent/rpent/__init__.py": "",
        f"liberopro/bddl_files/{SUITE}/a.bddl": "(:language put it there)",
        f"liberopro/init_files/{SUITE}/a.pruned_init": "init",
        "assets/scenes/s.xml": "<x/>",
        "sam3.pt": "weights",
        "libs/empty/lessons/none.md": "",
        "libs/transfer/lessons/cross.md": "## card\n",
        "libs/transfer/skills/moves.py": '"""Moves."""\n',
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    (tmp_path / "libero_config").mkdir(exist_ok=True)
    return {
        "python": "/bin/true",
        "lean": str(tmp_path / "lean"),
        "rpent_code": str(tmp_path / "rpent"),
        "liberopro": str(tmp_path / "liberopro"),
        "assets": str(tmp_path / "assets"),
        "libero_config": str(tmp_path / "libero_config"),
        "sam3_checkpoint": str(tmp_path / "sam3.pt"),
    }


def plan(tmp_path: Path, /, **overrides) -> Path:
    config = {
        "label": "test",
        "suite": SUITE,
        "task_indices": [2, 5],
        "seeds": [0, 1, 2],
        "arms": {
            "v0": {"library": str(tmp_path / "libs/empty")},
            "transfer": {"library": str(tmp_path / "libs/transfer")},
        },
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
        "execution": {
            "robot": "libero_rgb",
            "primitives": "no-vla",
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
    """Writes the result an episode process would: the transfer arm succeeds on seed 1 only."""

    def __init__(self, why: str | None = None):
        self.calls: list[dict] = []
        self.envs: list[dict] = []
        self.why = why
        self.lock = threading.Lock()

    def __call__(self, output: Path, env: dict, timeout_s: float, started) -> int:
        request = json.loads((output / "request.json").read_text())
        started(4242)
        with self.lock:
            self.calls.append(request)
            self.envs.append(env)
        arm, _, _ = parse_task(request["task"])
        seed = request["config"]["seeds"][request["layout"]]
        run = output / "episode"
        run.mkdir(parents=True)
        success = arm == "transfer" and seed == 1
        result = {
            # LIBERO: no verdict record; a failed episode still ran its predicate to the end
            "status": self.why or ("completed" if success else "call_budget"),
            "environment_success": success,
            "verdict": None,
            "generation": {"requests": 12, "input_tokens": 1000},
            "prompt": {"sha256": "p"},
            "library": {"sha256": library_sha256(Path(request["view"]))},
            "identity": request["identity"],
        }
        (run / "result.json").write_text(json.dumps(result))
        return 0


def gpus(root: Path) -> base.Gpus:
    config = json.loads((root / "config.json").read_text())
    return base.Gpus(config, free=lambda: {0: 100.0, 1: 100.0}, held=lambda: {}, poll_s=0.01)


def evaluation(root: Path, episodes: FakeEpisodes) -> LiberoEvaluation:
    return LiberoEvaluation(root, launch=episodes, gpus=gpus(root))


def test_init_freezes_libraries_and_interleaves_the_arms(tmp_path):
    root = tmp_path / "root"
    config = init(root, plan(tmp_path), sam3_hash=fake_sam3)
    assert config["order"] == [
        f"v0/{SUITE}-t2", f"transfer/{SUITE}-t2", f"v0/{SUITE}-t5", f"transfer/{SUITE}-t5",
    ]  # fmt: skip
    assert config["tasks"][f"v0/{SUITE}-t2"] == {"layouts": 3, "pool": 3}
    for name, record in config["libraries"].items():
        assert library_sha256(root / "libraries" / name) == record["sha256"]
        assert record["revision"] == parse_task(name)[0]
    assert (root / "libraries" / f"transfer/{SUITE}-t5" / "skills" / "moves.py").exists()
    assert config["identity"] == identity(config)
    with pytest.raises(ConfigError, match="initialised"):
        init(root, plan(tmp_path), sam3_hash=fake_sam3)


def test_init_refuses_what_would_not_run(tmp_path):
    with pytest.raises(ConfigError, match="sha256"):
        init(tmp_path / "a", plan(tmp_path), sam3_hash=lambda _p: "x" * 64)
    path = plan(tmp_path)
    config = json.loads(path.read_text())
    config["execution"]["robot"] = "robodojo_xpl_rgb"
    path.write_text(json.dumps(config))
    with pytest.raises(ConfigError, match="libero_rgb"):
        init(tmp_path / "b", path, sam3_hash=fake_sam3)
    with pytest.raises(ConfigError, match="no bddl_files"):
        init(tmp_path / "c", plan(tmp_path, suite="libero_goal_swap"), sam3_hash=fake_sam3)
    with pytest.raises(ConfigError, match="seeds"):
        init(tmp_path / "d", plan(tmp_path, seeds=[0, 0]), sam3_hash=fake_sam3)


def test_runs_every_arm_task_and_seed_and_pairs_them(tmp_path):
    root = tmp_path / "root"
    init(root, plan(tmp_path, seeds=[3, 4, 1]), sam3_hash=fake_sam3)
    episodes = FakeEpisodes()
    report = evaluation(root, episodes).run()
    assert len(episodes.calls) == 12
    # layout n is seed seeds[n]; every request names the copied library of its arm and task
    seen = {(c["task"], c["config"]["seeds"][c["layout"]]) for c in episodes.calls}
    assert seen == {
        (t, s) for t in json.loads((root / "config.json").read_text())["order"] for s in (3, 4, 1)
    }
    assert {c["view"] for c in episodes.calls} == {str(root / "libraries" / t) for t, _ in seen}
    env = episodes.envs[0]
    assert "CUDA_VISIBLE_DEVICES" not in env and "HTTP_PROXY" not in env
    assert env["PYTHONPATH"].split(":") == [str(tmp_path / "lean/src"), str(tmp_path / "rpent")]
    assert env["RPENT_ROOT"] == str(tmp_path / "rpent") and env["MUJOCO_GL"] == "egl"
    # failures without a verdict record are judged episodes, not infrastructure
    assert report["arms"]["v0"] == {"episodes": 6, "successes": 0, "missing": 0}
    assert report["arms"]["transfer"]["successes"] == 2
    paired = report["paired"]["transfer_vs_v0"]
    assert paired["tasks"]["t2"] == {
        "pairs": 3,
        "v0_only": 0,
        "transfer_only": 1,
        "both": 0,
        "neither": 2,
    }
    assert (paired["transfer_only"], paired["v0_only"], paired["sign_test_p"]) == (2, 0, 0.5)
    entries = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
    assert {e["kind"] for e in entries} == {"policy"}
    # a second run has nothing left to do
    again = FakeEpisodes()
    evaluation(root, again).run()
    assert again.calls == []


def test_provider_failures_pause_the_run(tmp_path):
    root = tmp_path / "root"
    init(root, plan(tmp_path, concurrency=1), sam3_hash=fake_sam3)
    episodes = FakeEpisodes(why="planner_error")
    evaluation(root, episodes).run()
    assert len(episodes.calls) == 3
    assert "provider" in json.loads((root / "PAUSED.json").read_text())["reason"]


def test_changed_code_refuses_to_run(tmp_path):
    root = tmp_path / "root"
    init(root, plan(tmp_path), sam3_hash=fake_sam3)
    (tmp_path / "rpent/robots/libero/tools.py").write_text("changed")
    with pytest.raises(RuntimeError, match="changed since init"):
        LiberoEvaluation(root, launch=FakeEpisodes(), gpus=None)


def test_helpers():
    assert parse_task(f"transfer/{SUITE}-t7") == ("transfer", SUITE, 7)
    assert sign_test(0, 0) is None and sign_test(5, 0) == 0.0625 and sign_test(3, 3) == 1.0
