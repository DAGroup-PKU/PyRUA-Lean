"""The learn campaign's RoboCasa365 branch offline: config, identity, the worker's sources."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyrualean.evolve.library import Library
from pyrualean.learn import config as configuration
from pyrualean.learn import worker as workers
from pyrualean.learn.campaign import init
from pyrualean.learn.episode import execution_identity

TASK = "TurnOnSinkFaucet"
SINK = """class ManipulateSinkFaucet:
    def _check_success(self):
        return self.sink.get_handle_state(env=self)["water_on"]


class TurnOnSinkFaucet(ManipulateSinkFaucet):
    pass


class TurnOffSinkFaucet(ManipulateSinkFaucet):
    pass
"""


def make_config(tmp_path: Path, **overrides) -> Path:
    packages = tmp_path / "site-packages"
    atomic = packages / "robocasa" / "environments" / "kitchen" / "atomic"
    atomic.mkdir(parents=True)
    (atomic / "kitchen_sink.py").write_text(SINK)
    (atomic / "kitchen_doors.py").write_text("class OpenCabinet:\n    pass\n")
    (packages / "robosuite").mkdir()
    (packages / "robosuite" / "x.py").write_text("x = 1\n")
    config = {
        "benchmark": "robocasa",
        "tasks": [TASK],
        "dev": {"eval_seed": 1, "layouts": [0, 1]},
        "test": {"eval_seed": 2, "layouts": [0, 1]},
        "scene_seeds": {"dev": [1, 2], "test": [7, 8]},
        "rounds": 2,
        "sim_python": "/bin/true",
        "sim_packages": str(packages),
        "robocasa_assets": str(tmp_path / "assets"),
        "sam3_checkpoint": "/dev/null",
        "slots": {"gpus": [0], "per_gpu": 2},
        "execution": {
            "model": "m",
            "base_url": "https://x/v1",
            "api_key_env": "KEY_FOR_TEST",
            "robot": "robocasa_rgb",
            "primitives": "no-vla",
            "split": "target",
            "max_episode_steps": 10000,
        },
        "budgets": {
            "max_launches": 10,
            "max_hours": 1,
            "max_infra_retries": 1,
            "episode_watchdog_seconds": 6000,
        },
    }
    config.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_a_robocasa_config_takes_robocasa_tasks_and_scene_seeds(tmp_path):
    config = configuration.load(make_config(tmp_path))
    assert configuration.robocasa(config)
    assert configuration.scene_seed(config, "dev", TASK, 1) == 2
    assert configuration.scene_seed(config, "test", TASK, 0) == 7
    override = {"dev": {TASK: [1, 6]}}
    config = configuration.load(make_config(tmp_path / "a", seed_overrides=override))
    assert configuration.scene_seeds(config, "dev", TASK) == [1, 6]
    with pytest.raises(configuration.ConfigError, match="disjoint"):
        configuration.load(make_config(tmp_path / "b", scene_seeds={"dev": [1, 2], "test": [2, 3]}))
    with pytest.raises(configuration.ConfigError, match="one distinct dev scene seed"):
        configuration.load(make_config(tmp_path / "e", seed_overrides={"dev": {TASK: [1]}}))
    execution = dict(config["execution"])
    del execution["split"]
    with pytest.raises(configuration.ConfigError, match="execution.split"):
        configuration.load(make_config(tmp_path / "c", execution=execution))
    with pytest.raises(configuration.ConfigError, match="tasks must be distinct names from"):
        configuration.load(make_config(tmp_path / "d", benchmark="robodojo", source_root="/x"))


def test_the_identity_covers_the_kitchen_sources(tmp_path):
    config = configuration.load(make_config(tmp_path))
    before = execution_identity(config)
    sink = Path(config["sim_packages"]) / "robocasa/environments/kitchen/atomic/kitchen_sink.py"
    sink.write_text(SINK + "\n# changed\n")
    assert execution_identity(config) != before


def test_the_worker_reads_its_task_file_and_no_other(tmp_path):
    path = make_config(tmp_path)
    config = configuration.load(path)
    sources = workers.scoring_sources(config, TASK)
    assert [p.name for p in sources] == ["kitchen_sink.py"]
    denied = workers.other_judges(config, TASK, tmp_path / "evidence")
    assert any(rule.endswith("kitchen_doors.py)") for rule in denied)
    assert not any(rule.endswith("kitchen_sink.py)") for rule in denied)

    root = tmp_path / "campaign"
    init(root, path)
    library = Library(root / "library" / TASK)
    empty = root / "library" / ".empty"
    empty.mkdir(parents=True)
    revision = library.freeze(empty, parent=None, note="v0")
    session = root / "tasks" / TASK / "worker-r1"
    prepare = dict(
        task=TASK, round_index=1, parent=revision, library=library, campaign=root,
        config_path=root / "config.json", config=config, episodes="", history="",
    )  # fmt: skip
    workers.prepare(session, **prepare)
    brief = (session / "BRIEF.md").read_text()
    assert "RoboCasa365 kitchen task" in brief and "kitchen_sink.py" in brief
    assert "robocasa_rgb.py" in brief and "robodojo_xpl_rgb.py" not in brief


def test_a_blind_worker_gets_the_runs_without_kitchen_records_and_no_benchmark(tmp_path):
    path = make_config(tmp_path)
    config = configuration.load(path)
    root = tmp_path / "campaign"
    init(root, path)
    run = root / "runs" / "dev" / TASK / "v0-l0-r0-a0" / "episode"
    (run / "workspace").mkdir(parents=True)
    (run / "workspace" / "cell_001.py").write_text("robo.home()\n")
    for name in ("codex_events.jsonl", "prompt.txt", "calls.jsonl", "kitchen.log", "host.json"):
        (run / name).write_text("{}\n")
    verdict = {"solved": False, "over": True, "step": 600, "step_limit": 600}
    (run / "result.json").write_text(json.dumps({"environment_success": False, "verdict": verdict}))
    library = Library(root / "library" / TASK)
    empty = root / "library" / ".empty"
    empty.mkdir(parents=True)
    revision = library.freeze(empty, parent=None, note="v0")
    session = root / "tasks" / TASK / "worker-r1"
    workers.prepare(
        session, task=TASK, round_index=1, parent=revision, library=library, campaign=root,
        config_path=root / "config.json", config=config, episodes="", history="", scoring="off",
    )  # fmt: skip
    shown = workers.evidence_path(session, run)
    assert sorted(p.name for p in shown.iterdir()) == [
        "calls.jsonl", "codex_events.jsonl", "prompt.txt", "result.json", "workspace",
    ]  # fmt: skip
    brief = (session / "BRIEF.md").read_text()
    assert "RoboCasa365 kitchen task" in brief and "You are not shown how the benchmark" in brief
    assert "kitchen_sink.py" not in brief and "`messages.json` (the model's turns" in brief
    codex = {**config, "execution": {**config["execution"], "runtime": "codex"}}
    other = root / "tasks" / TASK / "worker-r1-codex"
    workers.prepare(
        other, task=TASK, round_index=1, parent=revision, library=library, campaign=root,
        config_path=root / "config.json", config=codex, episodes="", history="", scoring="off",
    )  # fmt: skip
    assert "`codex_events.jsonl` (the model's turns" in (other / "BRIEF.md").read_text()
    argv = workers.command(config, session)
    packages = str(Path(config["sim_packages"]).resolve())
    assert f"Read(/{packages}/**)" in argv[argv.index("--disallowedTools") :]
    assert not any("kitchen_sink.py" in arg for arg in argv)
    assert "--add-dir" not in argv
