"""``--runtime replay``: saved cells run through the arm's toolkit without a model."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.replay_runtime import replay_cells, run_replay_agent


def _cells(root, *sources):
    root.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(sources, start=1):
        (root / f"cell_{index:03d}.py").write_text(source)
    return root


def test_cells_are_replayed_in_numeric_order(tmp_path):
    root = _cells(tmp_path / "src", "a = 1", "b = 2")
    (root / "cell_010.py").write_text("c = 3")
    (root / "notes.py").write_text("ignored = True")
    assert [p.name for p in replay_cells(root)] == ["cell_001.py", "cell_002.py", "cell_010.py"]


def test_an_empty_or_missing_replay_dir_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        replay_cells(tmp_path / "nowhere")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        replay_cells(tmp_path / "empty")


def test_replay_runs_the_cells_through_the_sandbox_and_ledger(robo, backend, tmp_path):
    source = _cells(
        tmp_path / "src",
        "target = [0.05, -0.10, 0.6]\nprint('planned')",
        "robo.move_to(target); print(robo.state().eef_pos)",
    )
    cells = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    generation = run_replay_agent(cells, source=source, budget_s=60, out_dir=tmp_path)
    assert generation["status"] == "completed" and generation["cells_run"] == 2
    assert generation["total_tokens"] == 0 and generation["runtime"] == "replay"
    results = json.loads((tmp_path / "replay_results.json").read_text())
    assert results[0]["stdout"] == "planned\n"
    assert results[1]["moved"] is True and "(0.05, -0.1, 0.6)" in results[1]["stdout"]
    assert "move_to" in [record.tool for record in robo._ledger.records]
    assert cells.finish["status"] == "failure" and "replayed 2 of 2" in cells.finish["summary"]


def test_replay_stops_when_the_owner_ends_the_episode(robo, tmp_path):
    source = _cells(tmp_path / "src", "x = 1", "y = 2", "z = 3")
    cells = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    asked = iter([False, True])
    generation = run_replay_agent(
        cells, source=source, budget_s=60, out_dir=tmp_path, should_stop=lambda: next(asked)
    )
    assert generation["cells_run"] == 1 and generation["status"] == "completed"


def test_replay_needs_the_cells_arm(robo, tmp_path):
    program = ArmToolkit(robo, arm="program", workspace=tmp_path / "ws")
    with pytest.raises(ValueError, match="--arm cells"):
        run_replay_agent(
            program, source=_cells(tmp_path / "s", "x = 1"), budget_s=5, out_dir=tmp_path
        )


def test_play_episode_wires_the_replay_runtime(tmp_path, monkeypatch, backend):
    for module in ("mcp", "uvicorn", "httpx"):
        pytest.importorskip(module, reason=f"the MCP server needs {module} (the dev extra has it)")
    from pyrualean import play
    from pyrualean.robots import get_adapter

    monkeypatch.chdir(tmp_path)
    card = SimpleNamespace(
        robot="libero",
        max_env_steps=600,
        task_language="put the bowl on the plate",
        images={},
        render=lambda: "# Task\n\nput the bowl on the plate",
    )
    host = SimpleNamespace(
        boot=lambda cell, out_dir, rpent_root=None, **options: (backend, []),
        dump_card=lambda toolkit, cell, out_dir: card,
        make_backend=lambda toolkit: toolkit,
    )
    adapter = SimpleNamespace(**vars(get_adapter("libero")), host_module=lambda: host)
    monkeypatch.setattr(play, "get_adapter", lambda name: adapter)
    monkeypatch.setattr(play, "resolve_rpent_root", lambda explicit=None: tmp_path)
    cell = SimpleNamespace(
        suite="libero_object_swap", task=2, seed=0, max_episode_steps=600, cuda_device=0, tag="t2s0"
    )
    source = _cells(tmp_path / "saved", "robo.move_to([0.05, -0.10, 0.6])")
    result = play.play_episode(
        cell,
        tmp_path / "out",
        arm="cells",
        model="replay",
        budget_s=30,
        max_turns=10,
        runtime="replay",
        replay_dir=str(source),
    )
    assert result["status"] == "completed" and result["host"]["runtime"] == "replay"
    assert result["host"]["replay_dir"] == str(source.resolve())
    assert result["generation"]["cells_run"] == 1 and result["stateful_calls"] >= 1
    assert (tmp_path / "out" / "workspace" / "cell_001.py").read_text() == (
        "robo.move_to([0.05, -0.10, 0.6])"
    )
    # a setup error is recorded, as for --runtime openai without --base-url
    missing = play.play_episode(cell, tmp_path / "out2", arm="cells", model="m", runtime="replay")
    assert (
        missing["error"]["type"] == "ValueError" and "--replay-dir" in missing["error"]["message"]
    )
    assert missing["environment_success"] is False and missing["generation"] is None


def test_replay_stops_at_a_cell_that_raised(robo, tmp_path):
    source = _cells(tmp_path / "src", "1/0", "print('must not run')")
    cells = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")
    generation = run_replay_agent(cells, source=source, budget_s=60, out_dir=tmp_path)
    assert generation["cells_run"] == 1
    assert generation["stopped_by"] == "exception in cell_001.py"
    assert "exception in cell_001.py" in cells.finish["summary"]
