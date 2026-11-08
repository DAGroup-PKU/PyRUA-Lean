"""The cells / program tool surfaces over the fake backend."""

from __future__ import annotations

import json

import pytest

from pyrualean.arms import ArmToolkit, arm_contract


@pytest.fixture
def cells(robo, tmp_path):
    return ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws")


@pytest.fixture
def program(robo, tmp_path):
    return ArmToolkit(robo, arm="program", workspace=tmp_path / "ws", max_programs=2)


def test_tool_specs_per_arm(cells, program):
    assert [t["name"] for t in cells.get_tools_spec()] == ["python", "finish"]
    assert [t["name"] for t in program.get_tools_spec()] == ["run_program", "finish"]
    assert "at most 2" in program.get_tools_spec()[0]["description"]


def test_cells_persist_state_and_echo_trailing_expression(cells, backend):
    first = cells.execute_tool(
        "python", {"code": "x = 21\ndef twice(v):\n    return 2 * v\nprint('hi')"}
    )
    assert first.result["stdout"] == "hi\n" and first.result["moved"] is False
    assert first.images == []
    second = cells.execute_tool("python", {"code": "twice(x)"})
    assert second.result["value"] == "42"
    third = cells.execute_tool(
        "python", {"code": "robo.move_to([0.05, -0.10, 0.6]); robo.state().eef_pos"}
    )
    assert third.result["moved"] is True and third.result["env_steps"] > 0
    assert third.result["state"]["eef_pos"] == [0.05, -0.1, 0.6]
    assert len(third.images) == 2 and third.result["images"] == [
        "agentview_high.png",
        "wrist_high.png",
    ]
    assert third.result["primitive_calls"][0]["tool"] == "move_to"
    blocks = third.content_blocks
    assert blocks[0]["type"] == "text" and blocks[1]["type"] == "image"
    assert json.loads(blocks[0]["text"])["moved"] is True


def test_cells_report_exceptions_and_episode_end(cells, backend):
    boom = cells.execute_tool("python", {"code": "1/0"})
    assert "ZeroDivisionError" in boom.result["exception"]
    cells.execute_tool(
        "python",
        {"code": "robo.move_to([0.05, -0.10, 0.6]); robo.pi0_pick('pick up the black bowl')"},
    )
    cells.execute_tool(
        "python",
        {
            "code": (
                "robo.set_gripper(1); robo.move_to([0.05, 0.20, 0.55], gripper=1); robo.release()"
            )
        },
    )
    assert backend.solved()
    after = cells.execute_tool("python", {"code": "robo.move_to([0, 0, 0.68])"})
    assert after.result["episode_finished"] == "terminated" and after.result["episode_done"] is True
    done = cells.execute_tool("finish", {"status": "success", "summary": "placed"})
    assert done.result["environment_terminated"] is True and cells.finish["status"] == "success"
    assert [t["tool"] for t in cells.turns][-1] == "finish"


def test_program_runs_fresh_namespace_and_enforces_budget(program, backend, tmp_path):
    one = program.execute_tool(
        "run_program",
        {
            "code": (
                "SHARED = 1\ndef run(robo):\n    print('p1')\n"
                "    robo.move_to([0.05, -0.10, 0.6])\n"
            )
        },
    )
    assert one.result["status"] == "completed" and one.result["stdout"] == "p1\n"
    assert one.result["moved"] and one.result["programs_left"] == 1
    helper = program.workspace / "helper.py"
    helper.write_text("def go(robo):\n    return robo.state().eef_pos\n")
    two = program.execute_tool(
        "run_program",
        {
            "code": (
                "from helper import go\ndef run(robo):\n    print(go(robo))\n"
                "    print('SHARED' in globals())\n"
            )
        },
    )
    assert "(0.05, -0.1, 0.6)" in two.result["stdout"] and "False" in two.result["stdout"]
    assert two.result["programs_left"] == 0
    three = program.execute_tool("run_program", {"code": "def run(robo):\n    pass\n"})
    assert "budget exhausted" in three.result["error"]
    outside = ArmToolkit(backend and program.robo, arm="program", workspace=tmp_path / "ws2")
    assert (
        "inside the workspace"
        in outside.execute_tool("run_program", {"path": "../x.py"}).result["error"]
    )


def test_program_reports_crash_and_load_errors(program):
    crash = program.execute_tool(
        "run_program", {"code": "def run(robo):\n    raise KeyError('k')\n"}
    )
    assert crash.result["status"] == "error" and "KeyError" in crash.result["exception"]
    bad = program.execute_tool("run_program", {"code": "x = 1\n"})
    assert "callable 'run'" in bad.result["exception"]


def test_contracts_mention_budget_and_single_shot():
    single = arm_contract("program", budget_s=900, max_env_steps=10000, max_programs=1)
    assert "EXACTLY ONCE" in single and "900 s" in single
    multi = arm_contract("program", budget_s=900, max_env_steps=10000, max_programs=5)
    assert "at most 5" in multi
    cells = arm_contract("cells", budget_s=1200, max_env_steps=10000, max_programs=None)
    assert "persistent Python session" in cells and "1200 s" in cells


def test_on_demand_images_only_when_shown(robo, tmp_path):
    tk = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", images="on-demand")
    moved = tk.execute_tool("python", {"code": "robo.move_to([0.05, -0.10, 0.6])"})
    assert moved.result["moved"] is True and moved.images == [] and "images" not in moved.result
    shown = tk.execute_tool("python", {"code": "robo.show('wrist'); robo.state()"})
    assert shown.result["images"] == ["wrist_high.png"] and len(shown.images) == 1
    again = tk.execute_tool("python", {"code": "1"})
    assert again.images == []  # requests are consumed by the turn that made them
    none = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws2", images="none")
    assert none.execute_tool("python", {"code": "robo.move_to([0.05, -0.10, 0.6])"}).images == []
    assert "robo.show" in arm_contract(
        "cells", budget_s=1, max_env_steps=1, max_programs=None, images="on-demand"
    )


def test_pure_feedback_returns_only_program_output(robo, tmp_path):
    tk = ArmToolkit(
        robo, arm="cells", workspace=tmp_path / "ws", images="on-demand", feedback="pure"
    )
    out = tk.execute_tool(
        "python", {"code": "m = robo.move_to([0.05, -0.10, 0.6]); print(m.reached); m.final_dist_m"}
    )
    assert set(out.result) == {"cell", "stdout", "stderr", "elapsed_s", "value"}
    assert out.result["stdout"] == "True\n" and out.result["value"] == "0.0"
    assert out.images == []
    shown = tk.execute_tool("python", {"code": "robo.show('agentview')"})
    assert shown.result["images"] == ["agentview_high.png"] and len(shown.images) == 1
    assert tk.turns[0]["moved"] is True and tk.turns[0]["env_steps"] > 0
    text = arm_contract(
        "cells", budget_s=1, max_env_steps=1, max_programs=None, images="on-demand", feedback="pure"
    )
    assert "exactly what your code produced" in text and "robo.show" in text


def test_display_and_log_helpers(robo, tmp_path):
    pytest.importorskip("PIL")
    tk = ArmToolkit(
        robo, arm="cells", workspace=tmp_path / "ws", images="on-demand", feedback="pure"
    )
    out = tk.execute_tool(
        "python",
        {
            "code": (
                "img = robo.image('wrist', resolution='low'); display(img[:10, :10]); log('kept')"
            )
        },
    )
    assert out.result["stdout"] == "kept\n"
    assert out.result["images"] == ["display_0"] and len(out.images) == 1
    assert out.images[0][:8] == b"\x89PNG\r\n\x1a\n"


def test_arm_contract_guides_note_is_opt_in():
    from pyrualean.arms import GUIDES_NOTE

    kwargs = dict(budget_s=1200, max_env_steps=10000, max_programs=None)
    assert GUIDES_NOTE not in arm_contract("cells", **kwargs)
    with_guides = arm_contract("cells", guides=True, **kwargs)
    assert with_guides.rstrip().endswith(GUIDES_NOTE)
    assert "strict_hybrid_guide.md" in with_guides
    assert GUIDES_NOTE in arm_contract(
        "program", guides=True, max_programs=3, budget_s=900, max_env_steps=10000
    )


def test_copy_guides_copies_the_three_files_and_hashes_them(tmp_path):
    import hashlib

    from pyrualean.play import GUIDE_NAMES, copy_guides

    src = tmp_path / "guides-src"
    src.mkdir(parents=True)
    for name in GUIDE_NAMES:
        (src / name).write_text(f"# {name}\n")
    digests = copy_guides(src, tmp_path / "ws" / "guides")
    assert set(digests) == set(GUIDE_NAMES)
    assert digests["env_calibration.md"] == hashlib.sha256(b"# env_calibration.md\n").hexdigest()
    assert (
        tmp_path / "ws" / "guides" / "env_calibration.md"
    ).read_text() == "# env_calibration.md\n"
    (src / "env_calibration.md").unlink()
    import pytest

    with pytest.raises(FileNotFoundError):
        copy_guides(src, tmp_path / "ws2")


def test_arm_contract_budget_modes():
    kwargs = dict(budget_s=1200, max_env_steps=10000, max_programs=None)
    wall = arm_contract("cells", **kwargs)
    calls = arm_contract("cells", budget_mode="calls", **kwargs)
    assert "1200 s of wall clock" in wall and "1200" not in calls
    assert "number of `python` calls in an episode is limited" in calls
    prog = arm_contract("program", budget_mode="calls", max_programs=3, budget_s=9, max_env_steps=5)
    assert "number of `run_program` calls" in prog
    import pytest

    with pytest.raises(ValueError):
        arm_contract("cells", budget_mode="hours", **kwargs)


def test_python_tool_call_budget(tmp_path, robo):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path, max_turns=2, feedback="pure")
    first = kit.execute_tool("python", {"code": "print(1)"})
    second = kit.execute_tool("python", {"code": "print(2)"})
    third = kit.execute_tool("python", {"code": "print(3)"})
    assert "error" not in first.result and "error" not in second.result
    assert "call budget exhausted" in third.result["error"]
    assert kit.turn_budget_exhausted
    # further motion raises EpisodeFinished inside a cell; the tool still answers
    fourth = kit.execute_tool("python", {"code": "print(4)"})
    assert "call budget exhausted" in fourth.result["error"]
