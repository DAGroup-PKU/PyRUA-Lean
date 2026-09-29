"""The no-VLA primitive set: hidden in the reference, refused at call time, same knowledge."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from pyrualean import LiberoRobot, TaskCard, api_reference, knowledge_text, render_prompt
from pyrualean._robot import NO_VLA_MESSAGE, PRIMITIVE_SETS
from pyrualean.arms import ArmToolkit, robot_words
from pyrualean.play import build_prompt, play_episode
from pyrualean.prompt import _public_members, knowledge_stem, vla_hidden
from pyrualean.robocasa import RobocasaRobot
from pyrualean.robots import get_adapter
from pyrualean.robotwin import RobotwinRobot

#: Any mention of a VLA in model-facing text (Pi0.5, LingBot-VLA, RLDX-1, "VLA").
VLA_TOKENS = re.compile(r"pi0|lingbot|rldx|vla", re.IGNORECASE)
ROBOTS = ("libero", "robotwin", "robocasa")


def _card() -> TaskCard:
    return TaskCard(
        suite="libero_object_swap",
        task=2,
        seed=0,
        task_language="Pick the salad dressing and place it in the basket",
        object_names=["basket_1", "salad_dressing_1"],
        eef_pos=[-0.148, 0.0, 0.261],
        eef_quat=[1, 0, 0, 0],
        gripper_opening=0.078,
    )


def test_primitive_sets_and_the_vla_names_per_robot():
    assert PRIMITIVE_SETS == ("full", "no-vla")
    assert LiberoRobot.VLA_PRIMITIVES == ("pi0_pick", "pi0_doubled")
    assert RobotwinRobot.VLA_PRIMITIVES == ("lingbot_act",)
    assert RobocasaRobot.VLA_PRIMITIVES == ("rldx_skill", "rldx_arm")
    assert vla_hidden(LiberoRobot) == (frozenset(), frozenset())
    names, types = vla_hidden(LiberoRobot, "no-vla")
    assert names == {"pi0_pick", "pi0_doubled"} and types == {"Pick", "Contact"}
    assert vla_hidden(RobotwinRobot, "no-vla")[1] == {"VlaRun"}
    assert vla_hidden(RobocasaRobot, "no-vla")[1] == {"Skill"}
    with pytest.raises(ValueError, match="primitives"):
        vla_hidden(LiberoRobot, "some")


@pytest.mark.parametrize("robot", ROBOTS)
def test_no_vla_reference_hides_the_vla_and_nothing_else(robot):
    cls = get_adapter(robot).robot_cls
    full = api_reference(cls)
    assert api_reference(cls, primitives="full") == full
    text = api_reference(cls, primitives="no-vla")
    for name in cls.VLA_PRIMITIVES:
        assert f"robo.{name}(" in full and f"robo.{name}(" not in text
    for type_name in vla_hidden(cls, "no-vla")[1]:
        assert f"class {type_name}(" in full and f"class {type_name}(" not in text
    assert not VLA_TOKENS.search(text), sorted(set(VLA_TOKENS.findall(text)))
    for name in _public_members(cls):
        if name not in cls.VLA_PRIMITIVES:
            assert f"robo.{name}" in text, name
    result_types = tuple(getattr(cls, "RESULT_TYPES", ()) or ())
    for result in result_types:
        if result.__name__ not in vla_hidden(cls, "no-vla")[1]:
            assert f"class {result.__name__}(" in text
    assert "Exceptions:" in text and "EpisodeFinished" in text


def test_reference_edits_must_match_exactly_once():
    class Drifted(LiberoRobot):
        VLA_REFERENCE_EDITS = (("no such sentence", ""),)

    assert "no such" not in api_reference(Drifted)  # the full reference ignores the edits
    with pytest.raises(RuntimeError, match="drifted"):
        api_reference(Drifted, primitives="no-vla")


def test_hidden_primitives_are_refused_with_the_message(backend):
    robo = LiberoRobot(backend, primitives="no-vla")
    message = NO_VLA_MESSAGE.format(name="pi0_pick")
    assert message == "pi0_pick is not available in this run (no-VLA primitive set)"
    with pytest.raises(RuntimeError, match=re.escape(message)):
        robo.pi0_pick("pick up the bowl", max_chunks=3)
    with pytest.raises(RuntimeError, match="pi0_doubled is not available"):
        robo.pi0_doubled("open the drawer")
    assert [name for name, _ in backend.calls] == []  # nothing reached the backend
    records = robo._ledger.records
    assert [r.tool for r in records] == ["pi0_pick", "pi0_doubled"]
    assert records[0].ok is False and records[0].stateful is False
    assert records[0].error == f"RuntimeError: {message}"
    assert records[0].kwargs == {"args": ["pick up the bowl"], "max_chunks": 3}
    assert robo.move_to([0.05, -0.10, 0.6]).reached  # the analytic primitives still work
    assert LiberoRobot(backend).pi0_pick("pick up the bowl").chunks_used > 0  # default set
    with pytest.raises(ValueError, match="primitives"):
        LiberoRobot(backend, primitives="vla-only")


def test_the_cell_output_shows_the_refusal(backend, tmp_path):
    robo = LiberoRobot(backend, primitives="no-vla")
    pure = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", feedback="pure")
    out = pure.execute_tool("python", {"code": "robo.pi0_pick('pick up the bowl')"})
    assert (
        "RuntimeError: pi0_pick is not available in this run (no-VLA primitive set)"
        in out.result["exception"]
    )
    rich = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws2")
    out = rich.execute_tool("python", {"code": "robo.pi0_doubled('x')"})
    assert out.result["moved"] is False and "no-VLA primitive set" in out.result["exception"]
    assert out.result["primitive_calls"][-1]["tool"] == "pi0_doubled"
    assert out.result["primitive_calls"][-1]["ok"] is False


#: Lines of the no-VLA knowledge files that lost a VLA clause; every other
#: line is verbatim from the full document.
REWRITTEN = {
    "libero": {
        "- Long pushes can destabilise the simulator; keep pushes short and capped.",
        "  x/y within 0.30 m; `step_clip` 0.025 empty or box / 0.015 cans / 0.012 tall",
    },
    "robotwin": {
        "  the hold becomes uncertain or a contact cannot be interpreted.",
        "  same primitive target or the same hand-written recovery twice.",
    },
    "robocasa": {
        "- Every base motion - `navigate_to`, `move_base` - moves the arm and all",
        "  three cameras with it.  Re-localise from a fresh world map afterwards (the",
        "  arm servo recalibrates itself on its next call); never reuse pixels or",
        "  world points taken before the base moved.",
        "  grasp from the wrist image and the object moving with the gripper.",
        "  closes and lifts: a coarse fallback for simple, well-localised objects.",
        "  `Grasp.stage` names the stage that stalled.",
        "  became uncertain, the object moved unexpectedly).",
        "  step, and re-localise after any base motion or contact.",
        "- Only `state().success` (`robo.done`) proves task success; stop acting as",
        "  soon as it fires.  Primitive success is not task success.",
        "  target, out of reach (drive closer), missed or lost grasp, stuck base (back",
        "  off, new approach direction), premature release, unstable placement.",
        "  Change one meaningful variable and verify; do not repeat the same",
        "  primitive target twice.",
    },
}
DROPPED_SECTIONS = {
    "libero": "## The VLA skills",
    "robotwin": "## The VLA",
    "robocasa": "## The VLA",
}


@pytest.mark.parametrize("robot", ROBOTS)
def test_no_vla_knowledge_is_the_same_document_minus_the_vla(robot):
    assert knowledge_stem(robot) == robot and knowledge_stem(robot, "no-vla") == f"{robot}-novla"
    full = knowledge_text(robot)
    novla = knowledge_text(robot, primitives="no-vla")
    assert knowledge_text(robot, primitives="full") == full
    assert not VLA_TOKENS.search(novla), sorted(set(VLA_TOKENS.findall(novla)))
    assert DROPPED_SECTIONS[robot] in full and DROPPED_SECTIONS[robot] not in novla
    full_lines = set(full.splitlines())
    rewritten = {line for line in novla.splitlines() if line not in full_lines}
    assert rewritten == REWRITTEN[robot]
    headings = [line for line in full.splitlines() if line.startswith("#")]
    kept = [line for line in novla.splitlines() if line.startswith("#")]
    assert kept == [h for h in headings if h != DROPPED_SECTIONS[robot]]
    assert novla.startswith(full.splitlines()[0]) and len(novla) < len(full)


def test_contract_names_only_the_remaining_services():
    libero = get_adapter("libero")
    assert robot_words(libero)["svc"] == ", a frozen Pi0.5 VLA and a SAM3 segmentation service"
    words = robot_words(libero, "no-vla")
    assert words["svc"] == ", a SAM3 segmentation service"
    assert words["svc_nl"] == ", a\nSAM3 segmentation service"
    for robot in ("robotwin", "robocasa"):
        adapter = get_adapter(robot)
        assert adapter.services and adapter.services_novla == ""
        assert robot_words(adapter, "no-vla")["svc"] == ""
        assert robot_words(adapter, "no-vla")["svc_nl"] == ""


def test_prompts_without_the_vla():
    card = _card()
    kwargs = dict(
        budget_s=7200,
        max_programs=None,
        images="on-demand",
        feedback="pure",
        guides=True,
        budget_mode="calls",
    )
    full = build_prompt("cells", card, **kwargs)
    assert build_prompt("cells", card, primitives="full", **kwargs) == full
    novla = build_prompt("cells", card, primitives="no-vla", **kwargs)
    assert "frozen Pi0.5 VLA" in full and "robo.pi0_pick(" in full
    assert not VLA_TOKENS.search(novla), sorted(set(VLA_TOKENS.findall(novla)))
    assert "`LiberoRobot` bound to the live simulator, a\nSAM3 segmentation service" in novla
    assert "robo.segment(" in novla and "## Perception" in novla
    assert "strict_hybrid_guide.md" in novla and "Pick the salad dressing" in novla
    bundle = render_prompt(card, budget_s=900, primitives="no-vla")
    assert "live simulator, a SAM3 segmentation service) and calls" in bundle.system
    assert not VLA_TOKENS.search(bundle.system)
    default = render_prompt(card, budget_s=900).system
    assert default == render_prompt(card, budget_s=900, primitives="full").system
    assert "live simulator, a frozen Pi0.5 VLA and a SAM3 segmentation service)" in default


def test_play_episode_records_the_primitive_set_even_when_boot_fails(tmp_path, monkeypatch):
    from pyrualean import play

    def boot(cell, out_dir, *, rpent_root=None, **options):
        raise RuntimeError("no simulator here")

    adapter = SimpleNamespace(
        name="libero",
        robot_cls=LiberoRobot,
        knowledge="libero",
        guides_subdir="",
        guide_names=(),
        host_module=lambda: SimpleNamespace(boot=boot),
    )
    monkeypatch.setattr(play, "get_adapter", lambda name: adapter)
    cell = SimpleNamespace(
        suite="libero_object_swap", task=2, seed=0, max_episode_steps=10000, cuda_device=6
    )
    result = play_episode(cell, tmp_path / "run", arm="cells", model="m", primitives="no-vla")
    assert result["status"] == "boot_error" and result["error"]["message"] == "no simulator here"
    assert result["host"]["primitive_set"] == "no-vla"
    written = json.loads((tmp_path / "run" / "result.json").read_text())
    assert written["host"]["primitive_set"] == "no-vla"
    default = play_episode(cell, tmp_path / "run2", arm="cells", model="m")
    assert default["host"]["primitive_set"] == "full"
    with pytest.raises(ValueError, match="primitives"):
        play_episode(cell, tmp_path / "run3", arm="cells", model="m", primitives="none")
