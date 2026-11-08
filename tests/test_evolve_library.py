"""Library revisions: validation, freezing, task views and the promotion ledger."""

from __future__ import annotations

import json

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.evolve.library import Library, LibraryError, diff, parse_lessons, validate
from pyrualean.play import copy_library

CARD = """# Grasp lessons

## grasp-001
stage: grasp
symptom: gripper closed but the charger did not lift
applies: thin object lying flat
do: approach with pitch 90 at grasp_yaw, close, lift 3 cm, re-locate before transport
status: hypothesis
evidence: dev/plug_in_charger_l0/cell_009 (observed, fix not verified)
"""

SKILL = '''"""Skills for grasping."""

LIFT_M = 0.03


def lift_and_check(robo, arm, target):
    """Lift by LIFT_M and say whether the target moved with the gripper."""
    before = robo.locate(target)
    robo.move_to([*before.center_xyz[:2], before.center_xyz[2] + LIFT_M], arm=arm)
    return robo.locate(target)
'''


def _tree(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_a_valid_library_passes(tmp_path):
    root = _tree(
        tmp_path / "lib",
        {
            "shared/lessons/grasp.md": CARD,
            "tasks/plug_in_charger/skills/grasping.py": SKILL,
        },
    )
    assert validate(root) == []
    assert parse_lessons(CARD)[0]["status"] == "hypothesis"


@pytest.mark.parametrize(
    ("name", "text", "complaint"),
    [
        ("shared/skills/run.py", "robo.move_to([0, 0, 1])\n", "would run at import"),
        ("shared/skills/run.py", "import os\nos.system('x')\n", "would run at import"),
        ("shared/skills/peek.py", "def f(g):\n    return g.__globals__\n", "__globals__"),
        ("shared/skills/bad.py", "def f(:\n", "does not parse"),
        ("shared/notes.txt", "hello", "not a skill"),
        ("shared/lessons/l.md", "## x\nstage: grasp\n", "missing"),
        ("shared/lessons/l.md", CARD.replace("hypothesis", "proven"), "status"),
        ("shared/lessons/l.md", CARD.replace("stage: grasp", "stage: dance"), "stage"),
        ("shared/lessons/l.md", CARD + CARD.split("\n", 1)[1], "duplicate"),
    ],
)
def test_validation_refuses(tmp_path, name, text, complaint):
    root = _tree(tmp_path / "lib", {name: text})
    assert any(complaint in p for p in validate(root)), validate(root)


def test_a_frozen_revision_is_read_only_content_addressed_and_tamper_evident(tmp_path):
    library = Library(tmp_path / "L")
    source = _tree(tmp_path / "src", {"shared/lessons/grasp.md": CARD})
    first = library.freeze(source, parent=None)
    assert library.freeze(source, parent=None) == first  # same content, same revision
    manifest = library.manifest(first)
    assert manifest["parent"] is None and set(manifest["files"]) == {"shared/lessons/grasp.md"}
    path = library.revisions / first / "shared/lessons/grasp.md"
    with pytest.raises(PermissionError):
        path.write_text("changed")
    path.chmod(0o644)
    path.write_text(CARD.replace("hypothesis", "accepted"))
    with pytest.raises(LibraryError, match="modified"):
        library.verify(first)


def test_freezing_an_invalid_candidate_is_refused(tmp_path):
    library = Library(tmp_path / "L")
    source = _tree(tmp_path / "src", {"shared/skills/run.py": "robo.home()\n"})
    with pytest.raises(LibraryError, match="import"):
        library.freeze(source, parent=None)
    assert not library.revisions.exists() or not any(library.revisions.iterdir())


def test_a_task_view_holds_shared_and_its_own_files_and_drops_retired_cards(tmp_path):
    library = Library(tmp_path / "L")
    retired = CARD.replace("grasp-001", "grasp-000").replace("hypothesis", "retired")
    source = _tree(
        tmp_path / "src",
        {
            "shared/lessons/grasp.md": CARD + "\n" + retired,
            "tasks/plug_in_charger/skills/grasping.py": SKILL,
            "tasks/hang_mugs_random/lessons/mugs.md": CARD.replace("grasp-001", "mug-001"),
        },
    )
    revision = library.freeze(source, parent=None)
    plug = library.view(revision, "plug_in_charger")
    assert sorted(p.relative_to(plug).as_posix() for p in plug.rglob("*.*")) == [
        "lessons/grasp.md",
        "skills/grasping.py",
    ]
    text = (plug / "lessons/grasp.md").read_text()
    assert "grasp-001" in text and "grasp-000" not in text
    mugs = library.view(revision, "hang_mugs_random")
    assert (mugs / "lessons/mugs.md").is_file() and not (mugs / "skills").exists()


def test_a_view_refuses_a_name_that_is_both_shared_and_the_tasks(tmp_path):
    library = Library(tmp_path / "L")
    source = _tree(
        tmp_path / "src",
        {
            "shared/skills/grasping.py": SKILL,
            "tasks/plug_in_charger/skills/grasping.py": SKILL.replace("0.03", "0.05"),
        },
    )
    revision = library.freeze(source, parent=None)
    with pytest.raises(LibraryError, match="both shared and"):
        library.view(revision, "plug_in_charger")


def test_an_episode_imports_the_view_it_was_given(robo, tmp_path):
    library = Library(tmp_path / "L")
    source = _tree(tmp_path / "src", {"tasks/plug_in_charger/skills/grasping.py": SKILL})
    view = library.view(library.freeze(source, parent=None), "plug_in_charger")
    workspace = tmp_path / "ws"
    record = copy_library(view, workspace / "library")
    assert "skills/grasping.py" in record["files"]
    cells = ArmToolkit(robo, arm="cells", workspace=workspace)
    out = cells.execute_tool("python", {"code": "from library.grasping import LIFT_M; LIFT_M"})
    assert out.result["value"] == "0.03"


def test_promotion_records_every_decision_and_a_tie_keeps_the_parent(tmp_path):
    library = Library(tmp_path / "L")
    base = library.freeze(_tree(tmp_path / "v0", {"shared/lessons/grasp.md": CARD}), parent=None)
    library.set_active(base)
    work = library.candidate("try1")
    (work / "shared/lessons/grasp.md").write_text(CARD.replace("3 cm", "5 cm"))
    kept = library.promote(work, gate={"accepted": False, "reason": "tie"}, evidence=["dev/x"])
    assert kept is None and library.active() == base
    work2 = library.candidate("try2")
    (work2 / "shared/lessons/grasp.md").write_text(CARD.replace("3 cm", "4 cm"))
    new = library.promote(work2, gate={"accepted": True, "reason": "better"}, evidence=["dev/y"])
    assert library.active() == new != base
    ledger = library.promotions()
    assert [e["accepted"] for e in ledger] == [False, True]
    assert ledger[1]["parent"] == base and ledger[1]["evidence"] == ["dev/y"]
    assert diff(library, base, new) == {
        "added": [],
        "removed": [],
        "changed": ["shared/lessons/grasp.md"],
    }
    assert json.loads((library.root / "active.json").read_text()) == {"revision": new}
    with pytest.raises(LibraryError, match="exists"):
        library.candidate("try1")


def test_the_root_revision_may_be_empty_and_adds_nothing_to_the_prompt(tmp_path):
    from types import SimpleNamespace

    from pyrualean.play import build_prompt

    library = Library(tmp_path / "L")
    (tmp_path / "empty").mkdir()
    base = library.freeze(tmp_path / "empty", parent=None, note="zero-shot baseline")
    assert library.manifest(base)["files"] == {}
    view = library.view(base, "plug_in_charger")
    card = SimpleNamespace(robot="robodojo_xpl_rgb", max_env_steps=400, render=lambda: "# Task")
    common = dict(budget_s=600, max_programs=None, primitives="no-vla")
    plain = build_prompt("cells", card, **common)
    with_view = build_prompt("cells", card, library=view, **common)
    from pyrualean.play import LIBRARY_NOTE

    assert "Lessons from the skill library" not in with_view
    # the empty baseline's prompt is the plain one plus the note that a library exists
    assert with_view.replace("\n\n" + LIBRARY_NOTE, "", 1) == plain
    # a later candidate that empties the library is refused
    with pytest.raises(LibraryError, match="empty library"):
        library.freeze(tmp_path / "empty", parent=base)
