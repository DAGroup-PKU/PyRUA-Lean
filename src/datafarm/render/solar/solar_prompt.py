#!/usr/bin/env python3
"""SolarRenderer prompt for a T-layout white carrier: the six-part form of SolarRenderer's caption skill.

    SOLAR_PYTHON solar_prompt.py FREE.json OUT.txt [--duration SECONDS]   # assemble + check; OK or the problems, exit 1
    SOLAR_PYTHON solar_prompt.py --generic "dual-arm AgileX Piper robot" OUT.txt [--duration SECONDS]

FREE.json holds the free parts {"summary", "detailed_description", "overall_soundscape", "non_diegetic_music"} and
optionally "structure_objects": one parenthesis-free phrase naming how the carrier marks the task objects (e.g. "a red
rounded apple, and a pale grey kitchen scale whose display is a black rectangle when off and a flat light blue rectangle
when lit"). The structure sentence for the three-camera white carrier is fixed here; the first frame is the real frame 0
in the same layout (caption input kind "full"). --generic writes a task-neutral prompt for a robot at a table.

Write the free parts state-neutral: describe appearance (materials, colours, light, the robot), say that every drawer,
lid, door and object is wherever the structure shows it, and never narrate task progress ("puts toys into the open
drawers" made Solar open a drawer ~500 frames early). The same prompt serves every window of a chain. An appearance that
changes during the episode is described in terms of what the carrier shows in each state.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def condition(objects: str | None = None) -> tuple[str, str]:
    """(what the structure video is, how closely it is followed) for the three-camera white carrier."""
    marks = (f", while each small task object carries a simple identifying shape and flat colour ({objects.strip().rstrip('.')})"
             if objects and objects.strip() else ", while small task objects may carry a simple identifying shape and flat colour")
    what = ("a simple simulator rendering of the scene, from the same cameras as the target video and in the same "
            "three-camera composite: the table, furniture, walls and robot arms are untextured light grey" + marks +
            "; it gives the layout, the geometry, the robot motion, every object's position and identity, and the camera "
            "of every frame")
    kept = ("(frame-by-frame layout, geometry, robot and object motion, object identity, camera): attribute_transfer - the "
            "layout, shapes, motion and camera are followed exactly, every drawer, lid and door is open or closed and every "
            "object is placed exactly as {v} shows at that moment, and every object keeps the identity its shape and "
            "colour mark; real materials, textures, fine colours and lighting are the target video's own; the flat "
            "untextured look of {v} is not shown")
    return what, kept


GENERIC_FREE = {
    "summary": ("A {robot} works at a table with the objects of its task, moving exactly as the structure shows. The "
                "frame is a three-camera composite: the head camera is fixed and the two wrist cameras ride on the "
                "grippers."),
    "detailed_description": (
        "The target video is a realistic robot-camera recording of a real indoor work area, with the soft, even light, "
        "natural low-gloss surfaces, slight wide-angle perspective and modest sensor grain of small robot cameras. "
        "Every frame is the same synchronized three-camera composite: a wide head-camera view fills the top of the "
        "picture and two smaller wrist-camera views sit side by side beneath it, the left-wrist camera at bottom left "
        "and the right-wrist camera at bottom right, with plain black bars down the left and right edges; the three "
        "panels keep their places and sizes and always show the same moment. [Shot 1] The scene is exactly the real "
        "work area of the first frame: the same table and its surface, the same furniture, walls and background, the "
        "same objects in the same places, the same robot with the same link colours, joints, grippers and cables, and "
        "the same light, colour balance and shadows. Throughout the clip the robot arms move exactly as the structure "
        "shows, at times reaching, grasping, lifting, carrying and setting down an object, and the grippers open and "
        "close with it. Every object is wherever the structure shows it at that moment and keeps its own shape, size, "
        "colour, material and texture in every frame; an object moves only while a gripper holds or pushes it, "
        "nothing is duplicated, invented or removed, and any part of the table that has been cleared stays bare. "
        "Each drawer, lid or door is open or closed exactly as the structure shows at that moment, and none moves "
        "unless a gripper moves it. The wrist views show close-ups of the grippers and of the objects near them, with "
        "the same materials and colours as the head view, and they move with the grippers. The head camera never "
        "moves. Surfaces keep their real matte or glossy look, edges and seams stay sharp, small printed details and "
        "labels stay legible where they are visible, and the shadows of the arms and objects fall softly on the table "
        "under the same indoor light. The robot links keep their real colour and finish throughout, and nothing of the "
        "flat grey simulator look appears anywhere in any panel."),
    "overall_soundscape": ("Quiet indoor room tone with the soft whirr of the robot joint motors, light clicks of the "
                           "grippers and gentle taps and scrapes as objects are picked up and set down."),
    "non_diegetic_music": "N/A",
}


def generic_free(robot: str) -> dict:
    return {k: v.replace("{robot}", robot) for k, v in GENERIC_FREE.items()}


def build(free: dict, duration: float = 600.0) -> tuple[str, list]:
    """(prompt text, problems); empty problems = usable. Needs SOLAR_REPO (the caption skill's check and assemble)."""
    import solar_paths
    skill = solar_paths.caption_skill()
    if str(skill) not in sys.path:
        sys.path.insert(0, str(skill))
    from check import check_free, check_prompt
    from conditions import assemble
    f, c = check_free(free, duration)
    probs = []
    if not c["pass"]:
        probs = [f"free parts: {k}" for k in ("fields", "length", "english", "no_labels", "cuts") if not c[k]]
        if not c["length"]:
            probs.append(f"detailed_description words {c['words']}")
    text = assemble(f, "full", condition(free.get("structure_objects")))
    probs += check_prompt(text, duration=duration)
    return text, probs


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("free", nargs="?", help="FREE.json")
    ap.add_argument("out")
    ap.add_argument("--generic", metavar="ROBOT", help="write the task-neutral prompt for this robot instead of FREE.json")
    ap.add_argument("--duration", type=float, default=600.0, help="clip seconds the caption check assumes")
    a = ap.parse_args(argv)
    if bool(a.free) == bool(a.generic):
        ap.error("give FREE.json or --generic ROBOT")
    free = generic_free(a.generic) if a.generic else json.loads(Path(a.free).read_text())
    text, probs = build(free, a.duration)
    if probs:
        print("PROBLEMS: " + "; ".join(probs))
        return 1
    Path(a.out).write_text(text)
    print("OK", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
