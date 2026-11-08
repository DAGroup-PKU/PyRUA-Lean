"""Write OUT_DIR/BRIEF.md (OUT_DIR made by prep.py) from BRIEF.template.md:

    python transfer/xbench/brief.py SOURCE OUT_DIR --setting wo_ce|w_ce [--target libero|robocasa]

SOURCE is the frozen source library given to prep.py (its directory name is the library's name in
the brief); --setting says how its source campaigns ran (wo_ce: --worker-scoring off, w_ce: on),
which changes one sentence on the improvers."""

import argparse
from pathlib import Path

SIDE = {
    "wo_ce": "Its improvers saw only each episode's trajectory and whether it succeeded; they never saw "
    "the benchmark's judging code.",
    "w_ce": "Its improvers could read their own task's benchmark checking code, and what they wrote may "
    "rely on it; a rule that only describes how one task is judged does not transfer.",
}
TARGET = {
    "libero": (
        "The target is LIBERO-PRO: one Franka arm with a two-finger gripper, RGB cameras only, robot API\n"
        "`libero_rgb` without its VLA primitives, a policy of the same kind writing Python cells.",
        "libero_rgb",
    ),
    "robocasa": (
        "The target is RoboCasa365: a PandaOmron mobile manipulator in simulated kitchens, one Franka arm\n"
        "with a two-finger gripper on a mobile base with a torso lift, RGB cameras only, robot API\n"
        "`robocasa_rgb` (no VLA primitives), a policy of the same kind writing Python cells. Its tasks run\n"
        "in many kitchens and differ in their native-action budget (the official horizon of each task;\n"
        "`target/system_prompt.md` shows one task's).",
        "robocasa_rgb",
    ),
}
ap = argparse.ArgumentParser()
ap.add_argument("source", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--setting", choices=sorted(SIDE), required=True)
ap.add_argument("--target", choices=sorted(TARGET), default="libero")
a = ap.parse_args()
out = a.out.resolve()
target_sentence, target_module = TARGET[a.target]
text = (Path(__file__).resolve().parent / "BRIEF.template.md").read_text()
for key, value in (
    ("{SOURCE}", a.source.resolve().name),
    ("{SIDE_SENTENCE}", SIDE[a.setting]),
    ("{DIR}", f"{out}/"),
    ("{TARGET_SENTENCE}", target_sentence),
):
    assert text.count(key) == 1, key
    text = text.replace(key, value)
assert text.count("{TARGET_MODULE}") == 2
text = text.replace("{TARGET_MODULE}", target_module)
(out / "BRIEF.md").write_text(text)
print(f"wrote {out / 'BRIEF.md'} ({len(text)} chars)")
