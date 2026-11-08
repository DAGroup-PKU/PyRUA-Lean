"""The actor's prompt of the paper, frozen: any change to the robots' API reference, knowledge or
contract changes these hashes (and would make results incomparable with the paper's)."""

import hashlib

import pytest

from pyrualean.play import build_prompt
from pyrualean.prompt import TaskCard
from pyrualean.robots import get_adapter

#: sha256 prefix and length of the prompt for a fixed card, per robot and primitive set:
#: {(images, feedback): ...}; the paper ran ("on-demand", "pure").
PAPER = {
    ("robodojo_xpl_rgb", "no-vla"): {("on-demand", "pure"): ("53e4f8a548a74bea", 48154),
                                     ("on-motion", "rich"): ("e6225b35fd6002d8", 47777)},
    ("libero_rgb", "no-vla"): {("on-demand", "pure"): ("263f81141e8b4dba", 32714),
                               ("on-motion", "rich"): ("b4fdf4d6e5063e64", 32377)},
    ("robocasa_rgb", "no-vla"): {("on-demand", "pure"): ("71ef5d9dc634b668", 40876),
                                 ("on-motion", "rich"): ("ff626a48a254c1d5", 40496)},
}  # fmt: skip


@pytest.mark.parametrize("robot,primitives", sorted(PAPER))
def test_the_paper_prompt_is_unchanged(robot, primitives):
    card = TaskCard(
        suite="s", task="t", seed=0, task_language="Do the task.", object_names=["a", "b"],
        robot=robot, facts={"arm": "x"}, max_env_steps=1234,
    )  # fmt: skip
    for (images, feedback), (prefix, length) in PAPER[(robot, primitives)].items():
        text = build_prompt(
            "cells", card, budget_s=5400, max_programs=100, images=images, feedback=feedback,
            adapter=get_adapter(robot), primitives=primitives,
        )  # fmt: skip
        assert (hashlib.sha256(text.encode()).hexdigest()[:16], len(text)) == (prefix, length)
