"""Plumbing check: the primitive sequence of one successful RPent RoboCasa365 run, replayed.

OpenDrawer, target split, seed 1 ("Open the left drawer.").  One of RPent's
own tool-calling episodes on this exact cell back-projected five pixels of
the drawer handle and then solved
the task with a single rldx_skill call (26 chunks, 208 env steps, status
"success").  This file only proves that the library drives the same
primitives to the same outcome.  It is NOT a policy that generalises and
must never be reported as a method result; use `pyrualean generate` /
`pyrualean play` for real policies.
"""


def run(robo):
    print("task:", robo.task)
    start = robo.state()
    print("base:", start.base_pos, "yaw:", round(start.base_yaw, 3), "eef:", start.eef_pos)
    print("progress:", start.task_progress)
    handle = robo.back_project_batch([[203, 60], [204, 65], [203, 70], [202, 76], [205, 68]])
    print("handle median:", handle.median_xyz, "valid:", handle.valid_count)
    skill = robo.rldx_skill()
    print("vla:", skill)
    print("done:", robo.done, "progress:", robo.state().task_progress)
