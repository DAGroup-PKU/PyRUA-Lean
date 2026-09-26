"""Plumbing check: a frozen command sequence transcribed from one successful RPent run.

libero_object_swap, task 2, seed 0 ("Pick the salad dressing and place it in
the basket").  The coordinates were copied from one of RPent's own successful
tool-calling episodes on this exact seed, so this file only proves that the
library drives the same primitives to the same outcome.  It is NOT a policy
that generalises and must never be reported as a method result; use
`pyrualean generate` for real policies.
"""


def run(robo):
    print("task:", robo.task)
    robo.move_to([-0.03, 0.22, 0.32], step_clip=0.02)
    robo.move_to([-0.13, 0.06, 0.30], step_clip=0.02)
    robo.move_to([-0.188, -0.072, 0.30], step_clip=0.012)
    pick = robo.pi0_pick(
        "pick up the salad dressing",
        max_chunks=8,
        lift_thresh=0.08,
        gripper_closed_thresh=0.06,
    )
    print("pick:", pick)
    robo.set_gripper(robo.CLOSE, steps=8)
    carry = dict(gripper=robo.CLOSE, step_clip=0.012, max_steps=100)
    robo.move_to([-0.195, -0.082, 0.30], **carry)
    robo.move_to([-0.12, 0.10, 0.30], **carry)
    robo.move_to([0.003, 0.263, 0.30], tol=0.006, **carry)
    release = robo.release(max_steps=30)
    print("release:", release)
    print("done:", robo.done)
