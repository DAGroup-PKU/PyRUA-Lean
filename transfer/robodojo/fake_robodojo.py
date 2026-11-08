"""A fake robodojo_xpl_rgb robot for smoke-testing distilled skills: every call is bound to the real
RobodojoOfficialColourRobot signature (a wrong argument raises) and returns the real result
dataclasses (a wrong field raises); motion is ideal unless block_below stops a descent.  A smoke
script of one library imports Fake, load, check and report from here:

    sys.path.insert(0, "REPO/transfer/robodojo")
    from fake_robodojo import check, load, report
    m = load("DIR/out/skills/moves.py", "moves")
    check("place", lambda r: m.place_from_above(r, "left", 0.15, 0.05, 0.765, 0.02), lambda o: o["ok"])
    sys.exit(report())

It checks signatures and result fields, not physics: a skill that passes has not been run in
simulation."""

import dataclasses
import importlib.util
import inspect
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pyrualean.robodojo import ArmState, Gripper, Home, Move, State  # noqa: E402
from pyrualean.robodojo_xpl_rgb import (
    ObjectSight,
    SeenPoint,
    Segment,
)  # noqa: E402
from pyrualean.robodojo_xpl_rgb import (
    RobodojoOfficialColourRobot as Real,
)

K = [[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]]


def meta(origin):
    t = np.eye(4)
    t[:3, 3] = origin  # identity rotation: the camera looks straight down (-z), +y up
    return {"intrinsic_matrix": K, "extrinsic_matrix": t.tolist(), "shape": [480, 640, 3]}


def sight(**kw):
    fields = {f.name: None for f in dataclasses.fields(ObjectSight)}
    fields.update(
        found=True,
        score=0.9,
        center_xyz=(0.1, 0.0, 0.7825),
        top_z=0.80,
        width_m=0.04,
        length_m=0.06,
        height_m=0.035,
        grasp_yaw=10.0,
        views=2,
        note=None,
        camera="cam_head",
        long_axis_deg=10.0,
    )
    fields.update({k: v for k, v in kw.items() if k in fields})
    return ObjectSight(**fields)


def load(path, name):
    spec = importlib.util.spec_from_file_location(f"library.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def bound(fn):
    sig = inspect.signature(getattr(Real, fn.__name__))

    def call(self, *a, **kw):
        sig.bind(self, *a, **kw)  # raises TypeError on an argument the real API does not take
        return fn(self, *a, **kw)

    call.__name__ = fn.__name__
    return call


class Fake:
    CLOSE, OPEN, KEEP = Real.CLOSE, Real.OPEN, Real.KEEP
    LEFT, RIGHT = getattr(Real, "LEFT", "left"), getattr(Real, "RIGHT", "right")

    def __init__(s, block_below=None, sight_kw=None):
        s.p = {"left": [-0.1, 0.0, 1.0], "right": [0.1, 0.0, 1.0]}
        s.ang = {"left": [90.0, 0.0, 0.0], "right": [90.0, 0.0, 0.0]}
        s.g = {"left": 1.0, "right": 1.0}
        s.calls, s.block_below, s.sight_kw = [], block_below, sight_kw or {}
        s.metas = {}  # camera -> camera_meta; default a camera at (0, 0, 1.5) looking down

    def __getattr__(s, name):
        raise AttributeError(f"fake robot has no {name!r} (real has it: {hasattr(Real, name)})")

    def _arm(s, a):
        p, y, r = s.ang[a]
        return ArmState(
            tcp_pos=tuple(s.p[a]),
            pitch=p,
            yaw=y,
            roll=r,
            eef_pos=tuple(s.p[a]),
            eef_quat=(1, 0, 0, 0),
            gripper=s.g[a],
            joints=(0,) * 6,
        )

    @bound
    def state(s, step=-1):
        return State(step=1, task="t", left=s._arm("left"), right=s._arm("right"), truncated=False)

    @bound
    def move_to(s, xyz, arm="right", gripper=0, pitch=None, yaw=None, roll=None, path="auto"):
        s.calls.append(("move_to", arm, tuple(round(float(v), 3) for v in xyz), gripper, path))
        start, final = list(s.p[arm]), [float(v) for v in xyz]
        if s.block_below is not None and final[2] < s.block_below:
            final[2] = s.block_below
        s.p[arm] = final
        old = s.ang[arm]
        s.ang[arm] = [
            old[0] if pitch is None else pitch,
            old[1] if yaw is None else yaw,
            old[2] if roll is None else roll,
        ]
        if gripper == s.CLOSE:
            s.g[arm] = min(s.g[arm], 0.4)
        elif gripper == s.OPEN:
            s.g[arm] = 1.0
        dist = float(np.linalg.norm(np.subtract(xyz, final)))
        return Move(
            arm=arm,
            target_xyz=tuple(xyz),
            start_xyz=tuple(start),
            final_xyz=tuple(final),
            dist_to_target_m=dist,
            orientation_error_deg=0.0,
            pitch=s.ang[arm][0],
            yaw=s.ang[arm][1],
            roll=s.ang[arm][2],
            reached=dist < 0.005,
            path=path,
            ik_error=None,
            note=None,
            blocked=dist >= 0.005,
            steps_used=5,
            truncated=False,
        )

    @bound
    def set_gripper(s, arm, gripper, opening=None):
        s.calls.append(("set_gripper", arm, gripper, opening))
        s.g[arm] = 0.4 if gripper == s.CLOSE else (opening or 1.0)
        return Gripper(
            arm=arm,
            commanded="closed" if gripper == s.CLOSE else "open",
            opening=s.g[arm],
            holding=gripper == s.CLOSE,
            width_m=0.04 if gripper == s.CLOSE else None,
            steps_used=3,
            step=10,
            step_limit=700,
            truncated=False,
        )

    @bound
    def home(s, arm="both"):
        s.calls.append(("home", arm))
        return Home(arm=arm, reached=True, max_joint_error_rad=0.0, steps_used=3, truncated=False)

    @bound
    def locate(s, text_prompt, camera="cam_head", min_score=0.2):
        s.calls.append(("locate", camera))
        return sight(camera=camera, **s.sight_kw)

    @bound
    def camera_meta(s, camera="cam_head", *, step=-1):
        return s.metas.get(camera) or meta([0.0, 0.0, 1.5])

    @bound
    def segment(s, text_prompt, camera="cam_head", min_score=0.2):
        s.calls.append(("segment", camera))
        return Segment(
            found=True, centroid=(240.0, 370.0), area_px=400, score=0.9, box=None, camera=camera
        )

    @bound
    def back_project(s, row, col, camera="cam_head"):
        fields = {f.name: None for f in dataclasses.fields(SeenPoint)}
        fields.update(
            {
                k: v
                for k, v in dict(
                    xyz=(0.0, 0.0, 0.765), found=True, camera=camera, row=row, col=col
                ).items()
                if k in fields
            }
        )
        return SeenPoint(**fields)

    @bound
    def show(s, camera="cam_head"):
        s.calls.append(("show", camera))


OK = True


def pixel(origin, p):
    """(row, col) of world point p in a meta(origin) camera."""
    d = np.asarray(p) - np.asarray(origin)
    s = -d[2]
    return (240.0 - 500.0 * d[1] / s, 320.0 + 500.0 * d[0] / s)


def check(name, fn, expect=None, fake=None, **fake_kw):
    global OK
    r = (fake or Fake)(**fake_kw)
    try:
        out = fn(r)
        good = bool(expect(out)) if expect else True
        OK &= good
        print(f"{name:34s} {'OK ' if good else 'BAD'} -> {str(out)[:160]}")
        print("   calls:", r.calls[:8])
    except Exception:
        OK = False
        print(f"{name:34s} FAIL")
        traceback.print_exc(limit=4)
    return r


def report():
    print("ALL OK" if OK else "SOME FAILED")
    return 0 if OK else 1
