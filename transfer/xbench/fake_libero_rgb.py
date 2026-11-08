"""A fake libero_rgb robot for smoke-testing translated skills: every call is bound to the real
LiberoColourRobot signature (a wrong argument raises TypeError) and returns the real result
dataclasses of pyrualean.libero_rgb (a wrong field raises); motion is ideal unless ``block_below``
stops a descent.  The libero_rgb counterpart of transfer/robodojo/fake_robodojo.py.  A smoke script
of one library imports Fake, load, check and report from here (sys.exit(report())).  It checks
signatures and result fields, not physics: a skill that passes has not been run in simulation."""

import dataclasses
import importlib.util
import inspect
import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pyrualean.libero_rgb import (  # noqa: E402
    Gripper,
    Move,
    ObjectSight,
    SeenPoint,
    Segment,
    State,
)
from pyrualean.libero_rgb import (
    LiberoColourRobot as Real,
)

TABLE_Z = 0.90


def sight(**kw):
    fields = {f.name: None for f in dataclasses.fields(ObjectSight)}
    fields.update(
        found=True,
        score=0.9,
        center_xyz=(0.05, 0.0, TABLE_Z + 0.03),
        top_z=TABLE_Z + 0.06,
        height_m=0.06,
        length_m=0.08,
        width_m=0.05,
        long_axis_deg=10.0,
        grasp_yaw=100.0,
        views=2,
        centroid=(512, 512),
        area_px=4000,
        camera="agentview",
        note=None,
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
        self.calls.append((fn.__name__, a, kw))
        return fn(self, *a, **kw)

    call.__name__ = fn.__name__
    return call


class Fake:
    CLOSE, OPEN, KEEP = Real.CLOSE, Real.OPEN, Real.KEEP
    CAMERAS = Real.CAMERAS

    def __init__(s, block_below=None, sight_kw=None, holding=True):
        s.p = [-0.10, 0.0, 1.10]
        s.ang = [90.0, 0.0, -90.0]
        s.g = 1.0
        s.cmd = Real.OPEN
        s.calls, s.block_below, s.sight_kw, s.holding = [], block_below, sight_kw or {}, holding
        s.step = 0

    def __getattr__(s, name):
        raise AttributeError(f"fake robot has no {name!r} (real has it: {hasattr(Real, name)})")

    @property
    def task(s):
        return "a task"

    @property
    def done(s):
        return False

    @bound
    def state(s, step=-1):
        p, y, r = s.ang
        return State(
            step=s.step,
            task="a task",
            tcp_pos=tuple(s.p),
            pitch=p,
            yaw=y,
            roll=r,
            eef_quat=(0.0, 1.0, 0.0, 0.0),
            gripper=s.g,
            object_names=("thing_1",),
            terminated=False,
            truncated=False,
        )

    @bound
    def move_to(s, xyz, gripper=0, pitch=None, yaw=None, roll=None):
        start = tuple(s.p)
        target = tuple(float(v) for v in xyz)
        if math.hypot(target[0] - start[0], target[1] - start[1]) > 0.30:
            # the real API: "ValueError: an argument was invalid (e.g. a planar move over 0.30 m)"
            raise ValueError("planar move over 0.30 m")
        final = list(target)
        blocked = s.block_below is not None and target[2] < s.block_below
        if blocked:
            final[2] = s.block_below
        for i, v in enumerate((pitch, yaw, roll)):
            if v is not None:
                s.ang[i] = float(v)
        if gripper != Real.KEEP:
            s.cmd = gripper
            s.g = 1.0 if gripper == Real.OPEN else (0.6 if s.holding else 0.0)
        s.p = final
        s.step += 10
        d = math.dist(final, target)
        return Move(
            target_xyz=target,
            start_xyz=start,
            final_xyz=tuple(final),
            dist_to_target_m=d,
            orientation_error_deg=0.0,
            pitch=s.ang[0],
            yaw=s.ang[1],
            roll=s.ang[2],
            reached=d < 0.01,
            note=None,
            blocked=blocked,
            steps_used=10,
            terminated=False,
            truncated=False,
        )

    @bound
    def set_gripper(s, gripper, opening=None):
        s.cmd = gripper
        closed = gripper == Real.CLOSE
        s.g = (0.6 if s.holding else 0.0) if closed else (opening if opening is not None else 1.0)
        s.step += 12
        return Gripper(
            commanded="closed" if closed else "open",
            opening=s.g,
            holding=closed and s.holding,
            width_m=(0.05 if s.holding else 0.0) if closed else None,
            steps_used=12,
            terminated=False,
            truncated=False,
        )

    @bound
    def locate(s, text_prompt, camera="agentview", min_score=0.2):
        return sight(camera=camera, **s.sight_kw)

    @bound
    def segment(s, text_prompt, camera="agentview", min_score=0.2):
        return Segment(found=True, centroid=(512, 512), area_px=4000, score=0.9, camera=camera)

    @bound
    def back_project(s, row, col, camera="agentview"):
        return SeenPoint(
            pixel=(row, col), xyz=(0.05, 0.0, TABLE_Z), depth_m=1.0, camera=camera, surface="object"
        )

    @bound
    def camera_meta(s, camera="agentview", *, step=-1):
        return {
            "camera": camera,
            "intrinsic_matrix": [[800.0, 0, 512], [0, 800.0, 512], [0, 0, 1]],
            "extrinsic_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.6], [0, 0, 0, 1]],
            "shape": [1024, 1024, 3],
        }

    @bound
    def show(s, camera="agentview"):
        return None


FAILS = []


def check(name, fn, expect=None, fake=None, **fake_kw):
    """Run fn(fake); record a failure if it raises or expect(result, fake) is false."""
    fake = fake or Fake(**fake_kw)
    try:
        out = fn(fake)
        ok = expect(out, fake) if expect else True
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {str(out)[:160]}")
        if not ok:
            FAILS.append(name)
    except Exception:
        print(f"FAIL {name}:\n" + traceback.format_exc(limit=3))
        FAILS.append(name)
    return fake


def report():
    print(f"{len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
    return 1 if FAILS else 0
