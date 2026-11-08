"""A fake robocasa_rgb robot for smoke-testing translated skills: every call is bound to the real
RobocasaColourRobot signature (a wrong argument raises TypeError, an arm other than "right" raises
ValueError as the real one does) and returns the real result dataclasses of pyrualean.robocasa_rgb
(a wrong field raises); motion is ideal unless ``block_below`` stops a descent.  The robocasa_rgb
counterpart of fake_libero_rgb.py.  A smoke script of one library imports Fake, load, check and
report from here (sys.exit(report())).  It checks signatures and result fields, not physics: a
skill that passes has not been run in simulation."""

import dataclasses
import importlib.util
import inspect
import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pyrualean.robocasa_rgb import (  # noqa: E402
    ArmState,
    BaseMove,
    Gripper,
    Home,
    Move,
    ObjectSight,
    SeenPoint,
    Segment,
    State,
    Torso,
)
from pyrualean.robocasa_rgb import (
    RobocasaColourRobot as Real,
)

COUNTER_Z = 0.92  # a kitchen counter top in the robot frame (z up from the floor)


def sight(**kw):
    fields = {f.name: None for f in dataclasses.fields(ObjectSight)}
    fields.update(
        found=True,
        score=0.9,
        center_xyz=(0.0, 0.55, COUNTER_Z + 0.03),
        top_z=COUNTER_Z + 0.06,
        height_m=0.06,
        length_m=0.08,
        width_m=0.05,
        long_axis_deg=10.0,
        grasp_yaw=100.0,
        views=2,
        centroid=(240, 320),
        area_px=4000,
        camera="agentview_left",
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
        args = sig.bind(
            self, *a, **kw
        )  # raises TypeError on an argument the real API does not take
        arm = args.arguments.get("arm", "right")
        if arm != "right":
            raise ValueError(f"this robot has one arm, 'right'; got {arm!r}")
        cam = args.arguments.get("camera")
        if cam is not None and cam not in Real.CAMERAS:
            raise ValueError(f"camera must be one of {Real.CAMERAS}, got {cam!r}")
        self.calls.append((fn.__name__, a, kw))
        return fn(self, *a, **kw)

    call.__name__ = fn.__name__
    return call


class Fake:
    CLOSE, OPEN, KEEP, RIGHT = Real.CLOSE, Real.OPEN, Real.KEEP, Real.RIGHT
    CAMERAS = Real.CAMERAS
    ARMS = Real.ARMS

    def __init__(s, block_below=None, sight_kw=None, holding=True):
        s.p = [0.0, 0.35, 1.10]
        s.ang = [90.0, 0.0, 0.0]
        s.g = 0.5
        s.cmd = Real.OPEN
        s.base = [0.0, 0.0, 0.0]
        s.torso = 0.0
        s.calls, s.block_below, s.sight_kw, s.holding = [], block_below, sight_kw or {}, holding
        s.step = 0

    def __getattr__(s, name):
        absent = name in Real.ABSENT_MEMBERS
        raise AttributeError(
            f"fake robot has no {name!r} (real has it: {hasattr(Real, name) and not absent})"
        )

    @property
    def task(s):
        return "a task"

    def _arm(s):
        p, y, r = s.ang
        return ArmState(
            tcp_pos=tuple(s.p),
            pitch=p,
            yaw=y,
            roll=r,
            eef_pos=(s.p[0], s.p[1], s.p[2] + 0.1),
            eef_quat=(0.0, 1.0, 0.0, 0.0),
            gripper=s.g,
            joints=(0.0,) * 7,
        )

    @bound
    def state(s, step=-1):
        return State(
            step=s.step,
            task="a task",
            right=s._arm(),
            base=tuple(s.base),
            torso=s.torso,
            truncated=False,
        )

    @bound
    def move_to(s, xyz, arm="right", gripper=0, pitch=None, yaw=None, roll=None, path="auto"):
        if path not in ("auto", "line", "joint"):
            raise ValueError(f"path must be 'auto', 'line' or 'joint', got {path!r}")
        start = tuple(s.p)
        target = tuple(float(v) for v in xyz)
        final = list(target)
        blocked = s.block_below is not None and target[2] < s.block_below
        if blocked:
            final[2] = s.block_below
        for i, v in enumerate((pitch, yaw, roll)):
            if v is not None:
                s.ang[i] = float(v)
        if gripper != Real.KEEP:
            s.cmd = gripper
            s.g = 0.97 if gripper == Real.OPEN else (0.6 if s.holding else 0.0)
        s.p = final
        s.step += 1
        d = math.dist(final, target)
        return Move(
            arm="right",
            target_xyz=target,
            start_xyz=start,
            final_xyz=tuple(final),
            dist_to_target_m=d,
            orientation_error_deg=0.0,
            pitch=s.ang[0],
            yaw=s.ang[1],
            roll=s.ang[2],
            reached=d < 0.01,
            path="line" if path != "joint" else "joint",
            ik_error=None,
            note=None,
            blocked=blocked,
            steps_used=10,
            truncated=False,
        )

    @bound
    def set_gripper(s, arm, gripper, opening=None):
        if opening is not None and (gripper != Real.OPEN or not 0 < opening <= 1):
            raise ValueError("opening goes with robo.OPEN and must be in (0, 1]")
        s.cmd = gripper
        closed = gripper == Real.CLOSE
        s.g = (0.6 if s.holding else 0.0) if closed else (opening if opening is not None else 0.97)
        s.step += 1
        return Gripper(
            arm="right",
            commanded="closed" if closed else "open",
            opening=s.g,
            holding=closed and s.holding,
            width_m=(0.05 if s.holding else 0.0) if closed else None,
            steps_used=12,
            step=s.step,
            step_limit=600,
            truncated=False,
        )

    @bound
    def home(s, arm="right"):
        s.p = [0.0, 0.35, 1.10]
        s.ang = [90.0, 0.0, 0.0]
        s.step += 1
        return Home(
            arm="right", reached=True, max_joint_error_rad=0.0, steps_used=30, truncated=False
        )

    @bound
    def move_base(s, dx=0.0, dy=0.0, dyaw=0.0):
        s.step += 1
        return BaseMove(
            moved=(float(dx), float(dy)),
            turned_deg=float(dyaw),
            dist_to_target_m=0.0,
            yaw_error_deg=0.0,
            reached=True,
            blocked=False,
            steps_used=20,
            truncated=False,
        )

    @bound
    def set_torso(s, height):
        s.torso = float(height)
        s.step += 1
        return Torso(height_m=s.torso, reached=True, steps_used=10, truncated=False)

    @bound
    def locate(s, text_prompt, camera="agentview_left", min_score=0.2):
        return sight(camera=camera, **s.sight_kw)

    @bound
    def segment(s, text_prompt, camera="agentview_left", min_score=0.2):
        return Segment(
            found=True, centroid=(240, 320), area_px=4000, score=0.9, box=None, camera=camera
        )

    @bound
    def back_project(s, row, col, camera="agentview_left"):
        return SeenPoint(
            pixel=(row, col),
            xyz=(0.0, 0.55, COUNTER_Z),
            depth_m=1.0,
            camera=camera,
            surface="object",
            normal=(0.0, 0.0, 1.0),
            views=1,
        )

    @bound
    def camera_meta(s, camera="agentview_left", *, step=-1):
        return {
            "camera": camera,
            "intrinsic_matrix": [[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]],
            "extrinsic_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.4], [0, 0, 0, 1]],
            "shape": [480, 640, 3],
        }

    @bound
    def show(s, camera="agentview_left"):
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
