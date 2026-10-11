#!/usr/bin/env python
"""Do PyRUA-Lean's own RoboDojo tools work in the simulator?  A scripted run, no planner, no VLA.

    python examples/robodojo_check_primitives.py --task stack_blocks --seed 0 --cuda-device 0 \
        --out <dir> --objects "purple block,white block,yellow block" [--stack] [--side]
    python examples/robodojo_check_primitives.py --task swap_blocks --seed 10 --cuda-device 0 \
        --out <dir> --press button --presses 3

Run it with the interpreter of RPent's ``robodojo`` environment, the RoboDojo environment
variables set (docs/setup.md) and ``src/`` of this repository on ``PYTHONPATH``.

For each object: ``locate`` it (mask + depth: centre, top, heading), open the nearer arm's
gripper, ``move_pose`` the grasp point above it pointing straight down with the fingers across
its narrow side, descend along a line, ``grip`` (a full stroke: does it hold?), lift 12 cm (still
holding, and gone from where it was?).  With --stack the objects are stacked in the given
order on the first one instead of being put back; then both arms go ``home`` and the
environment's own verdict is read.  Every tool result, the native actions it spent and a head
image per phase are written to <out>.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from pyrualean.hosts import robodojo_kinematics as kin
from pyrualean.hosts import robodojo_tools as tools
from pyrualean.hosts import rpent_robodojo as host

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("--task", default="stack_blocks")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cuda-device", type=int, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--objects", default="purple block,white block,yellow block")
ap.add_argument("--stack", action="store_true")
ap.add_argument("--side", action="store_true", help="grasp from the side, level, at half height")
ap.add_argument("--press", default=None, help="instead of grasps: locate this and press it")
ap.add_argument("--presses", type=int, default=3)
ap.add_argument("--depth", type=float, default=0.008)
ns = ap.parse_args()
ns.out.mkdir(parents=True, exist_ok=True)
cell = host.Cell("arx_x5", ns.task, ns.seed, cuda_device=ns.cuda_device)
log: list[dict] = []
result: dict = {"task": ns.task, "layout": ns.seed, "status": "boot_error", "error": None}
toolkit, daemons = None, []


def call(name: str, **kwargs) -> dict:
    t0 = time.time()
    before = int(toolkit._primitives.env.get_status().get("step") or 0)
    out = toolkit.execute_tool(name, kwargs).result
    payload = (
        (out.get("log") or {}).get("result") if isinstance(out, dict) and "log" in out else out
    )
    payload = payload if isinstance(payload, dict) else {"value": payload}
    after = int(toolkit._primitives.env.get_status().get("step") or 0)
    row = {
        "tool": name,
        "kwargs": kwargs,
        "native_actions": after - before,
        "wall_s": round(time.time() - t0, 2),
        "result": payload,
    }
    log.append(row)
    short = {
        k: payload.get(k)
        for k in (
            "reached",
            "dist_to_target_m",
            "orientation_error_deg",
            "ik_error",
            "blocked",
            "holding",
            "opening",
            "width_m",
            "found",
            "center_xyz",
            "top_z",
            "grasp_yaw",
            "length_m",
            "width_m",
            "error",
            "path",
            "steps_used",
            "note",
            "pressed",
            "pressed_depth_m",
            "stalled",
            "press_error",
            "lifted_m",
            "geometry_error",
            "robot_px_fraction",
        )
        if k in payload
    }
    spent = after - before
    print(
        f"{name:12s} {json.dumps(kwargs)[:90]:90s} -> {spent:3d} actions  {json.dumps(short)[:300]}"
    )
    return payload


def snapshot(tag: str) -> None:
    try:
        import cv2
        import numpy as np

        img = np.asarray(toolkit._primitives.env.get_obs()["vision"]["cam_head"]["color"])
        cv2.imwrite(str(ns.out / f"{len(log):03d}-{tag}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except Exception as exc:  # noqa: BLE001 - pictures are a convenience
        print("snapshot failed:", exc)


try:
    toolkit, daemons = host.boot(cell, ns.out)
    env = toolkit._primitives.env
    result.update(
        status="running",
        instruction=env.get_task_language(),
        step_limit=env.get_status().get("step_limit"),
    )
    print("task:", result["instruction"], "| budget", result["step_limit"], flush=True)
    snapshot("start")
    # 1. what does a gripper read when it closes on nothing?
    empty = call("grip", arm="left", gripper=1)
    call("grip", arm="left", gripper=-1)
    result["closed_on_nothing"] = empty.get("opening")
    names = [s.strip() for s in ns.objects.split(",") if s.strip()]
    if ns.press:
        names = []  # press mode: no grasps
        target = call("locate", text_prompt=ns.press)
        result["press_target"] = target
        presses = []
        if target.get("found") and "center_xyz" in target:
            px, py, _ = target["center_xyz"]
            arm = "left" if px < 0 else "right"
            top = target["top_z"]
            call("grip", arm=arm, gripper=1)
            for k in range(ns.presses):
                # a press composed from move_to: above, down a line past the surface, back up;
                # each press commanded deeper (does the stall depth follow the command?)  The
                # fingertips may not be commanded below the table top (0.765).
                below = (0.01, 0.02, 0.028, 0.028)[min(k, 3)]
                call("move_pose", xyz=[px, py, top + 0.05], arm=arm, pitch=90, gripper=1)
                res = call(
                    "move_pose", xyz=[px, py, top + 0.013 - below], arm=arm, gripper=1, path="line"
                )
                low = env.get_obs()
                tip = float(low["state"][f"{arm}_ee_pose"][2]) - kin.TIP_OFFSET  # pointing down
                res["tips_below_top_mm"] = round((top - tip) * 1000, 1)
                print(
                    f"   commanded tips {below * 1000:.0f} mm below the top: they stopped "
                    f"{(top - tip) * 1000:.1f} mm below it",
                    flush=True,
                )
                call("move_pose", xyz=[px, py, top + 0.05], arm=arm, gripper=1, path="line")
                try:
                    details = env.get_reward_details()
                except Exception as exc:  # noqa: BLE001 - informative only
                    details = {"error": str(exc)}
                print(
                    "   reward details after press",
                    k + 1,
                    json.dumps(details, default=str)[:600],
                    flush=True,
                )
                presses.append({"press": res, "reward_details": details})
                snapshot(f"press{k + 1}")
        result["presses"] = presses
    picks = []
    base = None
    for i, name in enumerate(names):
        obj = call("locate", text_prompt=name)
        if not obj.get("found") or "center_xyz" not in obj:
            picks.append({"object": name, "stage": "locate", "ok": False})
            continue
        cx, cy, _cz = obj["center_xyz"]
        top, yaw = obj["top_z"], obj["grasp_yaw"]
        arm = "left" if cx < 0 else "right"
        grasp_z = max(top - min(0.02, obj["height_m"] / 2), kin.TABLE_Z + 0.012)
        rec = {"object": name, "arm": arm, "located": obj}
        call("grip", arm=arm, gripper=-1)
        if ns.side:
            # level, pointing from the arm's base at the object, fingers open on either side of
            # it: come down onto it from above (an approach from short of it, towards the robot,
            # is out of reach this close to the base), to half its height
            bx, by = kin.BASES[arm][0][:2]
            heading = math.degrees(math.atan2(cy - by, cx - bx)) - 90.0
            grasp_z = max(kin.TABLE_Z + obj["height_m"] / 2, kin.TABLE_Z + 0.03)
            above = call(
                "move_pose",
                xyz=[cx, cy, top + 0.04],
                arm=arm,
                pitch=0,
                yaw=heading,
                roll=0,
                gripper=-1,
            )
            down = call("move_pose", xyz=[cx, cy, grasp_z], arm=arm, gripper=-1, path="line")
        else:
            above = call(
                "move_pose", xyz=[cx, cy, top + 0.08], arm=arm, pitch=90, yaw=yaw, gripper=-1
            )
            down = call("move_pose", xyz=[cx, cy, grasp_z], arm=arm, gripper=-1, path="line")
        snapshot(f"{i}-down")
        g = call("grip", arm=arm, gripper=1)
        up = call("move_pose", xyz=[cx, cy, top + 0.12], arm=arm, gripper=1, path="line")
        snapshot(f"{i}-lifted")
        # still apart after the lift = still holding; a held object is hard to segment (the
        # gripper covers it), so "gone from where it was" is the second half of the check
        opening = float(env.get_obs()["state"][f"{arm}_ee_joint_state"][0])
        again = call("locate", text_prompt=name)
        moved = (
            not again.get("found")
            or "center_xyz" not in again
            or (
                abs(again["center_xyz"][0] - cx) + abs(again["center_xyz"][1] - cy) > 0.03
                or again.get("top_z", 0) > top + 0.05
            )
        )
        lifted = bool(g.get("holding") and opening > tools.HOLD_MIN_OPENING and moved)
        rec.update(
            above=above.get("reached"),
            down=down.get("reached"),
            holding=g.get("holding"),
            width_m=g.get("width_m"),
            lifted=lifted,
            opening_after_lift=opening,
            top_after=again.get("top_z"),
        )
        if ns.stack and i > 0 and base is not None and lifted:
            bx, by, btop = base
            h = obj["height_m"]
            call("move_pose", xyz=[bx, by, btop + h + 0.06], arm=arm, gripper=1)
            call("move_pose", xyz=[bx, by, btop + h / 2 + 0.008], arm=arm, gripper=1, path="line")
            call("grip", arm=arm, gripper=-1)
            call("move_pose", xyz=[bx, by, btop + h + 0.08], arm=arm, gripper=-1, path="line")
            snapshot(f"{i}-placed")
            placed = call("locate", text_prompt=name)
            rec["placed_top"] = placed.get("top_z")
            base = (bx, by, placed.get("top_z", btop + h))
        else:
            if i == 0:
                base = (cx, cy, top)
            # put it back where it was
            call("move_pose", xyz=[cx, cy, grasp_z + 0.005], arm=arm, gripper=1, path="line")
            call("grip", arm=arm, gripper=-1)
            call("move_pose", xyz=[cx, cy, top + 0.10], arm=arm, gripper=-1, path="line")
        picks.append(rec)
    call("home", arm="both")
    snapshot("end")
    result["picks"] = picks
    result["reward_details"] = env.get_reward_details()
    result["success"] = bool(env.is_success())
    result["native_actions"] = env.get_status().get("step")
    result["status"] = "completed"
except BaseException as exc:  # noqa: BLE001 - recorded
    import traceback

    result["error"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()[-3000:]
    if result["status"] == "running":
        result["status"] = "error"
finally:
    host.stop_all(toolkit, daemons)
    result["log"] = log
    (ns.out / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        "PRIMS-CHECK",
        result["status"],
        result.get("error") or "",
        "| success",
        result.get("success"),
        "| picks",
        [(p.get("object"), p.get("holding"), p.get("lifted")) for p in result.get("picks", [])],
    )
