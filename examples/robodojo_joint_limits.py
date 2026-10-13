#!/usr/bin/env python
"""Where does each ARX-X5 joint stop in RoboDojo's simulator?  The URDF says +-10 rad (joint6
+-3.14), but joint5 stopped at -1.548 when PyRUA-Lean's IK asked for -pi.

    python examples/robodojo_joint_limits.py --cuda-device 0 --out <dir> [--task stack_blocks]

Run it like examples/robodojo_check_primitives.py.

From the starting pose, each joint of each arm alone is commanded to +3.0 and then -3.0 rad for
30 native actions (the others held at their start), then back to the start.  The stop, and
the lowest point of the arm by the host's FK at that stop (a stop with a frame near the table
top is contact, not a limit), go to <out>/limits.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyrualean.hosts import robodojo_kinematics as kin
from pyrualean.hosts import rpent_robodojo as host

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("--task", default="stack_blocks")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cuda-device", type=int, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--target", type=float, default=3.0)
ns = ap.parse_args()
ns.out.mkdir(parents=True, exist_ok=True)
cell = host.Cell("arx_x5", ns.task, ns.seed, cuda_device=ns.cuda_device)
toolkit, daemons = None, []
out: dict = {"status": "boot_error", "joints": {}}
try:
    toolkit, daemons = host.boot(cell, ns.out)
    env = toolkit._primitives.env

    def joints():
        st = env.get_obs()["state"]
        return {a: [float(v) for v in st[f"{a}_arm_joint_state"]] for a in kin.ARMS}

    start = joints()

    def drive(target: dict, n: int) -> dict:
        for _ in range(n):
            env.step(
                {
                    **{f"{a}_arm_joint_state": target[a] for a in kin.ARMS},
                    **{f"{a}_ee_joint_state": [1.0] for a in kin.ARMS},
                }
            )
        return joints()

    out["status"] = "running"
    for arm in kin.ARMS:
        for j in range(6):
            row = {}
            for sign in (+1, -1):
                target = {a: list(start[a]) for a in kin.ARMS}
                target[arm][j] = sign * ns.target
                got = drive(target, 30)[arm]
                low = min(float(f[2, 3]) for f in kin.frames(arm, got)[1:])
                tip = kin.tcp(kin.fk(arm, got), kin.TIP_OFFSET)
                row["+" if sign > 0 else "-"] = {
                    "stop": round(got[j], 4),
                    "lowest_frame_z": round(low, 4),
                    "tip_z": round(float(tip[2]), 4),
                    "others_moved": round(
                        max(
                            abs(a - b)
                            for k, (a, b) in enumerate(zip(got, start[arm], strict=True))
                            if k != j
                        ),
                        4,
                    ),
                }
                back = drive(start, 30)[arm]
                row.setdefault("returned", []).append(
                    round(max(abs(a - b) for a, b in zip(back, start[arm], strict=True)), 4)
                )
            out["joints"][f"{arm}.joint{j + 1}"] = row
            print(arm, f"joint{j + 1}", json.dumps(row), flush=True)
    out["status"] = "completed"
except BaseException as exc:  # noqa: BLE001 - recorded
    import traceback

    out["error"] = f"{type(exc).__name__}: {exc}"
    out["traceback"] = traceback.format_exc()[-3000:]
finally:
    host.stop_all(toolkit, daemons)
    (ns.out / "limits.json").write_text(json.dumps(out, indent=2) + "\n")
    (ns.out / "result.json").write_text(
        json.dumps({"status": out["status"], "error": out.get("error")}) + "\n"
    )
    print("JOINT-LIMITS", out["status"], out.get("error") or "")
