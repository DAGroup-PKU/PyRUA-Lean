"""Independent read-only physical trace metrics."""

import json
import numpy as np
from datafarm.common import sha256_bytes


def audit(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    records = data["records"]
    binding = data["binding"]
    if not records:
        raise ValueError("empty physical trace")
    steps = np.array([r["step"] for r in records])
    delta = np.diff(steps)
    if np.any(delta < 0):
        raise ValueError("nonmonotonic control clock")
    # Some callbacks observe the same physics step. Count it once, retaining last.
    ids = np.r_[np.flatnonzero(delta > 0), len(steps) - 1]
    records = [records[i] for i in ids]
    steps = steps[ids]
    dt = np.r_[np.diff(steps), 1] / binding["sample_hz"]
    phase = np.array([r["phase"] for r in records])
    limits = np.array(binding["physical_joint_limits_rad"])
    out = {
        "path": str(path.resolve()),
        "sha256": sha256_bytes(raw),
        "samples": len(records),
        "seconds": float(dt.sum()),
        "max_step_gap": int(np.diff(steps).max()) if len(steps) > 1 else 0,
        "binding": binding,
        "arms": {},
    }

    def stats(q, mask):
        margin = np.rad2deg(np.minimum(q - limits[:, 0], limits[:, 1] - q))
        v = margin[mask]
        d = dt[mask]
        bands = {}
        for t in [1, 5, 10]:
            near = (margin < t) & mask[:, None]
            best = np.zeros(6)
            elapsed = np.r_[0, np.cumsum(dt)]
            for j in range(6):
                transitions = np.diff(np.r_[False, near[:, j], False].astype(int))
                starts = np.flatnonzero(transitions == 1)
                ends = np.flatnonzero(transitions == -1)
                if len(starts):
                    best[j] = np.max(elapsed[ends] - elapsed[starts])
            bands[str(t)] = {
                "frames": (v < t).sum(0).tolist(),
                "fraction": (v < t).mean(0).tolist(),
                "seconds": ((v < t) * d[:, None]).sum(0).tolist(),
                "longest_seconds": best.tolist(),
            }
        return {
            "samples": int(mask.sum()),
            "min_margin_deg": v.min(0).tolist(),
            "quantiles_deg": {str(x): np.quantile(v, x, axis=0).tolist() for x in [0, 0.01, 0.05, 0.5]},
            "beyond_limits_frames": (v < 0).sum(0).tolist(),
            "max_violation_deg": np.maximum(0, -v.min(0)).tolist(),
            "bands": bands,
        }

    allmask = np.ones(len(records), bool)
    for side in ["left", "right"]:
        q = np.array([r[side] for r in records])
        assert q.shape == (len(records), 6) and np.isfinite(q).all()
        arm = {
            "actual": stats(q, allmask),
            "phases": {p: stats(q, phase == p) for p in sorted(set(phase))},
            "max_actual_step_deg": np.rad2deg(abs(np.diff(q, axis=0))).max(0).tolist() if len(q) > 1 else [0] * 6,
        }
        command_mask = np.array([side in r["commands"] for r in records])
        c = np.array([r["commands"].get(side, {}).get("qpos_rad", r[side])[:6] for r in records])
        arm["missing_command_samples"] = int((~command_mask).sum())
        arm["command"] = stats(c, command_mask) if command_mask.any() else None
        out["arms"][side] = arm
    policy = binding["policy"]
    target, joint = policy["target_margin_deg"], policy["target_joint"]
    out["joint_target_met"] = all(x["actual"]["min_margin_deg"][joint] >= target for x in out["arms"].values())
    # One joint's target is task-specific, never an all-joint verdict.
    out["actual_within_bounds"] = all(min(a["actual"]["min_margin_deg"]) >= 0 for a in out["arms"].values())
    out["commands_complete_and_within_bounds"] = all(
        a["missing_command_samples"] == 0 and a["command"] is not None and min(a["command"]["min_margin_deg"]) >= 0
        for a in out["arms"].values()
    )
    out["recorded_trace_passed"] = bool(
        out["joint_target_met"]
        and out["actual_within_bounds"]
        and out["commands_complete_and_within_bounds"]
        and out["max_step_gap"] == 1
    )
    out["coverage_scope"] = (
        "Recorded samples only; initial/video-frame/terminal coverage must be independently bound. No limit tolerance or home exemption."
    )
    return out
