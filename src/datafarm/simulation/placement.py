"""Per-episode initial object placements for simulated expert collection.

Two modes, chosen by the orchestrator from the user's goal and recorded in the scene's `placement_mode`:
  local_robustness  each manipulated object at its nominal position (derived from the real episode) plus a uniform
                    jitter, default +-30 mm: small validation runs and debugging.
  broad_table       each object sampled independently and uniformly over the reachable tabletop region: training
                    data and broad-deployment claims.  The region comes from grounding (`reachable_tabletop_xy`,
                    probed with the robot's kinematics and joint limits on the table plane).
Both modes keep a minimum distance between objects (default 80 mm) by resampling the whole layout.  The plan is a
deterministic function of its inputs and seed and carries a sha256 digest; the scene sampler reads it by index.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

SCHEMA = "datafarm.object-placement-plan/1"
MODES = ("local_robustness", "broad_table")
DEFAULT_JITTER_M = 0.030
DEFAULT_MIN_DISTANCE_M = 0.080
MAX_TRIES = 1000


def nominal_positions(scene: dict[str, Any], key: str = "objects_initial") -> dict[str, tuple[float, float]]:
    """{object: (x, y)} from scene[key]; entries may be {"xy": [x, y]}, {"x": x, "y": y} or [x, y, ...]."""
    out = {}
    for name, value in (scene.get(key) or {}).items():
        if isinstance(value, dict) and "xy" in value:
            x, y = value["xy"][:2]
        elif isinstance(value, dict) and "x" in value and "y" in value:
            x, y = value["x"], value["y"]
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            x, y = value[:2]
        else:
            raise ValueError(f"scene {key}.{name}: no x/y position")
        out[name] = (float(x), float(y))
    if not out:
        raise ValueError(f"scene has no object positions under {key!r}")
    return out


def reachable_bounds(reachable: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    region = reachable.get("both_arms", reachable)
    x, y = region.get("x", region.get("x_range")), region.get("y", region.get("y_range"))
    if x is None or y is None:
        raise ValueError("reachable_tabletop_xy needs x and y ranges")
    (x0, x1), (y0, y1) = (float(x[0]), float(x[1])), (float(y[0]), float(y[1]))
    if not (x0 < x1 and y0 < y1):
        raise ValueError(f"empty reachable region x={x}, y={y}")
    return (x0, x1), (y0, y1)


def _spaced(pos: dict[str, tuple[float, float]], min_dist: float) -> bool:
    pts = list(pos.values())
    return all(math.dist(pts[i], pts[j]) >= min_dist for i in range(len(pts)) for j in range(i + 1, len(pts)))


def _inside(p: tuple[float, float], bounds) -> bool:
    (x0, x1), (y0, y1) = bounds
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


def sample_layout(rng: random.Random, mode: str, nominals: dict, jitter: float, bounds, min_dist: float) -> dict:
    for _ in range(MAX_TRIES):
        if mode == "local_robustness":
            pos = {n: (x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter)) for n, (x, y) in nominals.items()}
            if bounds is not None and not all(_inside(p, bounds) for p in pos.values()):
                continue
        else:
            (x0, x1), (y0, y1) = bounds
            pos = {n: (rng.uniform(x0, x1), rng.uniform(y0, y1)) for n in nominals}
        if _spaced(pos, min_dist):
            return pos
    raise ValueError(f"no {mode} layout of {len(nominals)} objects with {min_dist} m spacing in {MAX_TRIES} tries")


def generate_plan(scene: dict, mode: str, count: int, seed: int, *, jitter_m: float = DEFAULT_JITTER_M,
                  reachable: dict | None = None, min_dist_m: float = DEFAULT_MIN_DISTANCE_M,
                  key: str = "objects_initial") -> dict:
    if mode not in MODES:
        raise ValueError(f"placement mode must be one of {MODES}")
    if count < 1:
        raise ValueError("count must be positive")
    nominals = nominal_positions(scene, key)
    bounds = reachable_bounds(reachable) if reachable is not None else None
    if mode == "broad_table" and bounds is None:
        raise ValueError("broad_table needs reachable_tabletop_xy (grounding)")
    if not _spaced(nominals, min_dist_m):
        raise ValueError(f"the nominal layout itself is closer than {min_dist_m} m")
    rng = random.Random(seed)
    episodes = [{"index": i, "object_xy": {n: [round(v, 6) for v in p]
                                           for n, p in sample_layout(rng, mode, nominals, jitter_m, bounds,
                                                                     min_dist_m).items()}}
                for i in range(count)]
    plan = {"schema": SCHEMA, "placement_mode": mode, "seed": seed, "episode_count": count,
            "object_names": sorted(nominals), "nominals": {n: list(p) for n, p in nominals.items()},
            "policy": {"jitter_m": jitter_m if mode == "local_robustness" else None, "min_object_distance_m": min_dist_m,
                       "reachable_tabletop_xy": reachable},
            "episodes": episodes}
    plan["coverage"] = {n: {"x_span_m": round(max(xs) - min(xs), 6), "y_span_m": round(max(ys) - min(ys), 6)}
                        for n in nominals
                        for xs, ys in [([e["object_xy"][n][0] for e in episodes],
                                        [e["object_xy"][n][1] for e in episodes])]}
    plan["digest"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return plan


def default_mode(target_count: int, request: str = "") -> str:
    """The orchestrator's default when the user did not choose: broad for training-scale goals, local otherwise."""
    text = request.lower()
    if any(k in text for k in ("validation", "validate", "验证")):
        return "local_robustness"
    if target_count >= 50 or any(k in text for k in ("training", "train ", "policy", "broad", "训练")):
        return "broad_table"
    return "local_robustness"
