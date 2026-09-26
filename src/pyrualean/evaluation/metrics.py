# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Episode records shared by the code arm and the RPent arm.

``environment_success`` is the benchmark's own predicate.  On the code arm it
is always a boolean (a crash, a timeout or a boot failure is a failed
episode); on the RPent arm it is ``None`` only when the artifacts do not
contain a ``states.json``, and such episodes are reported as *unknown*, never
silently dropped or counted as failures.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

#: Simulator steps per Pi0.5 action chunk in RPent's LIBERO stack.  The code
#: arm measures env steps exactly (frame counter); this constant is only used
#: to reconstruct RPent transcripts, whose ``states.json`` records chunks.
PI0_CHUNK_STEPS = 5


@dataclass(frozen=True)
class EpisodeMetrics:
    """One episode, comparable across systems."""

    system: str
    backend: str
    task: str
    seed: int | None = None
    environment_success: bool | None = None
    status: str | None = None
    calls: int | None = None
    stateful_calls: int | None = None
    env_steps: int | None = None
    turns: int | None = None
    wall_time_s: float | None = None
    startup_s: float | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    total_tokens: int | None = None
    source: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, source: str | None = None) -> EpisodeMetrics:
        allowed = {f.name for f in fields(cls)}
        payload = {k: value[k] for k in allowed if k in value}
        payload.setdefault("system", "unknown")
        payload.setdefault("backend", "unknown")
        payload.setdefault("task", "unknown")
        if source is not None and not payload.get("source"):
            payload["source"] = source
        if isinstance(payload.get("notes"), list):
            payload["notes"] = tuple(str(n) for n in payload["notes"])
        return cls(**payload)


def env_steps_from_states(
    steps: Iterable[dict[str, Any]], *, chunk_steps: int = PI0_CHUNK_STEPS
) -> int:
    """Reconstruct simulator steps from RPent ``states.json`` step records.

    A runtime that counts its own native actions (RoboTwin's
    ``episode_status.take_action_cnt``) is trusted as is; otherwise the steps
    are rebuilt from each LIBERO primitive's ``steps_used`` / ``chunks_used``.
    """

    steps = list(steps)
    native = [n for n in map(_native_count, steps) if n is not None]
    if native:
        return max(native)
    if _is_robocasa(steps):
        return _robocasa_env_steps(steps)
    total = 0
    for step in steps:
        command = step.get("command") or {}
        result = step.get("result") or {}
        action = command.get("action")
        if not action:
            continue
        if action in ("pi0_pick", "pi0_doubled"):
            total += int(result.get("chunks_used", 0)) * chunk_steps
        elif action == "set_gripper":
            total += int(result.get("steps", command.get("steps", 5)))
        else:
            used = int(result.get("steps_used", 0))
            # RPent's servos count the final convergence check as a step
            # (``len(traj)``), so a converged move records one more step than
            # the simulator executed; the code arm's frame counter does not.
            if _converged(action, command, result) and used > 0:
                used -= 1
            total += used
    return total


#: RoboCasa calibration steps the code arm's frame counter includes but RPent's
#: states.json never records (robots/robocasa/primitives.py): the arm jacobian
#: before the first arm servo (and again after every navigate_to, which resets
#: it) and the base heading before the first navigate_to.
_ROBOCASA_JACOBIAN_STEPS = 9
_ROBOCASA_HEADING_STEPS = 6


def _is_robocasa(steps: list[dict[str, Any]]) -> bool:
    return any("vla_desync" in (s.get("extras") or {}) for s in steps)


def _robocasa_command_steps(command: dict[str, Any], result: dict[str, Any]) -> int:
    action = command.get("action")
    if action in ("rldx_skill", "rldx_arm"):
        return int(result.get("steps_applied", 0))
    if action in ("move_to", "move_delta", "navigate_to"):
        return int(result.get("steps", 0))  # exact: steps to convergence, or max_steps
    if action in ("set_gripper", "release", "move_base"):
        return int(command.get("steps", 10))
    if action == "rotate_pitch":
        return int(command.get("n", 12))
    if action == "scripted_grasp":
        # open (4) + hover + descent + close (14) + lift: only the last servo is recorded
        return 18 + int(result.get("steps", 0))
    return 0


def _robocasa_env_steps(steps: list[dict[str, Any]]) -> int:
    """Simulator steps of an RPent RoboCasa run, the code arm's frame-counter unit."""

    total = 0
    arm_calibrated = False
    heading_calibrated = False
    for step in steps:
        command = step.get("command") or {}
        result = step.get("result") or {}
        action = command.get("action")
        if not action or (isinstance(result, dict) and result.get("error")):
            continue
        if action in ("move_to", "move_delta", "scripted_grasp") and not arm_calibrated:
            total += _ROBOCASA_JACOBIAN_STEPS
            arm_calibrated = True
        if action == "navigate_to":
            if not heading_calibrated:
                total += _ROBOCASA_HEADING_STEPS
                heading_calibrated = True
            arm_calibrated = False
        total += _robocasa_command_steps(command, result)
    return total


def _native_count(step: dict[str, Any]) -> int | None:
    status = (step.get("state") or {}).get("episode_status")
    if isinstance(status, dict) and status.get("take_action_cnt") is not None:
        return int(status["take_action_cnt"])
    return None


def _converged(action: str, command: dict[str, Any], result: dict[str, Any]) -> bool:
    if action in ("move_to", "move_pose"):
        tol = float(command.get("tol", 0.012))
        dist = result.get("final_dist_m")
        return dist is not None and float(dist) < tol
    if action in ("rotate_wrist", "rotate_pitch"):
        tol = float(command.get("tol", 0.02))
        err = result.get("final_err")
        return err is not None and abs(float(err)) < tol
    return False


def _int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def load_episode_result(path: str | Path) -> EpisodeMetrics:
    """Read a ``result.json`` written by :mod:`pyrualean.hosts.rpent_libero`."""

    location = Path(path)
    if location.is_dir():
        location = location / "result.json"
    data = json.loads(location.read_text(encoding="utf-8"))
    notes = [str(n) for n in data.get("notes", [])]
    if data.get("error"):
        err = data["error"]
        notes.append(f"{err.get('type')}: {str(err.get('message'))[:200]}")
    generation = data.get("generation") or {}
    arm = data.get("arm")
    system = str(data.get("system", "pyrualean"))
    if arm:
        system = f"{system}:{arm}"
    return EpisodeMetrics(
        system=system,
        backend=str(data.get("backend", "libero")),
        task=f"{data.get('suite')}:{data.get('task')}",
        seed=_int(data.get("seed")),
        environment_success=bool(data.get("environment_success", False)),
        status=data.get("status"),
        calls=_int(data.get("calls")),
        stateful_calls=_int(data.get("stateful_calls")),
        env_steps=_int(data.get("env_steps")),
        # Interactive arms: one tool call = one model decision, which is the
        # comparable notion of a turn.  Single-shot generation: the number of
        # Codex turns (normally 1).
        turns=(
            _int(data.get("tool_turns"))
            if data.get("arm")
            else (_int(generation.get("turns")) if generation else None)
        ),
        wall_time_s=_float(data.get("policy_wall_s")),
        startup_s=_float(data.get("startup_s")),
        input_tokens=_int(generation.get("input_tokens")),
        cached_input_tokens=_int(generation.get("cached_input_tokens")),
        output_tokens=_int(generation.get("output_tokens")),
        reasoning_output_tokens=_int(generation.get("reasoning_output_tokens")),
        total_tokens=_int(generation.get("total_tokens")),
        source=str(location),
        notes=tuple(notes),
    )


def summarize(runs: list[EpisodeMetrics]) -> dict[str, Any]:
    """Aggregate episodes; unknown native outcomes are reported, not hidden."""

    known = [r for r in runs if r.environment_success is not None]

    def mean(name: str) -> float | None:
        values = [getattr(r, name) for r in runs]
        values = [float(v) for v in values if v is not None]
        return sum(values) / len(values) if values else None

    statuses: dict[str, int] = {}
    for run in runs:
        statuses[str(run.status)] = statuses.get(str(run.status), 0) + 1
    return {
        "episodes": len(runs),
        "native_sr": (
            sum(bool(r.environment_success) for r in known) / len(known) if known else None
        ),
        "native_sr_denominator": len(known),
        "unknown_outcome_episodes": len(runs) - len(known),
        "statuses": statuses,
        "mean_calls": mean("calls"),
        "mean_stateful_calls": mean("stateful_calls"),
        "mean_env_steps": mean("env_steps"),
        "mean_turns": mean("turns"),
        "mean_wall_time_s": mean("wall_time_s"),
        "mean_input_tokens": mean("input_tokens"),
        "mean_cached_input_tokens": mean("cached_input_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "mean_reasoning_output_tokens": mean("reasoning_output_tokens"),
        "mean_total_tokens": mean("total_tokens"),
    }


__all__ = [
    "PI0_CHUNK_STEPS",
    "EpisodeMetrics",
    "env_steps_from_states",
    "load_episode_result",
    "summarize",
]
