"""Task adapter configuration: everything task-specific the harness needs, in one JSON file.

A task adapter says what the agent is asked to build (task/IR, verifier and its exact
pass criteria, runtime facts), what it may read and what it must never see, how the
harness runs a rollout of an agent-written expert on a development seed, how the harness
alone evaluates hidden holdout seeds, and the round budgets. Absolute paths live only in
adapter files (``configs/expert-harness/*.json``), never in the harness code.
"""

from __future__ import annotations

import copy
import dataclasses
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datafarm.expert.util import ConfigError, deep_merge, expand, expand_all, human_duration, sha256_file, sha256_json
from datafarm.expert.task_ir import PROFILE as STAGE_GRAPH_PROFILE, validate_stage_graph

ADAPTER_SCHEMA = "datafarm.expert-harness.adapter/1"
HOLDOUT_SCHEMA = "datafarm.expert-harness.holdout-seeds/1"

AGENT_BACKENDS = ("claude", "codex", "codex-metered", "fake")
ROLLOUT_BACKENDS = ("command", "local", "fake")
ISOLATION_MODES = ("bwrap", "unshare", "none")
ISOLATION_BASES = ("allowlist", "root_ro")
JUDGE_POINTS = ("release", "settled", "terminal", "any")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


# --------------------------------------------------------------------------- seeds


@dataclasses.dataclass(frozen=True)
class SeedSet:
    """Development seeds the agent may run: inclusive ranges plus explicit seeds."""

    ranges: tuple[tuple[int, int], ...] = ()
    explicit: frozenset[int] = frozenset()

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> "SeedSet":
        ranges = []
        for item in raw.get("ranges", []):
            if (not isinstance(item, (list, tuple)) or len(item) != 2
                    or not all(isinstance(v, int) and not isinstance(v, bool) for v in item)
                    or item[0] > item[1] or item[0] < 0):
                raise ConfigError(f"invalid seed range {item!r}")
            ranges.append((int(item[0]), int(item[1])))
        explicit = []
        for value in raw.get("explicit", []):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ConfigError(f"invalid explicit seed {value!r}")
            explicit.append(value)
        seeds = cls(tuple(ranges), frozenset(explicit))
        if not seeds.ranges and not seeds.explicit:
            raise ConfigError("the development seed set is empty")
        return seeds

    def __contains__(self, seed: object) -> bool:
        if not isinstance(seed, int) or isinstance(seed, bool):
            return False
        return seed in self.explicit or any(lo <= seed <= hi for lo, hi in self.ranges)

    def describe(self) -> str:
        parts = [f"{lo}–{hi}" if lo != hi else str(lo) for lo, hi in self.ranges]
        parts += [str(v) for v in sorted(self.explicit)]
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {"ranges": [list(r) for r in self.ranges], "explicit": sorted(self.explicit)}


@dataclasses.dataclass(frozen=True)
class HoldoutSeed:
    seed: int
    family: str | None = None
    spare: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "family": self.family, "spare": self.spare}


def load_holdout_seeds(path: str | Path) -> list[HoldoutSeed]:
    """Holdout seeds: JSON (``holdout-seeds/1``) or operator text lines ``<seed> [family]``."""

    text = Path(path).read_text(encoding="utf-8")
    seeds: list[HoldoutSeed] = []
    stripped = text.lstrip()
    if stripped.startswith("{"):
        data = json.loads(text)
        if data.get("schema") != HOLDOUT_SCHEMA:
            raise ConfigError(f"{path}: not a {HOLDOUT_SCHEMA} file")
        for item in data.get("seeds", []):
            seed = item.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
                raise ConfigError(f"{path}: invalid holdout seed {item!r}")
            seeds.append(HoldoutSeed(seed, item.get("family"), bool(item.get("spare", False))))
    else:
        for number, line in enumerate(text.splitlines(), start=1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = line.split()
            if not fields[0].isdigit():
                raise ConfigError(f"{path}:{number}: expected '<seed> [family]'")
            seeds.append(HoldoutSeed(int(fields[0]), fields[1] if len(fields) > 1 else None))
    if not [s for s in seeds if not s.spare]:
        raise ConfigError(f"{path}: no holdout seeds")
    if len({s.seed for s in seeds}) != len(seeds):
        raise ConfigError(f"{path}: duplicate holdout seeds")
    return seeds


def write_holdout_seeds(path: str | Path, seeds: Sequence[HoldoutSeed], *, note: str) -> None:
    from datafarm.expert.util import atomic_write_json

    atomic_write_json(path, {"schema": HOLDOUT_SCHEMA, "note": note,
                             "seeds": [s.to_dict() for s in seeds]}, mode=0o600)


# --------------------------------------------------------------------------- budgets


@dataclasses.dataclass(frozen=True)
class Budgets:
    wall_per_session_s: float
    total_wall_s: float
    max_counted_rollouts: int
    max_concurrent_rollouts: int
    max_sessions: int
    max_usd: float | None = None
    max_infra_failures: int = 20
    max_idle_sessions: int = 3
    final_grace_s: float = 300.0
    wait_cap_s: float = 200.0
    warn_before_wall_s: float = 900.0

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> "Budgets":
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - fields - {"comment"}
        if unknown:
            raise ConfigError(f"unknown budget keys {sorted(unknown)}")
        try:
            budgets = cls(**{k: v for k, v in raw.items() if k in fields})
        except TypeError as exc:
            raise ConfigError(f"incomplete budgets: {exc}") from exc
        for name in ("wall_per_session_s", "total_wall_s", "max_counted_rollouts",
                     "max_concurrent_rollouts", "max_sessions", "wait_cap_s"):
            value = getattr(budgets, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ConfigError(f"budget {name} must be a positive number")
        if budgets.max_usd is not None and budgets.max_usd <= 0:
            raise ConfigError("budget max_usd must be positive or null")
        if budgets.max_concurrent_rollouts > budgets.max_counted_rollouts:
            raise ConfigError("max_concurrent_rollouts exceeds max_counted_rollouts")
        if budgets.wait_cap_s > 240:
            raise ConfigError("wait_cap_s above 240 s lets the prompt cache expire between turns")
        return budgets

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def describe(self) -> dict[str, str]:
        return {
            "max_counted_rollouts": str(self.max_counted_rollouts),
            "max_concurrent_rollouts": str(self.max_concurrent_rollouts),
            "total_wall": human_duration(self.total_wall_s),
            "wall_per_session": human_duration(self.wall_per_session_s),
            "max_sessions": str(self.max_sessions),
            "max_usd": "no limit" if self.max_usd is None else f"${self.max_usd:g}",
            "max_infra_failures": str(self.max_infra_failures),
            "wait_cap": f"{int(self.wait_cap_s)} s",
        }


# --------------------------------------------------------------------------- adapter


class TaskAdapter:
    """A validated task adapter. ``raw`` is the JSON as written; accessors expand ``{vars}``."""

    def __init__(self, raw: Mapping[str, Any], path: str | Path | None = None) -> None:
        self.raw: dict[str, Any] = copy.deepcopy(dict(raw))
        self.path = Path(path).resolve() if path else None
        self.dir = self.path.parent if self.path else Path.cwd()
        self._vars = self._resolve_vars(self.raw.get("vars", {}))
        self.validate()

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, path: str | Path, overrides: Mapping[str, Any] | None = None) -> "TaskAdapter":
        target = Path(path)
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read adapter {target}: {exc}") from exc
        if overrides:
            raw = deep_merge(raw, overrides)
        from datafarm.runtime import expand as expand_runtime, load_bindings
        raw = expand_runtime(raw, load_bindings())
        return cls(raw, target)

    @staticmethod
    def _resolve_vars(raw_vars: Mapping[str, Any]) -> dict[str, str]:
        values: dict[str, str] = {}
        pending = dict(raw_vars)
        for _ in range(len(pending) + 1):
            progressed = False
            for name, template in list(pending.items()):
                try:
                    values[name] = expand(str(template), values)
                except ConfigError:
                    continue
                del pending[name]
                progressed = True
            if not pending:
                return values
            if not progressed:
                break
        raise ConfigError(f"unresolvable adapter vars: {sorted(pending)}")

    # ------------------------------------------------------------------ helpers

    @property
    def sha256(self) -> str:
        return sha256_json(self.raw)

    @property
    def vars(self) -> dict[str, str]:
        return dict(self._vars)

    def x(self, value: Any, *, keep_env: bool = False, **extra: Any) -> Any:
        """Expand ``{placeholders}`` from adapter vars plus ``extra``."""

        return expand_all(value, {**self._vars, **{k: str(v) for k, v in extra.items()}},
                          keep_env=keep_env)

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"adapter section {name!r} must be an object")
        return value

    # ------------------------------------------------------------------ fields

    @property
    def task_id(self) -> str:
        return self.raw["task_id"]

    @property
    def title(self) -> str:
        return self.raw.get("title", self.task_id)

    @property
    def available(self) -> bool:
        return self.raw.get("status", "available") == "available"

    @property
    def budgets(self) -> Budgets:
        return Budgets.from_config(self.section("budgets"))

    @property
    def dev_seeds(self) -> SeedSet:
        return SeedSet.from_config(self.section("seeds").get("dev", {}))

    @property
    def verifier(self) -> dict[str, Any]:
        return self.section("verifier")

    @property
    def verifier_version(self) -> str:
        return str(self.verifier.get("version", ""))

    @property
    def brief_template_path(self) -> Path:
        return (self.dir / self.section("brief")["template"]).resolve()

    @property
    def agent(self) -> dict[str, Any]:
        return self.section("agent")

    @property
    def rollout(self) -> dict[str, Any]:
        return self.section("rollout")

    @property
    def holdout(self) -> dict[str, Any]:
        return self.section("holdout")

    @property
    def expert_policy(self) -> dict[str, Any]:
        return self.section("expert")

    def isolation(self, which: str) -> dict[str, Any]:
        return dict(self.section("isolation").get(which, {}))

    def price_table(self) -> dict[str, Any]:
        name = self.agent.get("price_table")
        if not name:
            return {}
        path = (self.dir / name).resolve()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read price table {path}: {exc}") from exc
        return data.get("models", {})

    def task_ir_sha256(self) -> str:
        return sha256_json({"task_id": self.task_id, "task_ir": self.raw.get("task_ir"),
                            "verifier": self.verifier})

    def readable_paths(self) -> list[str]:
        return self._paths(self.section("workspace").get("readable", []))

    def protected_paths(self) -> list[str]:
        return self._paths(self.section("workspace").get("protected", []))

    def _paths(self, patterns: Iterable[str], *, expand_globs: bool = True) -> list[str]:
        out: list[str] = []
        for pattern in patterns:
            value = os.path.expanduser(self.x(pattern))
            if expand_globs and any(ch in value for ch in "*?["):
                out.extend(sorted(glob.glob(value)))
            else:
                out.append(value)
        return [os.path.normpath(p) for p in out]

    def shell_paths(self, patterns: Iterable[str]) -> list[str]:
        """Paths for a generated job script: ``{env:NAME}`` stays ``${NAME}`` (expanded on the
        GPU node), other globs are expanded here (all GPU nodes share the filesystem)."""

        out: list[str] = []
        for pattern in patterns:
            value = self.x(pattern, keep_env=True)
            if "${" in value:
                out.append(value)
            elif any(ch in value for ch in "*?["):
                out.extend(sorted(glob.glob(value)))
            else:
                out.append(os.path.normpath(os.path.expanduser(value)))
        return out

    def display_paths(self, patterns: Iterable[str]) -> list[str]:
        """Patterns expanded with adapter vars but not globbed (for the brief)."""

        return [self.x(p, keep_env=True) for p in patterns]

    def manifest_paths(self) -> list[str]:
        files = list(self.verifier.get("files", []))
        files += list(self.section("manifest").get("files", []))
        return self._paths(files)

    def brief_template_sha256(self) -> str:
        return sha256_file(self.brief_template_path)

    # ------------------------------------------------------------------ validation

    def validate(self) -> None:
        raw = self.raw
        if raw.get("schema") != ADAPTER_SCHEMA:
            raise ConfigError(f"adapter schema must be {ADAPTER_SCHEMA!r}")
        if not isinstance(raw.get("task_id"), str) or not _SLUG.match(raw["task_id"]):
            raise ConfigError("task_id must be a lowercase slug")
        if raw.get("status", "available") not in ("available", "unavailable"):
            raise ConfigError("status must be 'available' or 'unavailable'")
        if not self.available:
            missing = raw.get("missing_interfaces")
            if not isinstance(missing, list) or not missing:
                raise ConfigError("an unavailable adapter must list its missing_interfaces")
            return
        for name in ("brief", "verifier", "seeds", "budgets", "agent", "rollout", "workspace",
                     "isolation", "expert", "holdout"):
            if name not in raw:
                raise ConfigError(f"adapter is missing section {name!r}")
        self.budgets  # noqa: B018 - validates
        self.dev_seeds  # noqa: B018
        if not self.brief_template_path.is_file():
            raise ConfigError(f"brief template {self.brief_template_path} does not exist")
        verifier = self.verifier
        if not verifier.get("version") or not verifier.get("name"):
            raise ConfigError("verifier needs a version and a name")
        points = verifier.get("judge_points", [])
        if not points or any(p not in JUDGE_POINTS for p in points):
            raise ConfigError(f"verifier.judge_points must be a non-empty subset of {JUDGE_POINTS}")
        if not verifier.get("criteria"):
            raise ConfigError("verifier.criteria must state the exact pass criteria")
        task_ir = raw.get("task_ir")
        if isinstance(task_ir, Mapping) and task_ir.get("profile") == STAGE_GRAPH_PROFILE:
            # A typed task contract must be structurally sound and map onto the frozen judge
            # before any round is prepared; summary-form task_ir (older adapters) is unchanged.
            ids = verifier.get("criterion_ids")
            problems = validate_stage_graph(task_ir, judge_criteria=list(ids) if isinstance(ids, list) else None)
            if problems:
                raise ConfigError("task_ir stage graph is invalid: " + "; ".join(problems))
        agent = self.agent
        if agent.get("backend") not in AGENT_BACKENDS:
            raise ConfigError(f"agent.backend must be one of {AGENT_BACKENDS}")
        if not agent.get("model"):
            raise ConfigError("agent.model is required")
        rollout = self.rollout
        if rollout.get("backend") not in ROLLOUT_BACKENDS:
            raise ConfigError(f"rollout.backend must be one of {ROLLOUT_BACKENDS}")
        if rollout.get("backend") in ("command", "local"):
            for key in ("sim", "judge", "verdict"):
                if key not in rollout:
                    raise ConfigError(f"rollout.{key} is required for backend {rollout['backend']}")
            if not isinstance(rollout["sim"].get("argv"), list) or not rollout["sim"]["argv"]:
                raise ConfigError("rollout.sim.argv must be a non-empty list")
            if not rollout["verdict"].get("file") or not rollout["verdict"].get("success"):
                raise ConfigError("rollout.verdict needs 'file' and 'success'")
        if rollout.get("backend") == "command" and not rollout.get("submit", {}).get("argv"):
            raise ConfigError("rollout.submit.argv is required for the command backend")
        for which in ("agent", "rollout"):
            iso = self.isolation(which)
            if iso.get("mode") not in ISOLATION_MODES:
                raise ConfigError(f"isolation.{which}.mode must be one of {ISOLATION_MODES}")
            if iso.get("fallback") not in (None, *ISOLATION_MODES):
                raise ConfigError(f"isolation.{which}.fallback is invalid")
            if iso.get("base", "allowlist") not in ISOLATION_BASES:
                raise ConfigError(f"isolation.{which}.base must be one of {ISOLATION_BASES}")
            run_as = iso.get("run_as")
            if run_as is not None and not (isinstance(run_as, dict) and isinstance(run_as.get("uid"), int)
                                           and isinstance(run_as.get("gid"), int)):
                raise ConfigError(f"isolation.{which}.run_as must be null or {{uid, gid}}")
        policy = self.expert_policy
        if not policy.get("allowed_imports"):
            raise ConfigError("expert.allowed_imports must be a non-empty list")
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(policy.get("default_class", "X"))):
            raise ConfigError("expert.default_class must be a Python identifier")
        for pattern in (self.section("workspace").get("readable", [])
                        + self.section("workspace").get("protected", [])):
            value = self.x(pattern, keep_env=True)
            if not (value.startswith("/") or value.startswith("~") or value.startswith("${")):
                raise ConfigError(f"workspace path {pattern!r} must be absolute")
        for fact in self.raw.get("runtime_facts", []):
            if not fact.get("id") or not fact.get("text") or not isinstance(fact.get("must_mention", []), list):
                raise ConfigError(f"invalid runtime fact {fact!r}")

    def describe(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "title": self.title, "adapter": str(self.path),
                "adapter_sha256": self.sha256, "verifier": self.verifier_version,
                "budgets": self.budgets.to_dict(), "dev_seeds": self.dev_seeds.describe(),
                "agent": {"backend": self.agent.get("backend"), "model": self.agent.get("model")},
                "rollout_backend": self.rollout.get("backend")}


def validate_holdout_against_dev(seeds: Sequence[HoldoutSeed], dev: SeedSet) -> None:
    overlap = sorted(s.seed for s in seeds if s.seed in dev)
    if overlap:
        raise ConfigError(f"{len(overlap)} holdout seeds are also development seeds")
