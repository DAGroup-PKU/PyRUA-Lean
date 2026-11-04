"""Round orchestration: prepare a round, run its session loop, report its status.

``prepare_round`` renders and lints the brief (fail closed), copies the holdout seeds into
the harness-private area, installs the ``xh`` tool, seeds the workspace from a handoff
bundle and writes ``round.json``. ``RoundRunner.run`` then:

1. verifies the adapter is the one the round was prepared with, records the tamper
   baseline and runs an isolation probe inside the agent sandbox (fail closed on a leak);
2. loops over agent sessions (first with the brief, then ``--resume`` with a status
   prompt), ticking the broker while each session runs;
3. stops on: final submission, flag for a human, wall / USD / session / rollout budget,
   too many infrastructure failures, idle sessions, tampering, or a fatal agent error;
   pauses (exit 75, resumable, not charged) after >= 10 minutes of provider outage;
4. drains rollouts in flight, snapshots an automatic final submission if the agent made
   none, re-checks the manifest and writes the round result.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import os
import shutil
import signal
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping

from datafarm.agent.call_log import AgentCallRecord, append_call, new_call_id, read_calls, summarize

from datafarm.expert.adapter import TaskAdapter, load_holdout_seeds, validate_holdout_against_dev, write_holdout_seeds
from datafarm.expert.audit import audit_expert_source
from datafarm.expert.brief import lint_brief, render_brief
from datafarm.expert.handoff import assert_no_seed_leak, handoff_markdown, load_bundle
from datafarm.expert.isolation import SandboxSpec, chown_tree, make_empty_file, resolve_spec, run_probe, sandbox_env, wrap
from datafarm.expert.layout import RoundPaths
from datafarm.expert.library import ExpertLibrary
from datafarm.expert.manifest import ManifestGuard
from datafarm.expert.rollout import make_backend
from datafarm.expert.session import AgentBackend, SessionRunner, SessionSpec, classify_errors, make_agent_backend
from datafarm.expert.state import TERMINAL, EventLog, RoundState
from datafarm.expert.toolbox import Broker
from datafarm.expert.util import (Clock, ConfigError, HarnessError, UnsafePathError, atomic_write_bytes,
                   atomic_write_json, atomic_write_text, human_duration, read_json,
                   read_regular_nofollow, sha256_file, sha256_text, tree_digest, utc_now)

from datafarm.agent.context import repository_root

ROUND_SCHEMA = "datafarm.expert-harness.round/1"
EXIT_PAUSED = 75

FIRST_PROMPT = (repository_root() / "prompts/sessions/expert-first.txt").read_text()

CONTINUE_PROMPT = (repository_root() / "prompts/sessions/expert-continue.txt").read_text()

FINAL_CALL_PROMPT = (repository_root() / "prompts/sessions/expert-final.txt").read_text()


def harness_code_sha256() -> str:
    return tree_digest(Path(__file__).resolve().parent)["sha256"]


def make_round_id(task_id: str, now: float) -> str:
    stamp = utc_now(now).replace("-", "").replace(":", "")[:13]  # YYYYMMDDTHHMM
    return f"XR-{task_id}-{stamp}-{uuid.uuid4().hex[:4]}"


def _round_info(paths: RoundPaths, round_id: str, agent_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {"round_id": round_id, "round_dir": str(paths.root), "workspace": str(paths.workspace),
            "rollouts": str(paths.rollouts), "brief": str(paths.brief), "handoff": str(paths.handoff_in),
            "references": str(paths.references), "model": agent_cfg.get("model"),
            "backend": agent_cfg.get("backend")}


# --------------------------------------------------------------------------- prepare


def load_round_adapter(paths: RoundPaths, adapter_path: str | None = None) -> TaskAdapter:
    """The adapter a round was prepared with (recorded path + recorded overrides), verified by digest."""

    round_json = read_json(paths.round_json)
    recorded = round_json["adapter"]
    adapter = TaskAdapter.load(adapter_path or recorded["path"], overrides=recorded.get("overrides") or None)
    if adapter.sha256 != recorded["sha256"]:
        raise HarnessError("the adapter differs from the one this round was prepared with "
                           f"({adapter.sha256[:12]} != {recorded['sha256'][:12]})")
    return adapter


def prepare_round(adapter: TaskAdapter, round_dir: str | Path, *, clock: Clock, round_id: str | None = None,
                  holdout_seeds_path: str | None = None, handoff_dir: str | None = None,
                  seed_expert: str | None = None, library_dir: str | None = None,
                  references: int = 0, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not adapter.available:
        raise ConfigError(f"adapter {adapter.task_id} is unavailable: {adapter.raw.get('missing_interfaces')}")
    paths = RoundPaths(Path(round_dir))
    if paths.root.exists() and any(paths.root.iterdir()):
        raise HarnessError(f"{paths.root} already exists and is not empty")
    for protected in adapter.protected_paths():
        if str(paths.root) == protected or str(paths.root).startswith(protected.rstrip("/") + "/"):
            raise HarnessError(f"the round directory lies inside the protected path {protected}")
    now = clock.now()
    round_id = round_id or make_round_id(adapter.task_id, now)
    agent_cfg = adapter.agent
    source = holdout_seeds_path or adapter.holdout.get("seeds_file")
    if not source:
        raise ConfigError("a holdout seed file is required (adapter holdout.seeds_file or --holdout-seeds)")
    seeds = load_holdout_seeds(adapter.x(source))
    validate_holdout_against_dev(seeds, adapter.dev_seeds)
    bundle = load_bundle(handoff_dir) if handoff_dir else None
    handoff_md = handoff_markdown(bundle) if bundle else None
    library = ExpertLibrary(library_dir) if library_dir and references > 0 else None
    ref_entries = library.find(min_pass_rate=float(adapter.holdout.get("verified_pass_rate", 0.5)),
                               limit=references) if library else []
    info = _round_info(paths, round_id, agent_cfg)
    references_md = None
    if ref_entries:
        references_md = "\n".join(
            f"- `{paths.references}/{e['entry_id']}.py`: task `{e['task_id']}`, class `{e['expert_class']}`, "
            f"holdout {e['holdout'].get('passed')}/{e['holdout'].get('scored')} (verifier "
            f"{e['verifier_version']}); a reference, not a solution to copy blindly." for e in ref_entries)
    text = render_brief(adapter, round_info=info, handoff_md=handoff_md, references_md=references_md)
    report = lint_brief(text, adapter, holdout_seeds=[s.seed for s in seeds], round_info=info)
    if not report.ok:
        raise HarnessError("brief lint failed:\n- " + "\n- ".join(report.errors))

    paths.create()
    write_holdout_seeds(paths.holdout_seeds, seeds, note=f"private copy for round {round_id}")
    if bundle:
        for name in bundle.get("files", []):
            src = Path(handoff_dir) / name
            if name != "next-brief.preview.md" and src.is_file() and not src.is_symlink():
                shutil.copyfile(src, paths.handoff_in / name)
        assert_no_seed_leak(paths.handoff_in, [s.seed for s in seeds])
    default_file = adapter.expert_policy.get("default_file", "expert.py")
    if seed_expert:
        shutil.copyfile(seed_expert, paths.workspace / default_file)
    elif bundle and (paths.handoff_in / "expert.py").exists():
        shutil.copyfile(paths.handoff_in / "expert.py", paths.workspace / default_file)
    if library and ref_entries:
        library.export_references(ref_entries, paths.references)
    atomic_write_text(paths.brief, text, mode=0o444)
    tool_source = (Path(__file__).resolve().parent / "agent_tool.py").read_bytes()
    atomic_write_bytes(paths.tool, tool_source, mode=0o555)
    atomic_write_json(paths.bin_dir / "agent-settings.json",
                      {"permissions": {"deny": list(agent_cfg.get("disallowed_tools", ["WebFetch", "WebSearch"]))}},
                      mode=0o444)
    make_empty_file(paths.harness)
    run_as = adapter.isolation("agent").get("run_as")
    if run_as:
        for path in paths.agent_writable():
            chown_tree(path, int(run_as["uid"]), int(run_as["gid"]))
    round_json = {
        "schema": ROUND_SCHEMA, "round_id": round_id, "task_id": adapter.task_id, "title": adapter.title,
        "created_at": utc_now(now), "adapter": {"path": str(adapter.path), "sha256": adapter.sha256,
                                                "overrides": dict(overrides or {})},
        "brief_sha256": report.sha256, "brief_template_sha256": adapter.brief_template_sha256(),
        "task_ir_sha256": adapter.task_ir_sha256(), "harness_code_sha256": harness_code_sha256(),
        "verifier_version": adapter.verifier_version,
        "agent": {"backend": agent_cfg.get("backend"), "model": agent_cfg.get("model")},
        "rollout_backend": adapter.rollout.get("backend"), "budgets": adapter.budgets.to_dict(),
        "dev_seeds": adapter.dev_seeds.describe(),
        "holdout": {"count": len([s for s in seeds if not s.spare]),
                    "policy": "hidden; chosen and run only by the harness"},
        "handoff": None if not bundle else {"from_round": bundle.get("round_id"),
                                            "provenance": bundle.get("provenance"),
                                            "expert_sha256": (bundle.get("frozen_expert") or {}).get("sha256")},
        "references": [e["entry_id"] for e in ref_entries],
        "isolation": {"agent": adapter.isolation("agent").get("mode"),
                      "rollout": adapter.isolation("rollout").get("mode")},
        "brief_lint": {"ok": report.ok, "warnings": report.warnings},
    }
    atomic_write_json(paths.round_json, round_json, mode=0o444)
    state = RoundState.create(paths.state, round_id=round_id, created_at=utc_now(now))
    events = EventLog(paths, round_id, clock.now)
    events.emit("round_prepared", task_id=adapter.task_id, brief_sha256=report.sha256,
                adapter_sha256=adapter.sha256, model=agent_cfg.get("model"))
    events.emit("holdout_seeds_recorded", private=True, source=str(source), count=len(seeds),
                sha256=sha256_file(paths.holdout_seeds))
    return {"round_id": round_id, "round_dir": str(paths.root), "brief_sha256": report.sha256,
            "brief_warnings": report.warnings, "status": state.status}


# --------------------------------------------------------------------------- run


@contextlib.contextmanager
def round_lock(paths: RoundPaths) -> Iterator[None]:
    paths.harness.mkdir(parents=True, exist_ok=True)
    handle = open(paths.lock, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HarnessError(f"another harness process is running round {paths.root}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()} {socket.gethostname()}\n")
        handle.flush()
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def manifest_entries(adapter: TaskAdapter, paths: RoundPaths) -> list[str]:
    entries = adapter.manifest_paths() + [str(paths.holdout_seeds), str(paths.brief), str(paths.tool),
                                          str(adapter.brief_template_path)]
    if adapter.path:
        entries.append(str(adapter.path))
    return entries


def rollout_masks(adapter: TaskAdapter, paths: RoundPaths, *, purpose: str = "dev") -> list[str]:
    """Paths hidden from agent code while it runs in a rollout. Development rollouts hide the
    whole harness-private directory; holdout rollouts (whose run directories live there)
    hide its sensitive entries individually."""

    from datafarm.expert.holdout import holdout_masks

    iso = adapter.isolation("rollout")
    base = (adapter.shell_paths(adapter.section("workspace").get("protected", []))
            + adapter.shell_paths(iso.get("extra_masked", [])))
    return base + (holdout_masks(paths) if purpose == "holdout" else [str(paths.harness)])


def agent_sandbox(adapter: TaskAdapter, paths: RoundPaths, *, override: str | None = None) -> SandboxSpec:
    iso = adapter.isolation("agent")
    ro = adapter.readable_paths() + adapter._paths(iso.get("extra_ro", []))
    ro += [str(p) for p in (paths.round_json, paths.brief, paths.bin_dir, paths.handoff_in, paths.references,
                            paths.broker, paths.rollouts, paths.submissions, paths.ledger)]
    rw = [str(paths.workspace), str(paths.agent_home), str(paths.requests)]
    binds = [(str(paths.tmp), "/tmp", True)]
    for item in adapter.agent.get("auth", {}).get("ro_bind", []):
        binds.append((os.path.expanduser(adapter.x(item["src"])),
                      adapter.x(item["dest"], agent_home=str(paths.agent_home)), False))
    masked = adapter.protected_paths() + adapter._paths(iso.get("extra_masked", [])) + [str(paths.harness)]
    run_as = iso.get("run_as")
    spec = SandboxSpec(mode=override or iso.get("mode", "bwrap"), base=iso.get("base", "allowlist"),
                       ro=tuple(ro), rw=tuple(rw), masked=tuple(masked), binds=tuple(binds),
                       system_ro=tuple(iso.get("system_ro") or SandboxSpec.system_ro),
                       dev="minimal", share_net=bool(iso.get("share_net", True)),
                       run_as=(int(run_as["uid"]), int(run_as["gid"])) if run_as else None,
                       cwd=str(paths.workspace), empty_file=str(paths.harness / ".xh-empty"))
    if override:
        return resolve_spec(spec, fallback=None, allow_degraded=True)
    return resolve_spec(spec, fallback=iso.get("fallback"),
                        allow_degraded=bool(iso.get("allow_degraded_fallback", False)))


class RoundRunner:
    def __init__(self, paths: RoundPaths, adapter: TaskAdapter, *, clock: Clock,
                 agent_backend: AgentBackend | None = None, rollout_backend: Any = None,
                 rollout_backend_kind: str | None = None, isolation_override: str | None = None,
                 approve_agent_spend: bool = False, approve_gpu: bool = False,
                 tick_interval_s: float = 1.0, poll_interval_s: float | None = None,
                 term_grace_s: float = 15.0, drain_timeout_s: float | None = None,
                 backoff_s: tuple[float, ...] = (60.0, 120.0, 240.0, 300.0),
                 model_down_after_s: float = 600.0, flag_grace_s: float = 60.0,
                 extra_call_ledger: str | None = None) -> None:
        self.paths = paths
        self.adapter = adapter
        self.clock = clock
        self.isolation_override = isolation_override
        self.agent_backend = agent_backend or make_agent_backend(
            {**adapter.agent, "settings_file": str(paths.bin_dir / "agent-settings.json")}
            if adapter.agent.get("backend") == "claude" else adapter.agent, adapter.price_table())
        self.rollout_kind = rollout_backend_kind or adapter.rollout.get("backend")
        self.rollout_backend = rollout_backend
        self.approve_agent_spend = approve_agent_spend
        self.approve_gpu = approve_gpu
        self.tick_interval_s = tick_interval_s
        self.poll_interval_s = poll_interval_s if poll_interval_s is not None else float(
            adapter.rollout.get("poll_interval_s", 10.0))
        self.term_grace_s = term_grace_s
        self.drain_timeout_s = drain_timeout_s if drain_timeout_s is not None else float(
            adapter.rollout.get("max_wall_s", 3600))
        self.backoff_s = backoff_s
        self.model_down_after_s = model_down_after_s
        self.flag_grace_s = flag_grace_s
        self.extra_call_ledger = extra_call_ledger or adapter.agent.get("call_ledger_extra")
        self.interrupted = False
        self.fatal: str | None = None
        self.budgets = adapter.budgets
        self._last_state_save = float("-inf")

    # ------------------------------------------------------------------ helpers

    def _now(self) -> float:
        return self.clock.now()

    def _cost(self) -> float:
        return self._known_cost() + sum(self.state.bounded_unknown_costs.values())

    def _known_cost(self) -> float:
        return float(self.state.data.get("cost_usd_known") or 0.0)

    def _check_spend(self) -> None:
        if self.agent_backend.name != "fake" and not self.approve_agent_spend:
            raise HarnessError(f"running a {self.agent_backend.name} agent spends API budget; "
                               "pass --approve-agent-spend after the user approved it")
        if self.agent_backend.name == "codex" and self.budgets.max_usd is not None:
            raise HarnessError("Codex CLI has no per-session USD hard cap; a max_usd budget "
                               "would not be enforceable. Use an explicitly approved unpriced "
                               "round with max_usd=null and bounded sessions/wall/rollouts, "
                               "or use a backend with a hard spend limit")
        kind = getattr(self.rollout_backend, "name", self.rollout_kind)
        if kind == "command" and not self.approve_gpu:
            raise HarnessError("the command rollout backend submits GPU jobs; pass --approve-gpu "
                               "after the user approved it")

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.interrupted = True

    # ------------------------------------------------------------------ run

    def run(self) -> dict[str, Any]:
        with round_lock(self.paths):
            return self._run_locked()

    def _run_locked(self) -> dict[str, Any]:
        paths, adapter = self.paths, self.adapter
        round_json = read_json(paths.round_json)
        if round_json["adapter"]["sha256"] != adapter.sha256:
            raise HarnessError("the adapter changed since this round was prepared; run with the "
                               "original adapter or prepare a new round")
        self.state = RoundState.load(paths.state)
        state = self.state
        self.events = EventLog(paths, state.data["round_id"], self.clock.now)
        if state.status in TERMINAL:
            return self.summary()
        self._check_spend()
        self.sandbox = agent_sandbox(adapter, paths, override=self.isolation_override)
        self.guard = ManifestGuard(paths, manifest_entries(adapter, paths), state, self.events, self.clock.now,
                                   extra={"harness_code_sha256": harness_code_sha256()})
        now = self._now()
        if state.status == "prepared":
            state.status = "running"
            state.data["wall_started_at"] = now
            state.data["run_host"] = socket.gethostname()
            state.save()
            self.guard.record_start()
            self.events.emit("round_started", agent_backend=self.agent_backend.name,
                             model=self.agent_backend.model, isolation=self.sandbox.mode,
                             degraded=self.sandbox.degraded)
            if not self._probe_isolation():
                return self._finalize("error")
        else:
            if state.status == "paused_model_down":
                if state.data.get("paused_at") is not None:
                    state.data["paused_s"] = float(state.data.get("paused_s") or 0.0) + (now - float(state.data["paused_at"]))
                    state.data["last_tick_at"] = now  # pause already covers downtime
                state.data["paused_at"] = None
                state.data["outage"] = None
                state.status = "running"
                self.events.emit("round_resumed", after="paused_model_down")
            last = state.data.get("last_tick_at")
            if last is not None and now - float(last) > 120:
                state.data["paused_s"] = float(state.data.get("paused_s") or 0.0) + (now - float(last))
                self.events.emit("harness_downtime", seconds=round(now - float(last)))
            state.save()
            ok, _ = self.guard.check("resume")
            if not ok:
                return self._finalize("invalid_tamper")
            self._close_dangling_session()
        if self.rollout_backend is None:
            self.rollout_backend = make_backend(adapter, kind=self.rollout_kind,
                                                masked=rollout_masks(adapter, paths),
                                                isolation_override=self.isolation_override,
                                                clock_now=self.clock.now)
        seeds = {s.seed for s in load_holdout_seeds(paths.holdout_seeds)}
        self.broker = Broker(paths=paths, adapter=adapter, state=state, backend=self.rollout_backend,
                             events=self.events, clock=self.clock, holdout_seeds=seeds, cost_fn=self._known_cost,
                             poll_interval_s=self.poll_interval_s)
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            for sig in previous:
                with contextlib.suppress(ValueError):
                    signal.signal(sig, self._handle_signal)
            reason = self._loop()
        finally:
            for sig, handler in previous.items():
                with contextlib.suppress(ValueError, TypeError):
                    signal.signal(sig, handler)
        if reason in ("interrupted", "paused_model_down"):
            state.data["last_tick_at"] = self._now()
            state.save()
            self.broker.write_status(self._now(), force=True)
            return self.summary(exit_code=EXIT_PAUSED)
        return self._finalize(reason)

    def _probe_isolation(self) -> bool:
        paths = self.paths
        if self.sandbox.mode == "none":
            self.state.data["isolation"]["agent"] = {"mode": "none", "ok": False}
            self.state.violation("isolation", "agent ran without OS isolation", utc_now(self._now()))
            self.state.save()
            self.events.emit("isolation_probe", sandbox="agent", mode="none", ok=False)
            return True  # explicitly configured; recorded as a violation
        masked = [str(paths.harness), str(paths.holdout_seeds)]
        masked += [p for p in self.adapter.protected_paths() if os.path.exists(p)][:60]
        readonly = [str(paths.rollouts), str(paths.ledger)]
        readonly += [p for p in self.adapter.readable_paths() if os.path.isdir(p)][:3]
        writable = [str(paths.workspace), str(paths.requests)]
        env = sandbox_env(self._base_env(), [])
        result = run_probe(self.sandbox, masked=masked, readonly_dirs=readonly, writable_dirs=writable, env=env)
        summary = {"mode": result.mode, "ok": result.ok, "degraded": result.degraded,
                   "checked_masks": len(masked), "leaks": len(result.leaks), "umounted": len(result.umounted),
                   "writable_readonly": len(result.writable_readonly),
                   "unwritable_workspace": len(result.unwritable_workspace), "error": result.error}
        self.state.data["isolation"]["agent"] = summary
        if not result.ok:
            self.state.violation("isolation", f"agent sandbox probe failed: {summary}", utc_now(self._now()))
            self.fatal = "isolation probe failed"
        self.state.save()
        self.events.emit("isolation_probe", sandbox="agent", **{k: summary[k] for k in ("mode", "ok", "degraded")})
        self.events.emit("isolation_probe_detail", private=True, result=result.to_dict())
        return result.ok

    def _base_env(self) -> dict[str, str]:
        paths = self.paths
        env = {"HOME": str(paths.agent_home), "PATH": f"{paths.bin_dir}:/usr/local/bin:/usr/bin:/bin",
               "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TMPDIR": "/tmp", "TERM": "dumb", "USER": "agent",
               "XH_ROUND_DIR": str(paths.root), "XH_TOOL": str(paths.tool), "PYTHONDONTWRITEBYTECODE": "1"}
        if self.agent_backend.name == "claude":
            env["CLAUDE_CONFIG_DIR"] = str(paths.agent_home / ".claude")
        if self.agent_backend.name in ("codex", "codex-metered"):
            env["CODEX_HOME"] = str(paths.agent_home / ".codex")
        env.update({k: str(v) for k, v in (self.adapter.agent.get("env") or {}).items()})
        env.update(self.agent_backend.env())
        return env

    # ------------------------------------------------------------------ loop

    def _tick(self) -> None:
        self.broker.tick()
        now = self._now()
        self.state.data["last_tick_at"] = now
        if now - self._last_state_save > 30:  # heartbeat for downtime accounting after a crash
            self._last_state_save = now
            self.state.save()

    def _stop_reason(self) -> str | None:
        state, b = self.state, self.budgets
        if self.interrupted:
            return "interrupted"
        if not state.data["manifest"].get("ok", True):
            return "invalid_tamper"
        if self.fatal:
            return "error"
        if (self.agent_backend.name == "codex-metered"
                and state.data.get("sessions_unknown_cost", 0) > len(state.bounded_unknown_costs)):
            return "cost_unknown"
        stop = state.stop_reason
        if stop == "flagged_for_human":
            return "flagged_for_human"
        if stop == "done":
            return "completed"
        if stop == "infra_exhausted":
            return "infra_exhausted"
        now = self._now()
        if state.active_wall_s(now) >= b.total_wall_s:
            return "wall_exhausted"
        if b.max_usd is not None and self._cost() >= b.max_usd:
            return "usd_exhausted"
        if sum(1 for s in state.data["sessions"] if not s.get("api_outage")) >= b.max_sessions:
            return "sessions_exhausted"
        if int(state.data.get("idle_sessions", 0)) >= b.max_idle_sessions:
            return "stalled"
        if self._rollouts_exhausted() and state.data.get("final_call_used"):
            return "rollouts_exhausted"
        return None

    def _rollouts_exhausted(self) -> bool:
        return (self.broker.counted() >= self.budgets.max_counted_rollouts
                and not self.broker.in_flight_ids())

    def _session_stop(self) -> str | None:
        """Checked every tick while a session runs."""

        if self.interrupted:
            return "interrupted"
        now = self._now()
        stop = self.state.data.get("stop")
        if stop:
            waited = now - _ts(stop["utc"], now)
            if stop["reason"] == "flagged_for_human" and waited > self.flag_grace_s:
                return "flagged_for_human"
            if stop["reason"] == "done" and waited > self.budgets.final_grace_s:
                return "final_grace_exceeded"
        if self.state.active_wall_s(now) >= self.budgets.total_wall_s:
            return "round_wall"
        return None

    def _loop(self) -> str:
        consecutive_failures = 0
        while True:
            self.broker.session_index = len(self.state.data["sessions"])
            self._tick()
            self.broker.tick(force_poll=True)
            reason = self._stop_reason()
            if reason:
                return reason
            session = self._run_session()
            if session.get("fatal"):
                self.fatal = session["fatal"]
                continue
            if session.get("api_outage"):
                outage = self.state.data.get("outage") or {"started_at": session["started_at_ts"], "retries": 0}
                outage["retries"] = int(outage["retries"]) + 1
                self.state.data["outage"] = outage
                self.state.save()
                elapsed = self._now() - float(outage["started_at"])
                if elapsed >= self.model_down_after_s:
                    self.state.status = "paused_model_down"
                    self.state.data["paused_at"] = float(outage["started_at"])
                    self.state.data["outage"] = None
                    self.state.save()
                    self.events.emit("paused_model_down", outage_s=round(elapsed), retries=outage["retries"])
                    return "paused_model_down"
                delay = self.backoff_s[min(outage["retries"] - 1, len(self.backoff_s) - 1)]
                self.events.emit("api_outage_backoff", retries=outage["retries"], sleep_s=delay)
                self._sleep_ticking(delay)
                continue
            if session.get("failed"):
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    self.fatal = f"three consecutive failed sessions ({session.get('error_type')})"
            else:
                consecutive_failures = 0

    def _sleep_ticking(self, seconds: float) -> None:
        end = self._now() + seconds
        while self._now() < end and not self.interrupted:
            self._tick()
            self.clock.sleep(min(self.tick_interval_s * 5, max(0.0, end - self._now())))

    def _finished_since(self, since_ts: float | None) -> str:
        items = []
        for a in self.state.data["attempts"].values():
            outcome = a.get("outcome") or {}
            if a.get("ended_at_ts") is None or (since_ts is not None and float(a["ended_at_ts"]) <= since_ts):
                continue
            fields = outcome.get("fields") or {}
            short = ", ".join(f"{k}={v}" for k, v in list(fields.items())[:4])
            items.append(f"- {a['attempt_id']} seed {a['seed']} ({outcome.get('family')}): "
                         f"{outcome.get('outcome_class')}, success={outcome.get('success')}"
                         + (f", {short}" if short else ""))
        if not items:
            return ""
        return "Finished since your last session:\n" + "\n".join(items[-20:]) + "\n"

    def _prompt(self, index: int, final_call: bool, wall_s: float) -> str:
        paths, state = self.paths, self.state
        if not state.data.get("agent_session_id"):
            round_json = read_json(paths.round_json)
            return FIRST_PROMPT.format(round_id=round_json["round_id"], task_id=self.adapter.task_id,
                                       workspace=paths.workspace, brief=paths.brief,
                                       brief_text=paths.brief.read_text(encoding="utf-8"))
        if final_call:
            return FINAL_CALL_PROMPT.format(minutes=max(1, int(wall_s // 60)))
        now = self._now()
        view = self.broker.budget_view(now)
        previous = state.data["sessions"][-1] if state.data["sessions"] else None
        since = previous.get("ended_at_ts") if previous else None
        status = self.broker.status_view(now)
        messages = "".join(f"Note: {m}\n" for m in status["messages"])
        return CONTINUE_PROMPT.format(
            index=1 + sum(not s.get("api_outage") for s in state.data["sessions"]),
            max_sessions=self.budgets.max_sessions,
            counted=view["counted_rollouts"]["used"], max_counted=view["counted_rollouts"]["max"],
            in_flight=view["counted_rollouts"]["in_flight"], max_concurrent=self.budgets.max_concurrent_rollouts,
            active=human_duration(view["wall_s"]["active"]), total=human_duration(self.budgets.total_wall_s),
            usd=self._cost(), usd_max="" if self.budgets.max_usd is None else f" of ${self.budgets.max_usd:g}",
            finished=self._finished_since(since), messages=messages, brief=paths.brief,
            wait_cap=int(self.budgets.wait_cap_s))

    def _run_session(self) -> dict[str, Any]:
        paths, state, b = self.paths, self.state, self.budgets
        now = self._now()
        index = len(state.data["sessions"]) + 1
        self.broker.session_index = index
        remaining = max(60.0, b.total_wall_s - state.active_wall_s(now))
        final_call = self._rollouts_exhausted() and not state.data.get("final_call_used")
        wall_s = min(b.wall_per_session_s, remaining + 30.0)
        if final_call:
            wall_s = min(wall_s, b.final_grace_s)
            state.data["final_call_used"] = True
        resume_id = state.data.get("agent_session_id")
        new_id = None if resume_id or self.agent_backend.name in ("codex", "codex-metered") else str(uuid.uuid4())
        from datafarm.agent.context import build_context
        project_context = build_context('task-expert-author')
        prompt = project_context['messages'][0]['content'] + "\n\n" + self._prompt(index, final_call, wall_s)
        if self.agent_backend.name == 'codex-metered':
            prompt = ('The complete repository task-expert-author instruction bundle is loaded below and its hashes are recorded by the harness. Do not regenerate context or load unrelated orchestration stages. Work only through the granted files and xh broker. All model traffic must use the inherited xh_metered provider.\n\n' + prompt)
        if resume_id and self.agent_backend.name == 'codex-metered':
            prompt = self._prompt(index, final_call, wall_s)
        note_path = paths.harness / 'operator-diagnostic.txt'
        if os.path.lexists(note_path):
            note = read_regular_nofollow(paths.harness, note_path.name, max_bytes=8192).decode('utf-8')
            snapshot = paths.sessions / f'session-{index:03d}.operator-diagnostic.txt'
            snapshot.write_text(note, encoding='utf-8')
            snapshot.chmod(0o600)
            self.events.emit('operator_diagnostic', session=index, sha256=sha256_text(note),
                             bytes=len(note.encode('utf-8')))
            prompt += ('\n\nOperator diagnostic (additional evidence, not a verifier verdict or a '
                       'change to the task, physics, budgets or acceptance criteria):\n' + note)
            project_context['receipt']['instruction_bundle_reused_from_conversation'] = True
        project_context['receipt']['expanded_prompt_sha256'] = sha256_text(prompt)
        project_context['receipt']['session_templates'] = {
            name: sha256_file(repository_root() / 'prompts/sessions' / name)
            for name in ('expert-first.txt', 'expert-continue.txt', 'expert-final.txt')}
        atomic_write_json(paths.root / f"agent-instructions-{index:03d}.json", project_context['receipt'])
        max_budget = None if b.max_usd is None else max(0.01, b.max_usd - self._cost())
        spec = SessionSpec(index=index, prompt=prompt, cwd=str(paths.workspace), resume_id=resume_id,
                           new_session_id=new_id, max_budget_usd=max_budget)
        record = {"index": index, "started_at": utc_now(now), "started_at_ts": now,
                  "planned_session_id": resume_id or new_id, "resumed": bool(resume_id),
                  "final_call": final_call, "prompt_sha256": sha256_text(prompt), "ended_at": None}
        # an outage that ends here is not charged to the round
        outage = state.data.get("outage")
        state.data["sessions"].append(record)
        state.save()
        self.events.emit("session_started", index=index, resumed=bool(resume_id), final_call=final_call,
                         wall_cap_s=round(wall_s))
        requests_before = int(state.counters.get("rejected", 0)) + int(state.data["next_attempt"]) + int(
            state.data["next_submission"])
        with self.agent_backend.session(spec, paths.sessions):
            inner = self.agent_backend.command(spec)
            argv = wrap(self.sandbox, inner)
            env = sandbox_env(self._base_env(), self.adapter.agent.get("auth", {}).get("env_passthrough", []))
            transcript = paths.sessions / f"session-{index:03d}.jsonl"
            stderr_path = paths.sessions / f"session-{index:03d}.stderr.log"
            parser = self.agent_backend.parser()
            runner = SessionRunner(tick=self._tick, should_stop=self._session_stop,
                                   tick_interval_s=self.tick_interval_s, term_grace_s=self.term_grace_s)
            info = runner.run(argv, env=env, cwd=str(paths.workspace),
                              stdin_text=prompt if self.agent_backend.uses_stdin_prompt else None,
                              transcript=transcript, stderr_path=stderr_path, wall_s=wall_s, parser=parser)
        parsed = self.agent_backend.settle(parser.finish())
        if self.agent_backend.name == "codex-metered" and self.agent_backend.receipt.get("denied_local"):
            self.fatal = "metered request reservation denied; inspect compact handoff before resuming"
        api_outage, auth_error, error_type = classify_errors(parsed, info.stderr_tail, info.exit_code)
        end = self._now()
        if outage and not api_outage:
            state.data["paused_s"] = float(state.data.get("paused_s") or 0.0) + max(
                0.0, now - float(outage["started_at"]))
            state.data["outage"] = None
        no_conversation = bool(parsed.result_text and "No conversation found" in parsed.result_text)
        if info.stop_reason == "wall_cap":
            status = "timeout"
        elif info.stop_reason in ("flagged_for_human", "final_grace_exceeded", "round_wall", "interrupted"):
            status = "aborted"
        elif error_type == "max_budget":
            status = "budget_exhausted"
        elif error_type:
            status = "error"
        else:
            status = "ok"
        call = AgentCallRecord(
            call_id=new_call_id(), component="expert-harness/session", backend=self.agent_backend.call_backend,
            model=parsed.model or self.agent_backend.model, started_at=record["started_at"], status=status,
            ended_at=utc_now(end), prompt_sha256=record["prompt_sha256"],
            response_sha256=sha256_file(transcript) if transcript.exists() else None,
            session_id=parsed.session_id or record["planned_session_id"],
            input_tokens=parsed.input_tokens, output_tokens=parsed.output_tokens,
            cache_read_tokens=parsed.cache_read_tokens, cache_write_tokens=parsed.cache_write_tokens,
            cost_usd=parsed.cost_usd, cost_source=parsed.cost_source, turns=parsed.turns,
            latency_s=round(info.wall_s, 3), error_type=error_type, run_id=state.data["round_id"], attempt=index,
            extra={"stop_reason": info.stop_reason, "exit_code": info.exit_code, "tool_uses": parsed.tool_uses,
                   "resumed": bool(resume_id), "final_call": final_call, "api_outage": api_outage,
                   "transcript": str(transcript.relative_to(paths.root)), "isolation": self.sandbox.mode,
                   "task_id": self.adapter.task_id})
        append_call(paths.agent_calls, call)
        if self.extra_call_ledger:
            append_call(self.extra_call_ledger, call)
        record.update({
            "ended_at": utc_now(end), "ended_at_ts": end, "wall_s": round(info.wall_s, 1),
            "session_id": parsed.session_id, "cost_usd": parsed.cost_usd, "cost_source": parsed.cost_source,
            "input_tokens": parsed.input_tokens, "output_tokens": parsed.output_tokens,
            "cache_read_tokens": parsed.cache_read_tokens, "cache_write_tokens": parsed.cache_write_tokens,
            "turns": parsed.turns, "tool_uses": parsed.tool_uses, "stop_reason": info.stop_reason,
            "exit_code": info.exit_code, "status": status, "error_type": error_type, "api_outage": api_outage,
            "call_id": call.call_id})
        if parsed.cost_usd is not None:
            state.data["cost_usd_known"] = round(self._known_cost() + parsed.cost_usd, 6)
        else:
            state.data["sessions_unknown_cost"] = int(state.data.get("sessions_unknown_cost", 0)) + 1
        if no_conversation:
            state.data["agent_session_id"] = None  # start a fresh conversation next time
        elif parsed.session_id or record["planned_session_id"]:
            if parsed.events > 0 or resume_id:
                state.data["agent_session_id"] = parsed.session_id or record["planned_session_id"]
        requests_after = int(state.counters.get("rejected", 0)) + int(state.data["next_attempt"]) + int(
            state.data["next_submission"])
        if api_outage or requests_after > requests_before or parsed.tool_uses > 0 or final_call:
            state.data["idle_sessions"] = 0
        else:
            state.data["idle_sessions"] = int(state.data.get("idle_sessions", 0)) + 1
        state.save()
        self.events.emit("session_ended", index=index, status=status, stop_reason=info.stop_reason,
                         wall_s=round(info.wall_s, 1), cost_usd=parsed.cost_usd, turns=parsed.turns,
                         tool_uses=parsed.tool_uses, error_type=error_type)
        self.guard.check(f"session-{index}")
        result = {"index": index, "started_at_ts": now, "api_outage": api_outage,
                  "failed": status == "error" and not api_outage, "error_type": error_type}
        if auth_error:
            result["fatal"] = "agent CLI authentication failed inside the sandbox"
        return result

    def _close_dangling_session(self) -> None:
        """A harness crash can leave a started session without a ledger record."""

        sessions = self.state.data["sessions"]
        if sessions and sessions[-1].get("ended_at") is None:
            record = sessions[-1]
            now = self._now()
            call = AgentCallRecord(
                call_id=new_call_id(), component="expert-harness/session", backend=self.agent_backend.call_backend,
                model=self.agent_backend.model, started_at=record["started_at"], status="aborted",
                ended_at=utc_now(now), prompt_sha256=record.get("prompt_sha256"),
                session_id=record.get("planned_session_id"), error_type="harness_restart",
                run_id=self.state.data["round_id"], attempt=record["index"],
                extra={"stop_reason": "harness_restart"})
            append_call(self.paths.agent_calls, call)
            record.update({"ended_at": utc_now(now), "ended_at_ts": now, "status": "aborted",
                           "stop_reason": "harness_restart", "cost_usd": None, "call_id": call.call_id})
            self.state.data["sessions_unknown_cost"] = int(self.state.data.get("sessions_unknown_cost", 0)) + 1
            if record.get("planned_session_id"):
                self.state.data["agent_session_id"] = record["planned_session_id"]
            self.state.save()
            self.events.emit("session_closed_after_restart", index=record["index"])

    # ------------------------------------------------------------------ finalize

    def _auto_final(self) -> None:
        if any(s.get("final") for s in self.state.data["submissions"]):
            return
        policy = self.adapter.expert_policy
        rel = policy.get("default_file", "expert.py")
        try:
            data = read_regular_nofollow(self.paths.workspace, rel, max_bytes=int(policy.get("max_bytes", 400_000)))
        except (UnsafePathError, OSError) as exc:
            self.events.emit("auto_final_skipped", reason=str(exc)[:200])
            return
        class_name = policy.get("default_class")
        subs = self.state.data["submissions"]
        if subs:
            class_name = subs[-1]["expert_class"]
        audit = audit_expert_source(data, class_name, policy)
        if not audit.ok:
            self.events.emit("auto_final_skipped", reason="audit: " + audit.summary()[:200])
            return
        self.broker.record_submission(data, class_name, rel, note="harness snapshot of the workspace "
                                      "expert at round end (the agent made no final submission)",
                                      final=True, request_id=None, source="harness_auto_final")

    def _finalize(self, reason: str) -> dict[str, Any]:
        state = self.state
        status = {"completed": "completed", "interrupted": "running"}.get(reason, reason)
        if hasattr(self, "broker"):
            abandoned = self.broker.drain(self.drain_timeout_s, sleep_s=min(5.0, self.poll_interval_s))
            if abandoned:
                self.events.emit("rollouts_abandoned", attempts=abandoned)
            if reason not in ("invalid_tamper", "error", "cost_unknown", "flagged_for_human"):
                self._auto_final()
        if hasattr(self, "guard") and state.data["manifest"].get("start_digest"):
            ok, _ = self.guard.check("end")
            if not ok:
                status = "invalid_tamper"
        if status not in TERMINAL:
            status = "error"
        now = self._now()
        state.status = status
        state.data["ended_at"] = utc_now(now)
        state.data["active_wall_s_final"] = round(state.active_wall_s(now), 1)
        state.data["end_reason"] = reason
        state.save()
        if hasattr(self, "broker"):
            self.broker.write_status(now, force=True)
        summary = self.summary()
        atomic_write_json(self.paths.round_result, summary, mode=0o600)
        self.events.emit("round_finished", status=status, reason=reason,
                         counted=state.counters.get("counted"), successes=state.counters.get("successes"),
                         infra=state.counters.get("infra"), sessions=len(state.data["sessions"]),
                         cost_usd_known=self._known_cost(), budget_used_usd=self._cost(),
                         submissions=len(state.data["submissions"]))
        return summary

    def summary(self, exit_code: int = 0) -> dict[str, Any]:
        return round_status(self.paths, exit_code=exit_code, now=self._now())


def _ts(utc: str, default: float) -> float:
    try:
        return dt.datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return default


def round_status(paths: RoundPaths, *, exit_code: int = 0, now: float | None = None) -> dict[str, Any]:
    """Operator view (may include private facts such as holdout presence, never seeds)."""

    state = RoundState.load(paths.state)
    data = state.data
    calls = read_calls(paths.agent_calls) if paths.agent_calls.exists() else []
    now = time.time() if now is None else now
    finals = [s for s in data["submissions"] if s.get("final")]
    return {
        "round_id": data["round_id"], "status": data["status"], "exit_code": exit_code,
        "end_reason": data.get("end_reason"), "stop": data.get("stop"), "flag": data.get("flag"),
        "counters": data["counters"], "sessions": len(data["sessions"]),
        "active_wall_s": (data["active_wall_s_final"] if data.get("active_wall_s_final") is not None
                          else round(state.active_wall_s(now), 1)),
        "cost_usd_known": data.get("cost_usd_known"), "sessions_unknown_cost": data.get("sessions_unknown_cost"),
        "unknown_liability_usd": sum(state.bounded_unknown_costs.values()),
        "agent_calls": summarize(calls) if calls else None,
        "submissions": [{"submission_id": s["submission_id"], "final": s["final"], "source": s.get("source"),
                         "expert_sha256": s["expert_sha256"][:16]} for s in data["submissions"]],
        "final_submission": finals[-1]["submission_id"] if finals else None,
        "attempts": {"total": len(data["attempts"]),
                     "in_flight": sum(1 for a in data["attempts"].values() if a["status"] in ("pending_submit", "running"))},
        "max_in_flight_observed": data.get("max_in_flight_observed"),
        "violations": data.get("violations", []), "manifest": data.get("manifest"),
        "isolation": data.get("isolation"), "amendments": data.get("amendments", []),
        "holdout_evaluated": paths.holdout_results.exists(),
    }
