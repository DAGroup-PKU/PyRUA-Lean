"""Rollout backends: run one agent-written expert on one seed and classify the outcome.

A rollout is one generated job script per attempt (``job.sh`` in a harness-owned run
directory):

1. heartbeat + environment record (digests of physics/material files);
2. an isolation probe inside the exact sandbox the expert will run in (masked paths must
   stay hidden even after an ``umount`` attempt), fail closed on a leak;
3. the simulator step **inside** the sandbox (agent code executes only here; the run-dir
   root is read-only, only declared working directories and caches are writable);
   when the runner refuses a pre-existing output directory, it can write a new
   child of a writable working directory and the harness moves it afterward;
4. the judge steps **outside** the sandbox (verifier code is masked from step 3), writing
   verdict files into the run-dir root that step 3 could not touch;
5. ``harness-job.json`` with exit codes, written last.

Outcomes are classified by the harness, not by the agent: judged (counted), expert error
(counted: e.g. the trace capacity was exceeded or the plugin raised), and infrastructure
classes that are not counted (invalid initial state, simulator start-up failure, judge
failure, lost node, timeout, isolation failure, submission failure).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shlex
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from datafarm.expert.adapter import TaskAdapter
from datafarm.expert.isolation import PROBE_SCRIPT, job_wrapper_bash, make_empty_file
from datafarm.expert.util import ConfigError, HarnessError, get_dotted, read_json, sha256_bytes
from datafarm.common import sha256_text

JOB_SCHEMA = "datafarm.expert-harness.job/1"

COUNTED_CLASSES = frozenset({"judged", "expert_error"})
_STATUS_DETAIL = {"lost": "no heartbeat and no result; node or process gone",
                  "timeout": "no result within the rollout wall limit",
                  "submit_failed": "could not submit the job within the submission window",
                  "abandoned": "still running when the round ended; not waited for"}


@dataclasses.dataclass
class AttemptSpec:
    attempt_id: str
    purpose: str  # dev | holdout
    seed: int
    expert_class: str
    run_dir: Path
    job_name: str

    @property
    def expert_path(self) -> Path:
        return self.run_dir / "expert_snapshot.py"


@dataclasses.dataclass
class Outcome:
    outcome_class: str
    counted: bool
    success: bool | None
    fields: dict[str, Any]
    family: str | None
    init_valid: bool | None
    sim_exit: int | None
    wall_s: float | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class RolloutBackend(Protocol):
    name: str

    def launch(self, spec: AttemptSpec) -> dict[str, Any]: ...

    def poll(self, spec: AttemptSpec, receipt: Mapping[str, Any], now: float) -> str: ...

    def outcome(self, spec: AttemptSpec, receipt: Mapping[str, Any], status: str) -> Outcome: ...


class LaunchError(HarnessError):
    """The rollout could not be started (no GPU free, submit failed). Not an attempt."""


# --------------------------------------------------------------------------- classification


def _read_json_quiet(path: Path) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def _log_tail(path: Path, limit: int = 400_000) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def classify_run_dir(run_dir: Path, rollout_cfg: Mapping[str, Any], *, status: str = "finished") -> Outcome:
    """Classify a finished (or lost / timed-out) job directory."""

    if status in _STATUS_DETAIL:
        return Outcome(status, False, None, {}, None, None, None, None, _STATUS_DETAIL[status])
    job = _read_json_quiet(run_dir / "harness-job.json") or {}
    init_cfg = rollout_cfg.get("init_state", {})
    init = _read_json_quiet(run_dir / init_cfg["file"]) if init_cfg.get("file") else None
    family = get_dotted(init, init_cfg.get("family", "family")) if isinstance(init, dict) else None
    valid = get_dotted(init, init_cfg.get("valid", "valid")) if isinstance(init, dict) else None
    sim_exit = job.get("sim_exit")
    wall = job.get("sim_wall_s")

    def out(cls: str, success: bool | None, fields: dict[str, Any], detail: str) -> Outcome:
        return Outcome(cls, cls in COUNTED_CLASSES, success, fields, family,
                       None if valid is None else bool(valid), sim_exit, wall, detail)

    if job.get("isolation") == "unavailable":
        return out("isolation_unavailable", None, {}, "no usable sandbox on the execution node")
    if job.get("isolation") == "failed":
        return out("isolation_failed", None, {}, "sandbox probe found a readable masked path; "
                                                 "the expert was not run")
    if valid is False:
        return out("init_invalid", None, {}, "the initial-state generator reported an invalid start")
    verdict_cfg = rollout_cfg.get("verdict", {})
    verdict = _read_json_quiet(run_dir / verdict_cfg.get("file", "verdict.json"))
    if isinstance(verdict, dict) and get_dotted(verdict, verdict_cfg["success"]) is not None:
        fields = {name: get_dotted(verdict, path) for name, path in verdict_cfg.get("fields", {}).items()}
        success = bool(get_dotted(verdict, verdict_cfg["success"]))
        return out("judged", success, fields, "judged by " + str(verdict_cfg.get("name", "verifier")))
    classify = rollout_cfg.get("classify", {})
    log = _log_tail(run_dir / rollout_cfg.get("sim", {}).get("log", "sim.log"))
    # Native crashes often echo the entire command, including expert_snapshot.py.
    # That filename alone does not establish a Python expert exception.
    if sim_exit in (-11, 139, -6, 134, -9, 137, -15, 143, -25, 153):
        return out("sim_failure", None, {}, "simulator native crash or signal termination; no task verdict")
    for pattern in classify.get("infra_regex", []):
        match = re.search(pattern, log)
        if match:
            return out("sim_failure", None, {}, f"infrastructure pattern {pattern!r} in the log")
    requires = rollout_cfg.get("sim", {}).get("requires_output")
    produced = bool(requires) and (run_dir / requires).exists()
    failure = (_read_json_quiet(run_dir / Path(requires).parent / "production_failure.json")
               if requires else None)
    if (isinstance(failure, dict) and failure.get("expert_phase") is None
            and failure.get("completed_primitives") == 0
            and not re.search(r'File ["\'][^"\']*expert_snapshot\.py["\']', str(failure.get("traceback", "")))):
        return out("sim_failure", None, {}, "structured simulator failure before the expert was constructed: "
                   + str(failure.get("exception_message", "unknown"))[:200])
    for pattern in classify.get("expert_error_regex", []):
        if pattern == r"expert_snapshot\.py":
            # The launcher prints this path in its command on timeout too.
            # Require an actual Python traceback frame for this classifier.
            pattern = r'File ["\'][^"\']*expert_snapshot\.py["\']'
        match = re.search(pattern, log)
        if match:
            line = log[max(0, match.start() - 200): match.end() + 200].strip().splitlines()
            return out("expert_error", False, {}, "expert error: " + (line[-1] if line else pattern)[:300])
    if sim_exit not in (0, None) or not produced:
        return out("sim_failure", None, {}, f"simulator exit {sim_exit} without an expert error "
                                           "signature; treated as infrastructure")
    return out("judge_failure", None, {}, "the simulator finished but no verdict was produced")


# --------------------------------------------------------------------------- job script


def _q(value: str) -> str:
    """Quote for bash, leaving ``${VAR}`` references expandable."""

    if "${" in value:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"'
    return shlex.quote(value)


def build_job_script(adapter: TaskAdapter, spec: AttemptSpec, *, masked: Sequence[str],
                     isolation_override: str | None = None) -> str:
    cfg = adapter.rollout
    iso = adapter.isolation("rollout")
    values = {"run_dir": str(spec.run_dir), "seed": spec.seed, "expert_class": spec.expert_class,
              "expert_file": str(spec.expert_path), "attempt_id": spec.attempt_id,
              "job_name": spec.job_name, "purpose": spec.purpose}
    sim = cfg["sim"]
    sim_argv = adapter.x(sim["argv"], keep_env=True, **values)
    sim_cwd = adapter.x(sim.get("cwd", "{run_dir}"), keep_env=True, **values)
    log = sim.get("log", "sim.log")
    timeout_s = int(sim.get("timeout_s", 3600))
    writable = [str(spec.run_dir / d) for d in sim.get("writable_run_subdirs", ["native", "native_workdir"])]
    writable += adapter.shell_paths(iso.get("writable", []))
    mode = isolation_override or iso.get("mode", "bwrap")
    fallback = None if isolation_override else iso.get("fallback")
    run_as = iso.get("run_as")
    wrapper = job_wrapper_bash(mode=mode, fallback=fallback, base=iso.get("base", "root_ro"),
                               dev=iso.get("dev", "host"),
                               ro=[str(spec.run_dir), *adapter.shell_paths(iso.get("readable", []))],
                               rw=writable,
                               masked=list(masked),
                               tmpfs=adapter.shell_paths(iso.get("tmpfs", [])),
                               run_as=(run_as["uid"], run_as["gid"]) if run_as else None,
                               share_net=bool(iso.get("share_net", True)),
                               private_tmp=bool(iso.get("private_tmp", True)),
                               keep_caps=tuple(iso.get("keep_caps", ())))
    probe_args = " ".join(_q("m:" + p) for p in masked)
    record = " ".join(_q(p) for p in adapter.shell_paths(cfg.get("record_digests", [])))
    judge_lines = []
    verdict_outputs = set()
    for step in cfg.get("judge", []):
        argv = adapter.x(step["argv"], keep_env=True, **values)
        stdout = step.get("stdout", f"judge-{step['name']}.stdout")
        for output in step.get("outputs", []):
            verdict_outputs.add(output)
        judge_lines.append(" ".join(_q(a) for a in argv) + f" > {_q(str(spec.run_dir / stdout))} 2>&1; "
                           "XH_J+=($?)")
    requires = sim.get("requires_output")
    post_sim_move = sim.get("post_sim_move")
    move_lines: list[str] = []
    if post_sim_move:
        source_rel = Path(post_sim_move["from"])
        target_rel = Path(post_sim_move["to"])
        if (source_rel.is_absolute() or target_rel.is_absolute()
                or ".." in source_rel.parts or ".." in target_rel.parts
                or source_rel == target_rel):
            raise ConfigError("rollout.sim.post_sim_move must use distinct relative run-dir paths")
        source_path = spec.run_dir / source_rel
        target_path = spec.run_dir / target_rel
        move_lines = [
            '  if [ "$SIM_EXIT" = 0 ]; then',
            f'    if [ -d {_q(str(source_path))} ] && [ ! -e {_q(str(target_path))} ]; then',
            f'      mv -- {_q(str(source_path))} {_q(str(target_path))} || SIM_EXIT=95',
            '    else SIM_EXIT=95; fi',
            '  fi',
        ]
    judge_block = "\n  ".join(judge_lines) if judge_lines else ":"
    cleanup = " ".join(_q(str(spec.run_dir / o)) for o in sorted(verdict_outputs))
    hb = int(cfg.get("heartbeat_interval_s", 30))
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by datafarm.expert for attempt {spec.attempt_id} ({spec.purpose}); do not edit.",
        "set -o pipefail",
        f"RUN_DIR={_q(str(spec.run_dir))}",
        'XH_EMPTY="$RUN_DIR/.xh-empty"',
        'cd "$RUN_DIR" || exit 90',
        'date +%s > "$RUN_DIR/job.started"',
        f'( while :; do date +%s > "$RUN_DIR/.heartbeat.tmp" && mv -f "$RUN_DIR/.heartbeat.tmp" '
        f'"$RUN_DIR/heartbeat"; sleep {hb}; done ) &',
        "XH_HB=$!",
        "trap 'kill $XH_HB 2>/dev/null' EXIT",
        "# never hand credential-like variables to agent code",
        "for v in $(env | sed -n 's/^\\([A-Za-z_][A-Za-z0-9_]*\\)=.*/\\1/p' | "
        "grep -E '(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)'); do unset \"$v\"; done",
        "export PYTHONDONTWRITEBYTECODE=1",
        f'{{ date -Is; hostname; echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"; '
        f'echo "SEED={spec.seed}"; echo "ATTEMPT={spec.attempt_id}"; '
        f'{("sha256sum " + record + " 2>&1;") if record else ""} }} > "$RUN_DIR/environment.txt"',
        wrapper,
        "XH_BACKEND=",
        "XH_J=()",
        "ISO=ok; SIM_EXIT=null; SIM_WALL=null",
        "if ! xh_pick_backend; then ISO=unavailable; fi",
        'if [ "$ISO" = ok ] && [ "$XH_BACKEND" = none ]; then ISO=none; fi',
        'if [ "$ISO" = ok ]; then',
        f"  xh_sandboxed \"$RUN_DIR\" /bin/sh -c {shlex.quote(PROBE_SCRIPT)} xh-probe {probe_args} "
        '> "$RUN_DIR/isolation-probe.txt" 2>&1',
        '  if grep -qE "^(LEAK|UMOUNTED) " "$RUN_DIR/isolation-probe.txt"; then ISO=failed; fi',
        "fi",
        'if [ "$ISO" = ok ] || [ "$ISO" = none ]; then',
        "  START=$(date +%s)",
        f"  xh_sandboxed {_q(sim_cwd)} timeout --kill-after=30 {timeout_s} "
        + " ".join(_q(a) for a in sim_argv) + f' > "$RUN_DIR/{log}" 2>&1',
        "  SIM_EXIT=$?",
        "  SIM_WALL=$(( $(date +%s) - START ))",
        *move_lines,
        f"  rm -f {cleanup}" if cleanup else "  :",
        f"  if [ -e {_q(str(spec.run_dir / requires))} ]; then" if requires else "  if true; then",
        f"  {judge_block}",
        "  fi",
        "fi",
        'JUDGE=$(IFS=,; echo "${XH_J[*]}")',
        f'printf \'{{"schema": "{JOB_SCHEMA}", "attempt_id": "{spec.attempt_id}", "purpose": '
        f'"{spec.purpose}", "seed": {spec.seed}, "isolation": "%s", "isolation_backend": "%s", '
        '"sim_exit": %s, "sim_wall_s": %s, "judge_exits": [%s], "host": "%s", "ended_at": "%s"}\\n\' '
        '"$ISO" "$XH_BACKEND" "$SIM_EXIT" "$SIM_WALL" "$JUDGE" "$(hostname)" '
        '"$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/.harness-job.json.tmp" && '
        'mv -f "$RUN_DIR/.harness-job.json.tmp" "$RUN_DIR/harness-job.json"',
        'if [ "$ISO" = unavailable ]; then exit 97; fi',
        'if [ "$ISO" = failed ]; then exit 96; fi',
        "exit 0",
    ]
    return "\n".join(lines) + "\n"


def prepare_run_dir(spec: AttemptSpec, expert_source: bytes, adapter: TaskAdapter) -> None:
    spec.run_dir.mkdir(parents=True, exist_ok=False)
    for sub in adapter.rollout.get("sim", {}).get("writable_run_subdirs", ["native", "native_workdir"]):
        (spec.run_dir / sub).mkdir(parents=True, exist_ok=True)
    spec.expert_path.write_bytes(expert_source)
    os.chmod(spec.expert_path, 0o444)
    (spec.run_dir / "expert.sha256").write_text(f"{sha256_bytes(expert_source)}  expert_snapshot.py\n")
    make_empty_file(spec.run_dir)


# --------------------------------------------------------------------------- job backends


class JobRolloutBackend:
    """Shared logic for job-script backends (``command`` submits, ``local`` runs here)."""

    name = "job"

    def __init__(self, adapter: TaskAdapter, *, masked: Sequence[str],
                 isolation_override: str | None = None, clock_now: Callable[[], float] | None = None) -> None:
        self.adapter = adapter
        self.cfg = adapter.rollout
        self.masked = list(masked)
        self.isolation_override = isolation_override
        # Jobs are real processes writing real files: liveness uses the real clock, whatever
        # clock the harness uses for budgets (``clock_now`` is accepted for symmetry only).
        self.now = time.time
        self.stale_after_s = float(self.cfg.get("stale_after_s", 900))
        self.max_wall_s = float(self.cfg.get("max_wall_s", 3 * 3600))

    def write_job(self, spec: AttemptSpec) -> Path:
        script = build_job_script(self.adapter, spec, masked=self.masked,
                                  isolation_override=self.isolation_override)
        path = spec.run_dir / "job.sh"
        path.write_text(script, encoding="utf-8")
        os.chmod(path, 0o555)
        return path

    def poll(self, spec: AttemptSpec, receipt: Mapping[str, Any], now: float) -> str:
        now = self.now()
        run_dir = spec.run_dir
        if (run_dir / "harness-job.json").exists():
            return "finished"
        launched = float(receipt.get("launched_at_ts", now))
        if now - launched > self.max_wall_s:
            return "timeout"
        last = launched
        for name in ("heartbeat", "job.started"):
            try:
                last = max(last, (run_dir / name).stat().st_mtime)
                break
            except OSError:
                continue
        if now - last > self.stale_after_s:
            return "lost"
        return "running"

    def outcome(self, spec: AttemptSpec, receipt: Mapping[str, Any], status: str) -> Outcome:
        return classify_run_dir(spec.run_dir, self.cfg, status=status)


class CommandRolloutBackend(JobRolloutBackend):
    """Submit ``job.sh`` with the adapter's submit command (``gpu-scheduler submit`` in
    production). A failed submission raises :class:`LaunchError` (not an attempt)."""

    name = "command"

    def __init__(self, adapter: TaskAdapter, *, masked: Sequence[str], run: Callable[..., Any] = subprocess.run,
                 **kwargs: Any) -> None:
        super().__init__(adapter, masked=masked, **kwargs)
        self.run = run
        submit = self.cfg.get("submit", {})
        if not submit.get("argv"):
            raise ConfigError("rollout.submit.argv is required for the command backend")
        self.submit_cfg = submit

    def launch(self, spec: AttemptSpec) -> dict[str, Any]:
        job = self.write_job(spec)
        argv = self.adapter.x(self.submit_cfg["argv"], run_dir=str(spec.run_dir), job_script=str(job),
                              job_name=spec.job_name, attempt_id=spec.attempt_id, seed=spec.seed)
        try:
            proc = self.run(argv, capture_output=True, text=True,
                            timeout=float(self.submit_cfg.get("timeout_s", 120)))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise LaunchError(f"submit failed: {exc}") from exc
        if proc.returncode != 0:
            raise LaunchError(f"submit exit {proc.returncode}: "
                              + ((proc.stderr or "") + (proc.stdout or "")).strip()[-300:])
        receipt: dict[str, Any] = {"backend": self.name, "launched_at_ts": self.now()}
        for line in (proc.stdout or "").splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                if key.strip() in ("id", "host", "gpus", "pid", "log"):
                    receipt[key.strip()] = value.strip()
        return receipt


class LocalRolloutBackend(JobRolloutBackend):
    """Run ``job.sh`` as a local background process (dry runs, single-node operation)."""

    name = "local"

    def __init__(self, adapter: TaskAdapter, *, masked: Sequence[str], **kwargs: Any) -> None:
        super().__init__(adapter, masked=masked, **kwargs)
        self.procs: dict[str, subprocess.Popen[bytes]] = {}
        self.exit_seen: dict[str, float] = {}

    def launch(self, spec: AttemptSpec) -> dict[str, Any]:
        job = self.write_job(spec)
        log = open(spec.run_dir / "job.log", "wb")
        try:
            proc = subprocess.Popen(["bash", str(job)], cwd=spec.run_dir, stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        except OSError as exc:
            raise LaunchError(f"cannot start job: {exc}") from exc
        finally:
            log.close()
        self.procs[spec.attempt_id] = proc
        return {"backend": self.name, "launched_at_ts": self.now(), "pid": proc.pid,
                "host": socket.gethostname()}

    def poll(self, spec: AttemptSpec, receipt: Mapping[str, Any], now: float) -> str:
        status = super().poll(spec, receipt, now)
        now = self.now()
        proc = self.procs.get(spec.attempt_id)
        if status == "running" and proc is not None and proc.poll() is not None:
            # Process gone: give the result file a moment (it is written last), then lost.
            if not (spec.run_dir / "harness-job.json").exists():
                exited = self.exit_seen.setdefault(spec.attempt_id, now)
                if now - exited > 5:
                    return "lost"
            else:
                return "finished"
        return status

# --------------------------------------------------------------------------- fake backend

FAKE_FAMILIES = ("family_a", "family_b", "family_c", "family_d")


def fake_outcome(spec: AttemptSpec, source: bytes) -> Outcome:
    """Deterministic stand-in: ``FAKE_QUALITY = q`` in the expert sets the pass probability,
    ``FAKE_RAISE`` makes an expert error, seed 13 is an invalid start."""

    text = source.decode("utf-8", "replace")
    family = FAKE_FAMILIES[spec.seed % len(FAKE_FAMILIES)]
    if spec.seed == 13:
        return Outcome("init_invalid", False, None, {}, family, False, 0, 1.0, "fake invalid start")
    if "FAKE_RAISE" in text:
        return Outcome("expert_error", True, False, {}, family, True, 1, 1.0, "fake expert error")
    match = re.search(r"FAKE_QUALITY\s*=\s*([0-9.]+)", text)
    quality = float(match.group(1)) if match else 0.0
    score = int(sha256_text(f"{spec.seed}:{sha256_bytes(source)}")[:8], 16) / 0xFFFFFFFF
    success = score < quality
    err = round(20 + 100 * score, 1)
    return Outcome("judged", True, success, {"m1": success, "m2": success, "fold_max_xy_mm": [err, err]},
                   family, True, 0, 1.0, "fake judge")


class FakeRolloutBackend:
    """In-process backend for tests: finishes after ``polls_to_finish`` polls."""

    name = "fake"

    def __init__(self, outcome_fn: Callable[[AttemptSpec, bytes], Outcome] = fake_outcome,
                 polls_to_finish: int = 1, fail_launches: int = 0, lose: Sequence[str] = ()) -> None:
        self.outcome_fn = outcome_fn
        self.polls_to_finish = polls_to_finish
        self.fail_launches = fail_launches
        self.lose = set(lose)
        self.launched: list[str] = []
        self.polls: dict[str, int] = {}

    def launch(self, spec: AttemptSpec) -> dict[str, Any]:
        if self.fail_launches > 0:
            self.fail_launches -= 1
            raise LaunchError("no matching free GPU found (fake)")
        self.launched.append(spec.attempt_id)
        self.polls[spec.attempt_id] = 0
        return {"backend": self.name, "launched_at_ts": 0.0}

    def poll(self, spec: AttemptSpec, receipt: Mapping[str, Any], now: float) -> str:
        self.polls[spec.attempt_id] = self.polls.get(spec.attempt_id, 0) + 1
        if self.polls[spec.attempt_id] < self.polls_to_finish:
            return "running"
        if spec.attempt_id in self.lose:
            return "lost"
        return "finished"

    def outcome(self, spec: AttemptSpec, receipt: Mapping[str, Any], status: str) -> Outcome:
        if status != "finished":
            return classify_run_dir(spec.run_dir, {}, status=status)
        source = spec.expert_path.read_bytes() if spec.expert_path.exists() else b""
        return self.outcome_fn(spec, source)


def make_backend(adapter: TaskAdapter, *, kind: str | None, masked: Sequence[str],
                 isolation_override: str | None = None, clock_now: Callable[[], float] = time.time):
    kind = kind or adapter.rollout.get("backend")
    if kind == "command":
        return CommandRolloutBackend(adapter, masked=masked, isolation_override=isolation_override,
                                     clock_now=clock_now)
    if kind == "local":
        return LocalRolloutBackend(adapter, masked=masked, isolation_override=isolation_override,
                                   clock_now=clock_now)
    if kind == "fake":
        return FakeRolloutBackend()
    raise ConfigError(f"unknown rollout backend {kind!r}")


def summarize_outcome(outcome: Outcome) -> dict[str, Any]:
    return {"outcome_class": outcome.outcome_class, "counted": outcome.counted,
            "success": outcome.success, "family": outcome.family, "init_valid": outcome.init_valid,
            "fields": outcome.fields, "sim_exit": outcome.sim_exit, "wall_s": outcome.wall_s,
            "detail": outcome.detail}


def job_record(run_dir: Path) -> dict[str, Any] | None:
    value = _read_json_quiet(run_dir / "harness-job.json")
    return value if isinstance(value, dict) else None


def dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)
