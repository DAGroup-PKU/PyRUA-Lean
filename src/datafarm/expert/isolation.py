"""OS-level isolation for the agent CLI and for rollouts that execute agent-written code.

A brief alone cannot keep the authoring agent away from holdout seeds and verifier code: an
agent running as root makes "clean room" its own claim. Here the rules become mounts:

* ``bwrap`` (preferred): a new mount + PID namespace. ``allowlist`` base shows only the
  system directories and the paths the adapter lists as readable (read-only) plus the
  workspace (read-write); ``root_ro`` base shows the whole host read-only. Protected paths
  are masked (empty read-only tmpfs for directories, an empty file for files).
* ``unshare -m`` (fallback): a private mount namespace with the same masks; the host stays
  visible, so it only has denylist semantics.

Capabilities are always dropped (``--cap-drop ALL``) or the process switches to an
unprivileged uid with ``setpriv``. This matters: bwrap started by root keeps every
capability by default, and a process holding ``CAP_SYS_ADMIN`` can simply ``umount`` a
mask (see tests).
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from datafarm.expert.util import HarnessError

DEFAULT_SYSTEM_RO = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/etc",
                     "/run/systemd/resolve")
_KIND_ORDER = {"dir": -1, "ro": 0, "rw": 1, "tmpfs": 2, "mask": 3}
# Capabilities a sandbox may keep. Never CAP_SYS_ADMIN (umount a mask) or
# CAP_DAC_READ_SEARCH (open_by_handle_at reads files behind any mount).
ALLOWED_KEEP_CAPS = frozenset({"CAP_DAC_OVERRIDE", "CAP_FOWNER", "CAP_CHOWN"})


def check_keep_caps(caps: Sequence[str]) -> tuple[str, ...]:
    bad = [c for c in caps if c not in ALLOWED_KEEP_CAPS]
    if bad:
        raise HarnessError(f"capabilities {bad} would let sandboxed code escape its masks; "
                           f"allowed: {sorted(ALLOWED_KEEP_CAPS)}")
    return tuple(caps)


class IsolationUnavailable(HarnessError):
    pass


@dataclasses.dataclass(frozen=True)
class MountOp:
    kind: str  # ro | rw | tmpfs | mask_dir | mask_file
    src: str | None
    dest: str


@dataclasses.dataclass(frozen=True)
class SandboxSpec:
    mode: str  # bwrap | unshare | none
    base: str = "allowlist"  # allowlist | root_ro
    ro: tuple[str, ...] = ()
    rw: tuple[str, ...] = ()
    masked: tuple[str, ...] = ()
    binds: tuple[tuple[str, str, bool], ...] = ()  # (src, dest, writable)
    tmpfs: tuple[str, ...] = ()
    system_ro: tuple[str, ...] = DEFAULT_SYSTEM_RO
    dev: str = "minimal"  # minimal | host
    share_net: bool = True
    run_as: tuple[int, int] | None = None
    keep_caps: tuple[str, ...] = ()
    cwd: str = "/"
    empty_file: str = ""
    degraded: bool = False  # an allowlist spec running under unshare (denylist only)

    def replace(self, **changes: Any) -> "SandboxSpec":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["binds"] = [list(b) for b in self.binds]
        return data


def _norm(path: str) -> str:
    if not isinstance(path, str) or not path.startswith("/"):
        raise HarnessError(f"sandbox paths must be absolute: {path!r}")
    normalized = os.path.normpath(path)
    if normalized.startswith("//"):
        normalized = "/" + normalized.lstrip("/")
    return normalized


def _depth(path: str) -> int:
    return 0 if path == "/" else path.count("/")


def plan_mounts(spec: SandboxSpec, *, lexists: Callable[[str], bool] = os.path.lexists,
                isdir: Callable[[str], bool] = os.path.isdir) -> list[MountOp]:
    """Mount operations in application order: shallower paths first, so the most specific
    rule wins (a readable child re-bound inside a masked parent stays readable; a masked
    child inside a readable parent stays hidden). Missing sources are skipped."""

    entries: list[tuple[int, int, int, MountOp]] = []
    order = 0

    def add(kind: str, src: str | None, dest: str) -> None:
        nonlocal order
        dest_n = _norm(dest)
        key_kind = "mask" if kind.startswith("mask") else kind
        entries.append((_depth(dest_n), _KIND_ORDER[key_kind], order, MountOp(kind, src, dest_n)))
        order += 1

    for path in spec.ro:
        if lexists(path):
            add("ro", _norm(path), path)
    for path in spec.rw:
        if lexists(path):
            add("rw", _norm(path), path)
    for src, dest, writable in spec.binds:
        if lexists(src):
            add("rw" if writable else "ro", _norm(src), dest)
    for path in spec.tmpfs:
        add("tmpfs", None, path)
    bound = [_norm(p) for p in spec.ro] + [_norm(p) for p in spec.rw] + [_norm(d) for _, d, _ in spec.binds]
    bound += [_norm(p) for p in spec.tmpfs] + [_norm(p) for p in spec.system_ro]
    for path in spec.masked:
        if not lexists(path):
            continue
        if spec.base == "allowlist" and not any(_under(_norm(path), root) for root in bound):
            continue  # never bound, so already invisible; do not even create a mount point
        add("mask_dir" if isdir(path) and not os.path.islink(path) else "mask_file", None, path)
    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    ops = [item[3] for item in entries]
    return _with_parent_dirs(spec, ops, lexists)


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _with_parent_dirs(spec: SandboxSpec, ops: list[MountOp],
                      lexists: Callable[[str], bool] = os.path.lexists) -> list[MountOp]:
    """bwrap creates missing mount-point parents with mode 0700, which an unprivileged
    sandbox user cannot traverse. Create every *synthetic* parent (one not backed by a
    host bind) explicitly with mode 0755 first. A parent is synthetic when the deepest
    operation covering it is a mask/tmpfs, when nothing covers it in an allowlist
    sandbox (bwrap's own root tmpfs), or when it is covered by a bind from another
    source that has no such directory (the round's tmp bound to /tmp)."""

    host_roots = [("ro", p, p) for p in (spec.system_ro if spec.base == "allowlist" else ("/",))]
    covering = host_roots + [(op.kind, op.dest, op.src) for op in ops]
    dests = {op.dest for op in ops}
    needed: dict[str, int] = {}
    for op in ops:
        parent = os.path.dirname(op.dest)
        while parent and parent != "/":
            if parent not in dests and parent not in needed:
                best = None
                for kind, root, src in covering:
                    if root != parent and _under(parent, root) and (best is None or len(root) > len(best[1])):
                        best = (kind, root, src)
                synthetic = (best is None and spec.base == "allowlist") or (
                    best is not None and (best[0] in ("mask_dir", "tmpfs") or (
                        best[0] in ("ro", "rw") and best[2] not in (None, best[1])
                        and not lexists(best[2] + parent[len(best[1]):]))))
                if synthetic:
                    needed[parent] = _depth(parent)
            parent = os.path.dirname(parent)
    if not needed:
        return ops
    order = {op.dest: i for i, op in enumerate(ops)}
    merged = [(_depth(op.dest), _KIND_ORDER["mask" if op.kind.startswith("mask") else op.kind], order[op.dest], op)
              for op in ops]
    merged += [(depth, _KIND_ORDER["dir"], -1, MountOp("dir", None, path)) for path, depth in needed.items()]
    merged.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in merged]


def setpriv_prefix(run_as: tuple[int, int] | None) -> list[str]:
    setpriv = shutil.which("setpriv") or "/usr/bin/setpriv"
    prefix = [setpriv]
    if run_as is not None:
        prefix += [f"--reuid={int(run_as[0])}", f"--regid={int(run_as[1])}", "--clear-groups"]
    return prefix + ["--inh-caps=-all", "--bounding-set=-all", "--no-new-privs", "--"]


def bwrap_argv(spec: SandboxSpec, command: Sequence[str], *, bwrap: str = "bwrap",
               lexists: Callable[[str], bool] = os.path.lexists,
               isdir: Callable[[str], bool] = os.path.isdir) -> list[str]:
    ops = plan_mounts(spec, lexists=lexists, isdir=isdir)
    argv = [bwrap, "--die-with-parent", "--unshare-pid", "--unshare-ipc"]
    if not spec.share_net:
        argv.append("--unshare-net")
    if spec.base == "root_ro":
        argv += ["--ro-bind", "/", "/"]
    else:
        for path in spec.system_ro:
            argv += ["--ro-bind-try", path, path]
    argv += ["--dev-bind", "/dev", "/dev"] if spec.dev == "host" else ["--dev", "/dev"]
    argv += ["--proc", "/proc"]
    covered = {op.dest for op in ops}
    if spec.base == "allowlist" and "/tmp" not in covered:
        argv += ["--tmpfs", "/tmp"]
    remount: list[str] = []
    for op in ops:
        if op.kind == "dir":
            argv += ["--perms", "0755", "--dir", op.dest]
        elif op.kind == "ro":
            argv += ["--ro-bind", op.src or op.dest, op.dest]
        elif op.kind == "rw":
            argv += ["--bind", op.src or op.dest, op.dest]
        elif op.kind == "tmpfs":
            argv += ["--tmpfs", op.dest]
        elif op.kind == "mask_dir":
            argv += ["--tmpfs", op.dest]
            remount += ["--remount-ro", op.dest]
        elif op.kind == "mask_file":
            if not spec.empty_file:
                raise HarnessError("masking a file needs SandboxSpec.empty_file")
            argv += ["--ro-bind", spec.empty_file, op.dest]
    argv += remount
    inner = list(command)
    if spec.run_as is not None:
        argv += ["--cap-add", "CAP_SETUID", "--cap-add", "CAP_SETGID", "--cap-add", "CAP_SETPCAP"]
        inner = setpriv_prefix(spec.run_as) + inner
    else:
        argv += ["--cap-drop", "ALL"]
        for cap in check_keep_caps(spec.keep_caps):
            argv += ["--cap-add", cap]
    argv += ["--chdir", spec.cwd, "--", *inner]
    return argv


def unshare_argv(spec: SandboxSpec, command: Sequence[str], *,
                 lexists: Callable[[str], bool] = os.path.lexists,
                 isdir: Callable[[str], bool] = os.path.isdir) -> list[str]:
    """Fallback: private mount namespace with the same masks (denylist semantics only)."""

    if spec.base != "root_ro":
        raise IsolationUnavailable("unshare cannot hide the host tree; it needs a root_ro spec "
                                   "(set allow_degraded_fallback to accept denylist semantics)")
    ops = plan_mounts(spec, lexists=lexists, isdir=isdir)
    masked_dirs = [op.dest for op in ops if op.kind == "mask_dir"]
    lines = ["set -e"]
    for op in ops:
        dest = shlex.quote(op.dest)
        if op.kind == "dir":
            continue
        if any(op.dest.startswith(m + "/") for m in masked_dirs) and op.kind in ("ro", "rw"):
            # The source is hidden once its masked parent is mounted; unsupported here.
            continue
        if op.kind == "ro":
            lines.append(f"mount --bind {shlex.quote(op.src or op.dest)} {dest}")
            lines.append(f"mount -o remount,bind,ro {dest}")
        elif op.kind == "rw":
            # a bind below a read-only bind inherits read-only: remount it rw explicitly
            lines.append(f"mount --bind {shlex.quote(op.src or op.dest)} {dest}")
            lines.append(f"mount -o remount,bind,rw {dest}")
        elif op.kind == "tmpfs":
            lines.append(f"mount -t tmpfs -o mode=1777 tmpfs {dest}")
        elif op.kind == "mask_dir":
            lines.append(f"mount -t tmpfs -o ro,size=16k,mode=0555 tmpfs {dest}")
        elif op.kind == "mask_file":
            if not spec.empty_file:
                raise HarnessError("masking a file needs SandboxSpec.empty_file")
            lines.append(f"mount --bind {shlex.quote(spec.empty_file)} {dest}")
            lines.append(f"mount -o remount,bind,ro {dest}")
    lines.append(f"cd {shlex.quote(spec.cwd)}")
    lines.append("exec " + " ".join(shlex.quote(a) for a in setpriv_prefix(spec.run_as)) + ' "$@"')
    script = "\n".join(lines)
    return ["unshare", "--mount", "--propagation", "private", "--fork", "--pid", "--mount-proc",
            "--kill-child", "--", "/bin/sh", "-c", script, "xh-sandbox", *command]


_AVAILABLE: dict[str, tuple[bool, str]] = {}


def probe_available(mode: str) -> tuple[bool, str]:
    """Can this host create the sandbox? Cached per process."""

    if mode == "none":
        return True, "no isolation requested"
    if mode in _AVAILABLE:
        return _AVAILABLE[mode]
    if mode == "bwrap":
        exe = shutil.which("bwrap")
        if not exe:
            result = (False, "bwrap not found")
        else:
            cmd = [exe, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--unshare-pid",
                   "--die-with-parent", "--cap-drop", "ALL", "--", "/bin/true"]
            result = _try(cmd)
    elif mode == "unshare":
        exe = shutil.which("unshare")
        if not exe or not shutil.which("setpriv"):
            result = (False, "unshare or setpriv not found")
        else:
            result = _try([exe, "--mount", "--propagation", "private", "--fork", "--",
                           "/bin/sh", "-c", "mount -t tmpfs tmpfs /mnt && exit 0"])
    else:
        result = (False, f"unknown isolation mode {mode!r}")
    _AVAILABLE[mode] = result
    return result


def _try(cmd: list[str]) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{cmd[0]}: {exc}"
    if proc.returncode == 0:
        return True, "ok"
    return False, (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[:300]


def resolve_spec(spec: SandboxSpec, *, fallback: str | None, allow_degraded: bool,
                 available: Callable[[str], tuple[bool, str]] = probe_available) -> SandboxSpec:
    """Pick a usable mode: the preferred one, else the fallback; fail closed otherwise."""

    reasons = []
    for mode in [spec.mode] + ([fallback] if fallback and fallback != spec.mode else []):
        ok, why = available(mode)
        if not ok:
            reasons.append(f"{mode}: {why}")
            continue
        if mode == "unshare" and spec.base == "allowlist":
            if not allow_degraded:
                reasons.append("unshare: allowlist spec needs allow_degraded_fallback")
                continue
            return spec.replace(mode="unshare", base="root_ro", degraded=True)
        return spec.replace(mode=mode)
    raise IsolationUnavailable("no usable isolation: " + "; ".join(reasons))


def wrap(spec: SandboxSpec, command: Sequence[str]) -> list[str]:
    if spec.mode == "bwrap":
        return bwrap_argv(spec, command, bwrap=shutil.which("bwrap") or "bwrap")
    if spec.mode == "unshare":
        return unshare_argv(spec, command)
    if spec.mode == "none":
        return list(command)
    raise HarnessError(f"unknown isolation mode {spec.mode!r}")


def sandbox_env(base: Mapping[str, str], passthrough: Iterable[str]) -> dict[str, str]:
    """Exact environment for the sandboxed process. Secrets named in ``passthrough`` are
    copied from the harness environment into the child's environment only (never argv,
    never logs)."""

    env = {k: str(v) for k, v in base.items()}
    for name in passthrough:
        if name in os.environ and name not in env:
            env[name] = os.environ[name]
    return env


# --------------------------------------------------------------------------- probes

PROBE_SCRIPT = r"""
for arg in "$@"; do
  kind=${arg%%:*}; p=${arg#*:}
  case "$kind" in
    m)
      umount "$p" >/dev/null 2>&1 && echo "UMOUNTED $p"
      if [ -d "$p" ]; then
        first=$(ls -A "$p" 2>/dev/null | head -n 1)
        if [ -n "$first" ]; then echo "LEAK $p"; else echo "HIDDEN $p"; fi
      elif [ -f "$p" ] && [ -s "$p" ] && head -c 1 "$p" >/dev/null 2>&1; then
        echo "LEAK $p"
      else
        echo "HIDDEN $p"
      fi ;;
    r|w)
      probe="$p/.xh-probe-$$"
      if (set -C; : > "$probe") 2>/dev/null; then rm -f "$probe"; echo "WRITABLE $p"; else echo "READONLY $p"; fi ;;
  esac
done
"""


@dataclasses.dataclass
class ProbeResult:
    mode: str
    ok: bool
    leaks: list[str]
    umounted: list[str]
    writable_readonly: list[str]
    unwritable_workspace: list[str]
    lines: list[str]
    error: str | None = None
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def probe_args(masked: Iterable[str], readonly_dirs: Iterable[str], writable_dirs: Iterable[str]) -> list[str]:
    return ([f"m:{p}" for p in masked] + [f"r:{p}" for p in readonly_dirs]
            + [f"w:{p}" for p in writable_dirs])


def parse_probe(lines: Iterable[str], *, mode: str, readonly_dirs: Iterable[str],
                writable_dirs: Iterable[str], degraded: bool = False,
                error: str | None = None) -> ProbeResult:
    lines = [line.strip() for line in lines if line.strip()]
    leaks = [line.split(" ", 1)[1] for line in lines if line.startswith("LEAK ")]
    umounted = [line.split(" ", 1)[1] for line in lines if line.startswith("UMOUNTED ")]
    writable = {line.split(" ", 1)[1] for line in lines if line.startswith("WRITABLE ")}
    ro_bad = [p for p in readonly_dirs if p in writable]
    rw_bad = [p for p in writable_dirs if p not in writable]
    ok = error is None and not leaks and not umounted and not ro_bad and not rw_bad
    return ProbeResult(mode, ok, leaks, umounted, ro_bad, rw_bad, lines, error, degraded)


def run_probe(spec: SandboxSpec, *, masked: Sequence[str], readonly_dirs: Sequence[str] = (),
              writable_dirs: Sequence[str] = (), env: Mapping[str, str] | None = None,
              timeout: float = 60.0) -> ProbeResult:
    """Run the probe inside ``spec``: every masked path must be hidden (even after an
    ``umount`` attempt), read-only dirs unwritable, workspace dirs writable."""

    args = probe_args(masked, readonly_dirs, writable_dirs)
    argv = wrap(spec, ["/bin/sh", "-c", PROBE_SCRIPT, "xh-probe", *args])
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              env=dict(env) if env is not None else {"PATH": "/usr/bin:/bin"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        return parse_probe([], mode=spec.mode, readonly_dirs=readonly_dirs,
                           writable_dirs=writable_dirs, degraded=spec.degraded, error=str(exc))
    error = None if proc.returncode == 0 else f"probe exit {proc.returncode}: {proc.stderr.strip()[:300]}"
    return parse_probe(proc.stdout.splitlines(), mode=spec.mode, readonly_dirs=readonly_dirs,
                       writable_dirs=writable_dirs, degraded=spec.degraded, error=error)


def make_empty_file(directory: str | Path) -> str:
    path = Path(directory) / ".xh-empty"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".xh-empty.{uuid.uuid4().hex}")
        tmp.write_bytes(b"")
        os.chmod(tmp, 0o444)
        os.replace(tmp, path)
    return str(path)


def chown_tree(path: str | Path, uid: int, gid: int) -> None:
    """Give an unprivileged sandbox user its writable directories (never follows links)."""

    for current, dirs, files in os.walk(path, followlinks=False):
        os.lchown(current, uid, gid)
        for name in dirs + files:
            os.lchown(os.path.join(current, name), uid, gid)


# --------------------------------------------------------------------------- job scripts


def job_wrapper_bash(*, mode: str, fallback: str | None, base: str, dev: str, ro: Sequence[str],
                     rw: Sequence[str], masked: Sequence[str], run_as: tuple[int, int] | None,
                     tmpfs: Sequence[str] = (),
                     share_net: bool = True, private_tmp: bool = True,
                     keep_caps: Sequence[str] = ()) -> str:
    """Bash functions for a generated rollout job. ``xh_pick_backend`` chooses bwrap, then
    the fallback, and fails (the job then exits 97) when neither works; ``xh_sandboxed <cwd>
    <cmd...>`` runs a command inside the sandbox. Path existence and ``${HOME}``-relative
    paths are resolved on the execution node at run time."""

    def q(path: str) -> str:
        # keep ${VAR} expandable, quote everything else
        return '"' + path.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"'

    # Ordering (shallow first, so the most specific rule wins) happens at run time, after
    # ${HOME}-style variables have expanded on the execution node.
    entries = ([("ro", p) for p in ro] + [("rw", p) for p in rw]
               + [("tmpfs", p) for p in tmpfs] + [("mask", p) for p in masked])
    op_lines = "\n".join(f"      printf '%s\\t%s\\n' {kind} {q(path)}" for kind, path in entries)
    setpriv = " ".join(shlex.quote(a) for a in setpriv_prefix(run_as))
    base_args = "--ro-bind / /" if base == "root_ro" else " ".join(
        f"--ro-bind-try {p} {p}" for p in DEFAULT_SYSTEM_RO)
    dev_args = "--dev-bind /dev /dev" if dev == "host" else "--dev /dev"
    net = "" if share_net else "--unshare-net"
    tmp_args = "--tmpfs /tmp" if private_tmp else ""
    caps = ("--cap-add CAP_SETUID --cap-add CAP_SETGID --cap-add CAP_SETPCAP" if run_as
            else " ".join(["--cap-drop ALL"] + [f"--cap-add {c}" for c in check_keep_caps(keep_caps)]))
    return f"""
XH_ISOLATION_MODE={shlex.quote(mode)}
XH_ISOLATION_FALLBACK={shlex.quote(fallback or '')}
xh_op() {{
  local kind="$1" p="$2"
  case "$XH_BACKEND:$kind" in
    bwrap:ro) [ -e "$p" ] && XH_ARGS+=(--ro-bind "$p" "$p") ;;
    bwrap:rw) [ -e "$p" ] && XH_ARGS+=(--bind "$p" "$p") ;;
    bwrap:tmpfs) [ -d "$p" ] && XH_ARGS+=(--tmpfs "$p") ;;
    bwrap:mask)
      if [ -d "$p" ] && [ ! -L "$p" ]; then XH_ARGS+=(--tmpfs "$p"); XH_REMOUNT+=(--remount-ro "$p")
      elif [ -e "$p" ]; then XH_ARGS+=(--ro-bind "$XH_EMPTY" "$p"); fi ;;
    unshare:ro) [ -e "$p" ] && XH_MOUNTS+="mount --bind '$p' '$p' && mount -o remount,bind,ro '$p'"$'\\n' ;;
    unshare:rw) [ -e "$p" ] && XH_MOUNTS+="mount --bind '$p' '$p' && mount -o remount,bind,rw '$p'"$'\\n' ;;
    unshare:tmpfs) [ -d "$p" ] && XH_MOUNTS+="mount -t tmpfs -o size=256m,mode=0755 tmpfs '$p'"$'\\n' ;;
    unshare:mask)
      if [ -d "$p" ] && [ ! -L "$p" ]; then XH_MOUNTS+="mount -t tmpfs -o ro,size=16k,mode=0555 tmpfs '$p'"$'\\n'
      elif [ -e "$p" ]; then XH_MOUNTS+="mount --bind '$XH_EMPTY' '$p' && mount -o remount,bind,ro '$p'"$'\\n'; fi ;;
  esac
  return 0
}}
xh_collect() {{
  local _depth _prio kind p
  while IFS=$'\\t' read -r _depth _prio kind p; do
    xh_op "$kind" "$p"
  done < <(
    {{
{op_lines}
    }} | awk -F'\\t' 'BEGIN {{ OFS = "\\t" }} {{ d = gsub("/", "/", $2); pr = ($1 == "ro") ? 0 : (($1 == "rw") ? 1 : (($1 == "tmpfs") ? 2 : 3)); print d, pr, $1, $2 }}' \\
       | sort -t "$(printf '\\t')" -k1,1n -k2,2n -s
  )
  return 0
}}
xh_pick_backend() {{
  for m in "$XH_ISOLATION_MODE" "$XH_ISOLATION_FALLBACK"; do
    case "$m" in
      bwrap) if command -v bwrap >/dev/null 2>&1 && bwrap --ro-bind / / --dev /dev --proc /proc --unshare-pid --die-with-parent --cap-drop ALL -- /bin/true >/dev/null 2>&1; then XH_BACKEND=bwrap; return 0; fi ;;
      unshare) if command -v unshare >/dev/null 2>&1 && command -v setpriv >/dev/null 2>&1 && unshare --mount --propagation private --fork -- /bin/sh -c 'mount -t tmpfs tmpfs /mnt' >/dev/null 2>&1; then XH_BACKEND=unshare; return 0; fi ;;
      none) XH_BACKEND=none; return 0 ;;
    esac
  done
  return 1
}}
xh_sandboxed() {{
  local cwd="$1"; shift
  if [ "$XH_BACKEND" = bwrap ]; then
    XH_ARGS=(--die-with-parent --unshare-pid --unshare-ipc {net} {base_args} {dev_args} --proc /proc {tmp_args})
    XH_REMOUNT=()
    xh_collect
    bwrap "${{XH_ARGS[@]}}" "${{XH_REMOUNT[@]}}" {caps} --chdir "$cwd" -- {setpriv if run_as else ''} "$@"
  elif [ "$XH_BACKEND" = unshare ]; then
    XH_MOUNTS=""
    xh_collect
    unshare --mount --propagation private --fork --pid --mount-proc --kill-child -- /bin/bash -c "set -e
$XH_MOUNTS
cd '$cwd'
exec {setpriv} \\"\\$@\\"" xh-sandbox "$@"
  else
    (cd "$cwd" && "$@")
  fi
}}
"""
