"""Evaluator tamper manifest (after HELIX's ``evaluator_manifest``).

At round start the harness hashes everything that decides or feeds a verdict: judge and
verifier code, the verifier's contract package, the rollout runner and physics/material
files, the initial-state generator, the holdout seed list, the adapter and brief. It
re-hashes at the end of every session and before holdout evaluation. Any difference
invalidates the round unless an operator recorded an amendment (``tools/expert_harness.py amend``)
first -- which is how a mid-round rule change such as a verifier revision becomes an
auditable event instead of a silent one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from datafarm.expert.layout import RoundPaths
from datafarm.expert.state import EventLog, RoundState
from datafarm.expert.util import HarnessError, atomic_write_json, read_json, sha256_file, sha256_json, tree_digest, utc_now

MANIFEST_SCHEMA = "datafarm.expert-harness.manifest/1"


def describe_path(path: str) -> dict[str, Any]:
    if os.path.islink(path) and not os.path.exists(path):
        return {"kind": "dangling_link", "target": os.readlink(path)}
    if os.path.isdir(path):
        return tree_digest(path)
    if os.path.isfile(path):
        return {"kind": "file", "sha256": sha256_file(path), "bytes": os.path.getsize(path)}
    return {"kind": "missing"}


def build_manifest(paths: Iterable[str], *, extra: Mapping[str, str] | None = None,
                   created_at: str | None = None) -> dict[str, Any]:
    entries = {str(p): describe_path(str(p)) for p in sorted(set(str(p) for p in paths))}
    extra = dict(extra or {})
    return {"schema": MANIFEST_SCHEMA, "created_at": created_at, "entries": entries, "extra": extra,
            "digest": sha256_json({"entries": entries, "extra": extra})}


def diff_manifests(base: Mapping[str, Any], current: Mapping[str, Any]) -> list[dict[str, Any]]:
    changes = []
    b, c = base.get("entries", {}), current.get("entries", {})
    for path in sorted(set(b) | set(c)):
        if b.get(path) != c.get(path):
            changes.append({"path": path, "before": (b.get(path) or {}).get("sha256") or (b.get(path) or {}).get("kind"),
                            "after": (c.get(path) or {}).get("sha256") or (c.get(path) or {}).get("kind")})
    be, ce = base.get("extra", {}), current.get("extra", {})
    for key in sorted(set(be) | set(ce)):
        if be.get(key) != ce.get(key):
            changes.append({"path": f"extra:{key}", "before": be.get(key), "after": ce.get(key)})
    return changes


class ManifestGuard:
    """Round-level tamper checks with a stored baseline and recorded amendments."""

    def __init__(self, paths: RoundPaths, entries: Iterable[str], state: RoundState, events: EventLog,
                 clock_now, extra: Mapping[str, str] | None = None) -> None:
        self.paths = paths
        self.entries = sorted(set(str(e) for e in entries))
        self.state = state
        self.events = events
        self.now = clock_now
        self.extra = dict(extra or {})

    @property
    def baseline_path(self) -> Path:
        return self.paths.manifests / "baseline.json"

    def build(self) -> dict[str, Any]:
        return build_manifest(self.entries, extra=self.extra, created_at=utc_now(self.now()))

    def record_start(self) -> dict[str, Any]:
        manifest = self.build()
        atomic_write_json(self.paths.manifests / "start.json", manifest, mode=0o600)
        atomic_write_json(self.baseline_path, manifest, mode=0o600)
        self.state.data["manifest"]["start_digest"] = manifest["digest"]
        self.state.save()
        self.events.emit("manifest_baseline", private=True, digest=manifest["digest"],
                         entries=len(manifest["entries"]))
        self.events.emit("manifest_check", label="start", ok=True)
        return manifest

    def check(self, label: str) -> tuple[bool, list[dict[str, Any]]]:
        baseline = read_json(self.baseline_path)
        current = self.build()
        changes = diff_manifests(baseline, current)
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
        atomic_write_json(self.paths.manifests / f"{safe_label}.json", current, mode=0o600)
        info = self.state.data["manifest"]
        info["checks"] = int(info.get("checks", 0)) + 1
        info["last_label"] = label
        if changes:
            info["ok"] = False
            info.setdefault("changes", []).append({"label": label, "changes": changes[:50]})
            self.state.violation("tamper", f"{len(changes)} protected entries changed at {label}",
                                 utc_now(self.now()))
        self.state.save()
        # public event carries no paths or digests (holdout seed file is one of them)
        self.events.emit("manifest_check", label=label, ok=not changes, changed=len(changes))
        self.events.emit("manifest_check_detail", private=True, label=label, changes=changes)
        return not changes, changes

    def rebaseline(self, *, reason: str, operator: str) -> dict[str, Any]:
        if not self.baseline_path.exists():
            raise HarnessError("the round has not started, so there is no baseline to amend")
        if not reason.strip():
            raise HarnessError("an amendment needs a reason")
        before = read_json(self.baseline_path)
        current = self.build()
        changes = diff_manifests(before, current)
        atomic_write_json(self.baseline_path, current, mode=0o600)
        number = int(self.state.data["manifest"].get("amendments", 0)) + 1
        self.state.data["manifest"]["amendments"] = number
        self.state.data.setdefault("amendments", []).append(
            {"number": number, "utc": utc_now(self.now()), "operator": operator, "reason": reason,
             "changed": [c["path"] for c in changes if not c["path"].startswith("extra:")]})
        self.state.save()
        self.events.emit("amendment", number=number, reason=reason, operator=operator,
                         changed_entries=len(changes))
        self.events.emit("amendment_detail", private=True, number=number, changes=changes,
                         digest=current["digest"])
        return {"amendment": number, "changes": changes, "digest": current["digest"]}
