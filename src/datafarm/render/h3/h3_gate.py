#!/usr/bin/env python3
"""Production render gate: two complete-episode prefixes, then a fixed canary, then the rest of the batch.

  python h3_gate.py init --gate <name> --recipe <render recipe name> --task <task> \
      --prefix <ep> <ep> --canary <ep> x8.. [--rest <ep> ...] [--run-dir <run>]
  python h3_gate.py record --gate <name> --episode <ep> (--package <final package dir> | --failed "<reason>")
  python h3_gate.py status --gate <name>            # JSON: stage, per-episode verdicts, allowed episodes
  h3_enqueue.py --gate <name> ...                   # refuses episodes the gate has not released

<run>/gates/<name>.json fixes the recipe, the task and the episode lists when it is created; `init` never overwrites
it.  A changed prompt, reference, recipe or verifier needs a new gate: the old canary no longer applies.
Verdicts come only from the independent blind review: <run>/reviews/.mapping.json maps the recorded final package to
its review id, and reviews/<rid>/review/review.json (h3.review/1) holds the verdict.  A package without a finished
review is pending.  A recorded render failure counts as not accepted and is kept; it is never replaced.

Stages (docs/reference/rendering.md):
  prefix   only the prefix episodes may render; every prefix must be accepted.
  canary   prefix and canary episodes may render; at least 75% of the canary must be accepted and no blocking `kind`
           may occur in half or more of the canary episodes.
  open     every episode may render.
  stopped  a rule can no longer be met; nothing new renders.  Repair at the earliest owner and start a new gate.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h3_paths

SCHEMA = "h3.render-gate/1"
CANARY_MIN = 8
CANARY_PASS = 0.75


def gate_path(run: Path, name: str) -> Path:
    if not name or "/" in name or name.startswith("."):
        raise SystemExit(f"bad gate name {name!r}")
    return run / "gates" / f"{name}.json"


def load(run: Path, name: str) -> dict:
    path = gate_path(run, name)
    if not path.is_file():
        raise SystemExit(f"no gate {path}")
    gate = json.loads(path.read_text())
    if gate.get("schema") != SCHEMA:
        raise SystemExit(f"{path}: schema {gate.get('schema')!r}, not {SCHEMA}")
    return gate


def init(run: Path, name: str, recipe: str, task: str, prefix: list, canary: list, rest: list) -> dict:
    path = gate_path(run, name)
    if path.exists():
        raise SystemExit(f"{path} exists; a gate's episode lists are fixed (start a new gate for a new recipe)")
    episodes = prefix + canary + rest
    if len(prefix) != 2:
        raise SystemExit("the prefix is exactly 2 complete episodes")
    if len(canary) < CANARY_MIN:
        raise SystemExit(f"the canary needs at least {CANARY_MIN} episodes")
    if len(set(episodes)) != len(episodes):
        raise SystemExit("an episode appears twice")
    gate = {"schema": SCHEMA, "name": name, "recipe": recipe, "task": task, "prefix": prefix, "canary": canary,
            "rest": rest, "records": {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate, indent=1))
    return gate


def record(run: Path, name: str, episode: str, package: Path | None, failed: str | None) -> dict:
    gate = load(run, name)
    if episode not in gate["prefix"] + gate["canary"] + gate["rest"]:
        raise SystemExit(f"{episode} is not in gate {name}")
    if episode in gate["records"]:
        raise SystemExit(f"{episode} already recorded in gate {name}: {gate['records'][episode]} (results are kept)")
    if package is not None:
        package = Path(package).resolve()
        pkg = json.loads((package / "package.json").read_text())
        if (pkg.get("task"), pkg.get("episode")) != (gate["task"], episode):
            raise SystemExit(f"{package} renders {pkg.get('task')}/{pkg.get('episode')}, not {gate['task']}/{episode}")
        if not (package / "assembled.json").is_file():
            raise SystemExit(f"{package} is not an assembled final package")
        gate["records"][episode] = {"package": str(package)}
    else:
        gate["records"][episode] = {"failed": failed}
    gate_path(run, name).write_text(json.dumps(gate, indent=1))
    return gate["records"][episode]


def review_of(run: Path, package: str) -> tuple[str | None, dict | None]:
    """(review id, finished review) of a final package; (None, None) before h3_review_prep.py has seen it."""
    mapping = run / "reviews" / ".mapping.json"
    rids = [rid for rid, v in (json.loads(mapping.read_text()) if mapping.is_file() else {}).items()
            if v.get("subject") == package]
    if not rids:
        return None, None
    path = run / "reviews" / rids[0] / "review" / "review.json"
    try:
        review = json.loads(path.read_text())
    except (OSError, ValueError):
        return rids[0], None
    if review.get("verdict") not in ("accept", "reject") or not isinstance(review.get("scores"), dict):
        return rids[0], None
    return rids[0], review


def verdict(run: Path, rec: dict | None) -> dict:
    if rec is None:
        return {"verdict": "not_rendered"}
    if "failed" in rec:
        return {"verdict": "failed", "reason": rec["failed"]}
    rid, review = review_of(run, rec["package"])
    if review is None:
        return {"verdict": "pending", "review_id": rid}
    kinds = sorted({str(b.get("kind")) for b in review.get("blocking") or [] if isinstance(b, dict) and b.get("kind")})
    return {"verdict": review["verdict"], "review_id": rid, "scores": review["scores"], "blocking_kinds": kinds}


def status(run: Path, name: str) -> dict:
    gate = load(run, name)
    v = {ep: verdict(run, gate["records"].get(ep)) for ep in gate["prefix"] + gate["canary"] + gate["rest"]}
    decided = ("accept", "reject", "failed")
    prefix = [v[e]["verdict"] for e in gate["prefix"]]
    canary = [v[e]["verdict"] for e in gate["canary"]]
    need = math.ceil(CANARY_PASS * len(gate["canary"]))
    kinds: dict[str, int] = {}
    for e in gate["canary"]:
        for k in v[e].get("blocking_kinds", []):
            kinds[k] = kinds.get(k, 0) + 1
    common = sorted(k for k, n in kinds.items() if 2 * n >= len(gate["canary"]))
    reason = None
    if any(x in ("reject", "failed") for x in prefix):
        stage, reason = "stopped", "a prefix episode was not accepted"
    elif not all(x == "accept" for x in prefix):
        stage = "prefix"
    elif canary.count("accept") + sum(x not in decided for x in canary) < need:
        stage, reason = "stopped", f"fewer than {need}/{len(canary)} canary episodes can still be accepted"
    elif common:
        stage, reason = "stopped", f"blocking kind(s) {common} in half or more of the canary"
    elif not all(x in decided for x in canary):
        stage = "canary"
    else:
        stage = "open"
    allowed = {"prefix": gate["prefix"], "canary": gate["prefix"] + gate["canary"],
               "open": gate["prefix"] + gate["canary"] + gate["rest"], "stopped": []}[stage]
    return {"gate": name, "recipe": gate["recipe"], "task": gate["task"], "stage": stage, "reason": reason,
            "canary_accepted": canary.count("accept"), "canary_needed": need, "canary_total": len(canary),
            "common_blocking_kinds": common, "allowed": allowed, "episodes": v}


def check_enqueue(run: Path, name: str, recipe: str, task: str, episodes: list) -> None:
    """Exit unless the gate covers this recipe and task and has released every episode."""
    st = status(run, name)
    if (st["recipe"], st["task"]) != (recipe, task):
        raise SystemExit(f"gate {name} is for recipe {st['recipe']} / task {st['task']}, not {recipe} / {task}")
    held = [e for e in episodes if e not in st["allowed"]]
    if held:
        raise SystemExit(f"gate {name} is at stage {st['stage']}"
                         + (f" ({st['reason']})" if st["reason"] else "") + f"; not released: {held}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--recipe", required=True, help="render recipe name (as given to h3_enqueue.py)")
    i.add_argument("--task", required=True)
    i.add_argument("--prefix", nargs=2, required=True)
    i.add_argument("--canary", nargs="+", required=True)
    i.add_argument("--rest", nargs="*", default=[])
    r = sub.add_parser("record")
    r.add_argument("--episode", required=True)
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--package", type=Path)
    g.add_argument("--failed")
    s = sub.add_parser("status")
    for p in (i, r, s):
        p.add_argument("--gate", required=True)
        h3_paths.add_run_dir_arg(p)
    args = ap.parse_args()
    run = h3_paths.run_dir(args.run_dir)
    if args.cmd == "init":
        out = init(run, args.gate, args.recipe, args.task, args.prefix, args.canary, args.rest)
    elif args.cmd == "record":
        out = record(run, args.gate, args.episode, args.package, args.failed)
    else:
        out = status(run, args.gate)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
