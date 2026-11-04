#!/usr/bin/env python3
"""Check prompts, reviewers and verifier specs against the standing rules.

Rules such as "hard shadows are acceptable" or "exposure drift is only a gross guard" apply
in many places: prompt files, reviewer prompts embedded in tools, verifier spec defaults and
per-task specs. `configs/standing-rules.json` is the single machine-readable source; this
lint checks the rules that declare patterns or required values.

Violations listed under a rule's `known_violations` are reported but do not fail; any other
violation fails (exit 1). Listed entries that no longer match are reported as stale.

    python tools/lint_standing_rules.py            # human summary, exit 1 on new violations
    python tools/lint_standing_rules.py --json     # machine-readable report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA = "datafarm.standing-rules/1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = Path("configs/standing-rules.json")


def load_rules(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA}")
    seen = set()
    for rule in document.get("rules", []):
        rule_id = rule.get("id")
        if not rule_id or rule_id in seen:
            raise ValueError(f"{path}: rule ids must be present and unique ({rule_id!r})")
        seen.add(rule_id)
        for field in ("statement", "globs"):
            if not rule.get(field):
                raise ValueError(f"{path}: rule {rule_id} lacks {field}")
        for pattern in rule.get("forbidden", []):
            re.compile(pattern["regex"])
    return document


def iter_files(root: Path, globs: Iterable[str], exclude: set[Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            resolved = path.resolve()
            if resolved in seen or resolved in exclude:
                continue
            seen.add(resolved)
            yield path


def json_values(node: Any, key: str) -> Iterator[Any]:
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key:
                yield value
            yield from json_values(value, key)
    elif isinstance(node, list):
        for item in node:
            yield from json_values(item, key)


def numbers(value: Any) -> Iterator[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from numbers(item)


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def check_rule(root: Path, rule: dict[str, Any], exclude: set[Path]) -> list[dict[str, Any]]:
    waived = {item["path"] for item in rule.get("waivers", [])}
    known = {item["path"] for item in rule.get("known_violations", [])}
    findings: list[dict[str, Any]] = []

    def add(rel: str, line: int, why: str, excerpt: str) -> None:
        status = "waived" if rel in waived else "known" if rel in known else "new"
        findings.append(
            {"rule": rule["id"], "path": rel, "line": line, "why": why,
             "excerpt": excerpt.strip()[:160], "status": status}
        )

    for path in iter_files(root, rule["globs"], exclude):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in rule.get("forbidden", []):
            for match in re.finditer(pattern["regex"], text, re.IGNORECASE):
                start = text.rfind("\n", 0, match.start()) + 1
                end = text.find("\n", match.end())
                add(rel, line_of(text, match.start()), pattern["why"], text[start:end if end >= 0 else None])
        for requirement in rule.get("numeric_min", []):
            key, minimum = requirement["key"], float(requirement["min"])
            if path.suffix == ".json":
                try:
                    values = [v for found in json_values(json.loads(text), key) for v in numbers(found)]
                except ValueError:
                    values = []
                for value in values:
                    if value < minimum:
                        add(rel, 0, f"{requirement['why']} ({key}={value} < {minimum})", key)
            else:
                for match in re.finditer(rf"[\"']?{re.escape(key)}[\"']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text):
                    if float(match.group(1)) < minimum:
                        add(rel, line_of(text, match.start()),
                            f"{requirement['why']} ({key}={match.group(1)} < {minimum})", match.group(0))
        for requirement in rule.get("required_json", []):
            if path.suffix != ".json":
                continue
            key, allowed = requirement["key"], set(requirement.get("allowed", []))
            try:
                values = list(json_values(json.loads(text), key))
            except ValueError:
                add(rel, 0, "file is not valid JSON", rel)
                continue
            if not values:
                add(rel, 0, f"missing required key {key}", key)
            elif allowed and any(value not in allowed for value in values):
                add(rel, 0, f"{key} must be one of {sorted(allowed)}", json.dumps(values)[:120])
    return findings


def lint(root: Path, rules_path: Path) -> dict[str, Any]:
    document = load_rules(rules_path)
    exclude = {rules_path.resolve()}
    findings: list[dict[str, Any]] = []
    stale: list[dict[str, str]] = []
    for rule in document.get("rules", []):
        rule_findings = check_rule(root, rule, exclude)
        findings.extend(rule_findings)
        matched = {item["path"] for item in rule_findings if item["status"] == "known"}
        for entry in rule.get("known_violations", []):
            if entry["path"] not in matched:
                stale.append({"rule": rule["id"], "path": entry["path"]})
    counts = {status: sum(1 for item in findings if item["status"] == status) for status in ("new", "known", "waived")}
    return {"rules": len(document.get("rules", [])), "counts": counts, "findings": findings, "stale_known_violations": stale}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--rules", type=Path, default=None, help="default: <root>/configs/standing-rules.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    report = lint(root, (args.rules or root / DEFAULT_RULES).resolve())
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for item in report["findings"]:
            if item["status"] != "waived":
                print(f"[{item['status']}] {item['rule']} {item['path']}:{item['line']}: {item['why']}\n    {item['excerpt']}")
        for item in report["stale_known_violations"]:
            print(f"[stale] {item['rule']} {item['path']}: listed as a known violation but no longer matches; remove it from the rules file")
        c = report["counts"]
        print(f"{report['rules']} rules: {c['new']} new, {c['known']} known, {c['waived']} waived violation(s); "
              f"{len(report['stale_known_violations'])} stale known entr(ies)")
    return 1 if report["counts"]["new"] else 0


if __name__ == "__main__":
    sys.exit(main())
