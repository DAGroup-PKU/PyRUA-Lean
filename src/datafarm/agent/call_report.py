"""Grouped, denominator-preserving summaries of ``agent-calls.jsonl`` ledgers.

Every record counts: failed, timed-out, invalid and cached calls are reported next to
successful ones, and a call without a known cost is counted as *unknown* rather than
free. Duplicated ``call_id`` values (the same ledger passed twice) are counted once and
reported as a problem.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from datafarm.agent.call_log import AgentCallRecord, read_calls, summarize

REPORT_SCHEMA = "datafarm.agent-calls-report/1"
LEDGER_NAME = "agent-calls.jsonl"
GROUP_FIELDS = ("component", "backend", "model", "status")
_ALLOWED_GROUP_FIELDS = frozenset(
    {"component", "backend", "model", "status", "run_id", "error_type", "cost_source"}
)


def expand_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Files are used as given; directories contribute every ``agent-calls.jsonl`` below."""

    found: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            found.extend(sorted(path.rglob(LEDGER_NAME)))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"no ledger at {path}")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _read_lenient(path: Path) -> tuple[list[AgentCallRecord], list[dict[str, Any]]]:
    records: list[AgentCallRecord] = []
    problems: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            records.append(AgentCallRecord.from_dict(json.loads(raw)))
        except (ValueError, TypeError) as error:
            problems.append(
                {"kind": "invalid_line", "path": str(path), "line": number, "error": str(error)}
            )
    return records, problems


def load_records(
    paths: Sequence[Path], *, strict: bool = True
) -> tuple[list[AgentCallRecord], list[dict[str, Any]]]:
    """Read ledgers; strict mode raises on the first invalid line."""

    records: list[AgentCallRecord] = []
    problems: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if strict:
            items = read_calls(path)
        else:
            items, found = _read_lenient(path)
            problems.extend(found)
        for record in items:
            if record.call_id in seen:
                problems.append(
                    {"kind": "duplicate_call_id", "path": str(path), "call_id": record.call_id}
                )
                continue
            seen.add(record.call_id)
            records.append(record)
    return records, problems


def _percentile(sorted_values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile of already sorted values."""

    if not sorted_values:
        return None
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


def _extended_summary(records: Sequence[AgentCallRecord]) -> dict[str, Any]:
    base = summarize(records)
    latencies = sorted(r.latency_s for r in records if r.latency_s is not None)
    logical = {str(r.extra.get("logical_call_id") or r.call_id) for r in records}
    base.update(
        {
            "logical_calls": len(logical),
            "cache_hits": sum(r.status == "cache_hit" for r in records),
            "calls_with_known_latency": len(latencies),
            "latency_mean_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "latency_p50_s": _percentile(latencies, 0.5),
            "latency_p95_s": _percentile(latencies, 0.95),
            "latency_max_s": latencies[-1] if latencies else None,
        }
    )
    return base


def group_rows(
    records: Sequence[AgentCallRecord], by: Sequence[str] = GROUP_FIELDS
) -> list[dict[str, Any]]:
    unknown = set(by) - _ALLOWED_GROUP_FIELDS
    if unknown:
        allowed = sorted(_ALLOWED_GROUP_FIELDS)
        raise ValueError(f"cannot group by {sorted(unknown)}; allowed: {allowed}")
    groups: dict[tuple[str, ...], list[AgentCallRecord]] = {}
    for record in records:
        key = tuple(str(getattr(record, name) or "-") for name in by)
        groups.setdefault(key, []).append(record)
    rows = []
    for key in sorted(groups):
        rows.append({**dict(zip(by, key)), **_extended_summary(groups[key])})
    return rows


def build_report(
    records: Sequence[AgentCallRecord],
    *,
    by: Sequence[str] = GROUP_FIELDS,
    sources: Sequence[Path] = (),
    problems: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "sources": [str(path) for path in sources],
        "group_by": list(by),
        "totals": _extended_summary(records),
        "groups": group_rows(records, by),
        "problems": list(problems),
        "notes": [
            "calls counts every ledger record (retries, failures and cache hits included)",
            "cost_usd_known sums known costs only; calls_with_unknown_cost are not free",
        ],
    }


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 100 else f"{value:.1f}"
    return str(value)


def render_table(report: dict[str, Any]) -> str:
    """Plain-text table of the grouped rows plus a totals line."""

    by = list(report["group_by"])
    metrics = [
        ("calls", "calls"),
        ("logical_calls", "logical"),
        ("input_tokens", "in_tok"),
        ("output_tokens", "out_tok"),
        ("cache_read_tokens", "cache_rd"),
        ("cost_usd_known", "usd_known"),
        ("calls_with_unknown_cost", "usd_unknown"),
        ("latency_s", "lat_sum_s"),
        ("latency_p50_s", "lat_p50_s"),
        ("latency_max_s", "lat_max_s"),
    ]
    header = by + [label for _, label in metrics]

    def values(row: dict[str, Any]) -> list[str]:
        cells = []
        for key, _ in metrics:
            value = row.get(key)
            if key == "cost_usd_known" and value is not None:
                value = float(value)
            cells.append(_cell(value))
        return cells

    rows = [[_cell(row.get(name)) for name in by] + values(row) for row in report["groups"]]
    totals = report["totals"]
    rows.append(["TOTAL"] + ["" for _ in by[1:]] + values(totals))
    widths = [max(len(str(item)) for item in column) for column in zip(header, *rows)]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(str(cell).ljust(width) for cell, width in zip(cells, widths)).rstrip()

    output = [line(header), line(["-" * width for width in widths])]
    output.extend(line(row) for row in rows)
    status = ", ".join(f"{k}={v}" for k, v in sorted(totals["by_status"].items())) or "none"
    output.append(f"by status: {status}")
    if report.get("problems"):
        output.append(f"problems: {len(report['problems'])} (see JSON output)")
    return "\n".join(output)
