"""Brief (task book) rendering and fail-closed lint.

Hand-written task books grow and tend to omit facts that cost rollouts (trace capacity,
an underestimated rollout wall time) or to leak holdout information. Here a brief is rendered from a versioned template plus the task adapter
(so every number comes from the same config the harness enforces) and is linted before
any agent sees it: a missing required section or fact is an error, never a warning.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datafarm.expert.adapter import TaskAdapter
from datafarm.expert.util import HarnessError, get_dotted, sha256_text

TOOL = "xh"

# Harness-level rules every brief states in addition to the task's own forbidden list.
HARNESS_FORBIDDEN = (
    "SR-006: Producing new robot demonstrations with unchecked or sustained near-limit joint states. "
    "Bind limit sources and audit both arms/all six joints over actual full trajectories, including "
    "home/wait/recovery, next-state labels and trim connections; report per-phase margin quantiles, "
    "fractions below 1/5/10 degrees and longest dwell. Validate task-specific margin/occupancy/dwell "
    "profiles before scaling; candidate margins are not hardware limits. "
    "Replan and reacquire matched images/actions; never clip labels alone or hide home frames. "
    "Flag missing readback/limit evidence in REPORT.md; task success alone is not margin validation.",
    "Running anything on a GPU except through `xh rollout` (you have no scheduler "
    "credentials; direct submissions are impossible and attempts are logged).",
    "Running seeds outside the development set; the broker rejects them and records a rule "
    "violation.",
    "Trying to read, infer or reconstruct holdout seeds, verifier source code or any masked "
    "path; masked paths are empty or absent inside your sandbox by design.",
    "Editing files outside your workspace; everything else you can see is read-only.",
)

SECTION_RULES: tuple[tuple[str, str], ...] = (
    ("task", r"^task\b"),
    ("interface", r"expert interface"),
    ("verifier", r"^verifier"),
    ("runtime", r"runtime limits"),
    ("budget", r"^budget"),
    ("tools", r"^tools"),
    ("readable", r"readable paths"),
    ("forbidden", r"^forbidden"),
    ("deliverables", r"^deliverables"),
    ("submit", r"submit protocol"),
    ("holdout", r"holdout policy"),
    ("abort", r"stop or abort"),
)

_TEMPLATE_TOKEN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")


class BriefError(HarnessError):
    pass


@dataclasses.dataclass
class LintReport:
    ok: bool
    errors: list[str]
    warnings: list[str]
    sections: list[str]
    sha256: str
    chars: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_template(text: str, context: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        value = get_dotted(context, match.group(1))
        if value is None:
            raise BriefError(f"brief template references unknown value {{{{ {match.group(1)} }}}}")
        return str(value)

    return _TEMPLATE_TOKEN.sub(replace, text)


def masked_under_readable(adapter: TaskAdapter) -> list[str]:
    """Protected patterns that lie inside a readable root (must be named in the brief)."""

    readable = adapter.display_paths(adapter.section("workspace").get("readable", []))
    protected = adapter.display_paths(adapter.section("workspace").get("protected", []))
    roots = [re.split(r"[*?\[]", r, maxsplit=1)[0].rstrip("/") for r in readable]
    out = []
    for pattern in protected:
        prefix = re.split(r"[*?\[]", pattern, maxsplit=1)[0]
        if any(prefix.startswith(root + "/") for root in roots if root):
            out.append(pattern)
    return out


def brief_context(adapter: TaskAdapter, *, round_info: Mapping[str, Any] | None = None,
                  handoff_md: str | None = None, references_md: str | None = None) -> dict[str, Any]:
    budgets = adapter.budgets
    verifier = adapter.verifier
    brief_cfg = adapter.section("brief")
    info = dict(round_info or {})
    round_dir = info.get("round_dir", "<round>")
    facts = adapter.x(brief_cfg.get("facts", {}), keep_env=True)
    runtime = [f"**{f.get('title', f['id'])}.** {adapter.x(f['text'], keep_env=True)}"
               for f in adapter.raw.get("runtime_facts", [])]
    readable = adapter.display_paths(adapter.section("workspace").get("readable", []))
    masked = masked_under_readable(adapter)
    forbidden = [adapter.x(line, keep_env=True) for line in adapter.raw.get("forbidden", [])]
    deliverables = adapter.raw.get("deliverables", [])
    return {
        "task_id": adapter.task_id,
        "title": adapter.title,
        "facts": facts,
        "verifier": {
            "version": verifier["version"],
            "name": verifier["name"],
            "criteria_md": bullets(adapter.x(verifier["criteria"], keep_env=True)),
            "judge_points": " AND ".join(verifier["judge_points"]),
        },
        "budget": {**budgets.describe(), "wait_cap_s": str(int(budgets.wait_cap_s))},
        "dev_seeds": adapter.dev_seeds.describe(),
        "runtime_facts_md": bullets(runtime),
        "readable_md": bullets(f"`{p}`" for p in readable),
        "masked_md": bullets(f"`{p}` (masked: not readable)" for p in masked) or "- (none)",
        "forbidden_md": bullets(list(forbidden) + list(HARNESS_FORBIDDEN)),
        "deliverables_md": bullets(f"`{d['file']}`: {d['what']}" if isinstance(d, dict) else f"`{d}`"
                                   for d in deliverables),
        "tool": TOOL,
        "round": {
            "id": info.get("round_id", "<round-id>"),
            "dir": round_dir,
            "model": info.get("model", adapter.agent.get("model")),
            "backend": info.get("backend", adapter.agent.get("backend")),
            "workspace": info.get("workspace", f"{round_dir}/workspace"),
            "rollouts": info.get("rollouts", f"{round_dir}/rollouts"),
            "brief": info.get("brief", f"{round_dir}/brief.md"),
            "handoff": info.get("handoff", f"{round_dir}/handoff-in"),
            "references": info.get("references", f"{round_dir}/references"),
        },
        "handoff_md": handoff_md or adapter.x(brief_cfg.get("first_round_md", "This is the first "
                                                          "round for this task; there is no "
                                                          "previous expert."), keep_env=True),
        "references_md": references_md or "No verified reference experts were retrieved for this round.",
    }


def render_brief(adapter: TaskAdapter, *, round_info: Mapping[str, Any] | None = None,
                 handoff_md: str | None = None, references_md: str | None = None,
                 template_text: str | None = None) -> str:
    text = template_text if template_text is not None else adapter.brief_template_path.read_text(
        encoding="utf-8")
    context = brief_context(adapter, round_info=round_info, handoff_md=handoff_md,
                            references_md=references_md)
    return render_template(text, context)


def split_sections(text: str) -> dict[str, str]:
    """Map each ``## heading`` (lowercased) to its heading line plus body."""

    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current is not None:
                sections[current] = "\n".join(lines)
            current = line[3:].strip().lower()
            lines = [line[3:].strip()]
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines)
    return sections


def _find(sections: Mapping[str, str], pattern: str) -> tuple[str, str] | None:
    for heading, body in sections.items():
        if re.search(pattern, heading):
            return heading, body
    return None


def lint_brief(text: str, adapter: TaskAdapter, *, holdout_seeds: Sequence[int] | None = None,
               round_info: Mapping[str, Any] | None = None) -> LintReport:
    errors: list[str] = []
    warnings: list[str] = []
    sections = split_sections(text)
    found: dict[str, str] = {}
    for section_id, pattern in SECTION_RULES:
        hit = _find(sections, pattern)
        if hit is None:
            errors.append(f"missing required section '{section_id}' (a '## ' heading matching /{pattern}/)")
        else:
            found[section_id] = hit[1]
            if not "\n".join(hit[1].splitlines()[1:]).strip():
                errors.append(f"section '{section_id}' is empty")

    def need(section_id: str, tokens: Iterable[str], why: str, *, whole: bool = False) -> None:
        body = found.get(section_id)
        if body is None:
            return
        for token in tokens:
            if not token:
                continue
            present = (re.search(r"(?<![\w.])" + re.escape(token) + r"(?!\w|\.\d)", body) is not None
                       if whole else token in body)
            if not present:
                errors.append(f"section '{section_id}' must mention {token!r} ({why})")

    verifier = adapter.verifier
    need("verifier", [verifier["version"]], "verifier version")
    need("verifier", verifier.get("criteria_tokens", []), "exact pass criteria")
    points = verifier.get("judge_points", [])
    need("verifier", points, "every judge point")
    if len(points) > 1:
        need("verifier", ["AND"], "all judge points must pass, stated explicitly")
    for fact in adapter.raw.get("runtime_facts", []):
        need("runtime", [adapter.x(t, keep_env=True) for t in fact.get("must_mention", [])],
             f"runtime fact {fact['id']}")
    described = adapter.budgets.describe()
    need("budget", [described[k] for k in ("max_counted_rollouts", "max_concurrent_rollouts",
                                           "total_wall", "wall_per_session", "max_sessions")],
         "budget values enforced by the harness", whole=True)
    if adapter.budgets.max_usd is not None:
        need("budget", [described["max_usd"]], "USD budget", whole=True)
    need("budget", ["infra"], "how infrastructure failures are counted")
    need("tools", [f"{TOOL} rollout", f"{TOOL} wait", f"{TOOL} status", f"{TOOL} result",
                   f"{TOOL} submit", f"{TOOL} flag-for-human"], "broker commands")
    need("submit", [f"{TOOL} submit", "--final"], "submit protocol")
    need("holdout", ["hidden", "never"], "holdout policy")
    need("abort", [f"{TOOL} flag-for-human"], "abort channel")
    need("deliverables", [d["file"] if isinstance(d, dict) else str(d)
                          for d in adapter.raw.get("deliverables", [])], "deliverable files")
    need("readable", adapter.display_paths(adapter.section("workspace").get("readable", [])),
         "every readable path")
    readable_body = found.get("readable", "")
    for pattern in adapter.display_paths(adapter.section("workspace").get("protected", [])):
        exact = re.compile(re.escape(pattern) + r"(?=[`'\s/),;]|$)")
        for line in readable_body.splitlines():
            if exact.search(line) and not re.search(r"masked|not readable|except|forbidden", line, re.I):
                errors.append(f"protected path {pattern!r} is listed as readable")
    for pattern in masked_under_readable(adapter):
        if pattern not in text:
            errors.append(f"masked path {pattern!r} lies inside a readable root but the brief "
                          "does not name it")
    forbidden_body = found.get("forbidden", "")
    if sum(1 for line in forbidden_body.splitlines() if line.lstrip().startswith("- ")) < 3:
        errors.append("section 'forbidden' must list at least three forbidden actions")
    if "{{" in text or "}}" in text:
        errors.append("unresolved template placeholders remain")
    if holdout_seeds:
        leaked = sorted({s for s in holdout_seeds if re.search(rf"(?<![\w.]){s}(?![\w.])", text)})
        if leaked:
            errors.append(f"the brief mentions {len(leaked)} holdout seed value(s)")
    if round_info:
        for key in ("workspace", "rollouts"):
            if round_info.get(key) and round_info[key] not in text:
                errors.append(f"the brief does not state the round {key} path")
    if len(text) > 40_000:
        warnings.append(f"brief is {len(text)} characters; consider moving detail into handoff files")
    for line in text.splitlines():
        if "attempts.jsonl" in line and re.search(r"\b(write|append|keep)\b", line, re.I):
            warnings.append("the brief asks the agent to keep an attempts ledger; the harness "
                            "records attempts itself")
            break
    return LintReport(ok=not errors, errors=errors, warnings=warnings, sections=sorted(found),
                      sha256=sha256_text(text), chars=len(text))


def lint_file(path: str | Path, adapter: TaskAdapter, **kwargs: Any) -> LintReport:
    return lint_brief(Path(path).read_text(encoding="utf-8"), adapter, **kwargs)
