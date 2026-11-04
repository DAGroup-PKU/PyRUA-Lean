"""Calibrate a VLM reviewer against human labels and audit it against standing rules.

A VLM reviewer ("does this generated video preserve geometry / entities / ...?") may
only screen data after its agreement with humans has been measured. This module pairs
per-sample, per-criterion VLM verdicts with human labels and reports, per criterion:
confusion matrix, accuracy, Cohen's kappa, false-accept and false-reject rates (with
Wilson 95% intervals) and abstentions. Separately, it flags VLM verdicts and reviewer
prompts that contradict standing user rules (e.g. "hard shadows are not a defect").

Accepted VLM outputs (as written by the Seedance review scripts):

- ``<run>/episodes/<id>/visual-review/assessment.json``: first-pass screening with
  ``assessment.criteria.<name>.status`` in pass/fail/uncertain;
- ``.../visual-review/focused-recheck/assessment.json``: a targeted second check that
  overrides the named criteria, applied only when its ``original_assessment_sha256``
  equals the SHA-256 of the first-pass file bytes;
- ``<run>/review/visual-screen-summary.json``: already merged rows;
- any JSON object (or list of objects) ``{"id": ..., "criteria": {name: {"status": ...}}}``.

Class conventions: human labels and VLM verdicts are mapped to ``pass`` / ``fail`` /
``abstain``. ``fail`` means "defect present". A *false accept* is a VLM ``pass`` on a
human ``fail``; a *false reject* is a VLM ``fail`` on a human ``pass``.
"""

from __future__ import annotations

from datafarm.common import sha256_file
import csv
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from datafarm.verify.stats import cohen_kappa, wilson_interval
from datafarm.common import sha256_text

PASS, FAIL, ABSTAIN = "pass", "fail", "abstain"
CLASSES = (PASS, FAIL, ABSTAIN)
OUTCOME = "@outcome"
MAPPING_SCHEMA = "datafarm.vlm-calibration-mapping/1"
RULES_SCHEMA = "datafarm.standing-rules/1"
REPORT_SCHEMA = "datafarm.vlm-reviewer-calibration/1"
RULE_SCOPES = frozenset({"criterion", "evidence", "prompt"})

DEFAULT_HUMAN_VOCABULARY: Mapping[str, Sequence[str]] = {
    PASS: ("pass", "passed", "ok", "good", "accept", "accepted", "acceptable",
           "approved", "looks real", "passable"),
    FAIL: ("fail", "failed", "bad", "reject", "rejected", "defect", "defective",
           "not approved", "has defect", "looks fake"),
    ABSTAIN: ("abstain", "unsure", "uncertain", "skip", "skipped", "n/a",
              "not sure", "cannot judge"),
}
DEFAULT_VLM_VOCABULARY: Mapping[str, Sequence[str]] = {
    PASS: ("pass", "sampled_pass"),
    FAIL: ("fail", "flagged"),
    ABSTAIN: ("uncertain", "abstain"),
}


# --------------------------------------------------------------------------- vocabulary


def _norm(label: Any) -> str:
    return str(label).strip().casefold()


@dataclass(frozen=True)
class Vocabulary:
    table: Mapping[str, str]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Sequence[str]]) -> "Vocabulary":
        table: dict[str, str] = {}
        for klass, labels in mapping.items():
            if klass not in CLASSES:
                raise ValueError(f"vocabulary class must be one of {CLASSES}: {klass!r}")
            for label in labels:
                key = _norm(label)
                if table.get(key, klass) != klass:
                    raise ValueError(f"label {label!r} maps to two classes")
                table[key] = klass
        return cls(table)

    def classify(self, label: Any) -> str | None:
        """Class for ``label``; ``None`` for an empty label; ``ValueError`` if unknown."""

        if label is None or _norm(label) == "":
            return None
        key = _norm(label)
        if key not in self.table:
            raise ValueError(f"label {label!r} is not in the vocabulary")
        return self.table[key]


def derive_outcome(statuses: Mapping[str, str], vocabulary: Vocabulary) -> str:
    """Same rule as the Seedance summary script: any fail -> flagged, else any
    uncertain -> uncertain, else sampled_pass."""

    classes = {vocabulary.classify(value) for value in statuses.values()}
    if FAIL in classes:
        return "flagged"
    if ABSTAIN in classes or None in classes:
        return "uncertain"
    return "sampled_pass"


# --------------------------------------------------------------------------- VLM reviews


@dataclass
class VlmReview:
    sample_id: str
    statuses: dict[str, str]
    evidence: dict[str, str]
    source: str
    stage: str
    stored_outcome: str | None = None
    prompts: list[tuple[str, str]] = field(default_factory=list)  # (origin, text)
    rechecked: list[str] = field(default_factory=list)
    outcome: str | None = None  # derived


def _criteria_block(criteria: Any) -> tuple[dict[str, str], dict[str, str]]:
    statuses: dict[str, str] = {}
    evidence: dict[str, str] = {}
    if not isinstance(criteria, Mapping):
        raise ValueError("criteria must be an object")
    for name, value in criteria.items():
        if isinstance(value, Mapping):
            statuses[name] = str(value.get("status"))
            if isinstance(value.get("evidence"), str):
                evidence[name] = value["evidence"]
        elif isinstance(value, str):
            statuses[name] = value
    return statuses, evidence


def _prompt_from_metadata(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    system = data.get("system") if isinstance(data, Mapping) else None
    return [(str(path), system)] if isinstance(system, str) else []


def _load_assessment_file(
    path: Path, *, apply_rechecks: bool, problems: list[dict[str, Any]]
) -> VlmReview | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assessment = data.get("assessment") if isinstance(data, Mapping) else None
    if not isinstance(assessment, Mapping) or "criteria" not in assessment:
        problems.append({"kind": "unrecognised_assessment", "path": str(path)})
        return None
    statuses, evidence = _criteria_block(assessment["criteria"])
    sample_id = str(data.get("id") or path.parent.parent.name)
    review = VlmReview(
        sample_id=sample_id,
        statuses=statuses,
        evidence=evidence,
        source=str(path),
        stage="initial",
        stored_outcome=data.get("outcome"),
        prompts=_prompt_from_metadata(path.parent / "request-metadata.json"),
    )
    recheck = path.parent / "focused-recheck" / "assessment.json"
    if apply_rechecks and recheck.exists():
        revised = json.loads(recheck.read_text(encoding="utf-8"))
        expected = revised.get("original_assessment_sha256")
        if expected != sha256_file(path) or not isinstance(revised.get("criteria"), Mapping):
            problems.append(
                {"kind": "recheck_binding_mismatch", "sample_id": sample_id, "path": str(recheck)}
            )
        else:
            new_statuses, new_evidence = _criteria_block(revised["criteria"])
            review.statuses.update(new_statuses)
            review.evidence.update(new_evidence)
            review.rechecked = sorted(new_statuses)
            review.stage = "initial+focused_recheck"
            review.stored_outcome = None
            review.prompts.extend(_prompt_from_metadata(recheck.parent / "request-metadata.json"))
    return review


def _generic_reviews(data: Any, origin: str) -> list[VlmReview]:
    items = data if isinstance(data, list) else [data]
    reviews = []
    for item in items:
        if not isinstance(item, Mapping) or "id" not in item or "criteria" not in item:
            raise ValueError(f"{origin}: expected objects with 'id' and 'criteria'")
        statuses, evidence = _criteria_block(item["criteria"])
        reviews.append(
            VlmReview(
                sample_id=str(item["id"]),
                statuses=statuses,
                evidence=evidence,
                source=origin,
                stage="generic",
                stored_outcome=item.get("outcome"),
            )
        )
    return reviews


def _summary_reviews(data: Mapping[str, Any], origin: str) -> list[VlmReview]:
    reviews = []
    for row in data.get("rows", []):
        if not isinstance(row, Mapping) or not isinstance(row.get("criteria"), Mapping):
            continue  # unreviewed rows carry no criteria
        statuses, evidence = _criteria_block(row["criteria"])
        reviews.append(
            VlmReview(
                sample_id=str(row["id"]),
                statuses=statuses,
                evidence=evidence,
                source=origin,
                stage="summary+focused_recheck" if row.get("focused_recheck") else "summary",
                stored_outcome=row.get("outcome"),
            )
        )
    return reviews


def load_vlm_reviews(
    paths: Iterable[str | Path],
    *,
    apply_rechecks: bool = True,
    vocabulary: Vocabulary | None = None,
) -> tuple[dict[str, VlmReview], list[dict[str, Any]]]:
    vocab = vocabulary or Vocabulary.from_mapping(DEFAULT_VLM_VOCABULARY)
    problems: list[dict[str, Any]] = []
    loaded: list[VlmReview] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            files = sorted(
                f for f in path.rglob("assessment.json") if f.parent.name == "visual-review"
            )
            for file in files:
                review = _load_assessment_file(
                    file, apply_rechecks=apply_rechecks, problems=problems
                )
                if review is not None:
                    loaded.append(review)
            if not files:
                summary = next(
                    (
                        candidate
                        for candidate in (
                            path / "review" / "visual-screen-summary.json",
                            path / "visual-screen-summary.json",
                        )
                        if candidate.exists()
                    ),
                    None,
                )
                if summary is None:
                    problems.append({"kind": "no_reviews_found", "path": str(path)})
                else:
                    data = json.loads(summary.read_text(encoding="utf-8"))
                    loaded.extend(_summary_reviews(data, str(summary)))
        elif path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, Mapping) and isinstance(data.get("assessment"), Mapping):
                review = _load_assessment_file(
                    path, apply_rechecks=apply_rechecks, problems=problems
                )
                if review is not None:
                    loaded.append(review)
            elif isinstance(data, Mapping) and isinstance(data.get("rows"), list):
                loaded.extend(_summary_reviews(data, str(path)))
            elif isinstance(data, Mapping) and "original_assessment_sha256" in data:
                raise ValueError(
                    f"{path}: a focused re-check is applied from its visual-review directory"
                )
            else:
                loaded.extend(_generic_reviews(data, str(path)))
        else:
            raise FileNotFoundError(f"no VLM review output at {path}")
    reviews: dict[str, VlmReview] = {}
    for review in loaded:
        try:
            review.outcome = derive_outcome(review.statuses, vocab)
        except ValueError as error:
            problems.append(
                {"kind": "unknown_vlm_status", "sample_id": review.sample_id, "error": str(error)}
            )
            continue
        if review.stored_outcome and review.stage in ("initial", "summary", "generic"):
            if _norm(review.stored_outcome) != _norm(review.outcome):
                problems.append(
                    {
                        "kind": "stored_outcome_mismatch",
                        "sample_id": review.sample_id,
                        "stored": review.stored_outcome,
                        "derived": review.outcome,
                    }
                )
        if review.sample_id in reviews:
            problems.append(
                {"kind": "duplicate_sample", "sample_id": review.sample_id, "path": review.source}
            )
            continue
        reviews[review.sample_id] = review
    return reviews, problems


# --------------------------------------------------------------------------- human labels

_ID_COLUMNS = ("sample_id", "sample", "id")


def _row_id(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    for column in _ID_COLUMNS:
        if column in row and str(row[column]).strip():
            return column, str(row[column]).strip()
    return None, None


def _rows_to_labels(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for row in rows:
        id_column, sample = _row_id(row)
        if sample is None:
            raise ValueError("every human-label row needs a sample_id")
        target = labels.setdefault(sample, {})
        if "criterion" in row and "label" in row:
            criterion = str(row["criterion"]).strip()
            if criterion in target:
                raise ValueError(f"duplicate human label for {sample}/{criterion}")
            target[criterion] = "" if row["label"] is None else str(row["label"])
        else:
            for key, value in row.items():
                if key in (id_column, "note", "notes", "rater", "comment"):
                    continue
                if key in target:
                    raise ValueError(f"duplicate human label for {sample}/{key}")
                target[key] = "" if value is None else str(value)
    return labels


def read_human_label_file(path: str | Path) -> dict[str, dict[str, str]]:
    """Raw labels ``{sample_id: {criterion: label}}`` from JSON or CSV."""

    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".csv":
        return _rows_to_labels(csv.DictReader(io.StringIO(text)))
    data = json.loads(text)
    if isinstance(data, Mapping) and "labels" in data:
        data = data["labels"]
    if isinstance(data, list):
        return _rows_to_labels(data)
    if isinstance(data, Mapping):
        labels: dict[str, dict[str, str]] = {}
        for sample, criteria in data.items():
            if not isinstance(criteria, Mapping):
                raise ValueError(f"labels for {sample!r} must be an object")
            labels[str(sample)] = {
                str(name): str(value.get("label") if isinstance(value, Mapping) else value)
                for name, value in criteria.items()
            }
        return labels
    raise ValueError("unrecognised human label file")


def apply_id_map(
    labels: Mapping[str, Mapping[str, str]], id_map: Any, template: str = "{sample}"
) -> dict[str, dict[str, str]]:
    """Rename human sample ids, e.g. blind label ids to review sample ids.

    ``id_map`` is ``{human_id: review_id}`` or a list of objects with ``label_id`` whose
    fields fill ``template`` (e.g. a blind key ``[{"label_id": "L01", "sample": "043"}]``
    with template ``"sample-{sample}"``).
    """

    if isinstance(id_map, Mapping):
        mapping = {str(k): str(v) for k, v in id_map.items()}
    elif isinstance(id_map, list):
        mapping = {str(item["label_id"]): template.format(**item) for item in id_map}
    else:
        raise ValueError("id map must be an object or a list of objects with label_id")
    renamed: dict[str, dict[str, str]] = {}
    for sample, criteria in labels.items():
        target = mapping.get(sample, sample)
        if target in renamed:
            raise ValueError(f"two human samples map to {target!r}")
        renamed[target] = dict(criteria)
    return renamed


# --------------------------------------------------------------------------- mapping


@dataclass(frozen=True)
class CriterionMap:
    human: str
    vlm: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationMapping:
    criteria: tuple[CriterionMap, ...]
    human_vocabulary: Vocabulary
    vlm_vocabulary: Vocabulary

    @classmethod
    def load(
        cls, path: str | Path | None, *, human_criteria: Iterable[str] = ()
    ) -> "CalibrationMapping":
        data: Mapping[str, Any] = {}
        if path is not None:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data.get("schema") != MAPPING_SCHEMA:
                raise ValueError(f"mapping schema must be {MAPPING_SCHEMA!r}")
        human_vocab = Vocabulary.from_mapping(
            data.get("human_vocabulary") or DEFAULT_HUMAN_VOCABULARY
        )
        vlm_vocab = Vocabulary.from_mapping(data.get("vlm_vocabulary") or DEFAULT_VLM_VOCABULARY)
        entries: list[CriterionMap] = []
        if data.get("criteria"):
            for human, target in data["criteria"].items():
                vlm = target.get("vlm") if isinstance(target, Mapping) else target
                if isinstance(vlm, str):
                    vlm = [vlm]
                if not vlm or not all(isinstance(name, str) for name in vlm):
                    raise ValueError(f"mapping for {human!r} needs a list of VLM criteria")
                entries.append(CriterionMap(human=str(human), vlm=tuple(vlm)))
        else:
            entries = [CriterionMap(human=name, vlm=(name,)) for name in sorted(human_criteria)]
        return cls(tuple(entries), human_vocab, vlm_vocab)


def combine_vlm(
    review: VlmReview, criteria: Sequence[str], vocabulary: Vocabulary
) -> tuple[str | None, str | None]:
    """Conservative combination: any fail -> fail; otherwise any abstain or missing
    criterion -> abstain; all present and pass -> pass. ``(None, 'missing')`` if none of
    the criteria is present."""

    classes: list[str] = []
    missing: list[str] = []
    for name in criteria:
        raw = review.outcome if name == OUTCOME else review.statuses.get(name)
        klass = vocabulary.classify(raw) if raw is not None else None
        if klass is None:
            missing.append(name)
        else:
            classes.append(klass)
    if not classes:
        return None, "missing"
    if FAIL in classes:
        return FAIL, None
    if ABSTAIN in classes:
        return ABSTAIN, None
    if missing:
        return ABSTAIN, "missing:" + ",".join(missing)
    return PASS, None


# --------------------------------------------------------------------------- metrics


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _interval(numerator: int, denominator: int) -> list[float] | None:
    interval = wilson_interval(numerator, denominator)
    return None if interval is None else [round(interval[0], 4), round(interval[1], 4)]


def agreement_metrics(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Metrics for (human, vlm) class pairs; human in {pass, fail}, vlm adds abstain."""

    for human, vlm in pairs:
        if human not in (PASS, FAIL) or vlm not in CLASSES:
            raise ValueError(f"invalid pair ({human!r}, {vlm!r})")

    def count(human: str, vlm: str) -> int:
        return sum(1 for pair in pairs if pair == (human, vlm))

    human_pass = sum(1 for h, _ in pairs if h == PASS)
    human_fail = len(pairs) - human_pass
    decided = [pair for pair in pairs if pair[1] != ABSTAIN]
    agree = sum(1 for h, v in decided if h == v)
    false_accepts = count(FAIL, PASS)
    false_rejects = count(PASS, FAIL)
    abstentions = len(pairs) - len(decided)
    decided_fail = sum(1 for h, _ in decided if h == FAIL)
    decided_pass = len(decided) - decided_fail
    operational = [(h, FAIL if v == ABSTAIN else v) for h, v in pairs]
    op_agree = sum(1 for h, v in operational if h == v)
    op_rejects = sum(1 for pair in operational if pair == (PASS, FAIL))
    return {
        "n": len(pairs),
        "human_pass": human_pass,
        "human_fail": human_fail,
        "vlm_abstain": abstentions,
        "decided": len(decided),
        "confusion": {
            "human_pass": {f"vlm_{c}": count(PASS, c) for c in CLASSES},
            "human_fail": {f"vlm_{c}": count(FAIL, c) for c in CLASSES},
        },
        "accuracy": _rate(agree, len(decided)),
        "accuracy_ci95": _interval(agree, len(decided)),
        "cohen_kappa": cohen_kappa(decided),
        "false_accept_rate": _rate(false_accepts, human_fail),
        "false_accept_rate_ci95": _interval(false_accepts, human_fail),
        "false_accept_rate_decided": _rate(false_accepts, decided_fail),
        "false_reject_rate": _rate(false_rejects, human_pass),
        "false_reject_rate_ci95": _interval(false_rejects, human_pass),
        "false_reject_rate_decided": _rate(false_rejects, decided_pass),
        "abstention_rate": _rate(abstentions, len(pairs)),
        "abstain_as_reject": {
            "accuracy": _rate(op_agree, len(operational)),
            "cohen_kappa": cohen_kappa(operational),
            "false_reject_rate": _rate(op_rejects, human_pass),
        },
    }


def calibrate(
    reviews: Mapping[str, VlmReview],
    human_raw: Mapping[str, Mapping[str, str]],
    mapping: CalibrationMapping,
) -> dict[str, Any]:
    human: dict[str, dict[str, str | None]] = {
        sample: {name: mapping.human_vocabulary.classify(label) for name, label in labels.items()}
        for sample, labels in human_raw.items()
    }
    per_criterion: dict[str, Any] = {}
    pooled: list[tuple[str, str]] = []
    for entry in mapping.criteria:
        pairs: list[tuple[str, str]] = []
        details: list[dict[str, Any]] = []
        human_abstentions = vlm_missing = 0
        for sample in sorted(human):
            label = human[sample].get(entry.human)
            if label is None:
                continue
            if label == ABSTAIN:
                human_abstentions += 1
                continue
            review = reviews.get(sample)
            verdict, note = (None, "no_review")
            if review is not None:
                verdict, note = combine_vlm(review, entry.vlm, mapping.vlm_vocabulary)
            if verdict is None:
                vlm_missing += 1
                continue
            pairs.append((label, verdict))
            row = {"sample_id": sample, "human": label, "vlm": verdict}
            if note:
                row["note"] = note
            details.append(row)
        per_criterion[entry.human] = {
            "vlm_criteria": list(entry.vlm),
            **agreement_metrics(pairs),
            "human_abstentions": human_abstentions,
            "unpaired_missing_vlm": vlm_missing,
            "pairs": details,
        }
        pooled.extend(pairs)
    labelled = set(human)
    return {
        "status": "calibrated" if pooled else "no_paired_data",
        "human_samples": len(labelled),
        "reviewed_samples": len(reviews),
        "paired_samples": len(labelled & set(reviews)),
        "human_samples_without_review": sorted(labelled - set(reviews)),
        "per_criterion": per_criterion,
        "pooled": agreement_metrics(pooled),
    }


def vlm_summary(reviews: Mapping[str, VlmReview], vocabulary: Vocabulary) -> dict[str, Any]:
    per_criterion: dict[str, dict[str, int]] = {}
    outcomes: dict[str, int] = {}
    stages: dict[str, int] = {}
    for review in reviews.values():
        outcomes[review.outcome or "unknown"] = outcomes.get(review.outcome or "unknown", 0) + 1
        stages[review.stage] = stages.get(review.stage, 0) + 1
        for name, raw in review.statuses.items():
            counts = per_criterion.setdefault(name, {c: 0 for c in CLASSES})
            try:
                klass = vocabulary.classify(raw) or "missing"
            except ValueError:
                klass = "unknown"
            counts[klass] = counts.get(klass, 0) + 1
    return {
        "samples": len(reviews),
        "outcomes": dict(sorted(outcomes.items())),
        "stages": dict(sorted(stages.items())),
        "per_criterion": dict(sorted(per_criterion.items())),
    }


# --------------------------------------------------------------------------- standing rules


@dataclass(frozen=True)
class StandingRule:
    id: str
    statement: str
    source: str | None
    forbidden_criteria: frozenset[str]
    forbidden_phrases: tuple[str, ...]
    forbidden_patterns: tuple[re.Pattern[str], ...]
    scopes: frozenset[str]
    exempt_criteria: frozenset[str]

    def matches(self, text: str) -> list[tuple[str, int, int]]:
        """``(matched text, start, end)`` for every phrase/pattern hit (case-insensitive)."""

        found: list[tuple[str, int, int]] = []
        folded = text.casefold()
        for phrase in self.forbidden_phrases:
            index = folded.find(phrase.casefold())
            if index >= 0:
                found.append((phrase, index, index + len(phrase)))
        for pattern in self.forbidden_patterns:
            match = pattern.search(text)
            if match:
                found.append((pattern.pattern, match.start(), match.end()))
        return found


def load_standing_rules(path: str | Path) -> list[StandingRule]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        if data.get("schema") != RULES_SCHEMA:
            raise ValueError(f"standing rules schema must be {RULES_SCHEMA!r}")
        data = data.get("rules")
    if not isinstance(data, list) or not data:
        raise ValueError("standing rules must be a non-empty list")
    rules: list[StandingRule] = []
    seen: set[str] = set()
    audit_fields = ("forbidden_criteria", "forbidden_phrases", "forbidden_patterns")
    for raw in data:
        # Unified registry (configs/standing-rules.json): review fields live in a nested
        # ``review_audit`` block; rules without one (lint-only rules) are skipped. The
        # older flat form keeps its fields at the top level.
        if isinstance(raw.get("review_audit"), Mapping):
            item = {**raw, **raw["review_audit"]}
        elif any(raw.get(field) for field in audit_fields):
            item = raw
        else:
            continue
        rule_id = str(item.get("id") or "")
        if not rule_id or rule_id in seen:
            raise ValueError(f"standing rule ids must be unique and non-empty: {rule_id!r}")
        seen.add(rule_id)
        scopes = frozenset(item.get("scopes") or RULE_SCOPES)
        if not scopes <= RULE_SCOPES:
            raise ValueError(f"{rule_id}: scopes must be within {sorted(RULE_SCOPES)}")
        forbidden = frozenset(item.get("forbidden_criteria") or ())
        exempt = frozenset(item.get("exempt_criteria") or ())
        if forbidden & exempt:
            raise ValueError(f"{rule_id}: a criterion cannot be both forbidden and exempt")
        phrases = tuple(str(p) for p in item.get("forbidden_phrases") or () if str(p).strip())
        patterns = tuple(re.compile(p, re.I) for p in item.get("forbidden_patterns") or ())
        if not (forbidden or phrases or patterns):
            raise ValueError(f"{rule_id}: needs forbidden criteria, phrases or patterns")
        rules.append(
            StandingRule(
                id=rule_id,
                statement=str(item.get("statement") or ""),
                source=item.get("source"),
                forbidden_criteria=forbidden,
                forbidden_phrases=phrases,
                forbidden_patterns=patterns,
                scopes=scopes,
                exempt_criteria=exempt,
            )
        )
    if not rules:
        raise ValueError("standing rules define no review_audit rules")
    return rules


def _excerpt(text: str, start: int, end: int, context: int = 60) -> str:
    left, right = max(0, start - context), min(len(text), end + context)
    return ("…" if left else "") + text[left:right] + ("…" if right < len(text) else "")


def audit_standing_rules(
    reviews: Mapping[str, VlmReview],
    rules: Sequence[StandingRule],
    vocabulary: Vocabulary,
    *,
    extra_prompts: Sequence[tuple[str, str]] = (),
    max_details: int = 200,
) -> dict[str, Any]:
    """Flag VLM ``fail`` verdicts and reviewer prompts that contradict standing rules.

    Criterion-level hits (the criterion itself is forbidden) are definite conflicts;
    evidence- and prompt-level hits are phrase matches, i.e. *candidates* for a human to
    confirm.
    """

    conflicts: list[dict[str, Any]] = []
    flagged: dict[str, dict[str, set[str]]] = {}  # sample -> criterion -> kinds
    for sample in sorted(reviews):
        review = reviews[sample]
        for criterion in sorted(review.statuses):
            try:
                klass = vocabulary.classify(review.statuses[criterion])
            except ValueError:
                continue
            if klass != FAIL:
                continue
            evidence = review.evidence.get(criterion, "")
            for rule in rules:
                if criterion in rule.exempt_criteria:
                    continue
                kinds: set[str] = set()
                matched: list[str] = []
                excerpt = None
                if "criterion" in rule.scopes and criterion in rule.forbidden_criteria:
                    kinds.add("criterion")
                if "evidence" in rule.scopes and evidence:
                    hits = rule.matches(evidence)
                    if hits:
                        kinds.add("evidence")
                        matched = sorted({hit[0] for hit in hits})
                        excerpt = _excerpt(evidence, hits[0][1], hits[0][2])
                if not kinds:
                    continue
                flagged.setdefault(sample, {}).setdefault(criterion, set()).update(kinds)
                conflict = {
                    "rule_id": rule.id,
                    "sample_id": sample,
                    "criterion": criterion,
                    "kinds": sorted(kinds),
                }
                if matched:
                    conflict["matched"] = matched
                    conflict["excerpt"] = excerpt
                conflicts.append(conflict)

    prompts: dict[str, dict[str, Any]] = {}
    all_prompts = [p for review in reviews.values() for p in review.prompts] + list(extra_prompts)
    for origin, text in all_prompts:
        digest = sha256_text(text)
        entry = prompts.setdefault(digest, {"text": text, "origins": []})
        entry["origins"].append(origin)
    prompt_conflicts: list[dict[str, Any]] = []
    for digest, entry in sorted(prompts.items()):
        for rule in rules:
            if "prompt" not in rule.scopes:
                continue
            hits = rule.matches(entry["text"])
            if hits:
                prompt_conflicts.append(
                    {
                        "rule_id": rule.id,
                        "prompt_sha256": digest,
                        "used_by": len(entry["origins"]),
                        "example_origin": entry["origins"][0],
                        "matched": sorted({hit[0] for hit in hits}),
                        "excerpts": [_excerpt(entry["text"], s, e, 50) for _, s, e in hits[:5]],
                    }
                )

    per_rule: dict[str, Any] = {}
    for rule in rules:
        mine = [c for c in conflicts if c["rule_id"] == rule.id]
        by_criterion: dict[str, int] = {}
        for conflict in mine:
            by_criterion[conflict["criterion"]] = by_criterion.get(conflict["criterion"], 0) + 1
        per_rule[rule.id] = {
            "statement": rule.statement,
            "source": rule.source,
            "fail_verdicts_flagged": len(mine),
            "criterion_level": sum("criterion" in c["kinds"] for c in mine),
            "evidence_candidates_only": sum(c["kinds"] == ["evidence"] for c in mine),
            "samples_affected": len({c["sample_id"] for c in mine}),
            "by_criterion": dict(sorted(by_criterion.items())),
            "prompts_flagged": sum(1 for c in prompt_conflicts if c["rule_id"] == rule.id),
        }

    def outcomes_without(kinds_to_drop: set[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample, review in reviews.items():
            dropped = {
                criterion
                for criterion, kinds in flagged.get(sample, {}).items()
                if kinds & kinds_to_drop
            }
            kept = {k: v for k, v in review.statuses.items() if k not in dropped}
            outcome = derive_outcome(kept, vocabulary) if kept else "sampled_pass"
            counts[outcome] = counts.get(outcome, 0) + 1
        return dict(sorted(counts.items()))

    return {
        "rules": per_rule,
        "fail_verdicts_flagged": len(conflicts),
        "prompts_scanned": len(prompts),
        "prompt_conflicts": prompt_conflicts,
        "outcomes_if_criterion_conflicts_ignored": outcomes_without({"criterion"}),
        "outcomes_if_all_flagged_ignored": outcomes_without({"criterion", "evidence"}),
        "conflicts": conflicts[:max_details],
        "conflicts_truncated": max(0, len(conflicts) - max_details),
        "note": "criterion-level hits contradict a rule by construction; evidence and "
        "prompt hits are phrase matches that a human must confirm",
    }
