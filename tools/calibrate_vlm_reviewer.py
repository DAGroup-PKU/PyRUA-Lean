#!/usr/bin/env python3
"""Measure a VLM reviewer against human labels and check it against standing user rules.

Inputs:
  --vlm PATH...          review outputs: run directories (episodes/*/visual-review/
                         assessment.json, focused re-checks applied when hash-bound),
                         review/visual-screen-summary.json, or generic
                         {"id", "criteria"} JSON
  --human FILE           optional human labels, JSON or CSV (sample id -> criterion -> label)
  --mapping FILE         optional datafarm.vlm-calibration-mapping/1: human criterion -> VLM
                         criteria (use "@outcome" for the overall verdict) and label
                         vocabularies; default is an identity mapping
  --id-map FILE          optional human-id -> review-id map (object, or a blind key list
                         with label_id fields formatted through --id-template)
  --standing-rules FILE  optional datafarm.standing-rules/1 rules to audit verdicts/prompts
  --prompt-file FILE...  extra reviewer prompt sources to audit (e.g. a review script)

Per criterion it reports the confusion matrix, accuracy, Cohen's kappa, false-accept and
false-reject rates (Wilson 95% intervals) and abstentions. Without human labels only the
VLM verdict distribution and the rule audit are produced; agreement is then unmeasured.

Exit codes: 0 report written, 2 bad input, 3 human labels given but nothing paired
(unless --allow-empty).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datafarm.verify.reviewer_calibration import (  # noqa: E402
    REPORT_SCHEMA,
    CalibrationMapping,
    apply_id_map,
    audit_standing_rules,
    calibrate,
    load_standing_rules,
    load_vlm_reviews,
    read_human_label_file,
    vlm_summary,
)


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_summary(report: dict) -> None:
    vlm = report["vlm"]
    print(
        f"VLM reviews: {vlm['samples']} samples; outcomes {vlm['outcomes']}; "
        f"stages {vlm['stages']}"
    )
    print(f"{'criterion':32} pass  fail  abstain")
    for name, counts in vlm["per_criterion"].items():
        print(f"{name:32} {counts['pass']:4}  {counts['fail']:4}  {counts['abstain']:7}")
    calibration = report.get("calibration")
    if calibration is None:
        print("human labels: none supplied -> agreement with humans NOT measured")
    else:
        print(
            f"human labels: {calibration['human_samples']} samples, "
            f"{calibration['paired_samples']} with a VLM review; status {calibration['status']}"
        )
        print(f"{'human criterion':24} n  acc    kappa  FAR    FRR    abstain")
        rows = list(calibration["per_criterion"].items()) + [("POOLED", calibration["pooled"])]
        for name, metrics in rows:
            print(
                f"{name:24} {metrics['n']:<2} {_fmt(metrics['accuracy']):6} "
                f"{_fmt(metrics['cohen_kappa']):6} {_fmt(metrics['false_accept_rate']):6} "
                f"{_fmt(metrics['false_reject_rate']):6} {_fmt(metrics['abstention_rate'])}"
            )
    audit = report.get("standing_rules")
    if audit is not None:
        print(f"standing rules: {audit['fail_verdicts_flagged']} fail verdicts flagged")
        for rule_id, rule in audit["rules"].items():
            print(
                f"  {rule_id}: {rule['fail_verdicts_flagged']} flagged "
                f"({rule['criterion_level']} criterion-level, "
                f"{rule['evidence_candidates_only']} evidence-only candidates) in "
                f"{rule['samples_affected']} samples; prompts flagged {rule['prompts_flagged']}"
            )
        print(
            "  outcomes if criterion-level conflicts ignored: "
            f"{audit['outcomes_if_criterion_conflicts_ignored']}"
        )
        print(
            "  outcomes if all flagged verdicts ignored: "
            f"{audit['outcomes_if_all_flagged_ignored']}"
        )
    if report["problems"]:
        print(f"problems: {len(report['problems'])} (see JSON report)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vlm", nargs="+", type=Path, required=True)
    parser.add_argument("--human", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--id-map", type=Path)
    parser.add_argument("--id-template", default="{sample}")
    parser.add_argument("--standing-rules", type=Path)
    parser.add_argument("--prompt-file", nargs="*", type=Path, default=[])
    parser.add_argument("--no-rechecks", action="store_true", help="ignore focused re-checks")
    parser.add_argument("--max-conflict-details", type=int, default=200)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--out", help="write the JSON report here ('-' = stdout)")
    args = parser.parse_args(argv)
    try:
        human_raw = read_human_label_file(args.human) if args.human else None
        if human_raw is not None and args.id_map:
            id_map = json.loads(args.id_map.read_text(encoding="utf-8"))
            human_raw = apply_id_map(human_raw, id_map, args.id_template)
        criteria = sorted({name for labels in (human_raw or {}).values() for name in labels})
        mapping = CalibrationMapping.load(args.mapping, human_criteria=criteria)
        reviews, problems = load_vlm_reviews(
            args.vlm, apply_rechecks=not args.no_rechecks, vocabulary=mapping.vlm_vocabulary
        )
        report = {
            "schema": REPORT_SCHEMA,
            "inputs": {
                "vlm": [str(path) for path in args.vlm],
                "human": str(args.human) if args.human else None,
                "mapping": str(args.mapping) if args.mapping else None,
                "standing_rules": str(args.standing_rules) if args.standing_rules else None,
                "focused_rechecks_applied": not args.no_rechecks,
            },
            "vlm": vlm_summary(reviews, mapping.vlm_vocabulary),
            "calibration": None
            if human_raw is None
            else calibrate(reviews, human_raw, mapping),
            "standing_rules": None,
            "problems": problems,
        }
        if args.standing_rules:
            rules = load_standing_rules(args.standing_rules)
            extra = [
                (str(path), path.read_text(encoding="utf-8", errors="replace"))
                for path in args.prompt_file
            ]
            report["standing_rules"] = audit_standing_rules(
                reviews,
                rules,
                mapping.vlm_vocabulary,
                extra_prompts=extra,
                max_details=args.max_conflict_details,
            )
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if args.out == "-":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        if args.out:
            Path(args.out).write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        print_summary(report)
    calibration = report["calibration"]
    if calibration is not None and calibration["status"] == "no_paired_data":
        if not args.allow_empty:
            print("no (human, VLM) pairs: nothing was calibrated", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
