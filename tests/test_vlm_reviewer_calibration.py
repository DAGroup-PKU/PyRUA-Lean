from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from datafarm.verify.reviewer_calibration import (
    ABSTAIN,
    FAIL,
    PASS,
    CalibrationMapping,
    Vocabulary,
    DEFAULT_VLM_VOCABULARY,
    agreement_metrics,
    apply_id_map,
    audit_standing_rules,
    calibrate,
    load_standing_rules,
    load_vlm_reviews,
    read_human_label_file,
)
from datafarm.verify.stats import cohen_kappa, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "configs/standing-rules.json"
PROMPT = "Look for long hard shadows, beams, and require diffuse ambience. Return JSON only."


def _tool():
    spec = importlib.util.spec_from_file_location(
        "calibrate_vlm_reviewer_tool", ROOT / "tools/calibrate_vlm_reviewer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_assessment(run: Path, sample: str, criteria: dict, prompt: str | None = None) -> Path:
    folder = run / "episodes" / sample / "visual-review"
    folder.mkdir(parents=True)
    statuses = [status for status, _ in criteria.values()]
    outcome = "flagged" if "fail" in statuses else (
        "uncertain" if "uncertain" in statuses else "sampled_pass")
    data = {
        "id": sample,
        "model": "codex-test-model",
        "outcome": outcome,
        "assessment": {
            "criteria": {k: {"status": s, "evidence": e} for k, (s, e) in criteria.items()},
            "issues": [],
            "summary_zh": "test",
        },
        "human_reviewed": False,
    }
    (folder / "assessment.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    if prompt:
        (folder / "request-metadata.json").write_text(json.dumps({"system": prompt}))
    return folder


def write_recheck(folder: Path, criteria: dict, *, bound: bool = True) -> None:
    out = folder / "focused-recheck"
    out.mkdir()
    digest = hashlib.sha256((folder / "assessment.json").read_bytes()).hexdigest()
    data = {
        "id": folder.parent.name,
        "criteria": {k: {"status": s, "evidence": e} for k, (s, e) in criteria.items()},
        "original_assessment_sha256": digest if bound else "0" * 64,
    }
    (out / "assessment.json").write_text(json.dumps(data, ensure_ascii=False))


def build_run(run: Path) -> None:
    ok = ("pass", "no issues seen")
    write_assessment(run, "s1", {
        "drum_geometry": ok,
        "entity_identity_count": ok,
        "hard_shadows_beams": ("fail", "The generated frames show long, crisp hard shadows; the hard shadows themselves already constitute a defect."),
        "ambient_adoption": ("fail", "Does not match the requested neutral-white, soft, low-contrast diffuse ambient light."),
        "temporal_appearance": ok,
    }, prompt=PROMPT)
    write_assessment(run, "s2", {
        "drum_geometry": ("fail", "In head-view frame 0 the drum is enlarged by about 1.9x."),
        "entity_identity_count": ok,
        "hard_shadows_beams": ok,
        "ambient_adoption": ok,
        "temporal_appearance": ("fail", "Hard shadows flicker over time between frames 160 and 209."),
    })
    s3 = write_assessment(run, "s3", {
        "drum_geometry": ("fail", "suspected drum enlargement"),
        "entity_identity_count": ("uncertain", "insufficient resolution"),
        "hard_shadows_beams": ok,
        "ambient_adoption": ok,
        "temporal_appearance": ok,
    })
    write_recheck(s3, {"drum_geometry": ("pass", "independent frame recheck saw no enlargement")})
    s4 = write_assessment(run, "s4", {"drum_geometry": ok, "entity_identity_count": ok})
    write_recheck(s4, {"drum_geometry": ("fail", "unbound recheck")}, bound=False)


HUMAN = {
    "s1": {"geometry": "pass", "overall": "acceptable"},
    "s2": {"geometry": "fail", "overall": "reject"},
    "s3": {"geometry": "pass", "overall": "pass"},
    "s4": {"geometry": "fail", "overall": "unsure"},
    "s5": {"geometry": "pass"},
}
MAPPING = {
    "schema": "datafarm.vlm-calibration-mapping/1",
    "criteria": {"geometry": {"vlm": ["drum_geometry"]}, "overall": {"vlm": ["@outcome"]}},
}


class MetricTests(unittest.TestCase):
    def test_kappa_matches_hand_computed_example(self) -> None:
        # 20 agree-pass, 5 human-pass/VLM-fail, 10 human-fail/VLM-pass, 15 agree-fail:
        # po = 0.7, pe = (25*30 + 25*20) / 50^2 = 0.5, kappa = 0.4.
        pairs = [(PASS, PASS)] * 20 + [(PASS, FAIL)] * 5 + [(FAIL, PASS)] * 10 + [(FAIL, FAIL)] * 15
        metrics = agreement_metrics(pairs)
        self.assertAlmostEqual(metrics["cohen_kappa"], 0.4)
        self.assertAlmostEqual(metrics["accuracy"], 0.7)
        self.assertAlmostEqual(metrics["false_accept_rate"], 0.4)
        self.assertAlmostEqual(metrics["false_reject_rate"], 0.2)
        self.assertEqual(metrics["confusion"]["human_fail"],
                         {"vlm_pass": 10, "vlm_fail": 15, "vlm_abstain": 0})

    def test_kappa_undefined_and_abstentions(self) -> None:
        self.assertIsNone(cohen_kappa([(PASS, PASS)] * 4))
        metrics = agreement_metrics([(PASS, PASS), (FAIL, ABSTAIN), (PASS, ABSTAIN)])
        self.assertEqual(metrics["decided"], 1)
        self.assertAlmostEqual(metrics["abstention_rate"], 2 / 3)
        self.assertEqual(metrics["false_accept_rate"], 0.0)
        self.assertIsNone(metrics["false_accept_rate_decided"])
        self.assertAlmostEqual(metrics["abstain_as_reject"]["false_reject_rate"], 0.5)
        with self.assertRaises(ValueError):
            agreement_metrics([(ABSTAIN, PASS)])

    def test_wilson_interval_known_value(self) -> None:
        low, high = wilson_interval(8, 10)
        self.assertAlmostEqual(low, 0.4902, places=4)
        self.assertAlmostEqual(high, 0.9433, places=4)
        self.assertIsNone(wilson_interval(0, 0))


class CalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.run = self.tmp / "run"
        build_run(self.run)
        self.mapping_path = self.tmp / "mapping.json"
        self.mapping_path.write_text(json.dumps(MAPPING))
        self.human_path = self.tmp / "human.json"
        self.human_path.write_text(json.dumps({"labels": HUMAN}))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_loading_applies_only_hash_bound_rechecks(self) -> None:
        reviews, problems = load_vlm_reviews([self.run])
        self.assertEqual(sorted(reviews), ["s1", "s2", "s3", "s4"])
        self.assertEqual(reviews["s3"].statuses["drum_geometry"], "pass")
        self.assertEqual(reviews["s3"].stage, "initial+focused_recheck")
        self.assertEqual(reviews["s3"].outcome, "uncertain")
        self.assertEqual(reviews["s4"].statuses["drum_geometry"], "pass")
        self.assertEqual([p["kind"] for p in problems], ["recheck_binding_mismatch"])
        first_pass, _ = load_vlm_reviews([self.run], apply_rechecks=False)
        self.assertEqual(first_pass["s3"].statuses["drum_geometry"], "fail")

    def test_calibration_metrics_per_criterion(self) -> None:
        reviews, _ = load_vlm_reviews([self.run])
        mapping = CalibrationMapping.load(self.mapping_path)
        report = calibrate(reviews, read_human_label_file(self.human_path), mapping)
        geometry = report["per_criterion"]["geometry"]
        self.assertEqual((geometry["n"], geometry["decided"]), (4, 4))
        self.assertAlmostEqual(geometry["accuracy"], 0.75)
        self.assertAlmostEqual(geometry["cohen_kappa"], 0.5)
        self.assertAlmostEqual(geometry["false_accept_rate"], 0.5)  # s4: VLM pass, human fail
        self.assertEqual(geometry["false_reject_rate"], 0.0)
        self.assertEqual(geometry["unpaired_missing_vlm"], 1)  # s5 has no review
        overall = report["per_criterion"]["overall"]
        self.assertEqual((overall["n"], overall["human_abstentions"]), (3, 1))
        self.assertAlmostEqual(overall["false_reject_rate"], 0.5)  # s1 flagged, human ok
        self.assertAlmostEqual(overall["cohen_kappa"], 0.0)
        self.assertAlmostEqual(overall["abstain_as_reject"]["false_reject_rate"], 1.0)
        self.assertEqual(report["pooled"]["n"], 7)
        self.assertEqual(report["human_samples_without_review"], ["s5"])
        self.assertEqual(report["status"], "calibrated")

    def test_label_formats_and_id_map(self) -> None:
        long_csv = self.tmp / "long.csv"
        long_csv.write_text("sample_id,criterion,label,note\ns1,geometry,pass,\n"
                            "s2,geometry,fail,drum enlarged\n")
        self.assertEqual(read_human_label_file(long_csv),
                         {"s1": {"geometry": "pass"}, "s2": {"geometry": "fail"}})
        wide_csv = self.tmp / "wide.csv"
        wide_csv.write_text("sample_id,geometry,overall\ns1,pass,acceptable\n")
        self.assertEqual(read_human_label_file(wide_csv),
                         {"s1": {"geometry": "pass", "overall": "acceptable"}})
        blind_key = [{"label_id": "L01", "source": "neutral", "sample": "043"}]
        renamed = apply_id_map({"L01": {"appearance": "looks real"}}, blind_key, "sample-{sample}")
        self.assertEqual(renamed, {"sample-043": {"appearance": "looks real"}})
        mapping = CalibrationMapping.load(None, human_criteria=["geometry"])
        with self.assertRaises(ValueError):
            calibrate({}, {"s1": {"geometry": "maybe?"}}, mapping)

    def test_vocabulary_rejects_conflicts(self) -> None:
        with self.assertRaises(ValueError):
            Vocabulary.from_mapping({PASS: ["ok"], FAIL: ["OK"]})
        vocab = Vocabulary.from_mapping(DEFAULT_VLM_VOCABULARY)
        self.assertEqual(vocab.classify(" Flagged "), FAIL)
        self.assertIsNone(vocab.classify(""))

    def test_standing_rule_audit(self) -> None:
        reviews, _ = load_vlm_reviews([self.run])
        rules = load_standing_rules(RULES)
        audit = audit_standing_rules(
            reviews, rules, Vocabulary.from_mapping(DEFAULT_VLM_VOCABULARY)
        )
        lighting = audit["rules"]["SR-001-render-lighting"]
        self.assertEqual(lighting["fail_verdicts_flagged"], 2)
        self.assertEqual(lighting["criterion_level"], 1)
        self.assertEqual(lighting["evidence_candidates_only"], 1)
        self.assertEqual(lighting["by_criterion"], {"ambient_adoption": 1, "hard_shadows_beams": 1})
        flagged = {(c["sample_id"], c["criterion"]) for c in audit["conflicts"]}
        self.assertNotIn(("s2", "temporal_appearance"), flagged)  # exempt: change over time
        self.assertNotIn(("s2", "drum_geometry"), flagged)
        exposure = audit["rules"]["SR-002-exposure-gross-guard"]
        self.assertEqual(exposure["fail_verdicts_flagged"], 0)
        [prompt] = audit["prompt_conflicts"]
        self.assertEqual(prompt["matched"], ["diffuse ambience", "hard shadow"])
        self.assertEqual(audit["outcomes_if_criterion_conflicts_ignored"],
                         {"flagged": 2, "sampled_pass": 1, "uncertain": 1})
        self.assertEqual(audit["outcomes_if_all_flagged_ignored"],
                         {"flagged": 1, "sampled_pass": 2, "uncertain": 1})

    def test_rules_file_validation(self) -> None:
        bad = self.tmp / "rules.json"
        bad.write_text(json.dumps({"schema": "datafarm.standing-rules/1", "rules": [
            {"id": "r", "forbidden_criteria": ["x"], "exempt_criteria": ["x"]}]}))
        with self.assertRaises(ValueError):
            load_standing_rules(bad)

    def test_cli_modes(self) -> None:
        tool = _tool()
        out = self.tmp / "report.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = tool.main(["--vlm", str(self.run), "--standing-rules", str(RULES),
                              "--out", str(out)])
        self.assertEqual(code, 0)
        report = json.loads(out.read_text())
        self.assertIsNone(report["calibration"])
        self.assertEqual(report["standing_rules"]["fail_verdicts_flagged"], 2)
        unpaired = self.tmp / "unpaired.json"
        unpaired.write_text(json.dumps({"zz": {"geometry": "pass"}}))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(tool.main(["--vlm", str(self.run), "--human", str(unpaired)]), 3)
            self.assertEqual(tool.main(["--vlm", str(self.run), "--human", str(unpaired),
                                        "--allow-empty"]), 0)
            code = tool.main(["--vlm", str(self.run), "--human", str(self.human_path),
                              "--mapping", str(self.mapping_path), "--out", str(out)])
        self.assertEqual(code, 0)
        calibration = json.loads(out.read_text())["calibration"]
        self.assertAlmostEqual(calibration["per_criterion"]["geometry"]["cohen_kappa"], 0.5)


if __name__ == "__main__":
    unittest.main()
