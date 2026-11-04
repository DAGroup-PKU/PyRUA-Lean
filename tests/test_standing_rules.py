from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lint_standing_rules", REPO / "tools" / "lint_standing_rules.py")
lint_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lint_module)


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


RULES = {
    "schema": "datafarm.standing-rules/1",
    "rules": [
        {
            "id": "SR-T1",
            "since": "2026-09-22",
            "source": "test",
            "statement": "no soft-shadow requirements",
            "globs": ["configs/*.txt", "tools/*.py"],
            "forbidden": [{"regex": "shadows?\\s+should\\s+be\\s+soft", "why": "requires soft shadows"}],
            "waivers": [{"path": "tools/waived.py", "reason": "descriptive"}],
            "known_violations": [
                {"path": "tools/known.py", "owner": "x", "reason": "in progress"},
                {"path": "tools/fixed.py", "owner": "x", "reason": "already fixed"},
            ],
        },
        {
            "id": "SR-T2",
            "since": "2026-09-22",
            "source": "test",
            "statement": "exposure gross guard",
            "globs": ["configs/specs/*.json", "src/spec.py"],
            "numeric_min": [{"key": "background_mean_drift_max", "min": 60.0, "why": "gross guard"}],
        },
        {
            "id": "SR-T3",
            "since": "2026-09-21",
            "source": "test",
            "statement": "explicit state mode",
            "globs": ["configs/pi05-*.json"],
            "required_json": [{"key": "state_conditioning", "allowed": ["none", "eef_discrete"]}],
        },
    ],
}


class StandingRulesLintTest(unittest.TestCase):
    def run_lint(self, files: dict[str, str]) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, "configs/standing-rules.json", json.dumps(RULES))
            for rel, text in files.items():
                write(root, rel, text)
            return lint_module.lint(root, root / "configs" / "standing-rules.json")

    def test_forbidden_phrase_statuses_and_stale_known_entries(self) -> None:
        report = self.run_lint(
            {
                "configs/prompt.txt": "Generated shadows should be softened.\n",
                "tools/known.py": 'PROMPT = "shadow should be soft"\n',
                "tools/waived.py": '"""shadows should be soft in this old note"""\n',
                "tools/clean.py": 'PROMPT = "hard shadows are fine"\n',
            }
        )
        statuses = {(item["path"], item["status"]) for item in report["findings"]}
        self.assertIn(("configs/prompt.txt", "new"), statuses)
        self.assertIn(("tools/known.py", "known"), statuses)
        self.assertIn(("tools/waived.py", "waived"), statuses)
        self.assertNotIn("tools/clean.py", {item["path"] for item in report["findings"]})
        self.assertEqual(report["counts"]["new"], 1)
        self.assertEqual(report["stale_known_violations"], [{"rule": "SR-T1", "path": "tools/fixed.py"}])
        self.assertEqual(report["findings"][0]["line"], 1)

    def test_numeric_minimum_in_json_specs_and_python_defaults(self) -> None:
        report = self.run_lint(
            {
                "configs/specs/a.json": json.dumps({"thresholds": {"background_mean_drift_max": {"head": 40, "wrist": 70}}}),
                "configs/specs/b.json": json.dumps({"thresholds": {"background_mean_drift_max": 60}}),
                "src/spec.py": 'DEFAULTS = {\n    "background_mean_drift_max": 25.0,\n}\n',
            }
        )
        bad = sorted((item["path"], item["line"]) for item in report["findings"])
        self.assertEqual(bad, [("configs/specs/a.json", 0), ("src/spec.py", 2)])

    def test_required_json_key_and_allowed_values(self) -> None:
        report = self.run_lint(
            {
                "configs/pi05-ok.json": json.dumps({"training": {"state_conditioning": "none"}}),
                "configs/pi05-missing.json": json.dumps({"training": {}}),
                "configs/pi05-bad.json": json.dumps({"state_conditioning": "eef"}),
            }
        )
        flagged = sorted(item["path"] for item in report["findings"])
        self.assertEqual(flagged, ["configs/pi05-bad.json", "configs/pi05-missing.json"])

    def test_rules_file_must_have_unique_ids_and_valid_regex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.json"
            broken = json.loads(json.dumps(RULES))
            broken["rules"][1]["id"] = "SR-T1"
            path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(ValueError):
                lint_module.load_rules(path)

    def test_repository_has_no_new_violations(self) -> None:
        """Known violations may remain; new ones fail the suite."""

        report = lint_module.lint(REPO, REPO / "configs" / "standing-rules.json")
        new = [item for item in report["findings"] if item["status"] == "new"]
        self.assertEqual(new, [], json.dumps(new, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    unittest.main()
