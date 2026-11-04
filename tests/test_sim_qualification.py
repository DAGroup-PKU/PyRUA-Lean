from __future__ import annotations

import copy
import unittest

from datafarm.simulation.sim_qualification import (
    SimulatorVariant,
    VariantResult,
    apply_patch,
    build_certificate,
    evaluate_fold_probe,
    evaluate_hold_probe,
    canonical_json_sha256,
    verify_run_binding,
    select_variant,
    verify_certificate,
)

HELD = {"hang_z_max_m": 0.19, "held_dz_m": {"left": -0.004, "right": -0.006}, "hang_strain_p95": 1.10,
        "final_area_frac": 0.90, "final_height_mm": 12.0}
DROPPED = {"hang_z_max_m": 0.02, "held_dz_m": {"left": -0.184, "right": -0.184}, "hang_strain_p95": 1.02,
           "final_area_frac": 1.05, "final_height_mm": 5.0}
FOLDED = {"m1_at_release": True, "bottom_layer_capture_frac": 0.7, "robot_failures": []}
UNFOLDED = {"m1_at_release": False, "bottom_layer_capture_frac": 0.3, "robot_failures": []}


def result(variant, hang=HELD, fold=FOLDED):
    return VariantResult(variant, (evaluate_hold_probe("hang", hang), evaluate_fold_probe("fold", fold)))


class PatchTests(unittest.TestCase):
    def test_sets_and_multiplies_nested_values_without_mutating_the_base(self) -> None:
        base = {"pbd": {"friction": 0.5, "iterations": 32}, "name": "v21"}
        out = apply_patch(base, {"pbd.friction": 1.0, "pbd.iterations": "*2"})
        self.assertEqual(out["pbd"], {"friction": 1.0, "iterations": 64.0})
        self.assertEqual(base["pbd"]["friction"], 0.5)

    def test_unknown_path_is_rejected_instead_of_silently_added(self) -> None:
        with self.assertRaises(ValueError):
            apply_patch({"pbd": {"friction": 0.5}}, {"pbd.fricton": 1.0})

    def test_multiplier_needs_a_number(self) -> None:
        with self.assertRaises(ValueError):
            apply_patch({"mode": "max"}, {"mode": "*2"})


class ProbeCriteriaTests(unittest.TestCase):
    def test_a_held_hanging_cloth_passes(self) -> None:
        outcome = evaluate_hold_probe("hang", HELD)
        self.assertTrue(outcome.passed, outcome.failures)

    def test_cloth_sliding_out_of_closed_jaws_fails(self) -> None:
        outcome = evaluate_hold_probe("hang", DROPPED)
        self.assertFalse(outcome.passed)
        self.assertIn("grip_slipped", outcome.failures)
        self.assertIn("cloth_not_lifted", outcome.failures)

    def test_a_probe_that_never_reached_the_hang_fails_closed(self) -> None:
        outcome = evaluate_hold_probe("hang", {"final_area_frac": 1.0, "final_height_mm": 3.0})
        self.assertIn("hang_not_reached", outcome.failures)
        self.assertIn("grip_slip_unmeasured", outcome.failures)

    def test_an_overstretched_hang_fails(self) -> None:
        outcome = evaluate_hold_probe("hang", {**HELD, "hang_strain_p95": 1.40})
        self.assertEqual(outcome.failures, ("cloth_overstretched",))


    def test_hold_local_strain_limit_is_optional_and_required_when_set(self):
        self.assertTrue(evaluate_hold_probe("hang", {**HELD, "hang_strain_p999": 4.0}).passed)
        outcome = evaluate_hold_probe("hang", {**HELD, "hang_strain_p999": 4.0}, max_strain_p999=1.5)
        self.assertFalse(outcome.passed)
        self.assertIn("cloth_locally_overstretched", outcome.failures)
        self.assertTrue(evaluate_hold_probe("hang", {**HELD, "hang_strain_p999": 1.3}, max_strain_p999=1.5).passed)

    def test_hold_local_strain_unmeasured_fails_when_limit_set(self):
        outcome = evaluate_hold_probe("hang", HELD, max_strain_p999=1.5)
        self.assertIn("local_strain_unmeasured", outcome.failures)
    def test_flatness_after_the_scripted_lay_down_is_advisory(self) -> None:
        outcome = evaluate_hold_probe("hang", {**HELD, "final_area_frac": 0.7})
        self.assertTrue(outcome.passed)
        self.assertIn("laid_down_not_flat", outcome.advisories)

    def test_fold_probe_requires_both_folds_at_release(self) -> None:
        self.assertTrue(evaluate_fold_probe("fold", FOLDED).passed)
        failed = evaluate_fold_probe("fold", UNFOLDED)
        self.assertFalse(failed.passed)
        self.assertIn("bottom_layer_rarely_captured", failed.advisories)


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = SimulatorVariant("baseline")
        self.friction = SimulatorVariant("friction", material_patch={"pbd.friction": 2.0})
        self.lock = SimulatorVariant("lock", runtime_flags=("pinch_lock",))
        self.lock_adh = SimulatorVariant("lock_adh0", material_patch={"pbd.adhesion": 0.0}, runtime_flags=("pinch_lock",))

    def test_smallest_change_that_passes_every_required_probe_wins(self) -> None:
        results = [result(self.baseline, DROPPED), result(self.friction, DROPPED),
                   result(self.lock_adh), result(self.lock)]
        decision = select_variant(results, ["hang", "fold"])
        self.assertEqual(decision.selected, "lock")

    def test_no_variant_is_selected_when_none_passes(self) -> None:
        decision = select_variant([result(self.baseline, DROPPED), result(self.friction, DROPPED)], ["hang"])
        self.assertIsNone(decision.selected)

    def test_score_breaks_ties_between_equally_small_changes(self) -> None:
        a = SimulatorVariant("a", runtime_flags=("x",))
        b = SimulatorVariant("b", runtime_flags=("y",))
        results = [result(a, {**HELD, "final_area_frac": 0.5}), result(b)]
        self.assertEqual(select_variant(results, ["hang"]).selected, "b")

    def test_required_probes_must_be_declared(self) -> None:
        with self.assertRaises(ValueError):
            select_variant([result(self.lock)], [])


class CertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = [result(SimulatorVariant("baseline"), DROPPED),
                        result(SimulatorVariant("lock", runtime_flags=("pinch_lock",)))]
        self.decision = select_variant(self.results, ["hang", "fold"])
        self.cert = build_certificate(task_family="fold-cloth", decision=self.decision, results=self.results,
                                      required_probes=["hang", "fold"],
                                      file_digests={"lock": {"material.json": "ab" * 32}}, criteria={"hold": {}})

    def test_certificate_verifies_against_its_own_evidence(self) -> None:
        self.assertEqual(verify_certificate(self.cert, self.results), ())
        self.assertEqual(self.cert["selected_variant"]["runtime_flags"], ["pinch_lock"])

    def test_tampered_selection_or_evidence_is_detected(self) -> None:
        forged = copy.deepcopy(self.cert)
        forged["selected_variant"]["name"] = "baseline"
        forged["evidence"]["baseline"]["hang"]["passed"] = True
        failures = verify_certificate(forged, self.results)
        self.assertIn("selection_mismatch", failures)
        self.assertIn("evidence_mismatch:baseline:hang", failures)

    def test_certificate_needs_bound_files_and_a_selection(self) -> None:
        with self.assertRaises(ValueError):
            build_certificate(task_family="t", decision=self.decision, results=self.results,
                              required_probes=["hang"], file_digests={}, criteria={})
        none = select_variant([self.results[0]], ["hang"])
        with self.assertRaises(ValueError):
            build_certificate(task_family="t", decision=none, results=self.results[:1],
                              required_probes=["hang"], file_digests={}, criteria={})


if __name__ == "__main__":
    unittest.main()


class RunBindingTest(unittest.TestCase):
    MATERIAL = {"b": 2, "a": {"y": 1.5, "x": [1, 2]}}
    SCENE = {"pads": [0.015, 0.022]}

    def certificate(self):
        return {"selected_variant": {
            "files_sha256": {"material.json": "raw-m", "scene.json": "raw-s", "grasp_model.py": "gm"},
            "files_canonical_sha256": {"material.json": canonical_json_sha256(self.MATERIAL),
                                       "scene.json": canonical_json_sha256(self.SCENE)}}}

    def used(self, **override):
        used = {"material.json": canonical_json_sha256({"a": {"x": [1, 2], "y": 1.5}, "b": 2}),
                "scene.json": canonical_json_sha256(self.SCENE), "grasp_model.py": "gm"}
        used.update(override)
        return used

    def test_canonical_digest_ignores_key_order_and_whitespace(self):
        self.assertEqual(canonical_json_sha256(self.MATERIAL), canonical_json_sha256({"a": {"x": [1, 2], "y": 1.5}, "b": 2}))
        self.assertNotEqual(canonical_json_sha256(self.MATERIAL), canonical_json_sha256({"a": {"x": [2, 1], "y": 1.5}, "b": 2}))

    def test_binding_passes_for_the_certified_files(self):
        self.assertEqual(verify_run_binding(self.certificate(), self.used()), ())

    def test_binding_detects_changed_missing_and_extra_files(self):
        changed = verify_run_binding(self.certificate(), self.used(**{"grasp_model.py": "other"}))
        self.assertEqual(changed, ("mismatch:grasp_model.py",))
        used = self.used(); used.pop("scene.json"); used["extra.py"] = "x"
        self.assertEqual(set(verify_run_binding(self.certificate(), used)), {"not_reported:scene.json", "not_certified:extra.py"})

    def test_binding_needs_canonical_digests_for_json_and_bound_files(self):
        cert = self.certificate(); cert["selected_variant"]["files_canonical_sha256"] = {}
        self.assertIn("canonical_digest_missing:material.json", verify_run_binding(cert, self.used()))
        self.assertEqual(verify_run_binding({"selected_variant": {}}, self.used()), ("files_unbound",))


class FoldRestProbeTest(unittest.TestCase):
    RESTING = {"m1_at_release": True, "settled_passed": True, "creep_mm": 6.0, "rest_window_s": 15.0,
               "bottom_layer_capture_frac": 0.7, "robot_failures": []}

    def test_rest_requirements_are_optional(self):
        self.assertTrue(evaluate_fold_probe("fold", {**self.RESTING, "settled_passed": False, "creep_mm": 80.0}).passed)

    def test_settled_and_creep_required_when_set(self):
        judge = lambda m: evaluate_fold_probe("fold", m, require_settled=True, max_creep_m=0.015)
        self.assertTrue(judge(self.RESTING).passed)
        self.assertEqual(judge({**self.RESTING, "settled_passed": False}).failures, ("folds_not_in_place_when_settled",))
        self.assertEqual(judge({**self.RESTING, "creep_mm": 31.0}).failures, ("fold_creeps",))
        self.assertEqual(judge({**self.RESTING, "rest_window_s": 3.0}).failures, ("rest_window_short",))

    def test_unmeasured_rest_metrics_fail_when_required(self):
        m = {k: v for k, v in self.RESTING.items() if k not in ("settled_passed", "creep_mm")}
        failures = evaluate_fold_probe("fold", m, require_settled=True, max_creep_m=0.015).failures
        self.assertEqual(set(failures), {"settled_unmeasured", "creep_unmeasured"})

