import unittest
from dataclasses import replace

from datafarm.training import (
    AdmissionBinding,
    Pi05TrainingContract,
    audit_pi05_training_contract,
)


# A bare 64-hex digest does not bind admission. The fixture carries the
# result of a verified certificate (see tests/test_training_admission.py for
# the real issue/verify path).
VERIFIED_ADMISSION = AdmissionBinding(
    mode="certificate-verified",
    certificate_sha256="a" * 64,
    receipt_sha256="c" * 64,
    producer_id="datafarm-training-certificate-compiler-v1",
    verifier_id="datafarm-training-admission-root-v1",
    authority="admission-root",
    training_bundle_sha256="b" * 64,
    allowlist_sha256="9" * 64,
)


def valid_contract(**updates):
    values = {
        "admission_certificate_sha256": "a" * 64,
        "admission": VERIFIED_ADMISSION,
        "training_bundle_sha256": "b" * 64,
        "model_family": "pi05",
        "state_conditioning": "none",
        "discrete_state_input": False,
        "native_action_dim": 14,
        "model_action_dim": 32,
        "action_horizon": 50,
        "camera_keys": ("cam_high", "cam_left_wrist", "cam_right_wrist"),
        "source_fps": 30.0,
        "launch_argv": ("train", "--model.no-discrete-state-input"),
    }
    values.update(updates)
    return Pi05TrainingContract(**values)


class Pi05TrainingContractTests(unittest.TestCase):
    def test_explicit_state_free_contract_passes(self):
        self.assertTrue(audit_pi05_training_contract(valid_contract()).passed)

    def test_unbound_or_non_admitted_dataset_is_rejected(self):
        audit = audit_pi05_training_contract(
            valid_contract(admission_certificate_sha256="not-a-digest")
        )
        self.assertFalse(audit.passed)
        self.assertIn("admission_bound", audit.failures)

    def test_composite_carrier_cannot_replace_three_camera_inputs(self):
        audit = audit_pi05_training_contract(valid_contract(camera_keys=("t_shape",)))
        self.assertFalse(audit.passed)
        self.assertIn("three_independent_views", audit.failures)

    def test_state_mode_and_flag_contradiction_is_rejected(self):
        audit = audit_pi05_training_contract(
            valid_contract(
                discrete_state_input=True,
                launch_argv=("train", "--model.discrete-state-input"),
            )
        )
        self.assertFalse(audit.passed)
        self.assertIn("state_flag_matches_mode", audit.failures)
        self.assertIn("cli_state_flag_matches_mode", audit.failures)

    def test_unknown_or_state_on_mode_is_rejected(self):
        for mode in ("implicit-default", "eef_discrete"):
            audit = audit_pi05_training_contract(valid_contract(state_conditioning=mode))
            self.assertFalse(audit.passed)
            self.assertIn("state_mode_declared", audit.failures)


class AdmissionBindingContractTests(unittest.TestCase):
    """Admission is bound only by a verified certificate."""

    def test_bare_digest_without_verified_certificate_is_not_bound(self):
        # A bare semantic allowlist digest would be passed here.
        audit = audit_pi05_training_contract(valid_contract(admission=None))
        self.assertFalse(audit.passed)
        self.assertEqual(audit.statuses["admission_bound"], "fail")
        self.assertIn("admission_bound", audit.failures)

    def test_binding_must_match_the_certificate_and_bundle(self):
        for binding, label in (
            (replace(VERIFIED_ADMISSION, certificate_sha256="0" * 64), "different certificate"),
            (replace(VERIFIED_ADMISSION, training_bundle_sha256="0" * 64), "different bundle"),
            (replace(VERIFIED_ADMISSION, verifier_id=VERIFIED_ADMISSION.producer_id), "self-signed"),
            (replace(VERIFIED_ADMISSION, authority="render-contract-verifier"), "wrong authority"),
            (replace(VERIFIED_ADMISSION, receipt_sha256=None), "no receipt"),
        ):
            with self.subTest(label):
                audit = audit_pi05_training_contract(valid_contract(admission=binding))
                self.assertIn("admission_bound", audit.failures)


if __name__ == "__main__":
    unittest.main()
