from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from datafarm.training.cohort_auditors import (
    DENSE_ALLOWLIST_AUDITOR,
    PI05_COHORT_AUDITORS,
    build_claim,
)
from datafarm.core.admission import (
    ADMISSION_ROOT_ID,
    CERTIFICATE_COMPILER_ID,
    AdmissionError,
    AdmissionExpectation,
    TrainingAdmissionRoot,
    compile_certificate,
    issue_certificate,
    verify_certificate,
)
from datafarm.core.model import ArtifactRef, Verdict
from datafarm.core.store import ArtifactStore

from tests.admission_fixtures import sha, write_dense_cohort

TOOLS = Path(__file__).resolve().parents[1] / "tools"
NOW = "2026-09-26T12:00:00Z"


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"r1_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def issue(store_root: Path, dataset: Path, *extra: str) -> tuple[int, dict]:
    tool = load_tool("issue_admission_certificate")
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = tool.main(["--store", str(store_root), "--dataset-root", str(dataset),
                          "--created-at", NOW, *extra])
    text = out.getvalue() or err.getvalue()
    return code, json.loads(text)


class TrainingAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def expectation(self, cohort) -> AdmissionExpectation:
        provenance_path = cohort["dataset"] / "meta" / "lerobot-provenance.json"
        provenance = json.loads(provenance_path.read_text())
        return AdmissionExpectation(
            provenance_sha256=sha(provenance_path.read_bytes()),
            allowlist_sha256=provenance["semantic_allowlist_sha256"],
            repo_id=provenance["repo_id"],
            allowed_failed_checks=tuple(provenance["allowed_semantic_waivers"]),
            member_ids=tuple(item["sample_id"] for item in provenance["episodes_detail"]),
        )

    def test_issued_zero_waiver_certificate_verifies_and_binds_the_bundle(self) -> None:
        cohort = write_dense_cohort(self.root / "run")
        code, report = issue(self.root / "store", cohort["dataset"])
        self.assertEqual(code, 0, report)
        self.assertEqual(report["status"], "admitted")
        self.assertEqual(report["waiver_policy"], "zero-waiver")
        store = ArtifactStore(self.root / "store")
        verification = verify_certificate(store, report["certificate_sha256"], auditors=PI05_COHORT_AUDITORS,
                                          expectation=self.expectation(cohort))
        self.assertEqual(verification.member_count, 3)
        self.assertEqual(verification.verifier_id, ADMISSION_ROOT_ID)
        self.assertEqual(verification.producer_id, CERTIFICATE_COMPILER_ID)
        self.assertNotEqual(verification.verifier_id, verification.producer_id)
        self.assertEqual(verification.authority, "admission-root")
        self.assertEqual(verification.waiver_policy, "zero-waiver")
        self.assertIsNotNone(verification.data_files)
        document = store.document(ArtifactRef("admission-certificate", report["certificate_sha256"]))
        payload = document.payload
        self.assertEqual(payload["verdict"], "admit")
        self.assertEqual([m["sample_id"] for m in payload["admitted_set"]["members"]],
                         ["fixture-sample-000", "fixture-sample-001", "fixture-sample-002"])
        self.assertEqual({item["role"] for item in payload["reports"]},
                         {"dense-semantic-audit", "candidate-manifest", "distribution-acceptance"})
        for key in ("episode", "rendered_observation", "checks", "verdict", "cost", "training_bundle",
                    "repair_predecessor"):
            self.assertIn(key, payload)

    def test_dense_audit_binds_member_checks_and_rejects_unknown_versions(self) -> None:
        for case in ("valid", "missing_member", "failed_check", "reject_verdict", "unknown_schema"):
            with self.subTest(case=case):
                cohort = write_dense_cohort(self.root / case)
                audit_path = cohort["allowlist"].parent / "semantic-audit.json"
                audit = json.loads(audit_path.read_text())
                if case == "missing_member":
                    audit["records"].pop()
                elif case == "failed_check":
                    audit["records"][0]["checks"]["strict_all_seams"] = False
                    audit["records"][0]["failures"] = ["strict_all_seams"]
                elif case == "reject_verdict":
                    audit["records"][0]["verdict"] = "reject"
                elif case == "unknown_schema":
                    audit["schema_version"] = "datafarm_dense_semantic_audit/99.0.0"
                audit_path.write_text(json.dumps(audit))
                code, report = issue(self.root / f"store-{case}", cohort["dataset"])
                self.assertEqual(code, 0 if case == "valid" else 2, report)
                if case == "valid":
                    verification = verify_certificate(
                        ArtifactStore(self.root / f"store-{case}"), report["certificate_sha256"],
                        auditors=PI05_COHORT_AUDITORS, expectation=self.expectation(cohort))
                    self.assertEqual(verification.member_count, 3)
                    self.assertEqual(verification.waiver_policy, "zero-waiver")
                else:
                    check = "dense_audit_supported" if case == "unknown_schema" else "dense_audit_admits_members"
                    self.assertIn(f"evidence:{check}", report["failed_checks"])

    def test_allowlist_hash_posing_as_certificate_is_rejected(self) -> None:
        cohort = write_dense_cohort(self.root / "run")
        code, _ = issue(self.root / "store", cohort["dataset"])
        self.assertEqual(code, 0)
        store = ArtifactStore(self.root / "store")
        allowlist_sha = sha(cohort["allowlist"].read_bytes())
        with self.assertRaisesRegex(AdmissionError, "not an admission certificate"):
            verify_certificate(store, allowlist_sha, auditors=PI05_COHORT_AUDITORS)
        with self.assertRaisesRegex(AdmissionError, "SHA-256"):
            verify_certificate(store, "not-a-digest", auditors=PI05_COHORT_AUDITORS)

    def test_certificate_without_root_receipt_is_rejected(self) -> None:
        cohort = write_dense_cohort(self.root / "run")
        store = ArtifactStore(self.root / "store")
        refs, evidence = self._stored_evidence(store, cohort)
        claim, _ = build_claim(DENSE_ALLOWLIST_AUDITOR, refs, evidence)
        document = compile_certificate(claim, created_at=NOW, artifact_id="training-admission:unsigned")
        certificate = store.put_document(document)
        with self.assertRaisesRegex(AdmissionError, "no passing admission-root receipt"):
            verify_certificate(store, certificate.sha256, auditors=PI05_COHORT_AUDITORS)

    def test_self_signed_certificate_is_rejected(self) -> None:
        cohort = write_dense_cohort(self.root / "run")
        store = ArtifactStore(self.root / "store")
        refs, evidence = self._stored_evidence(store, cohort)
        claim, _ = build_claim(DENSE_ALLOWLIST_AUDITOR, refs, evidence)
        with self.assertRaisesRegex(AdmissionError, "cannot sign its own admission"):
            issue_certificate(store, claim, PI05_COHORT_AUDITORS, created_at=NOW, producer_id=ADMISSION_ROOT_ID)
        # A document produced under the root's identity is refused even if a
        # (differently named) root signs it.
        document = compile_certificate(claim, created_at=NOW, artifact_id="training-admission:self",
                                       producer_id=ADMISSION_ROOT_ID)
        certificate = store.put_document(document)
        other_root = TrainingAdmissionRoot(store, PI05_COHORT_AUDITORS, component_id="other-root")
        receipt = other_root.verify(document, certificate)
        self.assertIs(receipt.verdict, Verdict.REJECT)
        self.assertFalse(receipt.checks["certificate_signer_independent"])

    def test_waiver_in_zero_waiver_allowlist_is_rejected_but_recorded(self) -> None:
        cohort = write_dense_cohort(self.root / "run", failed_check="strict_all_seams")
        code, report = issue(self.root / "store", cohort["dataset"])
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("evidence:allowlist_checks_pass_or_waived", report["failed_checks"])
        store = ArtifactStore(self.root / "store")
        receipts = store.receipts_for(report["certificate_sha256"])
        self.assertEqual([item[1].verdict for item in receipts], [Verdict.REJECT])
        with self.assertRaisesRegex(AdmissionError, "reject"):
            verify_certificate(store, report["certificate_sha256"], auditors=PI05_COHORT_AUDITORS)

    def test_explicit_approved_waiver_is_admitted_and_named(self) -> None:
        cohort = write_dense_cohort(self.root / "run", failed_check="strict_all_seams",
                                    waiver_policy=["strict_all_seams"])
        code, report = issue(self.root / "store", cohort["dataset"])
        self.assertEqual(code, 0, report)
        self.assertEqual(report["waiver_policy"], "explicit-waivers")
        store = ArtifactStore(self.root / "store")
        verification = verify_certificate(store, report["certificate_sha256"], auditors=PI05_COHORT_AUDITORS,
                                          expectation=self.expectation(cohort))
        self.assertEqual(verification.allowed_failed_checks, ("strict_all_seams",))
        self.assertEqual(verification.granted_waivers, 1)
        # A launch that allows no waivers must not accept this certificate.
        strict = replace(self.expectation(cohort), allowed_failed_checks=())
        with self.assertRaisesRegex(AdmissionError, "waiver policy"):
            verify_certificate(store, report["certificate_sha256"], auditors=PI05_COHORT_AUDITORS,
                               expectation=strict)

    def test_provenance_that_hides_a_waiver_is_rejected(self) -> None:
        cohort = write_dense_cohort(self.root / "run", failed_check="strict_all_seams",
                                    waiver_policy=["strict_all_seams"], provenance_waivers=[])
        code, report = issue(self.root / "store", cohort["dataset"])
        self.assertEqual(code, 2)
        self.assertIn("evidence:provenance_waiver_policy_matches", report["failed_checks"])

    def test_binding_to_a_different_dataset_is_rejected(self) -> None:
        first = write_dense_cohort(self.root / "first")
        second = write_dense_cohort(self.root / "second", members=2)
        code, report = issue(self.root / "store", first["dataset"])
        self.assertEqual(code, 0)
        store = ArtifactStore(self.root / "store")
        with self.assertRaisesRegex(AdmissionError, "provenance digest"):
            verify_certificate(store, report["certificate_sha256"], auditors=PI05_COHORT_AUDITORS,
                               expectation=self.expectation(second))

    def test_tampered_evidence_in_store_fails_verification(self) -> None:
        cohort = write_dense_cohort(self.root / "run")
        code, report = issue(self.root / "store", cohort["dataset"])
        self.assertEqual(code, 0)
        store = ArtifactStore(self.root / "store")
        allowlist_ref = report["evidence"]["allowlist"]["sha256"]
        blob = store.root / "sha256" / allowlist_ref[:2] / f"{allowlist_ref}.blob"
        blob.write_bytes(blob.read_bytes() + b" ")
        with self.assertRaisesRegex(AdmissionError, "certificate_evidence_intact"):
            verify_certificate(store, report["certificate_sha256"], auditors=PI05_COHORT_AUDITORS)

    def test_forged_passing_receipt_for_a_bad_certificate_is_not_enough(self) -> None:
        cohort = write_dense_cohort(self.root / "run", failed_check="strict_all_seams")
        code, report = issue(self.root / "store", cohort["dataset"])
        self.assertEqual(code, 2)
        store = ArtifactStore(self.root / "store")
        (_, rejecting), = store.receipts_for(report["certificate_sha256"])
        forged = replace(rejecting, verdict=Verdict.PASS, checks={"forged": True}, reason="forged")
        store.put_receipt(forged)
        with self.assertRaises(AdmissionError) as caught:
            verify_certificate(store, report["certificate_sha256"], auditors=PI05_COHORT_AUDITORS)
        self.assertTrue(any("root check failed now" in item for item in caught.exception.failures))

    def test_unsupported_allowlist_schema_is_an_error(self) -> None:
        cohort = write_dense_cohort(self.root / "run")
        bogus = self.root / "bogus-allowlist.json"
        bogus.write_text(json.dumps({"schema_version": "something-else/9"}))
        code, report = issue(self.root / "store", cohort["dataset"], "--semantic-allowlist", str(bogus))
        self.assertEqual(code, 1)
        self.assertIn("no PI0.5 admission auditor", report["error"])

    def test_output_copy_is_the_canonical_certificate(self) -> None:
        cohort = write_dense_cohort(self.root / "run")
        output = self.root / "certificate.json"
        code, report = issue(self.root / "store", cohort["dataset"], "--output", str(output))
        self.assertEqual(code, 0)
        self.assertEqual(sha(output.read_bytes()), report["certificate_sha256"])

    def _stored_evidence(self, store: ArtifactStore, cohort):
        tool = load_tool("issue_admission_certificate")
        args = tool.parse_args(["--store", str(store.root), "--dataset-root", str(cohort["dataset"])])
        _, evidence = tool.collect_evidence(args)
        refs = {role: store.put_bytes(role, raw, media_type="application/vnd.datafarm.evidence+json")
                for role, raw in evidence.items()}
        return refs, evidence


if __name__ == "__main__":
    unittest.main()
