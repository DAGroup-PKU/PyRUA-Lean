"""Launch-time admission and train/deploy parity in datafarm.training.launch."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError:  # pragma: no cover - the tool imports the numpy-based dataset module
    np = None

from datafarm.core.admission import AdmissionError

from tests.admission_fixtures import sha, write_dense_cohort, write_json

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"r1_tool_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(np is None, "numpy is required to import datafarm.training.launch")
class Pi05LaunchAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        from datafarm.training import launch
        self.tool = launch
        self.issuer = load_tool("issue_admission_certificate")
        self.cohort = write_dense_cohort(self.root / "run")
        self.spec_path = self.root / "configs" / "pi05-fixture.json"
        write_json(self.spec_path, {"fixture": True})
        self.ledger = self.root / "ledger"
        self.ledger.mkdir()
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"DATAFARM_LEDGER_RUN_DIR", "DATAFARM_ADMISSION_CUTOVER"}}
        environment["DATAFARM_LEDGER_ROOT"] = str(self.ledger)
        patcher = mock.patch.dict(os.environ, environment, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ----------------------------------------------------------- fixtures

    def paths(self) -> dict:
        dataset = self.cohort["dataset"]
        provenance_path = dataset / "meta" / "lerobot-provenance.json"
        return {"dataset_root": dataset, "provenance_path": provenance_path,
                "provenance": json.loads(provenance_path.read_text()),
                "info": json.loads((dataset / "meta" / "info.json").read_text())}

    def spec(self, **updates) -> dict:
        value = {"repo_id": self.paths()["provenance"]["repo_id"], "allowed_semantic_waivers": [],
                 "run_manifest_dir": str(self.root / "manifests"), "state_conditioning": "none",
                 "assets_base_dir": str(self.root / "assets"), "base_config": "pi05_fixture"}
        value.update(updates)
        return value

    def args(self, **updates) -> argparse.Namespace:
        value = {"admission_certificate": None, "admission_store": None}
        value.update(updates)
        return argparse.Namespace(**value)

    def issue(self, *extra: str) -> dict:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = self.issuer.main(["--store", str(self.root / "store"), "--dataset-root",
                                     str(self.cohort["dataset"]), "--created-at", "2026-09-26T12:00:00Z", *extra])
        self.assertEqual(code, 0, out.getvalue())
        return json.loads(out.getvalue())

    def certified_binding(self):
        report = self.issue()
        binding, _ = self.tool.resolve_admission(
            self.args(admission_certificate=report["certificate_sha256"], admission_store=str(self.root / "store")),
            self.spec(), self.spec_path, self.paths())
        return binding

    @staticmethod
    def train(mode: str) -> list[str]:
        flag = "--model.no-discrete-state-input" if mode == "none" else "--model.discrete-state-input"
        return ["train.py", "--model.action-dim", "32", "--model.action-horizon", "50", flag]

    # ---------------------------------------------------------- admission

    def test_new_launch_without_certificate_is_refused(self) -> None:
        with self.assertRaisesRegex(self.tool.AdmissionRequired, "--admission-certificate is required"):
            self.tool.resolve_admission(self.args(), self.spec(), self.spec_path, self.paths())

    def test_verified_certificate_binds_the_launch(self) -> None:
        report = self.issue()
        binding, record = self.tool.resolve_admission(
            self.args(admission_certificate=report["certificate_sha256"], admission_store=str(self.root / "store")),
            self.spec(), self.spec_path, self.paths())
        self.assertEqual(binding.mode, "certificate-verified")
        self.assertTrue(record["dataset_files_verified"])
        audit = self.tool.audit_launch(self.spec(), self.paths(), self.train("none"), admission=binding)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["admission_mode"], "certificate-verified")
        self.assertEqual(audit["statuses"]["admission_bound"], "pass")

    def test_certificate_file_forms(self) -> None:
        output = self.root / "admission" / "certificate.json"
        report = self.issue("--output", str(output))
        store = self.root / "store"
        binding, _ = self.tool.resolve_admission(
            self.args(admission_certificate=str(output), admission_store=str(store)),
            self.spec(), self.spec_path, self.paths())
        self.assertEqual(binding.certificate_sha256, report["certificate_sha256"])
        in_store = store / "sha256" / report["certificate_sha256"][:2] / f"{report['certificate_sha256']}.json"
        binding, _ = self.tool.resolve_admission(self.args(admission_certificate=str(in_store)),
                                                 self.spec(), self.spec_path, self.paths())
        self.assertEqual(binding.certificate_sha256, report["certificate_sha256"])
        with self.assertRaisesRegex(self.tool.AdmissionRequired, "--admission-store is required"):
            self.tool.resolve_admission(self.args(admission_certificate=str(output)),
                                        self.spec(), self.spec_path, self.paths())
        with self.assertRaisesRegex(self.tool.AdmissionRequired, "does not exist"):
            self.tool.resolve_admission(self.args(admission_certificate=str(self.root / "missing.json"),
                                                  admission_store=str(store)),
                                        self.spec(), self.spec_path, self.paths())

    def test_allowlist_hash_posing_as_certificate_is_refused_at_launch(self) -> None:
        self.issue()
        allowlist_sha = self.paths()["provenance"]["semantic_allowlist_sha256"]
        with self.assertRaisesRegex(AdmissionError, "not an admission certificate"):
            self.tool.resolve_admission(
                self.args(admission_certificate=allowlist_sha, admission_store=str(self.root / "store")),
                self.spec(), self.spec_path, self.paths())

    def test_dataset_files_changed_after_certification_are_refused(self) -> None:
        report = self.issue()
        episode = next((self.cohort["dataset"] / "data").rglob("*.parquet"))
        episode.write_bytes(episode.read_bytes() + b"tamper")
        with self.assertRaisesRegex(AdmissionError, "dataset files changed"):
            self.tool.resolve_admission(
                self.args(admission_certificate=report["certificate_sha256"], admission_store=str(self.root / "store")),
                self.spec(), self.spec_path, self.paths())

    def test_certificate_for_another_repo_is_refused(self) -> None:
        report = self.issue()
        with self.assertRaisesRegex(AdmissionError, "different repo_id"):
            self.tool.resolve_admission(
                self.args(admission_certificate=report["certificate_sha256"], admission_store=str(self.root / "store")),
                self.spec(repo_id="example/other"), self.spec_path, self.paths())

    def test_launch_manifest_records_admission_and_keeps_the_superseded_one(self) -> None:
        manifests = self.root / "manifests"
        old = write_json(manifests / "launch-manifest.json", {"schema_version": "datafarm.pi05-train-launch/1.1.0"})
        paths = self.paths()
        paths.update({"distribution_admission": {"path": self.cohort["distribution"], "sha256": "d" * 64,
                                                 "contract": {"claim": "broad_deployment"}},
                      "paligemma_tokenizer": self.root / "tok.model", "paligemma_tokenizer_sha256": "t" * 64,
                      "metrics_jsonl": self.root / "metrics.jsonl"})
        spec = self.spec(trainer_source_sha256={}, batch_size=8, fsdp_devices=1)
        output = self.tool.write_launch_manifest(self.spec_path, spec, paths, ["cmd"], {"passed": True},
                                                 admission_record={"mode": "certificate-verified"})
        manifest = json.loads(output.read_text())
        self.assertEqual(manifest["schema_version"], "datafarm.pi05-train-launch/1.2.0")
        self.assertEqual(manifest["admission"]["mode"], "certificate-verified")
        superseded = manifests / f"launch-manifest.{sha(old)[:12]}.superseded.json"
        self.assertEqual(superseded.read_bytes(), old)

    # -------------------------------------------------------------- parity

    def test_joint_source_evidence_binds_exact_readbacks_without_a_new_threshold(self) -> None:
        dataset = self.root / "joint-dataset"
        dataset.mkdir()
        source = self.root / "source-state.json"
        source.write_text('{"actual": true}\n')
        terminal_path = self.root / "terminal.json"
        terminal = write_json(terminal_path, {
            "dataset": str(dataset),
            "episodes": [{"final_joint_dims_exactly_equal": True, "source_state_path": str(source),
                          "source_state_sha256": sha(source.read_bytes())}],
        })
        machine_path = self.root / "machine.json"
        machine = write_json(machine_path, {
            "tasks": {"fixture": {"episodes": 1, "episodes_detail": [{"rows": 3}], "checks": {
                "all_final_joint_dims_equal_final_source_readback": True,
                "all_gripper_columns_bit_identical_to_eef": True,
                "all_joint_state_action_rederived_exact": True,
                "all_rgb_timestamp_index_columns_exact": True,
                "all_source_state_hashes_match": True,
                "metadata_fields_and_units_match": True,
            }}}
        })
        spec = {"joint_source_evidence": {
            "terminal_joint_readback_manifest": {"path": str(terminal_path), "sha256": sha(terminal)},
            "independent_machine_audit": {"path": str(machine_path), "sha256": sha(machine)},
        }}
        report = self.tool.validate_joint_source_evidence(
            spec, dataset, {"total_episodes": 1, "total_frames": 3})
        self.assertEqual(report["terminal_joint_readback_manifest"]["episodes"], 1)
        source.write_text('{"tampered": true}\n')
        with self.assertRaisesRegex(ValueError, "Source joint readback drifted"):
            self.tool.validate_joint_source_evidence(spec, dataset, {"total_episodes": 1, "total_frames": 3})

    def no_eef_fixture(self, *, eef_check=None, command_check=True, unproven=True, verdict="pass_with_notes"):
        dataset = self.root / "joint-dataset-no-eef"
        dataset.mkdir(exist_ok=True)
        source = self.root / "source-no-eef.npz"
        source.write_bytes(b"readback")
        terminal_path = self.root / "terminal-no-eef.json"
        terminal = write_json(terminal_path, {
            "dataset": str(dataset),
            "episodes": [{"final_joint_dims_exactly_equal": True, "source_state_path": str(source),
                          "source_state_sha256": sha(source.read_bytes())}],
        })
        checks = {
            "all_final_joint_dims_equal_final_source_readback": True,
            "all_gripper_columns_bit_identical_to_eef": eef_check,
            "all_joint_state_action_rederived_exact": True,
            "all_rgb_timestamp_index_columns_exact": True,
            "all_source_state_hashes_match": True,
            "metadata_fields_and_units_match": True,
            "all_gripper_columns_bit_identical_to_independent_command_rederivation": command_check,
        }
        machine_path = self.root / "machine-no-eef.json"
        machine = write_json(machine_path, {
            "tasks": {"fixture": {"episodes": 1, "episodes_detail": [{"rows": 3}], "checks": checks}},
            "not_applicable_or_unproven": ["all_gripper_columns_bit_identical_to_eef"] if unproven else [],
            "hard_failures": [], "verdict": verdict,
        })
        spec = {"joint_source_evidence": {
            "terminal_joint_readback_manifest": {"path": str(terminal_path), "sha256": sha(terminal)},
            "independent_machine_audit": {"path": str(machine_path), "sha256": sha(machine)},
            "eef_baseline": {"status": "not_applicable", "reason": "labels derived from simulator export"},
        }}
        provenance = {"joint_conversion": {"gripper_source": {"label_m": "clip(cmd/0.04,0,1)*0.105"},
                                           "joint_source": "white export readback"}}
        return spec, dataset, provenance

    def test_joint_gripper_evidence_accepts_command_rederivation_only_without_eef_baseline(self) -> None:
        info = {"total_episodes": 1, "total_frames": 3}
        spec, dataset, provenance = self.no_eef_fixture()
        report = self.tool.validate_joint_source_evidence(spec, dataset, info, provenance)
        self.assertEqual(report["gripper_evidence"]["check"],
                         "all_gripper_columns_bit_identical_to_independent_command_rederivation")
        self.assertIn("all_gripper_columns_bit_identical_to_independent_command_rederivation",
                      report["independent_machine_audit"]["checks"])
        self.assertNotIn("all_gripper_columns_bit_identical_to_eef", report["independent_machine_audit"]["checks"])

        # Without the explicit spec declaration the EEF check stays mandatory.
        undeclared = json.loads(json.dumps(spec))
        del undeclared["joint_source_evidence"]["eef_baseline"]
        with self.assertRaisesRegex(ValueError, "missing a required exact check"):
            self.tool.validate_joint_source_evidence(undeclared, dataset, info, provenance)
        # A declaration without a reason is refused.
        unreasoned = json.loads(json.dumps(spec))
        unreasoned["joint_source_evidence"]["eef_baseline"] = {"status": "not_applicable"}
        with self.assertRaisesRegex(ValueError, "eef_baseline must be"):
            self.tool.validate_joint_source_evidence(unreasoned, dataset, info, provenance)
        # Provenance that names an EEF source dataset cannot skip the EEF bit identity.
        for key, value in (("source_dataset", "/data/eef"), ("gripper_bit_identity", "equals source EEF parquet")):
            named = {"joint_conversion": {**provenance["joint_conversion"], key: value}}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "names an EEF baseline"):
                self.tool.validate_joint_source_evidence(spec, dataset, info, named)
        with self.assertRaisesRegex(ValueError, "names an EEF baseline"):
            self.tool.validate_joint_source_evidence(spec, dataset, info, {**provenance, "eef_baseline": "/data/eef"})
        with self.assertRaisesRegex(ValueError, "requires the dataset provenance"):
            self.tool.validate_joint_source_evidence(spec, dataset, info)

    def test_joint_gripper_replacement_fails_closed_on_audit_disagreement(self) -> None:
        info = {"total_episodes": 1, "total_frames": 3}
        cases = (
            ({"eef_check": False}, "not applicable"),          # audit says the EEF check failed, not N/A
            ({"unproven": False}, "not applicable"),            # null without an N/A declaration
            ({"command_check": False}, "missing a required exact check"),
            ({"command_check": None}, "missing a required exact check"),
            ({"verdict": "fail"}, "did not pass"),
        )
        for updates, message in cases:
            with self.subTest(updates=updates):
                spec, dataset, provenance = self.no_eef_fixture(**updates)
                with self.assertRaisesRegex(ValueError, message):
                    self.tool.validate_joint_source_evidence(spec, dataset, info, provenance)

    # ---------------------------------------------------------- supervisor



if __name__ == "__main__":
    unittest.main()
