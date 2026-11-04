"""Fail-closed contract for admission-bound Piper PI0.5 training.

This module deliberately does not import a trainer.  It validates the semantic
launch contract before a backend adapter is allowed to materialize a command.

Every check reports ``pass`` or ``fail``; any ``fail`` blocks the launch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from datafarm.common import is_sha256


# PI0.5 can consume discretized state through its tokenizer, but this launcher trains state-free only.
STATE_CONDITIONING_MODES = ("none",)

PIPER_CAMERA_KEYS = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
)

CHECK_PASS = "pass"
CHECK_FAIL = "fail"

ADMISSION_MODE_CERTIFICATE = "certificate-verified"
ADMISSION_MODES = (ADMISSION_MODE_CERTIFICATE,)
ADMISSION_ROOT_AUTHORITY = "admission-root"



@dataclass(frozen=True)
class AdmissionBinding:
    """How a launch is bound to its Admission Certificate.

    A ``certificate-verified`` binding must be built from a successful
    ``datafarm.core.admission.verify_certificate`` result (see
    ``from_verification``).
    """

    mode: str
    certificate_sha256: str | None = None
    receipt_sha256: str | None = None
    producer_id: str | None = None
    verifier_id: str | None = None
    authority: str | None = None
    training_bundle_sha256: str | None = None
    allowlist_sha256: str | None = None

    @classmethod
    def from_verification(cls, verification: Any) -> "AdmissionBinding":
        return cls(
            mode=ADMISSION_MODE_CERTIFICATE,
            certificate_sha256=verification.certificate_sha256,
            receipt_sha256=verification.receipt_sha256,
            producer_id=verification.producer_id,
            verifier_id=verification.verifier_id,
            authority=verification.authority,
            training_bundle_sha256=verification.training_bundle_sha256,
            allowlist_sha256=verification.allowlist_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "certificate_sha256": self.certificate_sha256,
            "receipt_sha256": self.receipt_sha256,
            "producer_id": self.producer_id,
            "verifier_id": self.verifier_id,
            "authority": self.authority,
            "training_bundle_sha256": self.training_bundle_sha256,
            "allowlist_sha256": self.allowlist_sha256,
        }


@dataclass(frozen=True)
class Pi05TrainingContract:
    """Semantic inputs that must be frozen before a trainer command exists."""

    admission_certificate_sha256: str
    training_bundle_sha256: str
    model_family: str
    state_conditioning: str
    discrete_state_input: bool
    native_action_dim: int
    model_action_dim: int
    action_horizon: int
    camera_keys: tuple[str, ...]
    source_fps: float
    launch_argv: tuple[str, ...] = ()
    ablation_name: str | None = None
    # Admission: a bare digest is never enough; the binding carries the
    # verified certificate.
    admission: AdmissionBinding | None = None


@dataclass(frozen=True)
class Pi05TrainingAudit:
    passed: bool
    checks: Mapping[str, bool]
    failures: tuple[str, ...]
    statuses: Mapping[str, str] = field(default_factory=dict)
    admission_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "admission_mode": self.admission_mode,
            "checks": dict(self.checks),
            "statuses": dict(self.statuses),
            "failures": list(self.failures),
        }


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _admission_status(contract: Pi05TrainingContract) -> tuple[str, str | None]:
    binding = contract.admission
    if binding is None or binding.mode not in ADMISSION_MODES:
        return CHECK_FAIL, None
    bound = (
        is_sha256(binding.certificate_sha256)
        and binding.certificate_sha256 == contract.admission_certificate_sha256
        and is_sha256(binding.receipt_sha256)
        and bool(binding.producer_id)
        and bool(binding.verifier_id)
        and binding.producer_id != binding.verifier_id
        and binding.authority == ADMISSION_ROOT_AUTHORITY
        and is_sha256(contract.training_bundle_sha256)
        and binding.training_bundle_sha256 == contract.training_bundle_sha256
    )
    return (CHECK_PASS if bound else CHECK_FAIL), binding.mode


def audit_pi05_training_contract(contract: Pi05TrainingContract) -> Pi05TrainingAudit:
    """Validate a formal PI0.5 launch contract without invoking a trainer."""

    statuses: dict[str, str] = {}

    def put(name: str, passed: bool) -> None:
        statuses[name] = CHECK_PASS if passed else CHECK_FAIL

    admission_status, admission_mode = _admission_status(contract)
    statuses["admission_bound"] = admission_status
    put("training_bundle_bound", is_sha256(contract.training_bundle_sha256))
    put("pi05_model", contract.model_family == "pi05")

    state_mode_known = contract.state_conditioning in STATE_CONDITIONING_MODES
    positive_flag = "--model.discrete-state-input" in contract.launch_argv
    negative_flag = "--model.no-discrete-state-input" in contract.launch_argv

    put("state_mode_declared", state_mode_known)
    put("state_flag_matches_mode", state_mode_known and not contract.discrete_state_input)
    put("native_action_dimension", contract.native_action_dim == 14)
    put("model_dimensions", contract.model_action_dim == 32 and contract.action_horizon == 50)
    put("three_independent_views", contract.camera_keys == PIPER_CAMERA_KEYS)
    put("positive_clock", _positive(contract.source_fps))
    put("cli_state_flag_matches_mode", state_mode_known and negative_flag and not positive_flag)
    checks = {name: status == CHECK_PASS for name, status in statuses.items()}
    failures = tuple(name for name, status in statuses.items() if status == CHECK_FAIL)
    return Pi05TrainingAudit(
        passed=not failures,
        checks=checks,
        failures=failures,
        statuses=statuses,
        admission_mode=admission_mode,
    )
