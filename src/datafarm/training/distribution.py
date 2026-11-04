"""Setup-distribution acceptance contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from datafarm.common import is_sha256


PLAN_SCHEMA = "datafarm-distribution-plan/1.0.0"
RECEIPT_SCHEMA = "datafarm-distribution-acceptance/1.0.0"
CLAIMS = {"fixed_setup", "local_robustness", "broad_deployment"}



class DistributionContractError(ValueError):
    """A setup distribution is missing evidence required for its claim."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DistributionContractError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} is required")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    _require(type(value) is int and value > 0, f"{name} must be a positive integer")
    return value


def _nonnegative_vector(value: Any, name: str, *, positive: bool) -> tuple[float, float]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2,
        f"{name} must be a two-value list",
    )
    numbers = tuple(float(item) for item in value)
    _require(all(math.isfinite(item) for item in numbers), f"{name} must be finite")
    if positive:
        _require(all(item > 0.0 for item in numbers), f"{name} must be strictly positive")
    else:
        _require(all(item >= 0.0 for item in numbers), f"{name} must be non-negative")
    return numbers


def _string_set(value: Any, name: str) -> set[str]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{name} must be a list",
    )
    items = [_string(item, f"{name}[]") for item in value]
    _require(len(items) == len(set(items)), f"{name} contains duplicates")
    return set(items)


def _sha256(value: Any, name: str) -> str:
    digest = _string(value, name)
    _require(is_sha256(digest), f"{name} must be a lowercase SHA-256")
    return digest


def _validate_common(contract: Mapping[str, Any], *, target_task: str) -> tuple[str, set[str]]:
    _require(contract.get("status") in {"frozen", "accepted"}, "distribution status is not promotable")
    _require(_string(contract.get("target_task"), "target_task") == target_task, "distribution target_task mismatch")
    claim = _string(contract.get("claim"), "claim")
    _require(claim in CLAIMS, f"unsupported distribution claim {claim!r}")
    objects = _string_set(contract.get("variable_objects"), "variable_objects")
    _require(bool(objects), "variable_objects must be non-empty")
    return claim, objects


def validate_distribution_acceptance_contract(
    contract: Mapping[str, Any],
    *,
    target_task: str,
    required_claim: str,
) -> None:
    """Validate actual captured coverage before video generation or training."""

    _require(contract.get("schema_version") == RECEIPT_SCHEMA, "unsupported distribution-acceptance schema")
    claim, objects = _validate_common(contract, target_task=target_task)
    _require(claim == required_claim, f"distribution claim {claim!r} does not satisfy required {required_claim!r}")
    denominator = _positive_int(contract.get("frozen_denominator"), "frozen_denominator")
    _require(_positive_int(contract.get("accepted_count"), "accepted_count") == denominator, "fixed denominator is incomplete")
    _sha256(contract.get("frozen_plan_sha256"), "frozen_plan_sha256")
    _sha256(contract.get("cohort_manifest_sha256"), "cohort_manifest_sha256")
    _sha256(contract.get("coverage_authority_sha256"), "coverage_authority_sha256")

    _require(
        _positive_int(contract.get("unique_joint_layouts"), "unique_joint_layouts") == denominator,
        "joint layout identities are not unique across the frozen denominator",
    )
    relational = _positive_int(contract.get("unique_relational_layouts"), "unique_relational_layouts")
    if required_claim == "broad_deployment":
        _require(relational > 1, "shared rigid jitter preserves one relational layout; broad coverage rejected")

    required_world = _mapping(contract.get("required_world_span_xy_m"), "required_world_span_xy_m")
    observed = _mapping(contract.get("observed_per_object"), "observed_per_object")
    _require(set(required_world) == objects, "required world spans do not cover every variable object")
    _require(set(observed) == objects, "observed object statistics do not cover every variable object")
    for object_name in objects:
        required_span = _nonnegative_vector(
            required_world[object_name],
            f"required world span for {object_name}",
            positive=required_claim == "broad_deployment",
        )
        stats = _mapping(observed[object_name], f"observed_per_object.{object_name}")
        _require(
            _positive_int(stats.get("unique_positions"), f"{object_name}.unique_positions") > 1,
            f"{object_name} does not vary independently",
        )
        observed_span = _nonnegative_vector(stats.get("span_xy_m"), f"{object_name}.span_xy_m", positive=False)
        _require(
            all(observed_span[index] + 1e-12 >= required_span[index] for index in range(2)),
            f"{object_name} world-space coverage is below its deployment requirement",
        )

    required_image = _mapping(contract.get("required_image_span_px"), "required_image_span_px")
    observed_image = _mapping(contract.get("observed_image_span_px"), "observed_image_span_px")
    _require(set(required_image) == set(observed_image), "observed policy-camera set does not match requirements")
    for camera_name, per_object in required_image.items():
        required_objects = _mapping(per_object, f"required_image_span_px.{camera_name}")
        actual_objects = _mapping(observed_image[camera_name], f"observed_image_span_px.{camera_name}")
        _require(set(required_objects) == objects, f"{camera_name} requirements do not cover every variable object")
        _require(set(actual_objects) == objects, f"{camera_name} observations do not cover every variable object")
        for object_name in objects:
            required_span = _nonnegative_vector(
                required_objects[object_name],
                f"required image span for {camera_name}.{object_name}",
                positive=required_claim == "broad_deployment",
            )
            actual_span = _nonnegative_vector(
                actual_objects[object_name],
                f"observed image span for {camera_name}.{object_name}",
                positive=False,
            )
            _require(
                all(actual_span[index] + 1e-12 >= required_span[index] for index in range(2)),
                f"{camera_name}.{object_name} image-space coverage is below its deployment requirement",
            )
