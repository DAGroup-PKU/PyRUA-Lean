from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from datafarm.common import canonical_json_bytes as _canonical_json_bytes
from datafarm.common import is_sha256
from datafarm.common import sha256_bytes as sha256_hex
from datafarm.core.grounding import GroundingError, validate_quantities
from datafarm.core.model import ArtifactDocument, GateReceipt, IRKind

IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
}
PATH_KEYS = {"path", "file", "filepath", "filename", "directory", "root"}

REQUIRED_PAYLOAD_KEYS: dict[IRKind, frozenset[str]] = {
    IRKind.GROUNDING: frozenset(
        {
            "source",
            "transfer_scope",
            "camera_clock",
            "reference_roles",
            "system_identification",
            "rights",
        }
    ),
    IRKind.TASK: frozenset(
        {
            "demand",
            "roles",
            "stage_graph",
            "predicates",
            "forbidden_shortcuts",
            "skill_evidence",
        }
    ),
    IRKind.EPISODE: frozenset(
        {
            "setup",
            "actions",
            "state_trace",
            "contacts",
            "events",
            "frame_map",
            "replay",
        }
    ),
    IRKind.RENDER: frozenset(
        {
            "carrier",
            "immutable_invariants",
            "allowed_changes",
            "frame_map",
            "references",
            "backend",
        }
    ),
    IRKind.ADMISSION: frozenset(
        {
            "episode",
            "rendered_observation",
            "checks",
            "verdict",
            "cost",
            "training_bundle",
            "repair_predecessor",
        }
    ),
}


class SchemaError(ValueError):
    pass


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def _validate_timestamp(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SchemaError(f"{label} must be an RFC3339 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise SchemaError(f"{label} is not a valid timestamp") from exc


def _validate_portable_value(value: Any, *, label: str = "document") -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaError(f"{label} contains a non-finite float")
        return
    if isinstance(value, str):
        if PurePosixPath(value).is_absolute() or WINDOWS_ABSOLUTE_PATTERN.match(value):
            raise SchemaError(
                f"{label} contains an absolute filesystem path; use an ArtifactRef"
            )
        parsed = urlsplit(value)
        unsafe_url = parsed.username or parsed.password or parsed.query or parsed.fragment
        if parsed.scheme and unsafe_url:
            raise SchemaError(f"{label} contains a credentialed or mutable URL")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError(f"{label} contains a non-string object key")
            normalized = _normalized_key(key)
            if normalized in SENSITIVE_KEYS or normalized.endswith("_secret"):
                raise SchemaError(f"{label} contains forbidden secret field {key!r}")
            path_field = (
                normalized in PATH_KEYS
                or normalized.endswith("_path")
                or normalized.endswith("_file")
            )
            if path_field:
                raise SchemaError(
                    f"{label} contains filesystem locator field {key!r}; use an ArtifactRef"
                )
            _validate_portable_value(item, label=f"{label}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            _validate_portable_value(item, label=f"{label}[{index}]")
        return
    raise SchemaError(f"{label} contains unsupported JSON type {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    _validate_portable_value(value)
    try:
        return _canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"value is not canonical JSON: {exc}") from exc


def document_sha256(document: ArtifactDocument) -> str:
    """Content address of a document: SHA-256 of its canonical envelope."""

    return sha256_hex(canonical_json_bytes(document.to_dict()))


def payload_sha256(document: ArtifactDocument) -> str:
    """Digest of the canonical payload only (dedup key; not an address)."""

    return sha256_hex(canonical_json_bytes(dict(document.payload)))


def identity_sha256(document: ArtifactDocument) -> str:
    """Digest of the canonical envelope with ``created_at`` removed.

    Two executions that produce the same artifact id, producer, lineage,
    evidence and payload at different times share this digest. It is additive:
    the content address (which includes ``created_at``) is unchanged, so every
    stored artifact still verifies.
    """

    value = document.to_dict()
    value.pop("created_at", None)
    return sha256_hex(canonical_json_bytes(value))


def _validate_ref(value: Mapping[str, Any], label: str) -> None:
    required = {"kind", "sha256", "media_type"}
    if set(value) != required:
        raise SchemaError(f"{label} must contain exactly {sorted(required)}")
    if not isinstance(value["kind"], str) or not value["kind"]:
        raise SchemaError(f"{label}.kind must be non-empty")
    if not is_sha256(value["sha256"]):
        raise SchemaError(f"{label}.sha256 must be lowercase SHA-256")
    if not isinstance(value["media_type"], str) or "/" not in value["media_type"]:
        raise SchemaError(f"{label}.media_type must be a media type")


def validate_artifact(document: ArtifactDocument | Mapping[str, Any]) -> None:
    value = document.to_dict() if isinstance(document, ArtifactDocument) else dict(document)
    required = {
        "schema",
        "schema_version",
        "artifact_id",
        "created_at",
        "producer",
        "predecessors",
        "evidence",
        "payload",
    }
    if set(value) != required:
        raise SchemaError(f"artifact top-level fields must be exactly {sorted(required)}")
    try:
        parsed = ArtifactDocument.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(f"invalid artifact envelope: {exc}") from exc
    if parsed.schema_version != "1.0.0":
        raise SchemaError(f"unsupported schema version {parsed.schema_version}")
    if not IDENTIFIER_PATTERN.fullmatch(parsed.artifact_id):
        raise SchemaError("artifact_id must be a stable lowercase identifier")
    if not IDENTIFIER_PATTERN.fullmatch(parsed.producer):
        raise SchemaError("producer must be a stable lowercase component identifier")
    _validate_timestamp(parsed.created_at, "created_at")
    for field_name in ("predecessors", "evidence"):
        items = value[field_name]
        if not isinstance(items, list):
            raise SchemaError(f"{field_name} must be an array")
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise SchemaError(f"{field_name}[{index}] must be an object")
            _validate_ref(item, f"{field_name}[{index}]")
    if not isinstance(parsed.payload, Mapping):
        raise SchemaError("payload must be an object")
    missing = REQUIRED_PAYLOAD_KEYS[parsed.kind] - set(parsed.payload)
    if missing:
        raise SchemaError(
            f"{parsed.kind.value} payload is missing required sections: {sorted(missing)}"
        )
    if parsed.kind is IRKind.GROUNDING:
        identification = parsed.payload["system_identification"]
        if not isinstance(identification, Mapping) or "physical_quantities" not in identification:
            raise SchemaError("grounding system_identification must list physical_quantities")
        try:
            validate_quantities(identification["physical_quantities"],
                                bind=lambda ref: _validate_ref(ref, "system_identification evidence"))
        except GroundingError as exc:
            raise SchemaError(f"grounding system_identification: {exc}") from exc
    _validate_portable_value(value)


def validate_gate_receipt(receipt: GateReceipt | Mapping[str, Any]) -> None:
    value = receipt.to_dict() if isinstance(receipt, GateReceipt) else dict(receipt)
    required = {
        "schema",
        "schema_version",
        "stage",
        "subject_sha256",
        "verifier_id",
        "authority",
        "verdict",
        "checks",
        "evidence",
        "created_at",
        "reason",
    }
    if set(value) != required or value.get("schema") != "datafarm.gate-receipt":
        raise SchemaError("invalid gate receipt envelope")
    try:
        parsed = GateReceipt.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(f"invalid gate receipt: {exc}") from exc
    if parsed.schema_version != "1.0.0":
        raise SchemaError(f"unsupported receipt version {parsed.schema_version}")
    if not is_sha256(parsed.subject_sha256):
        raise SchemaError("receipt subject_sha256 must be lowercase SHA-256")
    if not IDENTIFIER_PATTERN.fullmatch(parsed.verifier_id):
        raise SchemaError("verifier_id must be a stable lowercase component identifier")
    if not parsed.checks or any(type(result) is not bool for result in parsed.checks.values()):
        raise SchemaError("receipt checks must be a non-empty boolean object")
    if parsed.verdict.value == "pass" and not all(parsed.checks.values()):
        raise SchemaError("a passing receipt cannot contain a failed check")
    _validate_timestamp(parsed.created_at, "created_at")
    for index, item in enumerate(value["evidence"]):
        _validate_ref(item, f"evidence[{index}]")
    _validate_portable_value(value)

