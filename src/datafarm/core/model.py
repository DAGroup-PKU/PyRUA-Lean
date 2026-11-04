from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class StageName(str, Enum):
    GROUNDING = "grounding"
    TASK = "task"
    EPISODE = "episode"
    RENDER = "render"
    ADMISSION = "admission"


class IRKind(str, Enum):
    GROUNDING = "grounding-ir"
    TASK = "task-ir"
    EPISODE = "episode-ir"
    RENDER = "render-ir"
    ADMISSION = "admission-certificate"


class Authority(str, Enum):
    REFERENCE_AUDITOR = "reference-auditor"
    TASK_VERIFIER = "task-verifier"
    SIMULATOR = "simulator"
    RENDER_CONTRACT_VERIFIER = "render-contract-verifier"
    ADMISSION_ROOT = "admission-root"


class Verdict(str, Enum):
    PASS = "pass"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    sha256: str
    media_type: str = "application/json"

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "sha256": self.sha256,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            kind=str(value["kind"]),
            sha256=str(value["sha256"]),
            media_type=str(value.get("media_type", "application/json")),
        )


@dataclass(frozen=True)
class ArtifactDocument:
    kind: IRKind
    artifact_id: str
    created_at: str
    producer: str
    predecessors: tuple[ArtifactRef, ...]
    evidence: tuple[ArtifactRef, ...]
    payload: Mapping[str, Any]
    schema_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"datafarm.{self.kind.value}",
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "producer": self.producer,
            "predecessors": [item.to_dict() for item in self.predecessors],
            "evidence": [item.to_dict() for item in self.evidence],
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactDocument":
        schema = str(value["schema"])
        prefix = "datafarm."
        if not schema.startswith(prefix):
            raise ValueError(f"unsupported artifact schema: {schema}")
        return cls(
            kind=IRKind(schema.removeprefix(prefix)),
            schema_version=str(value["schema_version"]),
            artifact_id=str(value["artifact_id"]),
            created_at=str(value["created_at"]),
            producer=str(value["producer"]),
            predecessors=tuple(
                ArtifactRef.from_dict(item) for item in value.get("predecessors", [])
            ),
            evidence=tuple(ArtifactRef.from_dict(item) for item in value.get("evidence", [])),
            payload=dict(value["payload"]),
        )

    @property
    def payload_sha256(self) -> str:
        """Digest of the canonical payload alone.

        The envelope digest (the content address) includes ``created_at`` and
        ``artifact_id``, so two executions that produce the same payload get
        different addresses. Producers and handlers use this digest for dedup;
        it never replaces the content address.
        """

        from datafarm.core.schema import payload_sha256  # local import: schema imports model

        return payload_sha256(self)

    @property
    def identity_sha256(self) -> str:
        """Digest of the envelope without ``created_at`` (timestamp-free identity)."""

        from datafarm.core.schema import identity_sha256  # local import: schema imports model

        return identity_sha256(self)


@dataclass(frozen=True)
class GateReceipt:
    stage: StageName
    subject_sha256: str
    verifier_id: str
    authority: Authority
    verdict: Verdict
    checks: Mapping[str, bool]
    evidence: tuple[ArtifactRef, ...]
    created_at: str
    reason: str = ""
    schema_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "datafarm.gate-receipt",
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "subject_sha256": self.subject_sha256,
            "verifier_id": self.verifier_id,
            "authority": self.authority.value,
            "verdict": self.verdict.value,
            "checks": dict(self.checks),
            "evidence": [item.to_dict() for item in self.evidence],
            "created_at": self.created_at,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GateReceipt":
        return cls(
            stage=StageName(value["stage"]),
            subject_sha256=str(value["subject_sha256"]),
            verifier_id=str(value["verifier_id"]),
            authority=Authority(value["authority"]),
            verdict=Verdict(value["verdict"]),
            checks=dict(value["checks"]),
            evidence=tuple(ArtifactRef.from_dict(item) for item in value.get("evidence", [])),
            created_at=str(value["created_at"]),
            reason=str(value.get("reason", "")),
            schema_version=str(value["schema_version"]),
        )


def refs(values: Sequence[Mapping[str, Any]]) -> tuple[ArtifactRef, ...]:
    return tuple(ArtifactRef.from_dict(value) for value in values)
