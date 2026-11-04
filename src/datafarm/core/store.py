from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from datafarm.common import atomic_write_bytes
from datafarm.core.model import ArtifactDocument, ArtifactRef, GateReceipt, IRKind
from datafarm.core.schema import (
    SchemaError,
    canonical_json_bytes,
    validate_artifact,
    validate_gate_receipt,
)
from datafarm.common import sha256_bytes


class ArtifactIntegrityError(RuntimeError):
    pass


class ArtifactStore:
    """Immutable content-addressed JSON store.

    The store owns locations. Documents contain digest references, never host
    filesystem paths, so a run can be moved without rewriting its evidence graph.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_document(self, document: ArtifactDocument) -> ArtifactRef:
        validate_artifact(document)
        return self._put(document.kind.value, document.to_dict())

    def put_receipt(self, receipt: GateReceipt) -> ArtifactRef:
        validate_gate_receipt(receipt)
        reference = self._put("gate-receipt", receipt.to_dict())
        self._index_receipt(receipt.subject_sha256, reference.sha256)
        return reference

    def receipts_for(self, subject_sha256: str) -> list[tuple[ArtifactRef, GateReceipt]]:
        """Return every stored gate receipt whose subject is ``subject_sha256``.

        The subject index under ``index/`` is a non-authoritative lookup aid:
        each hit is re-loaded by digest (integrity and schema checked) and its
        subject re-compared.
        """

        if len(subject_sha256) != 64 or any(char not in "0123456789abcdef" for char in subject_sha256):
            raise SchemaError("subject digest must be lowercase SHA-256")
        candidates: set[str] = set()
        directory = self._subject_index_dir(subject_sha256)
        if directory.is_dir():
            candidates.update(entry.name for entry in directory.iterdir() if len(entry.name) == 64)
        found: list[tuple[ArtifactRef, GateReceipt]] = []
        for digest in sorted(candidates):
            reference = ArtifactRef(kind="gate-receipt", sha256=digest)
            receipt = self.receipt(reference)
            if receipt.subject_sha256 == subject_sha256:
                found.append((reference, receipt))
        return found

    def _subject_index_dir(self, subject_sha256: str) -> Path:
        return self.root / "index" / "receipts-by-subject" / subject_sha256[:2] / subject_sha256

    def _index_receipt(self, subject_sha256: str, receipt_sha256: str) -> None:
        directory = self._subject_index_dir(subject_sha256)
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / receipt_sha256
        if not marker.exists():
            marker.touch()

    def put_json(self, kind: str, value: Mapping[str, Any]) -> ArtifactRef:
        """Store non-IR evidence after portable JSON validation."""

        return self._put(kind, dict(value))

    def put_bytes(self, kind: str, value: bytes, *, media_type: str) -> ArtifactRef:
        """Store immutable non-JSON evidence such as video, HDF5, or images."""

        if not isinstance(value, bytes):
            raise TypeError("blob value must be bytes")
        if "/" not in media_type or media_type == "application/json":
            raise ValueError("blob media_type must be non-JSON and contain '/'")
        digest = sha256_bytes(value)
        destination = self._path(digest, media_type)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != value:
                raise ArtifactIntegrityError(f"content collision at {digest}")
        else:
            atomic_write_bytes(destination, value)
        return ArtifactRef(kind=kind, sha256=digest, media_type=media_type)

    def _put(self, kind: str, value: Mapping[str, Any]) -> ArtifactRef:
        encoded = canonical_json_bytes(value)
        digest = sha256_bytes(encoded)
        destination = self._path(digest, "application/json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = destination.read_bytes()
            if existing != encoded:
                raise ArtifactIntegrityError(f"content collision at {digest}")
        else:
            atomic_write_bytes(destination, encoded)
        return ArtifactRef(kind=kind, sha256=digest)

    def load(self, reference: ArtifactRef) -> dict[str, Any]:
        path = self._path(reference.sha256, reference.media_type)
        try:
            encoded = path.read_bytes()
        except OSError as exc:
            raise ArtifactIntegrityError(f"missing artifact {reference.sha256}") from exc
        actual = sha256_bytes(encoded)
        if actual != reference.sha256:
            raise ArtifactIntegrityError(
                f"artifact digest mismatch: expected {reference.sha256}, got {actual}"
            )
        if reference.media_type != "application/json":
            raise ArtifactIntegrityError(
                f"artifact {reference.sha256} is {reference.media_type}; use load_bytes"
            )
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ArtifactIntegrityError(f"artifact {reference.sha256} is not JSON") from exc
        if not isinstance(value, dict):
            raise ArtifactIntegrityError(f"artifact {reference.sha256} is not an object")
        if reference.kind in {item.value for item in IRKind}:
            validate_artifact(value)
        elif reference.kind == "gate-receipt":
            validate_gate_receipt(value)
        return value

    def load_bytes(self, reference: ArtifactRef) -> bytes:
        path = self._path(reference.sha256, reference.media_type)
        try:
            encoded = path.read_bytes()
        except OSError as exc:
            raise ArtifactIntegrityError(f"missing artifact {reference.sha256}") from exc
        actual = sha256_bytes(encoded)
        if actual != reference.sha256:
            raise ArtifactIntegrityError(
                f"artifact digest mismatch: expected {reference.sha256}, got {actual}"
            )
        return encoded

    def document(self, reference: ArtifactRef) -> ArtifactDocument:
        return ArtifactDocument.from_dict(self.load(reference))

    def receipt(self, reference: ArtifactRef) -> GateReceipt:
        return GateReceipt.from_dict(self.load(reference))

    def _path(self, digest: str, media_type: str) -> Path:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SchemaError("artifact digest must be lowercase SHA-256")
        suffix = ".json" if media_type == "application/json" else ".blob"
        return self.root / "sha256" / digest[:2] / f"{digest}{suffix}"
