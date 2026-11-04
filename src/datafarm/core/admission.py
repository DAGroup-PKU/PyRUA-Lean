"""Cohort training-admission certificates: issue and verify.

A policy-training launch consumes one admitted *cohort*: a training bundle
(dataset) built from an exact set of samples that passed the render/semantic
verifiers. A bare digest of a semantic allowlist file would only prove that "a
64-hex string was present", so this module makes admission a real artifact:

* the **certificate** is an ``admission-certificate`` IR document produced by a certificate
  compiler. Its payload binds the admitted member ids (and per-member media
  digests), the allowlist digest, the digests of the audit reports it relies
  on, the training-bundle provenance digest, and an explicit waiver policy;
* the raw bytes of every piece of evidence are stored in the content-addressed
  ``ArtifactStore`` next to the certificate, so verification never depends on
  mutable run directories;
* an independent **admission root** (component id different from the
  producer, authority ``admission-root``) re-derives the claim from the stored
  evidence with a task-specific *auditor* and signs a gate receipt;
* ``verify_certificate`` is what a launcher calls: it loads the certificate by
  digest, requires a passing admission-root receipt, re-runs the root checks
  against the stored evidence, and checks the binding to the launch at hand.

The module is stdlib-only and task-agnostic; evidence formats are interpreted
by auditors supplied by adapters (for PI0.5: ``datafarm.training.cohort_auditors``).
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from datafarm.common import is_sha256, sha256_bytes, utc_now
from datafarm.core.model import (
    ArtifactDocument,
    ArtifactRef,
    Authority,
    GateReceipt,
    IRKind,
    StageName,
    Verdict,
)
from datafarm.core.schema import SchemaError, canonical_json_bytes, document_sha256
from datafarm.core.store import ArtifactIntegrityError, ArtifactStore

TRAINING_ADMISSION_PROFILE = "datafarm.training-admission/1.0.0"
CERTIFICATE_COMPILER_ID = "datafarm-training-certificate-compiler-v1"
ADMISSION_ROOT_ID = "datafarm-training-admission-root-v1"
TRUSTED_ADMISSION_ROOTS = frozenset({ADMISSION_ROOT_ID})
RAW_EVIDENCE_MEDIA_TYPE = "application/vnd.datafarm.evidence+json"
FILE_MANIFEST_SCHEMA = "datafarm.dataset-file-manifest"
ZERO_WAIVER = "zero-waiver"
EXPLICIT_WAIVERS = "explicit-waivers"
PROVENANCE_ROLE = "dataset-provenance"
ALLOWLIST_ROLE = "allowlist"
FILES_ROLE = "dataset-files"

_ID_UNSAFE = re.compile(r"[^a-z0-9._:-]+")


class AdmissionError(ValueError):
    """A certificate is missing, malformed, unsigned, or not bound to the launch."""

    def __init__(self, message: str, failures: Sequence[str] = ()):
        self.failures = tuple(failures)
        detail = f": {'; '.join(self.failures)}" if self.failures else ""
        super().__init__(f"{message}{detail}")


# --------------------------------------------------------------------- claim


@dataclass(frozen=True)
class AdmittedMember:
    sample_id: str
    digests: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"sample_id": self.sample_id, "digests": dict(sorted(self.digests.items()))}


@dataclass(frozen=True)
class GrantedWaiver:
    sample_id: str
    check: str
    authorized_by: str
    scope: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "check": self.check,
            "authorized_by": self.authorized_by,
            "scope": self.scope,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CohortAdmissionClaim:
    """What a certificate compiler asserts about one training bundle."""

    auditor_id: str
    members: tuple[AdmittedMember, ...]
    allowlist: ArtifactRef
    reports: Mapping[str, ArtifactRef]
    bundle: Mapping[str, ArtifactRef]
    repo_id: str
    episodes: int
    frames: int | None
    allowed_failed_checks: tuple[str, ...] = ()
    granted_waivers: tuple[GrantedWaiver, ...] = ()
    source_manifest_sha256: str | None = None
    producer_checks: Mapping[str, bool] = field(default_factory=dict)


class CohortEvidenceAuditor(Protocol):
    """Task-specific re-derivation of a cohort claim from raw evidence bytes."""

    component_id: str

    def audit(self, payload: Mapping[str, Any], evidence: Mapping[str, bytes]) -> Mapping[str, bool]:
        """Return named boolean checks; every check must hold to admit."""


def member_ids_digest(member_ids: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes(list(member_ids)))


def members_digest(members: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes([dict(item) for item in members]))


def certificate_artifact_id(repo_id: str, provenance_sha256: str) -> str:
    slug = _ID_UNSAFE.sub("-", repo_id.lower().replace("/", ":")).strip("-:.") or "bundle"
    return f"training-admission:{slug[:80]}:{provenance_sha256[:12]}"


def compile_certificate(
    claim: CohortAdmissionClaim,
    *,
    created_at: str,
    artifact_id: str,
    producer_id: str = CERTIFICATE_COMPILER_ID,
) -> ArtifactDocument:
    members = sorted(claim.members, key=lambda item: item.sample_id)
    member_dicts = [item.to_dict() for item in members]
    member_ids = [item.sample_id for item in members]
    allowed = sorted(set(claim.allowed_failed_checks))
    granted = sorted(
        (item.to_dict() for item in claim.granted_waivers),
        key=lambda item: (item["sample_id"], item["check"]),
    )
    reports = [{"role": role, "ref": ref.to_dict()} for role, ref in sorted(claim.reports.items())]
    bundle = [{"role": role, "ref": ref.to_dict()} for role, ref in sorted(claim.bundle.items())]
    payload = {
        "profile": TRAINING_ADMISSION_PROFILE,
        "auditor": claim.auditor_id,
        "episode": {
            "scope": "cohort",
            "member_count": len(members),
            "source_manifest_sha256": claim.source_manifest_sha256,
        },
        "rendered_observation": {
            "scope": "cohort",
            "member_digests_sha256": members_digest(member_dicts),
        },
        "checks": dict(sorted(claim.producer_checks.items())),
        "verdict": "admit",
        "cost": {"status": "unmeasured"},
        "training_bundle": {
            "repo_id": claim.repo_id,
            "episodes": claim.episodes,
            "frames": claim.frames,
            "artifacts": bundle,
        },
        "repair_predecessor": None,
        "admitted_set": {
            "allowlist": claim.allowlist.to_dict(),
            "member_ids_sha256": member_ids_digest(member_ids),
            "members": member_dicts,
        },
        "reports": reports,
        "waivers": {
            "policy": ZERO_WAIVER if not allowed and not granted else EXPLICIT_WAIVERS,
            "allowed_failed_checks": allowed,
            "granted": granted,
        },
    }
    evidence: list[ArtifactRef] = []
    for reference in [claim.allowlist, *claim.reports.values(), *claim.bundle.values()]:
        if reference not in evidence:
            evidence.append(reference)
    return ArtifactDocument(
        kind=IRKind.ADMISSION,
        artifact_id=artifact_id,
        created_at=created_at,
        producer=producer_id,
        predecessors=(),
        evidence=tuple(evidence),
        payload=payload,
    )


def evidence_roles(payload: Mapping[str, Any]) -> dict[str, ArtifactRef]:
    """Map every evidence role named in a certificate payload to its ref."""

    roles: dict[str, ArtifactRef] = {}

    def add(role: Any, value: Any) -> None:
        if not isinstance(role, str) or not role or role in roles:
            raise ValueError(f"evidence role {role!r} is missing or duplicated")
        roles[role] = ArtifactRef.from_dict(value)

    admitted = payload.get("admitted_set") or {}
    add(ALLOWLIST_ROLE, admitted.get("allowlist"))
    for item in payload.get("reports") or []:
        add(item.get("role"), item.get("ref"))
    for item in (payload.get("training_bundle") or {}).get("artifacts") or []:
        add(item.get("role"), item.get("ref"))
    return roles


# ---------------------------------------------------------------------- root


class TrainingAdmissionRoot:
    """Independent admission-root verifier for cohort certificates.

    It never trusts the producer's own ``checks``: it recomputes the member and
    waiver digests, re-loads every piece of evidence by digest from the store,
    and re-derives the claim with the auditor named in the payload.
    """

    def __init__(
        self,
        store: ArtifactStore,
        auditors: Mapping[str, CohortEvidenceAuditor],
        *,
        component_id: str = ADMISSION_ROOT_ID,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.store = store
        self.auditors = dict(auditors)
        self.component_id = component_id
        self.clock = clock

    def evaluate(self, subject: ArtifactDocument, subject_ref: ArtifactRef) -> dict[str, bool]:
        payload = subject.payload
        checks: dict[str, bool] = {}
        checks["certificate_digest_matches"] = document_sha256(subject) == subject_ref.sha256
        checks["certificate_profile"] = (
            subject.kind is IRKind.ADMISSION and payload.get("profile") == TRAINING_ADMISSION_PROFILE
        )
        checks["certificate_signer_independent"] = (
            subject.producer != self.component_id and subject.producer not in TRUSTED_ADMISSION_ROOTS
        )
        checks["certificate_verdict_admit"] = payload.get("verdict") == "admit"

        admitted = payload.get("admitted_set") if isinstance(payload.get("admitted_set"), Mapping) else {}
        members = admitted.get("members") if isinstance(admitted.get("members"), list) else []
        member_ids = [item.get("sample_id") for item in members if isinstance(item, Mapping)]
        members_ok = (
            bool(members)
            and len(member_ids) == len(members)
            and all(isinstance(item, str) and item for item in member_ids)
            and member_ids == sorted(set(member_ids))
            and all(
                isinstance(item.get("digests"), Mapping)
                and all(isinstance(key, str) and is_sha256(value) for key, value in item["digests"].items())
                for item in members
            )
        )
        episode = payload.get("episode") if isinstance(payload.get("episode"), Mapping) else {}
        observation = (
            payload.get("rendered_observation")
            if isinstance(payload.get("rendered_observation"), Mapping)
            else {}
        )
        checks["certificate_members_canonical"] = bool(
            members_ok
            and episode.get("member_count") == len(members)
            and admitted.get("member_ids_sha256") == member_ids_digest(member_ids)
            and observation.get("member_digests_sha256") == members_digest(members)
        )

        waivers = payload.get("waivers") if isinstance(payload.get("waivers"), Mapping) else {}
        allowed = waivers.get("allowed_failed_checks")
        granted = waivers.get("granted")
        waiver_ok = isinstance(allowed, list) and isinstance(granted, list)
        if waiver_ok:
            waiver_ok = allowed == sorted(set(allowed)) and all(isinstance(item, str) and item for item in allowed)
            member_set = set(member_ids)
            for item in granted:
                if not (
                    isinstance(item, Mapping)
                    and item.get("check") in allowed
                    and item.get("sample_id") in member_set
                    and all(isinstance(item.get(key), str) and item.get(key) for key in ("authorized_by", "reason"))
                ):
                    waiver_ok = False
            expected_policy = ZERO_WAIVER if not allowed and not granted else EXPLICIT_WAIVERS
            waiver_ok = waiver_ok and waivers.get("policy") == expected_policy
        checks["certificate_waiver_policy_consistent"] = bool(waiver_ok)

        evidence_bytes: dict[str, bytes] = {}
        try:
            roles = evidence_roles(payload)
            structure_ok = PROVENANCE_ROLE in roles and set(roles.values()) <= set(subject.evidence)
        except (KeyError, TypeError, ValueError, AttributeError):
            roles, structure_ok = {}, False
        checks["certificate_evidence_declared"] = structure_ok
        intact = bool(roles)
        for role, reference in roles.items():
            try:
                evidence_bytes[role] = self.store.load_bytes(reference)
            except (ArtifactIntegrityError, SchemaError):
                intact = False
        checks["certificate_evidence_intact"] = intact
        bundle = payload.get("training_bundle") if isinstance(payload.get("training_bundle"), Mapping) else {}
        checks["certificate_bundle_counts"] = bundle.get("episodes") == len(members) and bool(bundle.get("repo_id"))

        auditor = self.auditors.get(str(payload.get("auditor")))
        checks["certificate_auditor_known"] = auditor is not None
        if auditor is not None and intact and structure_ok:
            try:
                domain = dict(auditor.audit(payload, evidence_bytes))
            except Exception:  # an auditor crash is a rejection, never a pass
                domain = {"auditor_completed": False}
            if not domain:
                domain = {"auditor_completed": False}
            for key, value in domain.items():
                checks[f"evidence:{key}"] = value is True
        return checks

    def verify(
        self,
        subject: ArtifactDocument,
        subject_ref: ArtifactRef,
        context: Any = None,
    ) -> GateReceipt:
        checks = self.evaluate(subject, subject_ref)
        failed = sorted(name for name, value in checks.items() if not value)
        return GateReceipt(
            stage=StageName.ADMISSION,
            subject_sha256=subject_ref.sha256,
            verifier_id=self.component_id,
            authority=Authority.ADMISSION_ROOT,
            verdict=Verdict.REJECT if failed else Verdict.PASS,
            checks=checks,
            evidence=subject.evidence,
            created_at=self.clock(),
            reason=("failed checks: " + ", ".join(failed)) if failed else "cohort admitted",
        )


# -------------------------------------------------------------------- issue


@dataclass(frozen=True)
class IssuedCertificate:
    certificate: ArtifactRef
    receipt: ArtifactRef
    verdict: Verdict
    checks: Mapping[str, bool]
    reason: str

    @property
    def admitted(self) -> bool:
        return self.verdict is Verdict.PASS


def issue_certificate(
    store: ArtifactStore,
    claim: CohortAdmissionClaim,
    auditors: Mapping[str, CohortEvidenceAuditor],
    *,
    created_at: str | None = None,
    artifact_id: str | None = None,
    producer_id: str = CERTIFICATE_COMPILER_ID,
    root: TrainingAdmissionRoot | None = None,
) -> IssuedCertificate:
    """Compile, store, and root-sign a cohort certificate.

    A rejected certificate is stored together with its rejecting receipt so the
    attempt stays in the denominator; only a passing receipt admits.
    """

    provenance = claim.bundle.get(PROVENANCE_ROLE)
    if provenance is None:
        raise AdmissionError("a training-admission claim must bind the dataset provenance")
    document = compile_certificate(
        claim,
        created_at=created_at or utc_now(),
        artifact_id=artifact_id or certificate_artifact_id(claim.repo_id, provenance.sha256),
        producer_id=producer_id,
    )
    certificate = store.put_document(document)
    signer = root or TrainingAdmissionRoot(store, auditors)
    if signer.component_id == producer_id:
        raise AdmissionError("the certificate producer cannot sign its own admission")
    receipt = signer.verify(document, certificate)
    receipt_ref = store.put_receipt(receipt)
    return IssuedCertificate(certificate, receipt_ref, receipt.verdict, dict(receipt.checks), receipt.reason)


# ------------------------------------------------------------------- verify


@dataclass(frozen=True)
class AdmissionExpectation:
    """What the launch at hand requires the certificate to bind."""

    provenance_sha256: str | None = None
    allowlist_sha256: str | None = None
    repo_id: str | None = None
    allowed_failed_checks: tuple[str, ...] | None = None
    member_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class AdmissionVerification:
    certificate_sha256: str
    receipt_sha256: str
    producer_id: str
    verifier_id: str
    authority: str
    profile: str
    auditor_id: str
    repo_id: str
    training_bundle_sha256: str
    allowlist_sha256: str
    member_count: int
    member_ids_sha256: str
    waiver_policy: str
    allowed_failed_checks: tuple[str, ...]
    granted_waivers: int
    data_files: ArtifactRef | None
    checks: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "certificate_sha256": self.certificate_sha256,
            "receipt_sha256": self.receipt_sha256,
            "producer_id": self.producer_id,
            "verifier_id": self.verifier_id,
            "authority": self.authority,
            "profile": self.profile,
            "auditor_id": self.auditor_id,
            "repo_id": self.repo_id,
            "training_bundle_sha256": self.training_bundle_sha256,
            "allowlist_sha256": self.allowlist_sha256,
            "member_count": self.member_count,
            "member_ids_sha256": self.member_ids_sha256,
            "waiver_policy": self.waiver_policy,
            "allowed_failed_checks": list(self.allowed_failed_checks),
            "granted_waivers": self.granted_waivers,
            "data_files": None if self.data_files is None else self.data_files.to_dict(),
            "checks": dict(self.checks),
        }


def load_certificate(store: ArtifactStore, certificate_sha256: str) -> ArtifactDocument:
    if not is_sha256(certificate_sha256):
        raise AdmissionError("an admission certificate is referenced by a lowercase SHA-256 digest")
    reference = ArtifactRef(kind=IRKind.ADMISSION.value, sha256=certificate_sha256)
    try:
        document = store.document(reference)
    except (ArtifactIntegrityError, SchemaError, KeyError, TypeError, ValueError) as exc:
        raise AdmissionError(
            f"{certificate_sha256} is not an admission certificate in store {store.root}", [str(exc)]
        ) from exc
    if document.kind is not IRKind.ADMISSION:
        raise AdmissionError(f"{certificate_sha256} is a {document.kind.value}, not an admission certificate")
    return document


def verify_certificate(
    store: ArtifactStore,
    certificate_sha256: str,
    *,
    auditors: Mapping[str, CohortEvidenceAuditor],
    expectation: AdmissionExpectation | None = None,
    trusted_roots: frozenset[str] = TRUSTED_ADMISSION_ROOTS,
) -> AdmissionVerification:
    """Verify a stored certificate and its binding; raise ``AdmissionError``."""

    document = load_certificate(store, certificate_sha256)
    reference = ArtifactRef(kind=IRKind.ADMISSION.value, sha256=certificate_sha256)
    payload = document.payload
    failures: list[str] = []

    receipts = [
        (ref, receipt)
        for ref, receipt in store.receipts_for(certificate_sha256)
        if receipt.stage is StageName.ADMISSION
    ]
    passing = [
        (ref, receipt)
        for ref, receipt in receipts
        if receipt.verdict is Verdict.PASS
        and receipt.authority is Authority.ADMISSION_ROOT
        and receipt.verifier_id in trusted_roots
        and receipt.verifier_id != document.producer
        and all(receipt.checks.values())
    ]
    conflicting = [
        ref.sha256
        for ref, receipt in receipts
        if receipt.verifier_id in trusted_roots and receipt.verdict is not Verdict.PASS
    ]
    if not passing:
        failures.append("no passing admission-root receipt signs this certificate")
    if conflicting:
        failures.append(f"admission-root receipts reject this certificate: {sorted(conflicting)}")
    if document.producer in trusted_roots:
        failures.append("certificate producer is an admission root (self-signed)")

    fresh = TrainingAdmissionRoot(store, auditors).evaluate(document, reference)
    failures.extend(f"root check failed now: {name}" for name, value in sorted(fresh.items()) if not value)

    try:
        roles = evidence_roles(payload)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise AdmissionError("certificate evidence roles are malformed", [str(exc)]) from exc
    provenance = roles.get(PROVENANCE_ROLE)
    allowlist = roles.get(ALLOWLIST_ROLE)
    bundle = payload.get("training_bundle") or {}
    admitted = payload.get("admitted_set") or {}
    waivers = payload.get("waivers") or {}
    members = [item.get("sample_id") for item in admitted.get("members") or [] if isinstance(item, Mapping)]
    allowed = tuple(waivers.get("allowed_failed_checks") or ())

    if expectation is not None:
        if expectation.provenance_sha256 is not None and (
            provenance is None or provenance.sha256 != expectation.provenance_sha256
        ):
            failures.append("certificate does not bind this dataset's provenance digest")
        if expectation.allowlist_sha256 is not None and (
            allowlist is None or allowlist.sha256 != expectation.allowlist_sha256
        ):
            failures.append("certificate does not bind this dataset's semantic allowlist digest")
        if expectation.repo_id is not None and bundle.get("repo_id") != expectation.repo_id:
            failures.append("certificate admits a different repo_id")
        if expectation.allowed_failed_checks is not None and sorted(allowed) != sorted(
            expectation.allowed_failed_checks
        ):
            failures.append("certificate waiver policy differs from the launch's allowed waivers")
        if expectation.member_ids is not None and sorted(members) != sorted(expectation.member_ids):
            failures.append("certificate member set differs from the dataset's episodes")

    if failures:
        raise AdmissionError(f"admission certificate {certificate_sha256} does not verify", failures)
    receipt_ref, receipt = passing[0]
    return AdmissionVerification(
        certificate_sha256=certificate_sha256,
        receipt_sha256=receipt_ref.sha256,
        producer_id=document.producer,
        verifier_id=receipt.verifier_id,
        authority=receipt.authority.value,
        profile=str(payload.get("profile")),
        auditor_id=str(payload.get("auditor")),
        repo_id=str(bundle.get("repo_id")),
        training_bundle_sha256=provenance.sha256 if provenance else "",
        allowlist_sha256=allowlist.sha256 if allowlist else "",
        member_count=len(members),
        member_ids_sha256=str(admitted.get("member_ids_sha256")),
        waiver_policy=str(waivers.get("policy")),
        allowed_failed_checks=allowed,
        granted_waivers=len(waivers.get("granted") or []),
        data_files=roles.get(FILES_ROLE),
        checks=dict(fresh),
    )


# ------------------------------------------------------------- file manifest


def compute_file_manifest(root: str | Path) -> dict[str, Any]:
    """Digest every regular file under a dataset root (relative names only).

    The dataset directory must be immutable after certification: any added,
    removed, or changed file changes this manifest and fails verification.
    """

    base = Path(root)
    if not base.is_dir():
        raise AdmissionError(f"dataset root {base} does not exist")
    files = []
    for directory, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames.sort()
        for name in dirnames:
            if (Path(directory) / name).is_symlink():
                raise AdmissionError(f"dataset contains a symlinked directory: {name}")
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                raise AdmissionError(f"dataset contains a non-regular file: {path.relative_to(base)}")
            digest = hashlib.sha256()
            size = 0
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(block)
                    size += len(block)
            files.append({"name": path.relative_to(base).as_posix(), "bytes": size, "sha256": digest.hexdigest()})
    if not files:
        raise AdmissionError(f"dataset root {base} is empty")
    files.sort(key=lambda item: item["name"])
    return {"schema": FILE_MANIFEST_SCHEMA, "schema_version": "1.0.0", "files": files}


def file_manifest_digest(manifest: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(manifest)))
