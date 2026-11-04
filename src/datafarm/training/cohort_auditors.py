"""PI0.5 cohort-admission auditor and certificate claim builder.

The generic certificate machinery lives in ``datafarm.core.admission``. This
module interprets the evidence behind a dataset built by
``tools/build_pi05_dataset.py`` and re-derives, from raw bytes only, which
samples the training bundle admits: a ``datafarm.pi05-semantic-allowlist/1.1.0``
allowlist plus the dense semantic audit report it was cut from, optionally the
PI0.5 candidate manifest.

The certificate also binds the built LeRobot dataset: its ``lerobot-provenance.json``
and ``info.json`` bytes and, optionally, a digest manifest of every file.

Stdlib only, so certificates can be audited without the training stack.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from datafarm.common import is_sha256, sha256_bytes
from datafarm.core.admission import (
    ALLOWLIST_ROLE,
    FILES_ROLE,
    FILE_MANIFEST_SCHEMA,
    PROVENANCE_ROLE,
    AdmissionError,
    AdmittedMember,
    CohortAdmissionClaim,
    GrantedWaiver,
)
from datafarm.core.model import ArtifactRef

DENSE_ALLOWLIST_AUDITOR_ID = "datafarm-pi05-dense-allowlist-auditor-v1"
DENSE_ALLOWLIST_SCHEMA = "datafarm.pi05-semantic-allowlist/1.1.0"
DENSE_AUDIT_SCHEMA = "datafarm_dense_semantic_audit/1.1.0"
PROVENANCE_SCHEMA = "datafarm.pi05-lerobot-provenance/1.0.0"

INFO_ROLE = "dataset-info"
BUNDLE_ROLES = frozenset({PROVENANCE_ROLE, INFO_ROLE, FILES_ROLE})
DENSE_AUDIT_ROLE = "dense-semantic-audit"
CANDIDATE_ROLE = "candidate-manifest"
DISTRIBUTION_ROLE = "distribution-acceptance"
_PARQUET = re.compile(r"^data/chunk-\d+/episode_\d+\.parquet$")


@dataclass
class CohortDerivation:
    auditor_id: str
    members: tuple[AdmittedMember, ...] = ()
    allowed_failed_checks: tuple[str, ...] = ()
    granted_waivers: tuple[GrantedWaiver, ...] = ()
    repo_id: str | None = None
    episodes: int | None = None
    frames: int | None = None
    source_manifest_sha256: str | None = None
    checks: dict[str, bool] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def check(self, name: str, passed: bool, note: str = "") -> None:
        self.checks[name] = bool(passed)
        if not passed and note:
            self.notes[name] = note

    @property
    def failures(self) -> list[str]:
        return [
            f"{name}: {self.notes.get(name, 'failed')}"
            for name, passed in sorted(self.checks.items())
            if not passed
        ]


def _json_object(raw: bytes | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _bool_checks(value: Any) -> dict[str, bool] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    if any(not isinstance(key, str) or type(item) is not bool for key, item in value.items()):
        return None
    return dict(value)


def _member_digests(detail: Mapping[str, Any], generated: str | None) -> dict[str, str]:
    digests: dict[str, str] = {}
    if is_sha256(generated):
        digests["generated_video_sha256"] = str(generated)
    for key in ("training_video_sha256", "state_sha256", "canonical_hdf5_sha256"):
        if is_sha256(detail.get(key)):
            digests[key] = str(detail[key])
    return digests


def _provenance_checks(
    derivation: CohortDerivation,
    evidence: Mapping[str, bytes],
    member_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Check the dataset provenance and info; returns provenance details by sample id."""

    provenance = _json_object(evidence.get(PROVENANCE_ROLE))
    info = _json_object(evidence.get(INFO_ROLE))
    derivation.check("provenance_parsed", provenance is not None, "dataset provenance is not a JSON object")
    derivation.check("info_parsed", info is not None, "dataset info.json is not a JSON object")
    provenance = provenance or {}
    info = info or {}
    derivation.repo_id = provenance.get("repo_id") if isinstance(provenance.get("repo_id"), str) else None
    derivation.episodes = provenance.get("episodes") if type(provenance.get("episodes")) is int else None
    derivation.frames = provenance.get("frames") if type(provenance.get("frames")) is int else None
    source = provenance.get("source_manifest_sha256")
    derivation.source_manifest_sha256 = source if is_sha256(source) else None

    derivation.check(
        "provenance_schema_supported",
        provenance.get("schema_version") == PROVENANCE_SCHEMA,
        f"unsupported provenance schema {provenance.get('schema_version')!r}",
    )
    derivation.check(
        "provenance_training_eligible",
        provenance.get("training_eligible") is True,
        "provenance does not declare training_eligible=true",
    )
    allowlist_digest = sha256_bytes(evidence.get(ALLOWLIST_ROLE, b""))
    derivation.check(
        "provenance_binds_allowlist",
        provenance.get("semantic_allowlist_sha256") == allowlist_digest,
        "provenance semantic_allowlist_sha256 is not the certified allowlist",
    )
    declared = provenance.get("allowed_semantic_waivers")
    derivation.check(
        "provenance_waiver_policy_matches",
        isinstance(declared, list) and sorted(declared) == list(derivation.allowed_failed_checks),
        f"provenance allowed_semantic_waivers={declared!r} but the evidence grants "
        f"{list(derivation.allowed_failed_checks)!r}",
    )
    details = provenance.get("episodes_detail") if isinstance(provenance.get("episodes_detail"), list) else []
    detail_ids = [str(item.get("sample_id")) for item in details if isinstance(item, Mapping)]
    by_id = {str(item.get("sample_id")): item for item in details if isinstance(item, Mapping)}
    derivation.check(
        "provenance_members_match",
        bool(details)
        and len(detail_ids) == len(details) == len(set(detail_ids))
        and set(detail_ids) == set(member_ids)
        and derivation.episodes == len(member_ids),
        f"provenance episodes {sorted(set(detail_ids) ^ set(member_ids))[:5]} differ from the admitted set "
        f"(provenance episodes={derivation.episodes}, admitted={len(member_ids)})",
    )
    derivation.check(
        "provenance_generated_only",
        bool(details) and all(item.get("image_source") == "generated" for item in by_id.values()),
        "provenance contains a non-generated image source",
    )
    derivation.check(
        "info_matches_provenance",
        info.get("total_episodes") == derivation.episodes
        and info.get("total_frames") == derivation.frames
        and derivation.frames is not None,
        "info.json total_episodes/total_frames disagree with provenance",
    )
    if DISTRIBUTION_ROLE in evidence:
        derivation.check(
            "distribution_acceptance_bound",
            provenance.get("distribution_acceptance_sha256") == sha256_bytes(evidence[DISTRIBUTION_ROLE]),
            "provenance distribution_acceptance_sha256 is not the certified receipt",
        )
    if FILES_ROLE in evidence:
        manifest = _json_object(evidence[FILES_ROLE]) or {}
        files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
        names = {item.get("name"): item.get("sha256") for item in files if isinstance(item, Mapping)}
        parquet = [name for name in names if isinstance(name, str) and _PARQUET.fullmatch(name)]
        derivation.check(
            "dataset_files_cover_bundle",
            manifest.get("schema") == FILE_MANIFEST_SCHEMA
            and len(names) == len(files)
            and len(parquet) == derivation.episodes
            and names.get("meta/lerobot-provenance.json") == sha256_bytes(evidence[PROVENANCE_ROLE])
            and names.get("meta/info.json") == sha256_bytes(evidence[INFO_ROLE]),
            "dataset file manifest does not cover the certified provenance/info/episodes",
        )
    return by_id


def derive_dense_cohort(evidence: Mapping[str, bytes]) -> CohortDerivation:
    """Re-derive a zero-waiver (or explicit-waiver) dense-allowlist cohort."""

    derivation = CohortDerivation(DENSE_ALLOWLIST_AUDITOR_ID)
    allowlist = _json_object(evidence.get(ALLOWLIST_ROLE))
    derivation.check("allowlist_parsed", allowlist is not None, "allowlist is not a JSON object")
    allowlist = allowlist or {}
    derivation.check(
        "allowlist_schema_supported",
        allowlist.get("schema_version") == DENSE_ALLOWLIST_SCHEMA,
        f"unsupported allowlist schema {allowlist.get('schema_version')!r}",
    )
    policy = allowlist.get("waiver_policy") if isinstance(allowlist.get("waiver_policy"), Mapping) else {}
    allowed = policy.get("allowed_failed_checks")
    policy_ok = isinstance(allowed, list) and all(isinstance(item, str) and item for item in allowed)
    derivation.check("allowlist_waiver_policy_declared", policy_ok, "waiver_policy.allowed_failed_checks is missing")
    allowed_set = sorted(set(allowed)) if policy_ok else []
    derivation.allowed_failed_checks = tuple(allowed_set)

    reviews = allowlist.get("reviews") if isinstance(allowlist.get("reviews"), list) else []
    ids = [item.get("sample_id") for item in reviews if isinstance(item, Mapping)]
    unique = (
        bool(reviews)
        and len(ids) == len(reviews)
        and all(isinstance(item, str) and item for item in ids)
        and len(set(ids)) == len(ids)
    )
    derivation.check("allowlist_members_unique", unique, "allowlist reviews are empty or repeat a sample_id")
    derivation.check(
        "allowlist_reviews_accepted",
        bool(reviews) and all(isinstance(item, Mapping) and item.get("verdict") == "accept" for item in reviews),
        "a review in the allowlist is not an accept",
    )
    granted: list[GrantedWaiver] = []
    checks_ok = True
    stray_waivers = False
    media: dict[str, str] = {}
    review_checks: dict[str, dict[str, bool]] = {}
    for review in reviews:
        if not isinstance(review, Mapping):
            checks_ok = False
            continue
        sample = str(review.get("sample_id"))
        media[sample] = str(review.get("generated_video_sha256"))
        checks = _bool_checks(review.get("checks"))
        if checks is None:
            checks_ok = False
            continue
        review_checks[sample] = checks
        waivers = review.get("waivers")
        if waivers in (None, [], {}):
            waivers = {}
        if not isinstance(waivers, Mapping):
            stray_waivers = True
            checks_ok = False
            continue
        if waivers and not allowed_set:
            stray_waivers = True
        for name, passed in checks.items():
            if passed:
                continue
            grant = waivers.get(name)
            approved = (
                isinstance(grant, Mapping)
                and grant.get("status") == "approved"
                and name in allowed_set
                and all(isinstance(grant.get(key), str) and grant.get(key) for key in ("authorized_by", "reason"))
            )
            if not approved:
                checks_ok = False
                continue
            granted.append(
                GrantedWaiver(
                    sample_id=sample,
                    check=name,
                    authorized_by=str(grant["authorized_by"]),
                    scope=str(grant.get("scope") or ""),
                    reason=str(grant["reason"]),
                )
            )
    derivation.check(
        "allowlist_checks_pass_or_waived",
        checks_ok and bool(reviews),
        "a review has a failed check without an approved waiver in the allowlist policy",
    )
    derivation.check(
        "waiver_policy_consistent",
        not stray_waivers and (bool(allowed_set) or not granted),
        "waivers appear although the allowlist declares a zero-waiver policy",
    )
    derivation.check(
        "allowlist_media_digests",
        bool(media) and all(is_sha256(value) for value in media.values()),
        "an allowlist review lacks a generated_video_sha256",
    )
    derivation.granted_waivers = tuple(granted)
    member_ids = sorted(str(item) for item in ids if isinstance(item, str))

    by_id = _provenance_checks(derivation, evidence, member_ids)
    derivation.check(
        "provenance_media_match",
        bool(by_id) and all(by_id.get(sample, {}).get("generated_video_sha256") == media.get(sample) for sample in member_ids),
        "provenance generated_video_sha256 differs from the reviewed media",
    )

    audit = _json_object(evidence.get(DENSE_AUDIT_ROLE))
    derivation.check(
        "dense_audit_supported",
        audit is not None and audit.get("schema_version") == DENSE_AUDIT_SCHEMA,
        "dense semantic audit report is missing or has an unsupported schema",
    )
    records = audit.get("records") if audit and isinstance(audit.get("records"), list) else []
    by_record = {}
    duplicate_records = False
    for record in records:
        if isinstance(record, Mapping):
            key = str(record.get("sample_id"))
            duplicate_records |= key in by_record
            by_record[key] = record
    waived_by_member: dict[str, set[str]] = {}
    for item in granted:
        waived_by_member.setdefault(item.sample_id, set()).add(item.check)
    audit_ok = bool(records) and not duplicate_records
    for sample in member_ids:
        record = by_record.get(sample)
        if record is None or record.get("verdict") != "accept":
            audit_ok = False
            continue
        failures = record.get("failures") or []
        if not isinstance(failures, list) or not set(map(str, failures)) <= waived_by_member.get(sample, set()):
            audit_ok = False
        if _bool_checks(record.get("checks")) != review_checks.get(sample):
            audit_ok = False
    derivation.check(
        "dense_audit_admits_members",
        audit_ok,
        "a member is not an accept with identical checks in the dense semantic audit",
    )

    if CANDIDATE_ROLE in evidence:
        candidate = _json_object(evidence[CANDIDATE_ROLE]) or {}
        rows = candidate.get("records") if isinstance(candidate.get("records"), list) else []
        by_candidate = {str(row.get("sample_id")): row for row in rows if isinstance(row, Mapping)}
        candidate_ok = bool(rows) and all(
            by_candidate.get(sample, {}).get("technical_verdict") == "PASS"
            and by_candidate.get(sample, {}).get("provider_video_sha256") == media.get(sample)
            for sample in member_ids
        )
        source = candidate.get("source_dataset_sha256")
        if source is not None and derivation.source_manifest_sha256 is not None:
            candidate_ok = candidate_ok and source == derivation.source_manifest_sha256
        derivation.check(
            "candidate_manifest_admits_members",
            candidate_ok,
            "a member is not a technical PASS with the reviewed media in the candidate manifest",
        )

    derivation.members = tuple(
        AdmittedMember(sample, _member_digests(by_id.get(sample, {}), media.get(sample)))
        for sample in member_ids
    )
    return derivation


class Pi05CohortAuditor:
    """Re-derives a PI0.5 cohort and compares it with a certificate payload."""

    def __init__(
        self,
        component_id: str,
        derive: Callable[[Mapping[str, bytes]], CohortDerivation],
        required_roles: frozenset[str],
        optional_roles: frozenset[str],
    ) -> None:
        self.component_id = component_id
        self.derive = derive
        self.required_roles = required_roles
        self.optional_roles = optional_roles

    def audit(self, payload: Mapping[str, Any], evidence: Mapping[str, bytes]) -> dict[str, bool]:
        roles = set(evidence)
        checks = {
            "evidence_roles_complete": self.required_roles <= roles
            and roles <= self.required_roles | self.optional_roles
        }
        derivation = self.derive(evidence)
        checks.update(derivation.checks)
        admitted = payload.get("admitted_set") or {}
        waivers = payload.get("waivers") or {}
        bundle = payload.get("training_bundle") or {}
        episode = payload.get("episode") or {}
        expected_members = [item.to_dict() for item in sorted(derivation.members, key=lambda m: m.sample_id)]
        expected_granted = sorted(
            (item.to_dict() for item in derivation.granted_waivers),
            key=lambda item: (item["sample_id"], item["check"]),
        )
        checks["certificate_members_match_evidence"] = admitted.get("members") == expected_members
        checks["certificate_waivers_match_evidence"] = (
            waivers.get("allowed_failed_checks") == list(derivation.allowed_failed_checks)
            and waivers.get("granted") == expected_granted
        )
        checks["certificate_bundle_matches_evidence"] = (
            bundle.get("repo_id") == derivation.repo_id
            and bundle.get("episodes") == derivation.episodes
            and bundle.get("frames") == derivation.frames
        )
        checks["certificate_source_manifest_matches"] = (
            episode.get("source_manifest_sha256") == derivation.source_manifest_sha256
        )
        return checks


DENSE_ALLOWLIST_AUDITOR = Pi05CohortAuditor(
    DENSE_ALLOWLIST_AUDITOR_ID,
    derive_dense_cohort,
    frozenset({ALLOWLIST_ROLE, PROVENANCE_ROLE, INFO_ROLE, DENSE_AUDIT_ROLE}),
    frozenset({CANDIDATE_ROLE, DISTRIBUTION_ROLE, FILES_ROLE}),
)
PI05_COHORT_AUDITORS: dict[str, Pi05CohortAuditor] = {
    DENSE_ALLOWLIST_AUDITOR.component_id: DENSE_ALLOWLIST_AUDITOR,
}


def detect_auditor(allowlist: bytes) -> Pi05CohortAuditor:
    value = _json_object(allowlist) or {}
    schema = str(value.get("schema_version"))
    if schema == DENSE_ALLOWLIST_SCHEMA:
        return DENSE_ALLOWLIST_AUDITOR
    raise AdmissionError(
        f"no PI0.5 admission auditor supports allowlist schema {schema!r}; supported: {DENSE_ALLOWLIST_SCHEMA}"
    )


def build_claim(
    auditor: Pi05CohortAuditor,
    refs: Mapping[str, ArtifactRef],
    evidence: Mapping[str, bytes],
) -> tuple[CohortAdmissionClaim, CohortDerivation]:
    """Build the certificate compiler's claim from stored evidence.

    The claim records the derivation's own checks as producer checks; the
    admission root re-derives everything independently before signing.
    """

    if set(refs) != set(evidence):
        raise AdmissionError("evidence refs and bytes must name the same roles")
    derivation = auditor.derive(evidence)
    if not derivation.members:
        raise AdmissionError("no admitted members could be derived from the evidence", derivation.failures)
    if derivation.repo_id is None or derivation.episodes is None:
        raise AdmissionError("dataset provenance lacks repo_id/episodes", derivation.failures)
    claim = CohortAdmissionClaim(
        auditor_id=auditor.component_id,
        members=derivation.members,
        allowlist=refs[ALLOWLIST_ROLE],
        reports={role: ref for role, ref in refs.items() if role != ALLOWLIST_ROLE and role not in BUNDLE_ROLES},
        bundle={role: ref for role, ref in refs.items() if role in BUNDLE_ROLES},
        repo_id=derivation.repo_id,
        episodes=derivation.episodes,
        frames=derivation.frames,
        allowed_failed_checks=derivation.allowed_failed_checks,
        granted_waivers=derivation.granted_waivers,
        source_manifest_sha256=derivation.source_manifest_sha256,
        producer_checks=dict(derivation.checks),
    )
    return claim, derivation
