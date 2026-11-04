#!/usr/bin/env python3
"""Issue a PI0.5 training Admission Certificate for a built dataset.

Reads the dataset's ``meta/lerobot-provenance.json`` and ``meta/info.json``,
the semantic allowlist it pins, and the audit reports the allowlist was cut
from; copies their raw bytes into a content-addressed ``ArtifactStore``;
digests every dataset file; compiles an ``admission-certificate`` document
(producer ``datafarm-training-certificate-compiler-v1``); has the independent
admission root (``datafarm-training-admission-root-v1``) re-derive the claim from
the stored bytes and sign a gate receipt; and prints the certificate digest.

Exit status: 0 admitted, 2 rejected (the certificate and its rejecting receipt
are still stored so the attempt stays on record), 1 usage/evidence error.

Evidence: the semantic allowlist (``datafarm.pi05-semantic-allowlist/1.1.0``) and its
dense semantic audit, with the candidate manifest when present. Their defaults are the
siblings ``semantic-audit.json`` and ``pi05-candidate-manifest.json`` of the allowlist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datafarm.training.cohort_auditors import (  # noqa: E402
    CANDIDATE_ROLE,
    DENSE_AUDIT_ROLE,
    DISTRIBUTION_ROLE,
    INFO_ROLE,
    PI05_COHORT_AUDITORS,
    build_claim,
    detect_auditor,
)
from datafarm.core.admission import (  # noqa: E402
    ALLOWLIST_ROLE,
    FILES_ROLE,
    PROVENANCE_ROLE,
    RAW_EVIDENCE_MEDIA_TYPE,
    AdmissionError,
    compute_file_manifest,
    issue_certificate,
)
from datafarm.core.store import ArtifactStore  # noqa: E402


def _json(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise AdmissionError("evidence must be a JSON object")
    return value


def _optional_path(explicit: Path | None, fallback: Any, *, required: bool, role: str) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise AdmissionError(f"{role} evidence {explicit} does not exist")
        return explicit
    if isinstance(fallback, str) and fallback and Path(fallback).is_file():
        return Path(fallback)
    if isinstance(fallback, Path) and fallback.is_file():
        return fallback
    if required:
        raise AdmissionError(f"{role} evidence is required; pass it explicitly")
    return None


def collect_evidence(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, bytes]]:
    dataset = args.dataset_root
    provenance_path = dataset / "meta" / "lerobot-provenance.json"
    info_path = dataset / "meta" / "info.json"
    if not provenance_path.is_file() or not info_path.is_file():
        raise AdmissionError(f"{dataset} is not a built dataset (meta/lerobot-provenance.json, meta/info.json)")
    provenance = _json(provenance_path.read_bytes())
    allowlist_path = _optional_path(args.semantic_allowlist, provenance.get("semantic_allowlist"),
                                    required=True, role=ALLOWLIST_ROLE)
    sources: dict[str, Path] = {PROVENANCE_ROLE: provenance_path, INFO_ROLE: info_path, ALLOWLIST_ROLE: allowlist_path}
    detect_auditor(allowlist_path.read_bytes())
    sibling = allowlist_path.parent
    sources[DENSE_AUDIT_ROLE] = _optional_path(args.dense_audit, sibling / "semantic-audit.json",
                                               required=True, role=DENSE_AUDIT_ROLE)
    candidate = _optional_path(args.candidate_manifest, sibling / "pi05-candidate-manifest.json",
                               required=False, role=CANDIDATE_ROLE)
    if candidate is not None:
        sources[CANDIDATE_ROLE] = candidate
    distribution = _optional_path(args.distribution_acceptance, provenance.get("distribution_acceptance"),
                                  required=False, role=DISTRIBUTION_ROLE)
    if distribution is not None:
        sources[DISTRIBUTION_ROLE] = distribution
    return sources, {role: path.read_bytes() for role, path in sources.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, required=True, help="ArtifactStore root that receives the certificate")
    parser.add_argument("--dataset-root", type=Path, required=True, help="built LeRobot dataset to admit")
    parser.add_argument("--semantic-allowlist", type=Path, help="default: provenance semantic_allowlist")
    parser.add_argument("--dense-audit", type=Path, help="default: the allowlist's sibling semantic-audit.json")
    parser.add_argument("--candidate-manifest", type=Path,
                        help="default: the allowlist's sibling pi05-candidate-manifest.json")
    parser.add_argument("--distribution-acceptance", type=Path, help="default: provenance distribution_acceptance")
    parser.add_argument("--no-file-manifest", action="store_true",
                        help="do not bind a digest of every dataset file (not recommended)")
    parser.add_argument("--output", type=Path,
                        help="also write the canonical certificate bytes here (usable as --admission-certificate)")
    parser.add_argument("--created-at", help="RFC3339 UTC timestamp for the certificate (default: now)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sources, evidence = collect_evidence(args)
        store = ArtifactStore(args.store)
        refs = {
            role: store.put_bytes(role, raw, media_type=RAW_EVIDENCE_MEDIA_TYPE)
            for role, raw in sorted(evidence.items())
        }
        if not args.no_file_manifest:
            manifest = compute_file_manifest(args.dataset_root)
            refs[FILES_ROLE] = store.put_json("dataset-file-manifest", manifest)
            evidence[FILES_ROLE] = store.load_bytes(refs[FILES_ROLE])
        auditor = detect_auditor(evidence[ALLOWLIST_ROLE])
        claim, derivation = build_claim(auditor, refs, evidence)
        issued = issue_certificate(store, claim, PI05_COHORT_AUDITORS, created_at=args.created_at)
    except (AdmissionError, OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    certificate_bytes = store.load_bytes(issued.certificate)
    if args.output is not None and issued.admitted:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_bytes(certificate_bytes)
        temporary.replace(args.output)
    report = {
        "status": "admitted" if issued.admitted else "rejected",
        "certificate_sha256": issued.certificate.sha256,
        "receipt_sha256": issued.receipt.sha256,
        "store": str(store.root),
        "auditor": auditor.component_id,
        "members": len(claim.members),
        "waiver_policy": "zero-waiver" if not claim.allowed_failed_checks else "explicit-waivers",
        "repo_id": claim.repo_id,
        "evidence": {role: {"source": str(path), "sha256": refs[role].sha256} for role, path in sources.items()},
        "failed_checks": sorted(name for name, value in issued.checks.items() if not value),
        "evidence_notes": derivation.failures,
        "output": str(args.output) if args.output is not None and issued.admitted else None,
    }
    if FILES_ROLE in refs:
        report["evidence"][FILES_ROLE] = {"source": str(args.dataset_root), "sha256": refs[FILES_ROLE].sha256}
    print(json.dumps(report, indent=2))
    return 0 if issued.admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
