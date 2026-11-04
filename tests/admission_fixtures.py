"""Synthetic PI0.5 admission evidence shaped like real pipeline outputs.

Mirrors the formats of a production run: a
``datafarm.pi05-semantic-allowlist/1.1.0`` allowlist, the
``datafarm_dense_semantic_audit/1.1.0`` report it was cut from, the PI0.5
candidate manifest, and a built LeRobot dataset whose
``meta/lerobot-provenance.json`` pins the allowlist.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CHECKS = ("dense_three_panel_integrity", "opening_and_terminal_topology", "strict_all_seams")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(data)
    return data


def write_dense_cohort(
    root: Path,
    *,
    members: int = 3,
    failed_check: str | None = None,
    waiver_policy: list[str] | None = None,
    provenance_waivers: list[str] | None = None,
    repo_id: str = "example/piper_fixture_inline224_v1",
) -> dict[str, Path]:
    """Write a cohort; ``failed_check`` fails that check on the first member."""

    root = Path(root)
    ids = [f"fixture-sample-{index:03d}" for index in range(members)]
    media = {sample: sha(f"generated-{sample}".encode()) for sample in ids}
    policy = list(waiver_policy or [])
    reviews, records = [], []
    for index, sample in enumerate(ids):
        checks = {name: True for name in CHECKS}
        waivers: Any = []
        failures: list[str] = []
        if failed_check and index == 0:
            checks[failed_check] = False
            failures = [failed_check]
            if failed_check in policy:
                waivers = {failed_check: {"status": "approved", "authorized_by": "user",
                                          "scope": "fixture", "reason": "fixture waiver"}}
        reviews.append({"checks": checks, "generated_video_sha256": media[sample], "index": index,
                        "sample_id": sample, "verdict": "accept", "waivers": waivers})
        records.append({"checks": checks, "failures": failures, "index": index, "ordinal": index,
                        "sample_id": sample, "verdict": "accept"})
    audit_dir = root / "state" / "dense-audit-round-0001"
    allowlist = audit_dir / "pi05-semantic-allowlist-zero-waiver.json"
    write_json(allowlist, {"created_at": "2026-09-24T14:10:49Z",
                           "reviews": reviews,
                           "schema_version": "datafarm.pi05-semantic-allowlist/1.1.0",
                           "status": "passed_semantic100",
                           "waiver_policy": {"allowed_failed_checks": policy}})
    write_json(audit_dir / "semantic-audit.json", {
        "counts": {"accept": members}, "created_at": "2026-09-24T14:10:49Z", "failure_counts": {},
        "records": records, "rejected_indices": [],
        "schema_version": "datafarm_dense_semantic_audit/1.1.0", "status": "passed_semantic100",
    })
    source_manifest_sha = sha(b"white100-source-manifest")
    write_json(audit_dir / "pi05-candidate-manifest.json", {
        "record_count": members, "schema_version": "pi05_candidate/1.0.0",
        "source_dataset_sha256": source_manifest_sha, "technical_pass_count": members,
        "records": [{"sample_id": sample, "technical_verdict": "PASS",
                     "provider_video_sha256": media[sample],
                     "training_video_sha256": sha(f"training-{sample}".encode())} for sample in ids],
    })
    distribution = root / "distribution-acceptance.json"
    distribution_bytes = write_json(distribution, {"status": "accepted", "accepted_count": members})

    dataset = root / "lerobot" / repo_id
    frames = 0
    for index, sample in enumerate(ids):
        path = dataset / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"parquet-{sample}".encode() * 8)
        frames += 10
    write_json(dataset / "meta" / "info.json", {
        "fps": 30, "total_episodes": members, "total_frames": frames,
        "features": {"action": {"shape": [14]}, "observation.state": {"shape": [14]}},
    })
    details = [{"sample_id": sample, "image_source": "generated", "generated_video_sha256": media[sample],
                "training_video_sha256": sha(f"training-{sample}".encode()),
                "state_sha256": sha(f"state-{sample}".encode()), "output_episode_index": index}
               for index, sample in enumerate(ids)]
    write_json(dataset / "meta" / "lerobot-provenance.json", {
        "schema_version": "datafarm.pi05-lerobot-provenance/1.0.0",
        "training_eligible": True,
        "repo_id": repo_id,
        "episodes": members,
        "frames": frames,
        "semantic_allowlist": str(allowlist),
        "semantic_allowlist_sha256": sha(allowlist.read_bytes()),
        "allowed_semantic_waivers": list(provenance_waivers if provenance_waivers is not None else policy),
        "source_manifest_sha256": source_manifest_sha,
        "distribution_acceptance": str(distribution),
        "distribution_acceptance_sha256": sha(distribution_bytes),
        "episodes_detail": details,
    })
    return {"root": root, "dataset": dataset, "allowlist": allowlist, "audit_dir": audit_dir,
            "distribution": distribution}

