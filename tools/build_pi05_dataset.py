#!/usr/bin/env python3
"""Build a fresh inline-224 PI0.5 dataset from an admitted generated cohort.

The production scope requires technical PASS on every selected record, a zero-waiver
semantic allowlist, an accepted distribution receipt, and the frozen
terminal-home contract.  Media are split with an explicit carrier layout and
every episode passes a seam check and a first-frame round-trip gate.
Explicit exploratory subsets preserve their parent identities and missing
members; they require the same per-episode evidence and cannot claim production
eligibility or completion of the parent cohort.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import pyarrow.parquet as parquet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datafarm.common import read_json as load_json  # noqa: E402
from datafarm.training.dataset import (  # noqa: E402
    SEEDANCE_T_LAYOUT_1280X1080,
    PiperEefEncoder,
    audit_terminal_home,
    recovery_excisions,
    retained_record_indices,
    load_piper_fk_factory,
    materialize_episode,
    sha256_file,
    validate_frame_aligned_state,
)

from datafarm.training.cohort import cohort_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--task", required=True, help="task instruction written into the dataset")
    parser.add_argument("--excise-recovery", action="store_true",
                        help="drop deliberate misaligned-approach recovery detours from the training rows")
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--semantic-allowlist", type=Path, required=True)
    parser.add_argument("--distribution-acceptance", type=Path, required=True)
    parser.add_argument("--terminal-home-spec", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--target-repo-id", required=True)
    parser.add_argument("--piper-fk-source", type=Path, required=True)
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--batch-frames", type=int, default=16)
    parser.add_argument("--png-workers", type=int, default=24)
    parser.add_argument("--real-gripper-open-m", type=float, default=0.105)
    parser.add_argument("--expected-episodes", type=int, required=True,
                        help="Episode count of this build; source, candidate, allowlist and receipt must all match it.")
    parser.add_argument("--training-scope", choices=("production_policy", "exploratory_diagnostic"),
                        default="production_policy")
    parser.add_argument("--parent-execution-plan", type=Path,
                        help="Required for exploratory subsets; preserve all original source identities.")
    return parser.parse_args()



def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def write_state_only_episodes(
    root: Path, samples: list[dict[str, Any]], encoder: PiperEefEncoder, *, excise_recovery: bool = False
) -> None:
    """Write state/action-only parquet mirrors of the dataset rows for pre-audit."""

    import pyarrow as pa  # noqa: PLC0415

    for index, sample in enumerate(samples):
        payload = load_json(Path(sample["frame_aligned_robot_state"]["path"]))
        validate_frame_aligned_state(payload, expected_frames=int(sample["frame_count"]))
        excisions = recovery_excisions(payload) if excise_recovery else ()
        retained = retained_record_indices(payload, excisions)
        state, action = encoder.records_to_state_action([payload["records"][i] for i in retained])
        path = root / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(pa.table({
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 14)),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 14)),
        }), path)


def admitted_reviews(args: argparse.Namespace, records_by_id: dict[str, Any]) -> dict[str, dict[str, Any]]:
    allowlist = load_json(args.semantic_allowlist)
    reviews = allowlist.get("reviews") or []
    if allowlist.get("schema_version") != "datafarm.pi05-semantic-allowlist/1.1.0":
        raise ValueError("Unsupported semantic allowlist schema")
    if allowlist.get("waiver_policy", {}).get("allowed_failed_checks") != []:
        raise ValueError("Production semantic allowlist must contain zero waivers")
    if len(reviews) != args.expected_episodes:
        raise ValueError("Semantic allowlist count differs from expected episodes")
    by_id = {str(row["sample_id"]): row for row in reviews}
    if len(by_id) != len(reviews):
        raise ValueError("Semantic allowlist repeats a source identity")
    if set(by_id) != set(records_by_id):
        raise ValueError("Semantic allowlist/candidate identities differ")
    for sample_id, review in by_id.items():
        if review.get("verdict") != "accept" or review.get("waivers"):
            raise ValueError(f"{sample_id}: semantic review is not a zero-waiver accept")
        if not review.get("checks") or not all(value is True for value in review["checks"].values()):
            raise ValueError(f"{sample_id}: semantic checks are incomplete")
        if review.get("generated_video_sha256") != records_by_id[sample_id]["provider_video_sha256"]:
            raise ValueError(f"{sample_id}: semantic review video hash mismatch")
    return by_id


def main() -> None:
    args = parse_args()
    if args.target_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.target_root}")
    if args.batch_frames <= 0 or args.png_workers <= 0:
        raise ValueError("batch-frames and png-workers must be positive")

    manifest = load_json(args.source_manifest)
    samples = manifest.get("accepted", [])
    cohort = cohort_contract(args, samples)
    count = args.expected_episodes
    candidate = load_json(args.candidate_manifest)
    records = candidate.get("records", [])
    if manifest.get("status") != "complete":
        raise ValueError("Selected source manifest must be complete")
    if candidate.get("technical_pass_count") != count or len(records) != count:
        raise ValueError("Candidate count differs from expected technical-pass episodes")
    if candidate.get("source_dataset_sha256") != sha256_file(args.source_manifest):
        raise ValueError("Candidate/source manifest hash mismatch")
    records_by_id = {row["sample_id"]: row for row in records}
    if len(records_by_id) != len(records):
        raise ValueError("Candidate repeats a source identity")
    if set(records_by_id) != {row["sample_id"] for row in samples}:
        raise ValueError("Candidate/source sample IDs differ")
    if any(row.get("technical_verdict") != "PASS" for row in records):
        raise ValueError("Every candidate record must retain technical PASS")
    reviews = admitted_reviews(args, records_by_id)

    receipt = load_json(args.distribution_acceptance)
    if receipt.get("status") != "accepted" or receipt.get("accepted_count") != count:
        raise ValueError("Distribution acceptance count differs from the selected cohort")
    if cohort is not None and (receipt.get('training_scope') != args.training_scope
                               or receipt.get('cohort_selection') != cohort):
        raise ValueError("Exploratory receipt must bind the exact parent and selected identities")
    terminal_spec = load_json(args.terminal_home_spec)
    terminal_home_contract = terminal_spec.get("terminal_home_contract")
    if not isinstance(terminal_home_contract, dict):
        raise ValueError("terminal-home spec does not contain terminal_home_contract")

    template_info = load_json(args.template_root / "meta" / "info.json")
    template_episode = args.template_root / "data" / "chunk-000" / "episode_000000.parquet"
    if not template_episode.is_file():
        raise ValueError("Template must be an inline-224 dataset with data/chunk-000/episode_000000.parquet")
    schema = parquet.read_schema(template_episode)

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(args.trainer_root / "src"))
    from openpi.shared.image_tools import resize_with_pad  # noqa: PLC0415

    encoder = PiperEefEncoder(
        fk_factory=load_piper_fk_factory(args.piper_fk_source),
        real_gripper_open_m=args.real_gripper_open_m,
    )
    # Fail fast: audit terminal home on the exact state rows before any media
    # is decoded, so a source-trajectory defect costs seconds, not a full build.
    precheck_root = args.target_root.parent / f".{args.target_root.name}-terminal-precheck"
    shutil.rmtree(precheck_root, ignore_errors=True)
    try:
        write_state_only_episodes(precheck_root, samples, encoder, excise_recovery=args.excise_recovery)
        audit_terminal_home(precheck_root, terminal_home_contract)
    finally:
        shutil.rmtree(precheck_root, ignore_errors=True)

    args.target_root.mkdir(parents=True, exist_ok=False)
    total_frames = 0
    excised_total = 0
    excised_episodes: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.png_workers) as executor:
            for episode_index, sample in enumerate(samples):
                record = records_by_id[sample["sample_id"]]
                # The frame-retaining 24 Hz provider carrier is the reviewed
                # media; parquet timestamps are the original k/30 clock.
                length, stats, detail = materialize_episode(
                    sample=sample,
                    excise_recovery=args.excise_recovery,
                    generated_video=Path(record["provider_video"]).resolve(strict=True),
                    expected_video_sha256=record["provider_video_sha256"],
                    review=reviews[sample["sample_id"]],
                    output_episode=episode_index,
                    global_index=total_frames,
                    schema=schema,
                    target_root=args.target_root,
                    encoder=encoder,
                    resize_with_pad=resize_with_pad,
                    executor=executor,
                    batch_frames=args.batch_frames,
                )
                detail.update({
                    "source_sample": sample,
                    "technical_verdict": "PASS",
                    "semantic_verdict_at_training_admission": "PASS",
                    "training_video_path": record["training_video"],
                    "training_video_sha256": record["training_video_sha256"],
                })
                total_frames += length
                excised_total += int(detail.get("recovery_excised_frames", 0))
                if detail.get("recovery_excisions"):
                    excised_episodes.append({"sample_id": sample["sample_id"],
                                             "excisions": detail["recovery_excisions"]})
                episodes.append({"episode_index": episode_index, "tasks": [args.task], "length": length})
                episode_stats.append({"episode_index": episode_index, "stats": stats})
                details.append(detail)
                print(json.dumps({"status": "materialized", "episode": episode_index, "episodes": count,
                                  "sample_id": sample["sample_id"], "frames": length,
                                  "total_frames": total_frames}), flush=True)

        terminal_home_audit = audit_terminal_home(args.target_root, terminal_home_contract)

        meta = args.target_root / "meta"
        info = dict(template_info)
        info.update({"total_episodes": count, "total_frames": total_frames, "splits": {"train": f"0:{count}"}, "total_chunks": 1})
        write_json(meta / "info.json", info)
        write_jsonl(meta / "episodes.jsonl", episodes)
        shutil.copy2(args.template_root / "meta" / "tasks.jsonl", meta / "tasks.jsonl")
        write_jsonl(meta / "episodes_stats.jsonl", episode_stats)

        admission = {
            "schema_version": ("pi05_fixed_cohort_admission/1.0.0" if cohort is None
                               else "pi05_exploratory_admission/1.0.0"),
            "authorization_id": args.authorization_id,
            "candidate_manifest": str(args.candidate_manifest.resolve()),
            "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
            "record_count": count,
            "technical_pass_count": count,
            "semantic_status_counts": {"PASS": count},
            "training_verdict": "accept_zero_waiver_semantic" if cohort is None else "accept_zero_waiver_semantic_subset",
            "note": f"All {count} selected episodes passed the bound semantic checks.",
            "training_scope": args.training_scope,
            "cohort_selection": cohort,
        }
        write_json(meta / ("training-admission.json" if cohort is None else "exploratory-training-admission.json"), admission)
        builder = Path(__file__).resolve()
        module = ROOT / "src" / "datafarm" / "training" / "dataset.py"
        source_frames_total = sum(detail["source_frame_count"] for detail in details)
        if source_frames_total - excised_total - count != total_frames:
            raise ValueError("Source, recovery-excision and next-target frame counts do not reconcile")
        provenance = {
            "schema_version": "datafarm.pi05-lerobot-provenance/1.0.0",
            "training_eligible": True,
            "training_eligibility_reason": f"{count}/{count} selected technical and semantic checks pass with zero waivers; scope={args.training_scope}.",
            "repo_id": args.target_repo_id,
            "task": args.task,
            "episodes": count,
            "cohort_selection": cohort,
            "frames": total_frames,
            "recovery_excision": {
                "enabled": args.excise_recovery,
                "policy": "drop misaligned_approach..recenter_exact_joint_state; the detour revisits "
                          "the same joint state and gives a state-free policy two futures for one observation",
                "episodes_with_excision": len(excised_episodes),
                "frames_dropped": excised_total,
                "detail": excised_episodes,
            },
            "dataset_fps": 30,
            "episode_video_codec": None,
            "source_manifest": str(args.source_manifest.resolve()),
            "source_manifest_sha256": sha256_file(args.source_manifest),
            "semantic_allowlist": str(args.semantic_allowlist.resolve(strict=True)),
            "semantic_allowlist_sha256": sha256_file(args.semantic_allowlist),
            "allowed_semantic_waivers": [],
            "distribution_acceptance": str(args.distribution_acceptance.resolve(strict=True)),
            "distribution_acceptance_sha256": sha256_file(args.distribution_acceptance),
            "training_scope": args.training_scope,
            "production_policy_eligible": args.training_scope == "production_policy",
            "terminal_home_spec": str(args.terminal_home_spec.resolve(strict=True)),
            "terminal_home_spec_sha256": sha256_file(args.terminal_home_spec),
            "terminal_home_audit": terminal_home_audit,
            "clock_contract": {
                "control_and_state_hz": 30,
                "provider_carrier_fps": 24,
                "carrier_preserves_source_frame_indices": True,
                "all_source_frame_indices_retained": excised_total == 0,
                "dataset_timestamps": "k/30, after any declared recovery excisions",
                "frame_accounting": {
                    "source_frames": source_frames_total,
                    "recovery_excised_frames": excised_total,
                    "terminal_states_without_next_target": count,
                    "training_rows": total_frames,
                },
            },
            "state_action_contract": {"representation": "left XYZ-RPY-gripper + right XYZ-RPY-gripper", "dimension": 14, "action_pose_t": "observed puppet EEF pose at the next retained source frame", "action_gripper_t": "expert command at the current retained source frame"},
            "episodes_detail": details,
            "visual_materialization": {
                "schema_version": "datafarm_pi05_inline224/1.0.0",
                "source": "Seedance 1280x1080 three-view carrier",
                "carrier_layout": SEEDANCE_T_LAYOUT_1280X1080.to_dict(),
                "target": "three 224x224 PNG images embedded inline in parquet",
                "resize_authority": "openpi.shared.image_tools.resize_with_pad",
                "gates": ["per-episode carrier seam at declared layout", "first-frame exact round-trip"],
                "builder_source": str(builder),
                "builder_source_sha256": sha256_file(builder),
                "materializer_source": str(module),
                "materializer_source_sha256": sha256_file(module),
            },
        }
        write_json(meta / "lerobot-provenance.json", provenance)
        print(json.dumps({"status": "complete", "target_root": str(args.target_root),
                          "repo_id": args.target_repo_id, "episodes": count, "frames": total_frames},
                         indent=2), flush=True)
    except BaseException:
        shutil.rmtree(args.target_root, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
