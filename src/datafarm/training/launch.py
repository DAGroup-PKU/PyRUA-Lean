"""Preflight, normalize, smoke-test, prepare, or submit formal PI0.5 training.

The vendored openpi trainer is used read-only as the backend.  Before a launch manifest is written, the spec's explicit
``state_conditioning`` (``none``), the CLI state flag and the Admission Certificate
binding are audited with ``datafarm.training.audit_pi05_training_contract``.

Admission (``train-command``):

* ``--admission-certificate <digest-or-path> [--admission-store DIR]`` loads the
  certificate from its content-addressed store and verifies schema, digest,
  the passing admission-root receipt (verifier != producer), the evidence, and
  the binding to this dataset's provenance, allowlist, members and files;
* a launch without a verified certificate is refused.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from datafarm.runtime import expand, load_bindings, source_root
from datafarm.training import (
    STATE_CONDITIONING_MODES,
    AdmissionBinding,
    Pi05TrainingContract,
    audit_pi05_training_contract,
)
from datafarm.common import is_sha256, read_json, sha256_bytes, sha256_file
from datafarm.training.cohort_auditors import PI05_COHORT_AUDITORS
from datafarm.training.dataset import (
    audit_action_shift,
    audit_joint_action_shift,
    audit_terminal_home,
)
from datafarm.training.joint_actions import (
    MODES as JOINT_ACTION_MODES,
    validate_contract as validate_joint_action_contract,
    validate_provenance_contract as validate_joint_provenance_contract,
)
from datafarm.training.distribution import validate_distribution_acceptance_contract
from datafarm.core.admission import (
    AdmissionError,
    AdmissionExpectation,
    compute_file_manifest,
    file_manifest_digest,
    verify_certificate,
)
from datafarm.core.schema import canonical_json_bytes
from datafarm.core.store import ArtifactStore

ROOT = source_root()
LAUNCH_MANIFEST_SCHEMA = "datafarm.pi05-train-launch/1.2.0"


class AdmissionRequired(ValueError):
    """A new launch has no verifiable Admission Certificate."""


def resolve_certificate_reference(value: str, store: str | None) -> tuple[str, Path]:
    """Return (certificate digest, store root) for a digest or certificate file."""

    if is_sha256(value):
        if not store:
            raise AdmissionRequired("--admission-store is required when --admission-certificate is a digest")
        return value, Path(store)
    path = Path(value)
    if not path.is_file():
        raise AdmissionRequired(
            f"admission certificate {path} does not exist; issue it with "
            "tools/issue_admission_certificate.py after the dataset is built"
        )
    raw = path.read_bytes()
    try:
        canonical = canonical_json_bytes(json.loads(raw))
    except (ValueError, TypeError) as exc:
        raise AdmissionRequired(f"{path} is not a canonical certificate document: {exc}") from exc
    if canonical != raw:
        raise AdmissionRequired(f"{path} is not byte-identical to a canonical certificate document")
    digest = sha256_bytes(raw)
    if store:
        return digest, Path(store)
    resolved = path.resolve()
    if resolved.parent.parent.name == "sha256" and resolved.stem == digest:
        return digest, resolved.parents[2]
    raise AdmissionRequired("--admission-store is required when the certificate file is outside its store")


def verify_launch_certificate(value: str, store_arg: str | None, spec: dict[str, Any],
                              paths: dict[str, Any]) -> tuple[AdmissionBinding, dict[str, Any]]:
    digest, store_root = resolve_certificate_reference(value, store_arg)
    pinned = spec.get("admission_certificate")
    if isinstance(pinned, dict) and digest != pinned.get("sha256"):
        raise AdmissionRequired(
            f"admission certificate {digest} differs from spec binding {pinned.get('sha256')}"
        )
    if not (store_root / "sha256").is_dir():
        raise AdmissionRequired(f"admission store {store_root} does not exist or holds no artifacts")
    provenance = paths["provenance"]
    expectation = AdmissionExpectation(
        provenance_sha256=sha256_file(paths["provenance_path"]),
        allowlist_sha256=str(provenance.get("semantic_allowlist_sha256")),
        repo_id=spec["repo_id"],
        allowed_failed_checks=tuple(spec.get("allowed_semantic_waivers", [])),
        member_ids=tuple(str(item.get("sample_id")) for item in provenance.get("episodes_detail", [])),
    )
    verification = verify_certificate(
        ArtifactStore(store_root), digest, auditors=PI05_COHORT_AUDITORS, expectation=expectation
    )
    files_verified = False
    if verification.data_files is not None:
        manifest = compute_file_manifest(paths["dataset_root"])
        if file_manifest_digest(manifest) != verification.data_files.sha256:
            raise AdmissionError(
                f"admission certificate {digest} does not verify",
                ["dataset files changed since certification (file manifest digest differs)"],
            )
        files_verified = True
    record = {
        "mode": "certificate-verified",
        "store": str(store_root.resolve()),
        "verification": verification.to_dict(),
        "dataset_files_verified": files_verified,
    }
    return AdmissionBinding.from_verification(verification), record


def resolve_admission(args: argparse.Namespace, spec: dict[str, Any], spec_path: Path,
                      paths: dict[str, Any]) -> tuple[AdmissionBinding, dict[str, Any]]:
    """Bind this launch to a verified Admission Certificate; fail closed without one."""

    certificate = getattr(args, "admission_certificate", None)
    if not certificate:
        raise AdmissionRequired(
            "--admission-certificate is required. Issue one with tools/issue_admission_certificate.py "
            "--store <dir> --dataset-root <dataset> and pass --admission-certificate <digest-or-file> "
            "--admission-store <dir>."
        )
    return verify_launch_certificate(certificate, getattr(args, "admission_store", None), spec, paths)


def validate_training_distribution_admission(spec: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless policy training is bound to an accepted distribution."""

    if spec.get("schema_version") != "datafarm.pi05-finetune/1.1.0":
        raise ValueError("Formal training requires datafarm.pi05-finetune/1.1.0")
    target_task = spec.get("target_task")
    required_claim = spec.get("required_distribution_claim")
    if not isinstance(target_task, str) or not target_task.strip():
        raise ValueError("PI0.5 spec target_task is required")
    training_scope = str(spec.get("training_scope", "production_policy"))
    if required_claim == "broad_deployment":
        if training_scope != "production_policy":
            raise ValueError("broad_deployment training must use training_scope=production_policy")
    elif required_claim == "local_robustness":
        if training_scope != "exploratory_diagnostic":
            raise ValueError("local_robustness may train only with training_scope=exploratory_diagnostic")
        if spec.get("production_policy_eligible") is not False:
            raise ValueError("local_robustness exploratory training must set production_policy_eligible=false")
        if not str(spec.get("local_robustness_authorization_id", "")).strip():
            raise ValueError("local_robustness exploratory training requires explicit authorization")
    else:
        raise ValueError(f"Unsupported training distribution claim: {required_claim!r}")
    record = spec.get("distribution_acceptance")
    if not isinstance(record, dict):
        raise ValueError("PI0.5 spec must bind distribution_acceptance path and SHA-256")
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    actual_sha = sha256_file(path)
    if actual_sha != str(record.get("sha256", "")):
        raise ValueError(f"Distribution acceptance authority drifted: {actual_sha} != {record.get('sha256')}")
    receipt = read_json(path)
    validate_distribution_acceptance_contract(receipt, target_task=target_task.strip(), required_claim=required_claim)
    if provenance.get("distribution_acceptance_sha256") != actual_sha:
        raise ValueError("Dataset provenance is not bound to the accepted distribution receipt")
    if required_claim == "local_robustness":
        if provenance.get("training_scope") != "exploratory_diagnostic":
            raise ValueError("Dataset provenance is not marked exploratory_diagnostic")
        if provenance.get("production_policy_eligible") is not False:
            raise ValueError("Local-robustness dataset cannot be production-policy eligible")
    return {"path": path, "sha256": actual_sha, "contract": receipt, "training_scope": training_scope}


def state_conditioning(spec: dict[str, Any]) -> str:
    """Return the explicitly declared state mode; the launcher trains state-free only."""

    mode = spec.get("state_conditioning")
    if mode not in STATE_CONDITIONING_MODES:
        raise ValueError(
            f"Spec must declare state_conditioning explicitly as one of {STATE_CONDITIONING_MODES}"
        )
    return mode


def validate_joint_source_evidence(
    spec: dict[str, Any], dataset_root: Path, info: dict[str, Any], provenance: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Bind derived joint labels to exact source readbacks without inventing a home threshold."""

    evidence = spec.get("joint_source_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("Joint training requires joint_source_evidence")

    def load_bound(name: str) -> tuple[Path, dict[str, Any], str]:
        binding = evidence.get(name)
        if not isinstance(binding, dict):
            raise ValueError(f"joint_source_evidence.{name} must bind path and sha256")
        path = Path(str(binding.get("path") or "")).resolve(strict=True)
        actual = sha256_file(path)
        if actual != binding.get("sha256"):
            raise ValueError(f"Joint source evidence drifted for {name}: {actual} != {binding.get('sha256')}")
        return path, read_json(path), actual

    terminal_path, terminal, terminal_sha = load_bound("terminal_joint_readback_manifest")
    if Path(str(terminal.get("dataset") or "")).resolve() != dataset_root.resolve():
        raise ValueError("Terminal joint readback manifest belongs to a different dataset")
    episodes = terminal.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != int(info["total_episodes"]):
        raise ValueError("Terminal joint readback manifest does not cover every dataset episode")
    for episode in episodes:
        if episode.get("final_joint_dims_exactly_equal") is not True:
            raise ValueError("Terminal joint readback manifest contains a non-exact final joint label")
        source_path = Path(str(episode.get("source_state_path") or "")).resolve(strict=True)
        if sha256_file(source_path) != episode.get("source_state_sha256"):
            raise ValueError(f"Source joint readback drifted: {source_path}")

    audit_path, machine_audit, audit_sha = load_bound("independent_machine_audit")
    matching = [task for task in (machine_audit.get("tasks") or {}).values()
                if isinstance(task, dict) and int(task.get("episodes", -1)) == int(info["total_episodes"])
                and sum(int(item.get("rows", 0)) for item in task.get("episodes_detail", []))
                == int(info["total_frames"])]
    if len(matching) != 1:
        raise ValueError("Independent joint machine audit does not identify this dataset uniquely")
    checks = matching[0].get("checks") or {}
    gripper_evidence = _joint_gripper_evidence(evidence, checks, machine_audit, provenance)
    required = (
        "all_final_joint_dims_equal_final_source_readback",
        gripper_evidence["check"],
        "all_joint_state_action_rederived_exact",
        "all_rgb_timestamp_index_columns_exact",
        "all_source_state_hashes_match",
        "metadata_fields_and_units_match",
    )
    if any(checks.get(name) is not True for name in required):
        raise ValueError("Independent joint machine audit is missing a required exact check")
    return {
        "terminal_joint_readback_manifest": {"path": str(terminal_path), "sha256": terminal_sha,
                                               "episodes": len(episodes)},
        "independent_machine_audit": {"path": str(audit_path), "sha256": audit_sha,
                                       "checks": list(required)},
        "gripper_evidence": gripper_evidence,
    }


EEF_GRIPPER_CHECK = "all_gripper_columns_bit_identical_to_eef"
COMMAND_GRIPPER_CHECK = "all_gripper_columns_bit_identical_to_independent_command_rederivation"
# joint_conversion keys that name an EEF source dataset the joint labels were derived from.
_EEF_SOURCE_KEYS = ("source_dataset", "source_repo_id", "source_provenance_sha256", "gripper_bit_identity")


def _joint_gripper_evidence(evidence: dict[str, Any], checks: dict[str, Any], machine_audit: dict[str, Any],
                            provenance: dict[str, Any] | None) -> dict[str, Any]:
    """Choose the gripper bit-identity evidence; fail closed unless "no EEF baseline" is explicit.

    Default: the independent audit proves the gripper columns are bit-identical to the
    EEF source dataset. Only when the spec declares ``joint_source_evidence.eef_baseline``
    as not applicable (with a reason), the bound dataset provenance names no EEF source,
    and the independent audit itself records the EEF check as explicitly null/not
    applicable, may the audit's independent command re-derivation stand in for it.
    """

    declared = evidence.get("eef_baseline")
    if declared is None and "eef_baseline" not in evidence:
        return {"check": EEF_GRIPPER_CHECK, "eef_baseline": "bound"}
    if not (isinstance(declared, dict) and declared.get("status") == "not_applicable"
            and isinstance(declared.get("reason"), str) and declared["reason"].strip()):
        raise ValueError("joint_source_evidence.eef_baseline must be {status: not_applicable, reason: ...}")
    if provenance is None:
        raise ValueError("A no-EEF-baseline gripper claim requires the dataset provenance")
    conversion = provenance.get("joint_conversion") if isinstance(provenance.get("joint_conversion"), dict) else {}
    named = [key for key in _EEF_SOURCE_KEYS if conversion.get(key)]
    explicit = provenance.get("eef_baseline", conversion.get("eef_baseline"))
    explicit_ok = explicit in (None, "not_applicable") or (
        isinstance(explicit, dict) and explicit.get("status") == "not_applicable")
    if named or not explicit_ok:
        raise ValueError(f"Dataset provenance names an EEF baseline ({named or explicit}); "
                         "the EEF gripper bit-identity check is required")
    if not isinstance(conversion.get("gripper_source"), (dict, str)) or not conversion.get("gripper_source"):
        raise ValueError("Dataset provenance does not declare its joint_conversion.gripper_source")
    unproven = machine_audit.get("not_applicable_or_unproven")
    if EEF_GRIPPER_CHECK not in checks or checks[EEF_GRIPPER_CHECK] is not None or not (
            isinstance(unproven, list) and EEF_GRIPPER_CHECK in unproven):
        raise ValueError("Independent audit does not record the EEF gripper check as not applicable")
    if machine_audit.get("hard_failures") not in ([], None) or machine_audit.get("verdict") not in (
            "pass", "pass_with_notes"):
        raise ValueError("Independent joint machine audit did not pass")
    return {"check": COMMAND_GRIPPER_CHECK, "eef_baseline": "not_applicable",
            "reason": declared["reason"].strip(), "replaces": EEF_GRIPPER_CHECK}


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    state_conditioning(spec)
    mode = spec.get("action_representation")
    if mode not in {"absolute_eef", "delta_eef_current_rpy_v1", *JOINT_ACTION_MODES}:
        raise ValueError("action_representation must be declared explicitly and supported")
    expected_suffix = {
        "absolute_eef": "",
        "delta_eef_current_rpy_v1": "_delta_eef_current_rpy_v1",
        **{joint_mode: suffix for joint_mode, (suffix, _, _) in JOINT_ACTION_MODES.items()},
    }[mode]
    if expected_suffix and not spec["base_config"].endswith(expected_suffix):
        raise ValueError("Action representation and trainer config disagree")
    representation_suffixes = ("_delta_eef_current_rpy_v1", *(suffix for suffix, _, _ in JOINT_ACTION_MODES.values()))
    if not expected_suffix and spec["base_config"].endswith(representation_suffixes):
        raise ValueError("Action representation and trainer config disagree")
    joint_contract = None
    if mode in JOINT_ACTION_MODES:
        joint_contract = validate_joint_action_contract(spec.get("training_action_contract") or {}, mode)
    gpu_candidates = [int(value) for value in spec.get("gpu_candidates", [spec["fsdp_devices"]])]
    if not gpu_candidates or any(value <= 0 for value in gpu_candidates):
        raise ValueError("gpu_candidates must contain positive device counts")
    if len(set(gpu_candidates)) != len(gpu_candidates):
        raise ValueError("gpu_candidates must not contain duplicates")
    if any(int(spec["batch_size"]) % value != 0 for value in gpu_candidates):
        raise ValueError("Global batch_size must be divisible by every GPU candidate")

    trainer_root = Path(spec["trainer_root"])
    trainer_python = Path(spec["trainer_python"])
    lerobot_home = Path(spec["lerobot_home"])
    dataset_root = lerobot_home / spec["repo_id"]
    provenance_path = dataset_root / "meta" / "lerobot-provenance.json"
    info_path = dataset_root / "meta" / "info.json"
    tokenizer_path = Path(spec["paligemma_tokenizer"])
    training_pythonpath = Path(spec.get("training_pythonpath", trainer_root / "src"))
    required_paths = [
        trainer_root / "scripts" / "train.py",
        trainer_root / "src" / "openpi" / "training" / "config.py",
        trainer_root / "src" / "openpi" / "training" / "data_loader.py",
        trainer_root / "src" / "openpi" / "policies" /
        ("piper_policy.py" if mode in JOINT_ACTION_MODES else "piper_eef_xyz3d_policy.py"),
        trainer_python,
        Path(spec["base_checkpoint_params"]),
        tokenizer_path,
        training_pythonpath / "openpi" / "models" / "pi0.py",
        provenance_path,
        info_path,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required PI0.5 inputs: {missing}")

    mismatches = []
    for relative_path, expected in spec.get("trainer_source_sha256", {}).items():
        actual = sha256_file(trainer_root / relative_path)
        if actual != expected:
            mismatches.append({"path": str(trainer_root / relative_path), "expected": expected, "actual": actual})
    if mismatches:
        raise ValueError(f"trainer source authority drifted: {mismatches}")
    if spec.get("observability_protocol") is not None:
        if spec["observability_protocol"] != "training-observability-v1":
            raise ValueError("Unsupported training observability protocol")
        if int(spec.get("eval_interval", 0)) <= 0:
            raise ValueError("Observable training requires a positive eval_interval")
    tokenizer_hash = sha256_file(tokenizer_path)
    if tokenizer_hash != spec["paligemma_tokenizer_sha256"]:
        raise ValueError(f"PaliGemma tokenizer drifted: {tokenizer_hash} != {spec['paligemma_tokenizer_sha256']}")
    if spec.get("trainer_source_manifest"):
        from datafarm.core.source_lock import verify_source_lock
        lock = spec["trainer_source_manifest"]
        verify_source_lock(trainer_root, Path(lock["path"]), lock["sha256"])

    provenance = read_json(provenance_path)
    if mode in JOINT_ACTION_MODES:
        expected_provenance = str(spec.get("dataset_provenance_sha256") or "")
        actual_provenance = sha256_file(provenance_path)
        if actual_provenance != expected_provenance:
            raise ValueError(
                f"Joint dataset provenance drifted: {actual_provenance} != {expected_provenance}"
            )
        certificate = spec.get("admission_certificate")
        if not isinstance(certificate, dict):
            raise ValueError("Joint training spec must bind its action-conversion admission certificate")
        certificate_path = Path(str(certificate.get("path") or "")).resolve(strict=True)
        if sha256_file(certificate_path) != certificate.get("sha256"):
            raise ValueError("Joint action-conversion admission certificate drifted")
        if not (Path(str(certificate.get("store") or "")) / "sha256").is_dir():
            raise ValueError("Joint action-conversion admission store is missing")
    distribution_admission = validate_training_distribution_admission(spec, provenance)
    if provenance.get("repo_id") != spec["repo_id"]:
        raise ValueError("Dataset provenance repo_id does not match finetune spec")
    if not provenance.get("training_eligible", False):
        raise ValueError("Dataset is not training eligible")
    if not provenance.get("semantic_allowlist") or not provenance.get("semantic_allowlist_sha256"):
        raise ValueError("Training-eligible generated data must pin a semantic allowlist")
    if any(episode.get("image_source") != "generated" for episode in provenance.get("episodes_detail", [])):
        raise ValueError("Training-eligible dataset contains a non-generated image source")
    if sorted(provenance.get("allowed_semantic_waivers", [])) != sorted(spec.get("allowed_semantic_waivers", [])):
        raise ValueError("Dataset semantic waiver policy does not match the finetune spec")
    info = read_json(info_path)
    if int(info.get("fps", -1)) != int(spec.get("dataset_fps", 30)):
        raise ValueError(f"Expected a {spec.get('dataset_fps', 30)} Hz LeRobot dataset, got {info.get('fps')}")
    if int(info.get("total_episodes", 0)) < int(spec.get("minimum_episodes", 1)) or int(info.get("total_frames", 0)) <= 0:
        raise ValueError("LeRobot dataset is empty or below minimum_episodes")
    for key in ("observation.state", "action", "observation.images.cam_high",
                "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"):
        if key not in info.get("features", {}):
            raise ValueError(f"Dataset is missing feature {key}")
    if info["features"]["observation.state"]["shape"] != [14] or info["features"]["action"]["shape"] != [14]:
        raise ValueError("PI0.5 Piper state/action must be 14-D")

    if mode in JOINT_ACTION_MODES:
        joint_contract = validate_joint_provenance_contract(
            spec["training_action_contract"], provenance.get("state_action_contract") or {}, mode
        )
        joint_source_evidence = validate_joint_source_evidence(spec, dataset_root, info, provenance)
        action_shift_audit = audit_joint_action_shift(dataset_root, gripper_unit=str(joint_contract["gripper_unit"]))
        # Joint datasets inherit the already-admitted source home/collision evidence.
        # They must not be checked with EEF geometry or a newly invented joint tolerance.
        terminal_home_audit = {
            "status": "not_applicable",
            "reason": "joint mode binds source joint-home evidence; EEF terminal geometry is not applied",
            "source_evidence": joint_source_evidence,
        }
    else:
        action_shift_audit = audit_action_shift(dataset_root)
        terminal_home_audit = audit_terminal_home(dataset_root, spec.get("terminal_home_contract") or {})

    return {
        "trainer_root": trainer_root,
        "trainer_python": trainer_python,
        "lerobot_home": lerobot_home,
        "dataset_root": dataset_root,
        "provenance_path": provenance_path,
        "provenance": provenance,
        "info": info,
        "action_shift_audit": action_shift_audit,
        "terminal_home_audit": terminal_home_audit,
        "paligemma_tokenizer": tokenizer_path,
        "paligemma_tokenizer_sha256": tokenizer_hash,
        "training_pythonpath": training_pythonpath,
        "metrics_jsonl": Path(spec.get("metrics_jsonl", Path(spec["run_manifest_dir"]) / "training-metrics.jsonl")),
        "distribution_admission": distribution_admission,
        "joint_source_evidence": joint_source_evidence if mode in JOINT_ACTION_MODES else None,
    }


def effective_config(spec: dict[str, Any], paths: dict[str, Any]) -> tuple[Any, Any, Any]:
    os.environ["HF_LEROBOT_HOME"] = str(paths["lerobot_home"])
    sys.path.insert(0, str(paths["trainer_root"] / "src"))
    import openpi.shared.normalize as normalize  # noqa: PLC0415
    import openpi.training.config as config_module  # noqa: PLC0415
    import openpi.training.data_loader as data_loader  # noqa: PLC0415

    if spec.get("action_representation") == "delta_eef_current_rpy_v1":
        from datafarm.training.delta_eef import install
        install()
    elif spec.get("action_representation") in JOINT_ACTION_MODES:
        from datafarm.training.joint_actions import install
        install()
    config = config_module.get_config(spec["base_config"])
    config = dataclasses.replace(
        config,
        project_name=spec["project_name"],
        exp_name=spec["exp_name"],
        data=dataclasses.replace(config.data, repo_id=spec["repo_id"]),
        assets_base_dir=spec["assets_base_dir"],
        checkpoint_base_dir=spec["checkpoint_base_dir"],
        seed=int(spec["seed"]),
        batch_size=int(spec["batch_size"]),
        num_workers=int(spec["num_workers"]),
        num_train_steps=int(spec["num_train_steps"]),
        log_interval=int(spec["log_interval"]),
        save_interval=int(spec["save_interval"]),
        keep_period=int(spec["keep_period"]),
        wandb_enabled=bool(spec["wandb_enabled"]),
        fsdp_devices=int(spec["fsdp_devices"]),
    )
    return normalize, data_loader, config


def compute_norm_stats(spec: dict[str, Any], paths: dict[str, Any], max_frames: int | None) -> None:
    import numpy as np  # noqa: PLC0415
    import pyarrow.parquet as parquet  # noqa: PLC0415
    import tqdm  # noqa: PLC0415

    normalize, _, config = effective_config(spec, paths)
    horizon = int(config.model.action_horizon)
    parquet_files = sorted((paths["dataset_root"] / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No episode parquet files available for normalization")
    state_parts, action_window_parts = [], []
    remaining = max_frames
    for path in tqdm.tqdm(parquet_files, desc="Loading control-only parquet columns"):
        table = parquet.read_table(path, columns=["observation.state", "action"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != 14 or action.shape != state.shape:
            raise ValueError(f"Unexpected state/action arrays in {path}: {state.shape}, {action.shape}")
        take = len(state) if remaining is None else min(len(state), remaining)
        if take <= 0:
            break
        # LeRobot clamps future delta indices to the final frame of each episode.
        indices = np.minimum(
            np.arange(take, dtype=np.int64)[:, None] + np.arange(horizon, dtype=np.int64)[None, :],
            len(action) - 1,
        )
        state_parts.append(state[:take])
        windows = action[indices]
        if spec.get("action_representation") == "delta_eef_current_rpy_v1":
            from datafarm.training.delta_eef import convert
            windows = convert(windows, state[:take])
        elif spec.get("action_representation") in JOINT_ACTION_MODES:
            from datafarm.training.joint_actions import convert_for_mode
            windows = convert_for_mode(windows, state[:take], spec["action_representation"])
        action_window_parts.append(windows)
        if remaining is not None:
            remaining -= take
    if not state_parts:
        raise ValueError("No frames available for normalization")
    states = np.concatenate(state_parts, axis=0)
    action_windows = np.concatenate(action_window_parts, axis=0)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    batch_size = int(spec["batch_size"])
    for start in tqdm.trange(0, len(states), batch_size, desc="Computing control norm stats"):
        end = min(start + batch_size, len(states))
        stats["state"].update(states[start:end])
        stats["actions"].update(action_windows[start:end])
    output = config.assets_dirs / spec["repo_id"]
    output.mkdir(parents=True, exist_ok=True)
    normalize.save(output, {key: value.get_statistics() for key, value in stats.items()})
    print(json.dumps({"norm_stats": str(output), "frames": len(states),
                      "action_vectors": len(states) * horizon, "video_frames_decoded": 0}, indent=2))


def loader_smoke(spec: dict[str, Any], paths: dict[str, Any]) -> None:
    import jax  # noqa: PLC0415

    os.environ["OPENPI_DATA_HOME"] = str(paths["paligemma_tokenizer"].parents[1])
    _, data_loader, config = effective_config(spec, paths)
    smoke_batch_size = max(1, jax.device_count())
    loader = data_loader.create_data_loader(
        dataclasses.replace(config, batch_size=smoke_batch_size, num_workers=0),
        shuffle=False, num_batches=1, skip_norm_stats=False,
    )
    observation, actions = next(iter(loader))
    report = {
        "state_shape": list(observation.state.shape),
        "action_shape": list(actions.shape),
        "image_shapes": {key: list(value.shape) for key, value in observation.images.items()},
        "smoke_batch_size": smoke_batch_size,
        "jax_device_count": jax.device_count(),
    }
    if report["action_shape"][-2:] != [50, 32]:
        raise ValueError(f"Unexpected transformed action shape: {report['action_shape']}")
    print(json.dumps(report, indent=2))


def build_train_command(spec: dict[str, Any], paths: dict[str, Any]) -> list[str]:
    adapter = "none"
    if spec.get("action_representation") == "delta_eef_current_rpy_v1":
        adapter = "delta-eef"
    elif spec.get("action_representation") in JOINT_ACTION_MODES:
        adapter = "joint"
    return [
        str(paths["trainer_python"]),
        str(ROOT / "src/datafarm/training/run_trainer.py"), "--action-adapter", adapter, spec["base_config"],
        "--project-name", spec["project_name"],
        "--exp-name", spec["exp_name"],
        "--data.repo-id", spec["repo_id"],
        "--assets-base-dir", spec["assets_base_dir"],
        "--checkpoint-base-dir", spec["checkpoint_base_dir"],
        "--weight-loader.params-path", spec["base_checkpoint_params"],
        "--seed", str(spec["seed"]),
        "--batch-size", str(spec["batch_size"]),
        "--num-workers", str(spec["num_workers"]),
        "--num-train-steps", str(spec["num_train_steps"]),
        "--log-interval", str(spec["log_interval"]),
        "--save-interval", str(spec["save_interval"]),
        "--keep-period", str(spec["keep_period"]),
        "--fsdp-devices", str(spec["fsdp_devices"]),
        "--model.action-dim", "32",
        "--model.action-horizon", "50",
        "--model.max-token-len", "200",
        "--model.pi05",
        "--model.no-discrete-state-input",
        "--lr-schedule.warmup-steps", "1000",
        "--lr-schedule.peak-lr", "2.5e-5",
        "--lr-schedule.decay-steps", "30000",
        "--lr-schedule.decay-lr", "2.5e-6",
        "--optimizer.b1", "0.9",
        "--optimizer.b2", "0.95",
        "--optimizer.eps", "1e-8",
        "--optimizer.weight-decay", "1e-10",
        "--optimizer.clip-gradient-norm", "1.0",
        "--no-overwrite",
        "--no-resume",
        "--wandb-enabled" if spec["wandb_enabled"] else "--no-wandb-enabled",
    ]


def audit_launch(spec: dict[str, Any], paths: dict[str, Any], train: list[str], *,
                 admission: AdmissionBinding | None) -> dict[str, Any]:
    """Audit the launch contract: declared state mode, CLI flag, admission and dimensions."""

    mode = state_conditioning(spec)
    contract = Pi05TrainingContract(
        admission_certificate_sha256=str((admission.certificate_sha256 if admission else None) or ""),
        training_bundle_sha256=sha256_file(paths["provenance_path"]),
        model_family="pi05",
        state_conditioning=mode,
        discrete_state_input=False,
        native_action_dim=int(paths["info"]["features"]["action"]["shape"][0]),
        model_action_dim=int(train[train.index("--model.action-dim") + 1]),
        action_horizon=int(train[train.index("--model.action-horizon") + 1]),
        camera_keys=("cam_high", "cam_left_wrist", "cam_right_wrist"),
        source_fps=float(paths["info"]["fps"]),
        launch_argv=tuple(train),
        admission=admission,
    )
    audit = audit_pi05_training_contract(contract)
    if not audit.passed:
        raise ValueError(f"PI0.5 training contract failed: {list(audit.failures)}")
    return {"state_conditioning": mode, **audit.to_dict()}


def build_scheduler_command(spec: dict[str, Any], paths: dict[str, Any], train: list[str]) -> list[str]:
    job = [
        "env",
        "PYTHONUNBUFFERED=1",
        f"DATAFARM_TRAINER_ROOT={paths['trainer_root']}",
        f"PYTHONPATH={paths['training_pythonpath']}",
        f"HF_LEROBOT_HOME={paths['lerobot_home']}",
        f"OPENPI_DATA_HOME={paths['paligemma_tokenizer'].parents[1]}",
        f"TRAIN_METRICS_JSONL={paths['metrics_jsonl']}",
        f"TRAIN_EVAL_INTERVAL={int(spec.get('eval_interval', 1000))}",
        *train,
    ]
    candidates = ",".join(str(int(value)) for value in spec.get("gpu_candidates", [spec["fsdp_devices"]]))
    return [
        str(paths["trainer_python"]), str(ROOT / "tools/launch_pi05_training.py"), "submit",
        "--scheduler", spec["gpu_scheduler"], "--name", spec["exp_name"],
        "--cwd", str(paths["trainer_root"]), "--gpu-candidates", candidates, "--", *job,
    ]


def write_launch_manifest(spec_path: Path, spec: dict[str, Any], paths: dict[str, Any],
                          command: list[str], contract_audit: dict[str, Any],
                          admission_record: dict[str, Any] | None = None) -> Path:
    run_dir = Path(spec["run_manifest_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "launch-manifest.json"
    if output.is_file():
        # Keep the manifest being replaced as evidence.
        kept = run_dir / f"launch-manifest.{sha256_file(output)[:12]}.superseded.json"
        if not kept.exists():
            output.replace(kept)
    admission = paths["distribution_admission"]
    manifest = {
        "schema_version": LAUNCH_MANIFEST_SCHEMA,
        "status": "prepared_not_started",
        "admission": admission_record,
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
        "spec": str(spec_path),
        "spec_sha256": sha256_file(spec_path),
        "dataset_provenance_sha256": sha256_file(paths["provenance_path"]),
        "distribution_acceptance": {"path": str(admission["path"]), "sha256": admission["sha256"],
                                    "claim": admission["contract"].get("claim", spec.get("required_distribution_claim"))},
        "action_representation": spec.get("action_representation", "absolute_eef"),
        "training_action_contract": spec.get("training_action_contract"),
        "joint_source_evidence": paths.get("joint_source_evidence"),
        "training_scope": spec.get("training_scope", "production_policy"),
        "production_policy_eligible": spec.get("production_policy_eligible", True),
        "training_contract_audit": contract_audit,
        "trainer_source_sha256": spec["trainer_source_sha256"],
        "runtime_source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [
                ROOT / "src/datafarm/training/run_trainer.py",
                ROOT / "src/datafarm/training/checkpoint_restore.py",
                ROOT / "src/datafarm/training/delta_eef.py",
                ROOT / "src/datafarm/training/joint_actions.py",
            ]
        },
        "paligemma_tokenizer": str(paths["paligemma_tokenizer"]),
        "paligemma_tokenizer_sha256": paths["paligemma_tokenizer_sha256"],
        "observability": {
            "protocol": spec.get("observability_protocol"),
            "metrics_jsonl": str(paths["metrics_jsonl"]),
            "eval_interval": int(spec.get("eval_interval", 1000)),
            "eval_slice": "fixed first transformed batch with fixed RNG; not held-out",
        },
        "gpu_allocation": {
            "policy": "atomic_preference_order",
            "candidates": [int(value) for value in spec.get("gpu_candidates", [spec["fsdp_devices"]])],
            "global_batch_size": int(spec["batch_size"]),
            "authorization_id": spec.get("gpu_authorization_id"),
        },
        "command": command,
    }
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return output


def replace_fsdp_devices(command: list[str], devices: int) -> list[str]:
    result = list(command)
    matches = [index for index, token in enumerate(result) if token == "--fsdp-devices"]
    if len(matches) != 1 or matches[0] + 1 >= len(result):
        raise ValueError("Training command must contain exactly one --fsdp-devices value")
    result[matches[0] + 1] = str(devices)
    return result


def submit(args: argparse.Namespace) -> int:
    """Atomically prefer a larger single-host allocation; exit 2 if none is free."""

    job = args.job_command[1:] if args.job_command and args.job_command[0] == "--" else args.job_command
    if not job:
        raise SystemExit("a job command is required after --")
    candidates: list[int] = []
    for raw in args.gpu_candidates.split(","):
        value = int(raw.strip())
        if value <= 0:
            raise ValueError("GPU candidates must be positive")
        if value not in candidates:
            candidates.append(value)
    for devices in candidates:
        command = [args.scheduler, "submit", "--gpus", str(devices), "--name", args.name,
                   "--cwd", args.cwd, *replace_fsdp_devices(job, devices)]
        print(f"allocation_attempt_gpus={devices}", flush=True)
        result = subprocess.run(command, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
        if result.returncode == 0:
            print(f"selected_gpus={devices}", flush=True)
            return 0
        if result.returncode != 2:
            return int(result.returncode)
        print(f"allocation_unavailable_gpus={devices}", flush=True)
    return 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "norm-stats", "loader-smoke", "train-command"):
        command = sub.add_parser(name)
        command.add_argument("--spec", type=Path, required=True)
        command.add_argument("--max-frames", type=int)
        command.add_argument("--execute", action="store_true")
        command.add_argument("--admission-certificate",
                             help="Admission Certificate digest, or a canonical certificate file")
        command.add_argument("--admission-store", help="content-addressed store holding the certificate")
    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--scheduler", required=True)
    submit_parser.add_argument("--name", required=True)
    submit_parser.add_argument("--cwd", required=True)
    submit_parser.add_argument("--gpu-candidates", required=True)
    submit_parser.add_argument("job_command", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "submit":
        return submit(args)
    spec = expand(read_json(args.spec), load_bindings())
    paths = validate_spec(spec)
    mode = state_conditioning(spec)
    if args.command == "preflight":
        report = {
            "status": "passed",
            "dataset": str(paths["dataset_root"]),
            "episodes": paths["info"]["total_episodes"],
            "frames": paths["info"]["total_frames"],
            "state_conditioning": mode,
            "action_shift_audit": paths["action_shift_audit"],
            "terminal_home_passed": paths["terminal_home_audit"].get("passed"),
            "terminal_home_audit": paths["terminal_home_audit"],
            "admission": {"checked": False, "note": "the certificate is verified by train-command"},
        }
        if args.admission_certificate:
            _, record = verify_launch_certificate(args.admission_certificate, args.admission_store, spec, paths)
            report["admission"] = {"checked": True, **record}
        print(json.dumps(report, indent=2))
    elif args.command == "norm-stats":
        compute_norm_stats(spec, paths, args.max_frames)
    elif args.command == "loader-smoke":
        loader_smoke(spec, paths)
    elif args.command == "train-command":
        train = build_train_command(spec, paths)
        admission, admission_record = resolve_admission(args, spec, args.spec, paths)
        contract_audit = audit_launch(spec, paths, train, admission=admission)
        command = build_scheduler_command(spec, paths, train)
        manifest = write_launch_manifest(args.spec, spec, paths, command, contract_audit,
                                         admission_record=admission_record)
        print(json.dumps({"launch_manifest": str(manifest), "admission_mode": admission.mode,
                          "command": command}, indent=2))
        if args.execute:
            # Preserve the scheduler's explicit capacity refusal (2), so a
            # durable queue can distinguish it from an unknown submission.
            return subprocess.run(command, check=False).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
