"""Piper joint-position training contract and trainer config registration.

Datasets keep absolute joint targets.  In ``delta_joint_current_v1`` the model
sees each of the twelve arm joint targets relative to the current measured joint
state and inference applies the exact inverse using the fresh request state.  In
``absolute_joint_v1`` the model predicts the absolute joint targets directly, so
its output does not depend on the request state.  Both grippers stay absolute.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Mapping

import numpy as np


MODE = "delta_joint_current_v1"
SUFFIX = "_delta_joint_current_v1"
JOINT_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPER_INDICES = (6, 13)
FIELDS = (
    "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6",
    "left_gripper",
    "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6",
    "right_gripper",
)
DATASET_REPRESENTATION = "absolute_next_joint_current_gripper"
MODEL_REPRESENTATION = "delta_joint_current_absolute_gripper"
ABSOLUTE_MODE = "absolute_joint_v1"
ABSOLUTE_SUFFIX = "_absolute_joint_v1"
ABSOLUTE_MODEL_REPRESENTATION = "absolute_joint_absolute_gripper"
# mode -> (config suffix, model representation, relative to the current joint state)
MODES = {
    MODE: (SUFFIX, MODEL_REPRESENTATION, True),
    ABSOLUTE_MODE: (ABSOLUTE_SUFFIX, ABSOLUTE_MODEL_REPRESENTATION, False),
}
GRIPPER_RANGES = {"m": (0.0, 0.105), "normalized_0_1": (0.0, 1.0)}


def convert(actions: np.ndarray, state: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    """Convert absolute joint chunks to/from current-state-relative joint chunks."""

    actions = np.asarray(actions)
    state = np.asarray(state)
    if state.shape[-1] != 14 or actions.shape[-1] < 14:
        raise ValueError("delta joint actions require a measured 14D joint state and >=14D actions")
    if actions.ndim != state.ndim + 1 or actions.shape[:-2] != state.shape[:-1]:
        raise ValueError("one current joint-state anchor is required per action chunk")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("non-finite joint state/action")
    out = np.array(actions, dtype=np.float32, copy=True)
    anchor = np.expand_dims(state, -2)
    sign = 1 if inverse else -1
    out[..., JOINT_INDICES] += sign * anchor[..., JOINT_INDICES]
    return out


def convert_for_mode(actions: np.ndarray, state: np.ndarray, mode: str, *, inverse: bool = False) -> np.ndarray:
    """Map dataset absolute joint chunks into the model space of ``mode`` (or back)."""

    if mode not in MODES:
        raise ValueError(f"unsupported joint action mode {mode!r}")
    if MODES[mode][2]:
        return convert(actions, state, inverse=inverse)
    # Absolute targets: validate shapes exactly like the delta path, then pass through.
    convert(actions, state)
    return np.array(actions, dtype=np.float32, copy=True)


def validate_contract(contract: Mapping[str, object], mode: str = MODE) -> dict[str, object]:
    """Validate the explicit dataset/model unit contract used by the launcher."""

    fields_value = tuple(contract.get("fields") or ())
    units = tuple(contract.get("units") or ())
    if fields_value != FIELDS:
        raise ValueError(f"joint action fields must be {FIELDS}")
    if len(units) != 14 or any(units[index] != "rad" for index in JOINT_INDICES):
        raise ValueError("joint action units must declare radians for all twelve arm joints")
    gripper_units = {units[index] for index in GRIPPER_INDICES}
    if len(gripper_units) != 1 or next(iter(gripper_units), None) not in GRIPPER_RANGES:
        raise ValueError("both grippers must use one declared unit: 'm' or 'normalized_0_1'")
    if contract.get("dataset_representation") != DATASET_REPRESENTATION:
        raise ValueError(f"dataset_representation must be {DATASET_REPRESENTATION!r}")
    if mode not in MODES:
        raise ValueError(f"unsupported joint action mode {mode!r}")
    expected_model = MODES[mode][1]
    if contract.get("model_representation") != expected_model:
        raise ValueError(f"model_representation must be {expected_model!r} for {mode}")
    unit = next(iter(gripper_units))
    expected_range = GRIPPER_RANGES[unit]
    declared_range = tuple(float(value) for value in (contract.get("gripper_range") or ()))
    if declared_range != expected_range:
        raise ValueError(f"gripper_range for unit {unit!r} must be {list(expected_range)}")
    return {"fields": fields_value, "units": units, "gripper_unit": unit, "gripper_range": expected_range}


READBACK_SOURCE = "simulator actual joint readback at next retained source frame"
# Simulators whose export stores only the per-frame joint drive (PD) target, e.g. RoboTwin/SAPIEN white
# renders (HDF5 joint_action); the dataset must then audit target-vs-actual tracking.
DRIVE_TARGET_SOURCE = "simulator joint drive target at next retained source frame"
JOINT_ACTION_SOURCES = (READBACK_SOURCE, DRIVE_TARGET_SOURCE)


def validate_provenance_contract(
    training_contract: Mapping[str, object], dataset_contract: Mapping[str, object], mode: str = MODE
) -> dict[str, object]:
    """Bind the launch declaration to the dataset's stored state/action semantics."""

    validated = validate_contract(training_contract, mode)
    if tuple(dataset_contract.get("fields") or ()) != validated["fields"]:
        raise ValueError("Dataset joint field order disagrees with training_action_contract")
    if tuple(dataset_contract.get("units") or ()) != validated["units"]:
        raise ValueError("Dataset joint units disagree with training_action_contract")
    if dataset_contract.get("joint_unit") != "rad":
        raise ValueError("Dataset must declare joint_unit='rad'")
    if dataset_contract.get("gripper_unit") != validated["gripper_unit"]:
        raise ValueError("Dataset gripper unit disagrees with training_action_contract")
    range_key = "gripper_range_m" if validated["gripper_unit"] == "m" else "gripper_range"
    if tuple(float(value) for value in (dataset_contract.get(range_key) or ())) != validated["gripper_range"]:
        raise ValueError("Dataset gripper range disagrees with training_action_contract")
    joint_source = dataset_contract.get("action_joint_t")
    if joint_source not in JOINT_ACTION_SOURCES:
        raise ValueError(
            "Dataset joint action must be the next retained frame's simulator joint readback, or its declared "
            "joint drive target when the simulator export records no readback (with a tracking audit)"
        )
    if joint_source == DRIVE_TARGET_SOURCE and not isinstance(dataset_contract.get("joint_tracking_audit"), dict):
        raise ValueError("Drive-target joint labels must carry joint_tracking_audit against the actual pose")
    if dataset_contract.get("action_gripper_t") != "current retained source frame expert gripper command":
        raise ValueError("Dataset must keep gripper actions absolute at the current frame")
    return validated


def action_groups() -> dict[str, slice]:
    """Names used by the observable trainer's loss breakdown."""

    return {
        "left_joint_1_3": slice(0, 3),
        "left_joint_4_6": slice(3, 6),
        "left_gripper": slice(6, 7),
        "right_joint_1_3": slice(7, 10),
        "right_joint_4_6": slice(10, 13),
        "right_gripper": slice(13, 14),
        "left_arm": slice(0, 7),
        "right_arm": slice(7, 14),
    }


def install() -> None:
    """Register joint configs (every mode) without modifying the read-only trainer checkout."""

    from openpi.training import config as cfg

    source_types = (cfg.LeRobotPiperDataConfig, cfg.LeRobotPiperEefXyz3dDataConfig)
    suffixes = tuple(suffix for suffix, _, _ in MODES.values())
    joint_field_names = {field.name for field in fields(cfg.LeRobotPiperDataConfig)}
    for name, base in list(cfg._CONFIGS_DICT.items()):
        if name.endswith(suffixes) or not isinstance(base.data, source_types):
            continue
        available = {field.name: getattr(base.data, field.name) for field in fields(base.data)}
        for mode, (suffix, _, relative) in MODES.items():
            data_fields = {key: value for key, value in available.items() if key in joint_field_names}
            data_fields.update(use_delta_joint_actions=relative, adapt_to_pi=False)
            cfg._CONFIGS_DICT[name + suffix] = replace(
                base,
                name=name + suffix,
                data=cfg.LeRobotPiperDataConfig(**data_fields),
                policy_metadata={
                    **(base.policy_metadata or {}),
                    "action_representation": mode,
                    "dataset_actions": DATASET_REPRESENTATION,
                    "server_output": "absolute_joint",
                    "anchor": "request_state" if relative else "none",
                },
            )
