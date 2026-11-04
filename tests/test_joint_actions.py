from __future__ import annotations

import dataclasses
import sys
import types

import numpy as np
import pytest

from datafarm.training.joint_actions import (
    ABSOLUTE_MODE,
    ABSOLUTE_MODEL_REPRESENTATION,
    ABSOLUTE_SUFFIX,
    DATASET_REPRESENTATION,
    FIELDS,
    GRIPPER_INDICES,
    JOINT_INDICES,
    MODE,
    MODEL_REPRESENTATION,
    SUFFIX,
    action_groups,
    convert,
    convert_for_mode,
    install,
    validate_contract,
    validate_provenance_contract,
)


def contract(gripper_unit="m", gripper_range=(0.0, 0.105), model_representation=MODEL_REPRESENTATION):
    return {
        "fields": FIELDS,
        "units": tuple("rad" if index in JOINT_INDICES else gripper_unit for index in range(14)),
        "dataset_representation": DATASET_REPRESENTATION,
        "model_representation": model_representation,
        "gripper_range": gripper_range,
    }


def test_absolute_joint_roundtrip_uses_one_current_anchor_and_keeps_grippers():
    rng = np.random.default_rng(8)
    state = rng.normal(size=(3, 14)).astype(np.float32)
    state[:, GRIPPER_INDICES] = rng.uniform(0.0, 0.105, size=(3, 2))
    actions = rng.normal(size=(3, 50, 14)).astype(np.float32)
    actions[..., GRIPPER_INDICES] = rng.uniform(0.0, 0.105, size=(3, 50, 2))

    delta = convert(actions, state)
    reconstructed = convert(delta, state, inverse=True)

    np.testing.assert_allclose(reconstructed, actions, atol=1e-6)
    np.testing.assert_array_equal(delta[..., GRIPPER_INDICES], actions[..., GRIPPER_INDICES])
    np.testing.assert_allclose(delta[..., JOINT_INDICES],
                               actions[..., JOINT_INDICES] - state[:, None, JOINT_INDICES], atol=1e-7)


def test_joint_transform_rejects_missing_or_nonfinite_measured_anchor():
    with pytest.raises(ValueError, match="14D joint state"):
        convert(np.zeros((50, 14)), np.zeros(13))
    with pytest.raises(ValueError, match="one current joint-state anchor"):
        convert(np.zeros((2, 50, 14)), np.zeros(14))
    with pytest.raises(ValueError, match="non-finite"):
        convert(np.zeros((50, 14)), np.full(14, np.nan))


def test_explicit_units_accept_metres_or_normalized_grippers_only():
    validated = validate_contract(contract())
    assert validated["gripper_unit"] == "m"
    assert validated["gripper_range"] == (0.0, 0.105)
    assert validate_contract(contract("normalized_0_1", (0.0, 1.0)))["gripper_unit"] == "normalized_0_1"
    with pytest.raises(ValueError, match="radians"):
        validate_contract({**contract(), "units": ("m",) * 14})
    with pytest.raises(ValueError, match="gripper_range"):
        validate_contract(contract("m", (0.0, 1.0)))


def test_dataset_contract_must_match_units_and_next_readback_semantics():
    dataset = {
        "fields": FIELDS,
        "units": contract()["units"],
        "joint_unit": "rad",
        "gripper_unit": "m",
        "gripper_range_m": [0.0, 0.105],
        "action_joint_t": "simulator actual joint readback at next retained source frame",
        "action_gripper_t": "current retained source frame expert gripper command",
    }
    assert validate_provenance_contract(contract(), dataset)["gripper_unit"] == "m"
    with pytest.raises(ValueError, match="next retained"):
        validate_provenance_contract(contract(), {**dataset, "action_joint_t": "command.qpos_rad"})
    drive = {**dataset, "action_joint_t": "simulator joint drive target at next retained source frame"}
    with pytest.raises(ValueError, match="joint_tracking_audit"):
        validate_provenance_contract(contract(), drive)
    audited = {**drive, "joint_tracking_audit": {"rotation_vs_wrist_camera_deg_max": 3.1}}
    assert validate_provenance_contract(contract(), audited)["gripper_unit"] == "m"


def test_observable_metric_groups_name_joint_coordinates():
    groups = action_groups()
    assert "left_translation" not in groups
    assert groups["left_joint_1_3"] == slice(0, 3)
    assert groups["right_joint_4_6"] == slice(10, 13)


def test_install_converts_eef_base_to_joint_config_without_mutating_base(monkeypatch):
    @dataclasses.dataclass(frozen=True)
    class Assets:
        asset_id: str = "asset"

    @dataclasses.dataclass(frozen=True)
    class EefData:
        repo_id: str = "repo"
        assets: Assets = dataclasses.field(default_factory=Assets)
        base_config: object = None
        default_prompt: str | None = "prompt"
        repack_transforms: object = "three-cameras"
        action_sequence_keys: tuple[str, ...] = ("action",)

    @dataclasses.dataclass(frozen=True)
    class JointData(EefData):
        use_delta_joint_actions: bool = False
        adapt_to_pi: bool = True

    @dataclasses.dataclass(frozen=True)
    class Train:
        name: str
        data: object
        policy_metadata: dict | None = None

    base = Train("eef_base", EefData())
    cfg = types.SimpleNamespace(
        LeRobotPiperDataConfig=JointData,
        LeRobotPiperEefXyz3dDataConfig=EefData,
        _CONFIGS_DICT={base.name: base},
    )
    openpi = types.ModuleType("openpi")
    training = types.ModuleType("openpi.training")
    training.config = cfg
    openpi.training = training
    monkeypatch.setitem(sys.modules, "openpi", openpi)
    monkeypatch.setitem(sys.modules, "openpi.training", training)

    install()
    install()  # idempotent: generated configs are never used as new bases

    registered = cfg._CONFIGS_DICT[base.name + SUFFIX]
    assert registered.data.use_delta_joint_actions is True
    assert registered.data.adapt_to_pi is False
    assert registered.data.repack_transforms == "three-cameras"
    assert registered.policy_metadata["action_representation"] == MODE
    absolute = cfg._CONFIGS_DICT[base.name + ABSOLUTE_SUFFIX]
    assert absolute.data.use_delta_joint_actions is False
    assert absolute.data.adapt_to_pi is False
    assert absolute.data.repack_transforms == "three-cameras"
    assert absolute.policy_metadata["action_representation"] == ABSOLUTE_MODE
    assert absolute.policy_metadata["anchor"] == "none"
    assert sorted(cfg._CONFIGS_DICT) == sorted([base.name, base.name + SUFFIX, base.name + ABSOLUTE_SUFFIX])
    assert base.data == EefData()


def test_absolute_mode_passes_targets_through_independent_of_state():
    rng = np.random.default_rng(3)
    actions = rng.normal(size=(2, 50, 14)).astype(np.float32)
    state = rng.normal(size=(2, 14)).astype(np.float32)
    out = convert_for_mode(actions, state, ABSOLUTE_MODE)
    np.testing.assert_array_equal(out, actions)
    np.testing.assert_array_equal(convert_for_mode(actions, state + 1.0, ABSOLUTE_MODE), out)
    assert out is not actions
    np.testing.assert_allclose(convert_for_mode(actions, state, MODE), convert(actions, state))
    with pytest.raises(ValueError, match="one current joint-state anchor"):
        convert_for_mode(actions, state[0], ABSOLUTE_MODE)
    with pytest.raises(ValueError, match="unsupported"):
        convert_for_mode(actions, state, "absolute_eef")


def test_contract_model_representation_must_match_mode():
    absolute = contract(model_representation=ABSOLUTE_MODEL_REPRESENTATION)
    assert validate_contract(absolute, ABSOLUTE_MODE)["gripper_unit"] == "m"
    with pytest.raises(ValueError, match="model_representation"):
        validate_contract(absolute)  # default delta mode rejects an absolute declaration
    with pytest.raises(ValueError, match="model_representation"):
        validate_contract(contract(), ABSOLUTE_MODE)
