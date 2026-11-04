"""Current-observation-relative Piper EEF chunks (metres / wrapped Euler radians).

The grippers remain absolute. Every target in a chunk uses the SAME observation
anchor, never an adjacent target or a cumulative sum. This removes constant
coordinate offsets; it does not calibrate a pose-dependent kinematic error.
"""
from dataclasses import dataclass, replace
import numpy as np

MODE = "delta_eef_current_rpy_v1"
SUFFIX = "_delta_eef_current_rpy_v1"
POSE = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
ANGLES = (3, 4, 5, 10, 11, 12)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def convert(actions, state, *, inverse=False):
    actions = np.asarray(actions)
    state = np.asarray(state)
    if state.shape[-1] != 14 or actions.shape[-1] < 14:
        raise ValueError("delta EEF requires a measured 14D state and >=14D actions")
    if actions.ndim != state.ndim + 1 or actions.shape[:-2] != state.shape[:-1]:
        raise ValueError("one current observation anchor is required per action chunk")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("non-finite EEF state/action")
    out = np.array(actions, dtype=np.float32, copy=True)
    anchor = np.expand_dims(state, -2)
    out[..., POSE] += (1 if inverse else -1) * anchor[..., POSE]
    out[..., ANGLES] = wrap(out[..., ANGLES])
    return out


@dataclass(frozen=True)
class DeltaEef:
    inverse: bool = False

    def __call__(self, data):
        # Even state-free models need fresh measured state for reconstruction.
        state = np.asarray(data["state"])
        if self.inverse and state.shape[-1] == 32:
            state = state[..., :14]
        if state.shape[-1] != 14 or not np.isfinite(state).all():
            raise ValueError("delta EEF requires finite measured 14D state")
        if "actions" not in data:
            return data
        return {**data, "actions": convert(data["actions"], state, inverse=self.inverse)}


def install():
    """Register new configs without mutating the read-only trainer checkout."""
    from openpi.training import config as cfg
    from openpi import transforms

    @dataclass(frozen=True)
    class DeltaEefDataConfig(cfg.LeRobotPiperEefXyz3dDataConfig):
        def create(self, assets_dirs, model_config):
            data = super().create(assets_dirs, model_config)
            group = data.data_transforms
            return replace(data, data_transforms=transforms.Group(
                inputs=[*group.inputs, DeltaEef()],
                outputs=[DeltaEef(inverse=True), *group.outputs],
            ))

    for name, base in list(cfg._CONFIGS_DICT.items()):
        if name.endswith(SUFFIX) or not isinstance(base.data, cfg.LeRobotPiperEefXyz3dDataConfig):
            continue
        fields = {field.name: getattr(base.data, field.name)
                  for field in __import__('dataclasses').fields(base.data)}
        cfg._CONFIGS_DICT[name + SUFFIX] = replace(
            base, name=name + SUFFIX, data=DeltaEefDataConfig(**fields),
            policy_metadata={**(base.policy_metadata or {}), "action_representation": MODE,
                             "server_output": "absolute_eef", "anchor": "request_state"})
