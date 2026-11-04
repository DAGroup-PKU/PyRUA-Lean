# Modified by the DataFarm authors from openpi (Apache-2.0); see vendor/trainer/README.md.
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data
            
        if "rtc_obs" in data:
            rtc_norm_stats = self.norm_stats.copy()
            rtc_norm_stats["rtc_obs"] = {
                "prev_state": self.norm_stats["state"],
                "prev_action": self.norm_stats["actions"]
            }
            return apply_tree(
                data,
                rtc_norm_stats,
                self._normalize_quantile if self.use_quantiles else self._normalize,
                strict=self.strict,
            )

        result = apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )
        # Normalize standalone prev_action using actions norm stats so it matches
        # the same space as delta+normalized actions used for GPR conditioning.
        if "prev_action" in data and "actions" in self.norm_stats:
            normalize_fn = self._normalize_quantile if self.use_quantiles else self._normalize
            result["prev_action"] = normalize_fn(data["prev_action"], self.norm_stats["actions"])
        return result

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data

def _apply_delta_pose_block(
    output: np.ndarray,
    current: np.ndarray,
    reference: np.ndarray,
    pos_slice: slice,
    quat_slice: slice,
    handled: np.ndarray,
    expand_reference: bool = False,
    reference_quat: np.ndarray | None = None,
) -> None:
    """Apply delta pose transformation using 4x4 homogeneous transformation matrices.
    
    Computes relative pose in local frame: T_delta = T_ref^-1 @ T_curr
    This correctly handles position transformation in the rotated reference frame.
    
    Args:
        output: Output array to write delta values to.
        current: Current pose values [x, y, z, qw, qx, qy, qz].
        reference: Reference pose to subtract from (prev_state for state, state for actions).
        pos_slice: Slice for position dimensions (e.g., slice(0, 3)).
        quat_slice: Slice for quaternion dimensions (e.g., slice(3, 7)).
        handled: Boolean array marking which dimensions have been handled.
        expand_reference: If True, expand reference dims for broadcasting with action sequences.
        reference_quat: Optional reference quaternion for ensuring continuity in delta quaternion.
                       If provided, delta quaternion will be flipped if dot product is negative.
    """
    # Extract pose components
    curr_pose = np.concatenate([
        current[..., pos_slice],
        current[..., quat_slice]
    ], axis=-1)
    
    if expand_reference:
        ref_pose = np.concatenate([
            np.expand_dims(reference[..., pos_slice], axis=-2),
            np.expand_dims(reference[..., quat_slice], axis=-2)
        ], axis=-1)
    else:
        ref_pose = np.concatenate([
            reference[..., pos_slice],
            reference[..., quat_slice]
        ], axis=-1)
    
    # Convert to homogeneous transformation matrices
    T_curr = _pose_to_homogeneous(curr_pose)
    T_ref = _pose_to_homogeneous(ref_pose)
    
    # Compute relative transform in local frame: T_delta = T_ref^-1 @ T_curr
    # This ensures position is correctly transformed in the rotated reference frame
    T_ref_inv = _invert_homogeneous(T_ref)
    T_delta = np.einsum('...ij,...jk->...ik', T_ref_inv, T_curr)
    
    # Convert back to pose representation with quaternion continuity checking
    delta_pose = _homogeneous_to_pose(T_delta, reference_quat=reference_quat)
    
    # Write output
    output[..., pos_slice] = delta_pose[..., :3]
    output[..., quat_slice] = delta_pose[..., 3:7]
    
    handled[pos_slice] = True
    handled[quat_slice] = True



@dataclasses.dataclass(frozen=True)
class DeltaPose(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "prev_state" not in data or self.mask is None:
            return data

        prev_state, state = data["prev_state"], data["state"]
        mask = np.asarray(self.mask)
        dims = min(mask.shape[-1], state.shape[-1])
        delta = state.copy()
        handled = np.zeros(dims, dtype=bool)

        # Process left and right arm pose blocks with quaternion continuity
        if dims >= 7 and mask[:7].all():
            _apply_delta_pose_block(
                delta, state, prev_state, 
                slice(0, 3), slice(3, 7), handled,
                reference_quat=state[..., 3:7]
            )
        if dims >= 15 and mask[8:15].all():
            _apply_delta_pose_block(
                delta, state, prev_state,
                slice(8, 11), slice(11, 15), handled,
                reference_quat=state[..., 11:15]
            )

        data["state"] = delta

        # Convert actions into relative pose w.r.t current state
        if "actions" in data:
            actions = data["actions"]
            adims = min(dims, actions.shape[-1])
            a_delta = actions.copy()
            a_handled = np.zeros(adims, dtype=bool)

            if adims >= 7 and mask[:7].all():
                _apply_delta_pose_block(
                    a_delta, actions, state, 
                    slice(0, 3), slice(3, 7), a_handled, 
                    expand_reference=True,
                    reference_quat=actions[..., 3:7]
                )
            if adims >= 15 and mask[8:15].all():
                _apply_delta_pose_block(
                    a_delta, actions, state,
                    slice(8, 11), slice(11, 15), a_handled,
                    expand_reference=True,
                    reference_quat=actions[..., 11:15]
                )

            data["actions"] = a_delta

        # Convert prev_action into relative pose w.r.t current state
        if "rtc_obs" in data and "prev_action" in data["rtc_obs"]:
            prev_actions = data["rtc_obs"]["prev_action"].copy()
            prev_adims = min(dims, prev_actions.shape[-1])
            prev_a_delta = prev_actions.copy()
            prev_a_handled = np.zeros(prev_adims, dtype=bool)

            if prev_adims >= 7 and mask[:7].all():
                _apply_delta_pose_block(
                    prev_a_delta, prev_actions, state, 
                    slice(0, 3), slice(3, 7), prev_a_handled, 
                    expand_reference=True,
                    reference_quat=prev_actions[..., 3:7]
                )
            if prev_adims >= 15 and mask[8:15].all():
                _apply_delta_pose_block(
                    prev_a_delta, prev_actions, state,
                    slice(8, 11), slice(11, 15), prev_a_handled,
                    expand_reference=True,
                    reference_quat=prev_actions[..., 11:15]
                )

            data["rtc_obs"]["prev_action"] = prev_a_delta
        
        return data


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8
    return q / norm


def _pose_to_homogeneous(pose: np.ndarray) -> np.ndarray:
    """Convert pose [x, y, z, qw, qx, qy, qz] to 4x4 homogeneous transformation matrix.
    
    Args:
        pose: Pose vector with shape (..., 7)
        
    Returns:
        4x4 homogeneous transformation matrix with shape (..., 4, 4)
    """
    pos = pose[..., :3]  # [x, y, z]
    quat = pose[..., 3:7]  # [qw, qx, qy, qz]
    
    # Normalize quaternion
    quat = _quat_normalize(quat)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Build rotation matrix (3x3)
    R = np.zeros(quat.shape[:-1] + (3, 3), dtype=pose.dtype)
    R[..., 0, 0] = 1 - 2*(y*y + z*z)
    R[..., 0, 1] = 2*(x*y - w*z)
    R[..., 0, 2] = 2*(x*z + w*y)
    R[..., 1, 0] = 2*(x*y + w*z)
    R[..., 1, 1] = 1 - 2*(x*x + z*z)
    R[..., 1, 2] = 2*(y*z - w*x)
    R[..., 2, 0] = 2*(x*z - w*y)
    R[..., 2, 1] = 2*(y*z + w*x)
    R[..., 2, 2] = 1 - 2*(x*x + y*y)
    
    # Build homogeneous transformation matrix
    T = np.zeros(quat.shape[:-1] + (4, 4), dtype=pose.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    
    return T


def _homogeneous_to_pose(T: np.ndarray, reference_quat: np.ndarray | None = None) -> np.ndarray:
    """Convert 4x4 homogeneous transformation matrix to pose [x, y, z, qw, qx, qy, qz].
    
    Args:
        T: 4x4 homogeneous transformation matrix with shape (..., 4, 4)
        reference_quat: Optional reference quaternion to ensure continuity. If provided and 
                       dot product with extracted quat is negative, the extracted quat is negated.
                       Shape should be (..., 4)
        
    Returns:
        Pose vector with shape (..., 7)
    """
    pos = T[..., :3, 3]  # Extract position
    R = T[..., :3, :3]  # Extract rotation matrix
    
    # Convert rotation matrix to quaternion
    trace = np.trace(R, axis1=-2, axis2=-1)
    w = np.sqrt(np.maximum(1 + trace, 0)) / 2
    x = np.copysign(np.sqrt(np.maximum(1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2], 0)) / 2,
                    R[..., 2, 1] - R[..., 1, 2])
    y = np.copysign(np.sqrt(np.maximum(1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2], 0)) / 2,
                    R[..., 0, 2] - R[..., 2, 0])
    z = np.copysign(np.sqrt(np.maximum(1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2], 0)) / 2,
                    R[..., 1, 0] - R[..., 0, 1])
    
    quat = np.stack([w, x, y, z], axis=-1)
    quat = _quat_normalize(quat)
    
    # Ensure quaternion continuity with reference if provided
    if reference_quat is not None:
        # If dot product is negative, flip quaternion to same hemisphere
        dot_product = np.sum(quat * reference_quat, axis=-1, keepdims=True)
        quat = np.where(dot_product < 0, -quat, quat)
    
    return np.concatenate([pos, quat], axis=-1)


def _invert_homogeneous(T: np.ndarray) -> np.ndarray:
    """Compute the inverse of a homogeneous transformation matrix.
    
    For T = [R  t]
            [0  1]
    The inverse is T^-1 = [R^T  -R^T @ t]
                          [0    1      ]
    
    Args:
        T: 4x4 homogeneous transformation matrix with shape (..., 4, 4)
        
    Returns:
        Inverse matrix with shape (..., 4, 4)
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    
    R_T = np.transpose(R, axes=[*range(len(R.shape)-2), -1, -2])
    t_inv = -np.einsum('...ij,...j->...i', R_T, t)
    
    T_inv = np.zeros_like(T)
    T_inv[..., :3, :3] = R_T
    T_inv[..., :3, 3] = t_inv
    T_inv[..., 3, 3] = 1.0
    
    return T_inv


def _apply_absolute_pose_block(
    output: np.ndarray,
    prev: np.ndarray,
    rel: np.ndarray,
    pos_slice: slice,
    quat_slice: slice,
    handled: np.ndarray,
    expand_state: bool = False,
    reference_quat: np.ndarray | None = None,
) -> None:
    """Apply absolute pose transformation using 4x4 homogeneous transformation matrices.
    
    Computes absolute pose in global frame: T_curr = T_prev @ T_rel
    This correctly handles position transformation in the reference frame.
    
    Args:
        output: Output array to write absolute values to.
        prev: Previous absolute pose values [x, y, z, qw, qx, qy, qz].
        rel: Relative pose values to add (delta pose).
        pos_slice: Slice for position dimensions (e.g., slice(0, 3)).
        quat_slice: Slice for quaternion dimensions (e.g., slice(3, 7)).
        handled: Boolean array marking which dimensions have been handled.
        expand_state: If True, expand state dims for broadcasting with action sequences.
        reference_quat: Optional reference quaternion for ensuring continuity in resulting quaternion.
                       If provided, result quaternion will be flipped if dot product is negative.
    """
    # Extract pose components
    prev_pose = np.concatenate([
        prev[..., pos_slice],
        prev[..., quat_slice]
    ], axis=-1)
    
    rel_pose = np.concatenate([
        rel[..., pos_slice],
        rel[..., quat_slice]
    ], axis=-1)
    
    # Convert to homogeneous transformation matrices
    T_prev = _pose_to_homogeneous(prev_pose)
    T_rel = _pose_to_homogeneous(rel_pose)
    
    # Expand state dimension for broadcasting with action sequence if needed
    if expand_state:
        T_prev = np.expand_dims(T_prev, axis=-3)
    
    # Compute absolute pose: T_curr = T_prev @ T_rel
    T_curr = np.einsum('...ij,...jk->...ik', T_prev, T_rel)
    
    # Convert back to pose representation with quaternion continuity checking
    curr_pose = _homogeneous_to_pose(T_curr, reference_quat=reference_quat)
    
    # Write output
    output[..., pos_slice] = curr_pose[..., :3]
    output[..., quat_slice] = curr_pose[..., 3:7]
    
    handled[pos_slice] = True
    handled[quat_slice] = True


@dataclasses.dataclass(frozen=True)
class AbsolutePose(DataTransformFn):
    """Reconstruct absolute pose/state and actions from relative pose using 4x4 homogeneous matrices.

    Uses `prev_state` absolute pose and `state` relative pose (in local frame) to compute
    current absolute `state` via: T_curr = T_prev @ T_delta
    Then converts relative `actions` w.r.t. state into absolute `actions`.
    """

    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "prev_state" not in data or self.mask is None:
            return data

        prev_state, rel_state = data["prev_state"], data["state"]
        mask = np.asarray(self.mask)
        dims = min(mask.shape[-1], rel_state.shape[-1])

        abs_state = rel_state.copy()
        handled = np.zeros(dims, dtype=bool)

        # Process left and right arm pose blocks with quaternion continuity
        if dims >= 7 and mask[:7].all():
            _apply_absolute_pose_block(
                abs_state, prev_state, rel_state,
                slice(0, 3), slice(3, 7), handled,
                reference_quat=prev_state[..., 3:7]
            )
        if dims >= 15 and mask[8:15].all():
            _apply_absolute_pose_block(
                abs_state, prev_state, rel_state,
                slice(8, 11), slice(11, 15), handled,
                reference_quat=prev_state[..., 11:15]
            )

        # Fallback: element-wise add for masked indices not covered by pose blocks.
        data["state"] = abs_state

        # Convert relative actions (w.r.t current state) into absolute actions.
        if "actions" in data:
            actions = data["actions"]
            action_dims = actions.shape[-1]
            adims = min(dims, action_dims)
            abs_actions = actions.copy()
            a_handled = np.zeros(adims, dtype=bool)

            # Left arm pose block
            if adims >= 7 and mask[:7].all():
                _apply_absolute_pose_block(
                    abs_actions, abs_state, actions,
                    slice(0, 3), slice(3, 7), a_handled,
                    expand_state=True,
                    reference_quat=abs_state[..., 3:7]
                )

            # Right arm pose block
            if adims >= 15 and mask[8:15].all():
                _apply_absolute_pose_block(
                    abs_actions, abs_state, actions,
                    slice(8, 11), slice(11, 15), a_handled,
                    expand_state=True,
                    reference_quat=abs_state[..., 11:15]
                )

            # Fallback: element-wise add for masked indices not covered by pose blocks.
            data["actions"] = abs_actions

        return data

@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.mask is None:
            return data

        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        state = data["state"]

        # Apply delta to actions (training only - actions not present at inference)
        if "actions" in data:
            actions = data["actions"]
            actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
            data["actions"] = actions

        # Apply same delta to standalone prev_action independently of actions presence.
        # This MUST run at both training AND inference (GPR needs y_p in delta space).
        # Mirrors DeltaActions_Prev which does the same for rtc_obs["prev_action"].
        if "prev_action" in data:
            prev_actions = np.array(data["prev_action"], dtype=np.float32)
            prev_actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
            data["prev_action"] = prev_actions

        return data

@dataclasses.dataclass(frozen=True)
class DeltaActions_Prev(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "rtc_obs" not in data or self.mask is None:
            return data

        state, actions = data["state"].copy(), data["rtc_obs"]["prev_action"].copy()
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["rtc_obs"]["prev_action"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False
    advantage_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()
        
        if self.advantage_input:
            if (advantage := data.pop("advantage", None)) is not None:
                if not isinstance(advantage, str):
                    advantage = advantage.item()
                elif advantage.startswith("b'"):
                    advantage = advantage[2:-1]
            else: advantage = "advantage: positive"
        else:
            advantage = None

        tokens, token_masks = self.tokenizer.tokenize(prompt, state, advantage)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        # Pad standalone prev_action to model_action_dim so GPR noise output is correct dim.
        if "prev_action" in data:
            data["prev_action"] = pad_to_dim(np.asarray(data["prev_action"], dtype=np.float32), self.model_action_dim, axis=-1)
        return data

@dataclasses.dataclass(frozen=True)
class PadStatesAndActions_Prev(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "rtc_obs" not in data:
            return data
        data["rtc_obs"]["prev_state"] = pad_to_dim(data["rtc_obs"]["prev_state"], self.model_action_dim, axis=-1)
        data["rtc_obs"]["prev_action"] = pad_to_dim(data["rtc_obs"]["prev_action"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
