# Modified by the DataFarm authors from openpi (Apache-2.0); see vendor/trainer/README.md.
import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    advantage_input: bool = False

    # --------------- Gaussian Process prior settings ---------------
    # When enabled, replaces i.i.d. Gaussian noise with GP-structured noise
    # that introduces temporal correlations across the action horizon.
    # action_horizon=50 (predicted), execution_horizon=15 (executed).
    use_gp_prior: bool = False
    # Kernel function: "SE" (smooth), "OU" (rough/Brownian), "PE" (periodic)
    gp_kernel_type: str = "SE"
    # Length scale ℓ. Recommended: SE=5.0, OU=3.0, PE=2.0 (for exec_horizon=15)
    gp_length_scale: float = 5.0

    # --------------- Mini-batch OT coupling settings ---------------
    # When enabled, re-pairs GP noise samples with actions within each batch
    # using Sinkhorn OT to minimise the expected flow-matching transport cost.
    # Can be combined with use_gp_prior (GP for temporal structure, OT for pairing).
    use_ot_coupling: bool = False
    # Sinkhorn regularisation ε. Smaller → harder (permutation-like) assignment.
    # Larger → softer/more stable. Recommended range: [0.05, 0.5].
    ot_reg: float = 0.1
    # Number of Sinkhorn iterations (20 is sufficient for most batch sizes).
    ot_num_iters: int = 20

    # --------------- GPR (Gaussian Process Regression) settings ---------------
    # Conditions the initial noise on the previous chunk's executed actions y_p,
    # achieving cross-chunk temporal continuity without OT Coupling.
    # MUTUALLY EXCLUSIVE with use_ot_coupling.
    use_gpr: bool = False
    # Number of past action steps to condition on (execution horizon).
    gpr_past_horizon: int = 15

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        # Validate GP settings
        if self.use_gp_prior:
            valid_kernels = ("SE", "OU", "PE")
            if self.gp_kernel_type not in valid_kernels:
                raise ValueError(
                    f"gp_kernel_type must be one of {valid_kernels}, got {self.gp_kernel_type!r}"
                )
            if self.gp_length_scale <= 0:
                raise ValueError(
                    f"gp_length_scale must be positive, got {self.gp_length_scale}"
                )
        # Validate OT settings
        if self.use_ot_coupling:
            if self.ot_reg <= 0:
                raise ValueError(f"ot_reg must be positive, got {self.ot_reg}")
            if self.ot_num_iters < 1:
                raise ValueError(f"ot_num_iters must be >= 1, got {self.ot_num_iters}")
        # GPR and OT are mutually exclusive
        if self.use_gpr and self.use_ot_coupling:
            raise ValueError(
                "use_gpr and use_ot_coupling are mutually exclusive: "
                "OT re-pairs noise across the batch, destroying GPR's per-sample conditioning."
            )
        if self.use_gpr and self.gpr_past_horizon <= 0:
            raise ValueError(f"gpr_past_horizon must be positive, got {self.gpr_past_horizon}")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
