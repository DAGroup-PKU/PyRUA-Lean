# Modified by the DataFarm authors from openpi (Apache-2.0); see vendor/trainer/README.md.
import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.gp_utils_pytorch as _gp_utils_pt
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = self.sample_actions #torch.compile(self.sample_actions, mode="max-autotune") # remove torch.compile for debug

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            getattr(observation, "prev_state", None),
            getattr(observation, "prev_action", None),
            getattr(observation, "action_horizon", None),
            getattr(observation, "actions_during_latency", None),
            # gpr_prev_action: set only when RTC+GPR are both active (from obs['prev_action']).
            # In non-RTC mode this is None and prev_action is used for GPR instead.
            getattr(observation, "gpr_prev_action", None),
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state, prev_state, prev_action, action_horizon, actions_during_latency, _gpr_prev_action = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]

        # Preprocess observation first so prev_action is available for GPR noise conditioning
        images, img_masks, lang_tokens, lang_masks, state, prev_state, prev_action, action_horizon, actions_during_latency, gpr_prev_action = self._preprocess_observation(observation, train=False)

        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            # GPR data source: prefer gpr_prev_action (execution-history snapshot, from obs['prev_action']);
            # in non-RTC mode gpr_prev_action=None, so fall back to prev_action.
            gpr_src = gpr_prev_action if gpr_prev_action is not None else prev_action
            if self.config.use_gpr and gpr_src is not None:
                # GPR: condition noise on executed action history (gpr_prev_action from obs['prev_action']).
                # In RTC mode, gpr_prev_action = pre-slicer snapshot (gpr_prev_actions); in non-RTC mode
                # gpr_prev_action is None and prev_action is used as fallback.
                y_p = gpr_src[:, :self.config.gpr_past_horizon, :]
            elif self.config.use_gpr:
                # GPR cold start: no prior chunk yet → neutral zero prior.
                y_p = torch.zeros(
                    bsize, self.config.gpr_past_horizon, self.config.action_dim,
                    dtype=torch.float32, device=device
                )
            if self.config.use_gpr:
                noise = _gp_utils_pt.sample_gpr_noise(
                    y_p,
                    past_horizon=self.config.gpr_past_horizon,
                    action_horizon=self.config.action_horizon,
                    kernel_type=self.config.gp_kernel_type,
                    length_scale=self.config.gp_length_scale,
                )
                # gpr_src is delta+normalized+padded to 32-dim via transforms,
                # so noise is already the correct shape — no extra pad needed.
            elif self.config.use_gp_prior:
                # Unconditional GP noise (matches JAX pi0.py L297-305)
                noise = _gp_utils_pt.sample_gp_noise(
                    shape=actions_shape,
                    action_horizon=self.config.action_horizon,
                    kernel_type=self.config.gp_kernel_type,
                    length_scale=self.config.gp_length_scale,
                    device=device,
                )
            else:
                noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            # v_t = self.denoise_step(
            #     state,
            #     prefix_pad_masks,
            #     past_key_values,
            #     x_t,
            #     expanded_time,
            # )

            def denoise_step_partial_call(input_x_t, current_timestep=expanded_time):
                return self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    input_x_t,
                    current_timestep,
                )
            if actions_during_latency is None:
                # GPR-only mode OR no prev_action: use standard denoising
                v_t = denoise_step_partial_call(x_t, expanded_time)
            else:
                # Full RTC mode: prev_action + action_horizon both provided
                # Convert to Python int if they are Tensor
                inference_delay = actions_during_latency
                if inference_delay is not None:
                    if isinstance(inference_delay, Tensor):
                        inference_delay = int(inference_delay.item() if inference_delay.numel() == 1 else inference_delay[0].item())
                    else:
                        inference_delay = int(inference_delay)
                
                prev_chunk_left_over = prev_action
                execution_horizon = action_horizon
                if isinstance(execution_horizon, Tensor):
                    execution_horizon = int(execution_horizon.item() if execution_horizon.numel() == 1 else execution_horizon[0].item())
                else:
                    execution_horizon = int(execution_horizon)
                
                v_t = self.denoise_step_rtc(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=expanded_time,
                    execution_horizon=execution_horizon,
                    num_steps=num_steps,
                    mask_schedule="exp",
                    original_denoise_step_partial=denoise_step_partial_call,
                )
            
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        if actions_during_latency is not None and prev_action is not None:
            error = prev_action[0, :actions_during_latency, ...] - x_t[0, :actions_during_latency, ...]
            print(f"overlap part error: {error.norm()}")
        return x_t

    def denoise_step_rtc(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        execution_horizon,
        num_steps,
        mask_schedule,
        original_denoise_step_partial,

    ):
        tau = 1 - time

        if prev_chunk_left_over is None:
            # First step, no guidance - return v_t
            v_t = original_denoise_step_partial(x_t)
            return v_t
        
        x_t = x_t.clone().detach()

        squeezed = False
        if len(x_t.shape) < 3:
            # Add batch dimension
            x_t = x_t.unsqueeze(0)
            squeezed = True

        if len(prev_chunk_left_over.shape) < 3:
            # Add batch dimension
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

        # If the previous action chunk is to short then it doesn't make sense to use long execution horizon
        # because there is nothing to merge
        # Convert execution_horizon to int if it's a Tensor
        if execution_horizon is not None:
            if isinstance(execution_horizon, Tensor):
                execution_horizon = int(execution_horizon.item() if execution_horizon.numel() == 1 else execution_horizon[0].item())
            execution_horizon = int(execution_horizon)
            if execution_horizon > prev_chunk_left_over.shape[1]:
                execution_horizon = prev_chunk_left_over.shape[1]
        else:
            execution_horizon = prev_chunk_left_over.shape[1]

        batch_size = x_t.shape[0]
        action_chunk_size = x_t.shape[1]
        action_dim = x_t.shape[2]

        if prev_chunk_left_over.shape[1] < action_chunk_size or prev_chunk_left_over.shape[2] < action_dim:
            padded = torch.zeros(batch_size, action_chunk_size, action_dim).to(x_t.device)
            padded[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
            prev_chunk_left_over = padded

        assert prev_chunk_left_over.shape == x_t.shape, (
            "The padded previous chunk must be the same size as the input tensor"
        )

        weights = (
            self.get_prefix_weights(inference_delay, execution_horizon, action_chunk_size, mask_schedule)
            .to(x_t.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        with torch.enable_grad():
            v_t = original_denoise_step_partial(x_t)
            x_t.requires_grad_(True)

            x1_t = x_t - time * v_t  # noqa: N806
            err = (prev_chunk_left_over - x1_t) * weights
            grad_outputs = err.clone().detach()
            correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

        max_guidance_weight = torch.as_tensor(num_steps)
        tau_tensor = torch.as_tensor(tau)
        squared_one_minus_tau = (1 - tau_tensor) ** 2
        inv_r2 = (squared_one_minus_tau + tau_tensor**2) / (squared_one_minus_tau)
        c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
        guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        result = v_t - guidance_weight * correction

        # Remove the batch dimension if it was added
        if squeezed:
            result = result.squeeze(0)
            correction = correction.squeeze(0)
            x1_t = x1_t.squeeze(0)
            err = err.squeeze(0)

        return result

    def get_prefix_weights(self, start, end, total, mask_schedule):
        # Convert to Python int if they are Tensor
        if isinstance(start, Tensor):
            start = int(start.item() if start.numel() == 1 else start[0].item())
        if isinstance(end, Tensor):
            end = int(end.item() if end.numel() == 1 else end[0].item())
        if isinstance(total, Tensor):
            total = int(total.item() if total.numel() == 1 else total[0].item())
        
        start = min(start, end)

        if mask_schedule == "zeros":
            weights = torch.zeros(total)
            weights[:start] = 1.0
        elif mask_schedule == "ones":
            weights = torch.ones(total)
            weights[end:] = 0.0
        elif mask_schedule == "linear":
            lin_weights = self._linweights(start, end, total)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)
        elif mask_schedule == "exp":
            lin_weights = self._linweights(start, end, total)
            lin_weights = lin_weights * torch.expm1(lin_weights).div(math.e - 1)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)

        return weights

    def _linweights(self, start, end, total):
        # Ensure all parameters are Python int
        start = int(start) if not isinstance(start, int) else start
        end = int(end) if not isinstance(end, int) else end
        total = int(total) if not isinstance(total, int) else total
        
        skip_steps_at_end = max(total - end, 0)
        linspace_steps = total - skip_steps_at_end - start

        if end <= start or linspace_steps <= 0:
            return torch.tensor([])

        # Ensure linspace_steps is Python int
        linspace_steps = int(linspace_steps)
        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    def _add_trailing_zeros(self, weights, total, end):
        # Convert to Python int if they are Tensor
        if isinstance(total, Tensor):
            total = int(total.item() if total.numel() == 1 else total[0].item())
        if isinstance(end, Tensor):
            end = int(end.item() if end.numel() == 1 else end[0].item())
        
        zeros_len = total - end

        if zeros_len <= 0:
            return weights

        zeros_len = int(zeros_len)
        zeros = torch.zeros(zeros_len)
        return torch.cat([weights, zeros])

    def _add_leading_ones(self, weights, start, total):
        # Convert to Python int if they are Tensor
        if isinstance(start, Tensor):
            start = int(start.item() if start.numel() == 1 else start[0].item())
        if isinstance(total, Tensor):
            total = int(total.item() if total.numel() == 1 else total[0].item())
        
        ones_len = min(start, total)

        if ones_len <= 0:
            return weights

        ones_len = int(ones_len)
        ones = torch.ones(ones_len)
        return torch.cat([ones, weights])

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
