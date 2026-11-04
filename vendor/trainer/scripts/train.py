# Modified by the DataFarm authors from openpi (Apache-2.0); see vendor/trainer/README.md.
import dataclasses
import functools
import json
import logging
import os
import platform
from pathlib import Path
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


ACTIVE_ACTION_DIM = 14
ACTION_GROUPS = {
    "left_translation": slice(0, 3),
    "left_rotation": slice(3, 6),
    "left_gripper": slice(6, 7),
    "right_translation": slice(7, 10),
    "right_rotation": slice(10, 13),
    "right_gripper": slice(13, 14),
    "left_arm": slice(0, 7),
    "right_arm": slice(7, 14),
}
HORIZON_GROUPS = {
    "early_00_16": slice(0, 17),
    "middle_17_33": slice(17, 34),
    "late_34_49": slice(34, 50),
}


def _loss_breakdown(squared_error: at.Array, flow_time: at.Array) -> dict[str, at.Array]:
    """Reduce PI0.5 diagnostics to scalars without changing the objective."""

    metrics: dict[str, at.Array] = {
        "loss/active_dims": jnp.mean(squared_error[..., :ACTIVE_ACTION_DIM]),
        "loss/padded_dims": jnp.mean(squared_error[..., ACTIVE_ACTION_DIM:]),
    }
    for name, indices in ACTION_GROUPS.items():
        metrics[f"loss/group/{name}"] = jnp.mean(squared_error[..., indices])
    for index in range(ACTIVE_ACTION_DIM):
        metrics[f"loss/action_dim_{index:02d}"] = jnp.mean(squared_error[..., index])
    for name, indices in HORIZON_GROUPS.items():
        metrics[f"loss/horizon/{name}"] = jnp.mean(squared_error[:, indices, :])

    sample_loss = jnp.mean(squared_error, axis=(-2, -1))
    for name, lower, upper in (
        ("low_0_1over3", 0.0, 1.0 / 3.0),
        ("mid_1over3_2over3", 1.0 / 3.0, 2.0 / 3.0),
        ("high_2over3_1", 2.0 / 3.0, 1.001),
    ):
        mask = jnp.logical_and(flow_time >= lower, flow_time < upper)
        metrics[f"loss/flow_time/{name}_sum"] = jnp.sum(jnp.where(mask, sample_loss, 0.0))
        metrics[f"loss/flow_time/{name}_count"] = jnp.sum(mask)
    return metrics


def _finalize_interval_metrics(
    stacked_infos: dict[str, at.Array], *, flow_prefix: str = "loss/flow_time"
) -> dict[str, Any]:
    """Aggregate a logging interval, including count-weighted flow-time bins."""

    reduced = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
    for name in ("low_0_1over3", "mid_1over3_2over3", "high_2over3_1"):
        sum_key = f"{flow_prefix}/{name}_sum"
        count_key = f"{flow_prefix}/{name}_count"
        count = float(reduced[count_key])
        reduced[f"{flow_prefix}/{name}"] = float(reduced[sum_key]) / max(count, 1.0)
        del reduced[sum_key]
    return reduced


def _write_jsonl(stream: Any, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, sort_keys=True) + "\n")
    stream.flush()


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:


    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        # Only request auxiliary output for models that support it (Pi0, not Pi0FAST)
        # Pi0FAST doesn't use noise/flow matching, so it won't have aux data
        if hasattr(model, 'pi05') or 'Pi0' in model.__class__.__name__:
            loss, aux = model.compute_loss(rng, observation, actions, train=True, return_aux=True)
            chunked_loss = loss
            return jnp.mean(chunked_loss), aux
        else:
            # For models without aux support (e.g., Pi0FAST)
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            return jnp.mean(chunked_loss), {}


    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng, observation, actions)




    params = state.params.filter(config.trainable_filter)
    grad_norm = optax.global_norm(grads)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    update_norm = optax.global_norm(updates)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": grad_norm,
        "param_norm": optax.global_norm(kernel_params),
        "optimizer/learning_rate": config.lr_schedule.create()(state.step),
        "optimizer/update_norm": update_norm,
        "optimizer/update_to_param_ratio": update_norm / jnp.maximum(optax.global_norm(kernel_params), 1e-12),
        "optimizer/gradient_clipped": (
            grad_norm > getattr(config.optimizer, "clip_gradient_norm", jnp.inf)
        ).astype(jnp.float32),
        "numerics/loss_finite": jnp.isfinite(loss).astype(jnp.float32),
        "numerics/grad_norm_finite": jnp.isfinite(grad_norm).astype(jnp.float32),
    }
    if "squared_error" in aux:
        info.update(_loss_breakdown(aux["squared_error"], aux["time"]))
    # Noise capture is an opt-in legacy path.  Do not return large tensors for
    # ordinary PI0.5 training after their scalar diagnostics are reduced.
    if "pi05_piper_pick_cube" in config.name:
        info.update(
            {
                "aux": aux,
                "episode_index": observation.episode_index,
                "frame_index": observation.frame_index,
                "actions": actions,
            }
        )
    return new_state, info


@at.typecheck
def eval_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """Evaluate a frozen diagnostic batch with fixed flow noise/time."""

    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    loss, aux = model.compute_loss(rng, observation, actions, train=False, return_aux=True)
    metrics = {"eval_fixed/loss": jnp.mean(loss)}
    for key, value in _loss_breakdown(aux["squared_error"], aux["time"]).items():
        metrics[f"eval_fixed/{key}"] = value
    return metrics



def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images only for a fresh run. On resume, this would gather a sharded
    # batch before checkpoint restoration and attempt to append data at step 0
    # to an existing W&B history. Besides being redundant, that gather can
    # cause an avoidable NCCL memory spike when the node hosts another job.
    if not resuming:
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),

        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    peval_step = jax.jit(
        functools.partial(eval_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    # The first transformed batch is frozen as a deterministic diagnostic
    # slice.  It is not a held-out set because this experiment explicitly uses
    # all 100 episodes for training; the distinction is recorded in the schema.
    fixed_eval_batch = batch
    eval_rng = jax.random.key(config.seed + 1_000_003)

    metrics_path_value = os.environ.get("TRAIN_METRICS_JSONL")
    metrics_stream = None
    if metrics_path_value:
        metrics_path = Path(metrics_path_value)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_stream = metrics_path.open("a", encoding="utf-8", buffering=1)
        schema_path = metrics_path.with_name("metrics-schema.json")
        schema_path.write_text(
            json.dumps(
                {
                    "schema_version": "pi05_comprehensive_training_metrics/1.0.0",
                    "total_loss": "Mean flow-matching squared error over batch, 50-step horizon and 32 model dims.",
                    "active_action_dims": 14,
                    "padded_action_dims": 18,
                    "action_groups": {key: [value.start, value.stop] for key, value in ACTION_GROUPS.items()},
                    "horizon_groups": {key: [value.start, value.stop] for key, value in HORIZON_GROUPS.items()},
                    "flow_time_bins": ["[0,1/3)", "[1/3,2/3)", "[2/3,1]"],
                    "eval_fixed": "Fixed first transformed batch with fixed RNG; diagnostic slice, not held-out validation.",
                    "local_log": str(metrics_path),
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )


    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    interval_started = time.monotonic()
    interval_data_wait = 0.0
    interval_steps = 0
    eval_interval = int(os.environ.get("TRAIN_EVAL_INTERVAL", "1000"))
    try:
        for step in pbar:
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
            interval_steps += 1
            
        # Remove aux data for logging to wandb/console to avoid clutter/errors and memory usage
            info = {k: v for k, v in info.items() if k not in ["aux", "episode_index", "frame_index", "actions"]}

            infos.append(info)


            if step % config.log_interval == 0:
                stacked_infos = common_utils.stack_forest(infos)
                reduced_info = _finalize_interval_metrics(stacked_infos)
                now = time.monotonic()
                reduced_info["performance/seconds_per_step"] = (now - interval_started) / max(interval_steps, 1)
                reduced_info["performance/data_wait_seconds_per_step"] = interval_data_wait / max(interval_steps, 1)
                reduced_info["performance/samples_per_second"] = (
                    config.batch_size / max(reduced_info["performance/seconds_per_step"], 1e-12)
                )
                reduced_info["progress/seen_samples"] = (step + 1) * config.batch_size

                if eval_interval > 0 and step % eval_interval == 0:
                    with sharding.set_mesh(mesh):
                        eval_info = peval_step(eval_rng, train_state, fixed_eval_batch)
                    eval_reduced = _finalize_interval_metrics(
                        jax.tree.map(lambda value: value[None], eval_info),
                        flow_prefix="eval_fixed/loss/flow_time",
                    )
                    reduced_info.update(eval_reduced)

                console_keys = (
                    "loss",
                    "loss/group/left_arm",
                    "loss/group/right_arm",
                    "loss/group/left_gripper",
                    "loss/group/right_gripper",
                    "grad_norm",
                    "optimizer/learning_rate",
                    "performance/seconds_per_step",
                )
                info_str = ", ".join(
                    f"{key}={float(reduced_info[key]):.6g}"
                    for key in console_keys
                    if key in reduced_info
                )
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(reduced_info, step=step)
                if metrics_stream is not None:
                    _write_jsonl(
                        metrics_stream,
                        {"step": step, "wall_time_unix": time.time(), **{key: float(value) for key, value in reduced_info.items()}},
                    )
                infos = []
                interval_started = time.monotonic()
                interval_data_wait = 0.0
                interval_steps = 0

            data_wait_started = time.monotonic()
            batch = next(data_iter)
            interval_data_wait += time.monotonic() - data_wait_started

            if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
    finally:
        if metrics_stream is not None:
            metrics_stream.close()

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    



if __name__ == "__main__":
    main(_config.cli())
