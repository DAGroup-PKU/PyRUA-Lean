"""Restore the frozen trainer's state onto the requested device mesh."""


def with_target_sharding(state, sharding):
    import jax

    return jax.tree.map(
        lambda value, target: jax.ShapeDtypeStruct(
            value.shape, value.dtype, sharding=target
        ),
        state,
        sharding,
    )


def restore_state(checkpoint_manager, state, data_loader, step=None):
    """Keep parameters/optimizer/counters intact, overriding saved device layout."""
    del data_loader
    import jax
    import orbax.checkpoint as ocp
    from orbax.checkpoint import checkpoint_utils
    from openpi.shared import array_typing as at
    from openpi.training import checkpoints

    if any(getattr(x, "sharding", None) is None for x in jax.tree.leaves(state)):
        raise ValueError("Resume requires an explicit target sharding for every state leaf")
    with at.disable_typechecking():
        train_state, params = checkpoints._split_params(state)
        items = {"train_state": train_state, "params": {"params": params}}
        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(**{
                name: ocp.args.PyTreeRestore(
                    item=item,
                    restore_args=checkpoint_utils.construct_restore_args(item),
                )
                for name, item in items.items()
            }),
        )
    return checkpoints._merge_params(restored["train_state"], restored["params"])


def install(trainer_globals):
    """Patch only runtime hooks; the pinned trainer source stays read-only."""
    original_init = trainer_globals["init_train_state"]

    def init_train_state(config, init_rng, mesh, *, resume):
        state, sharding = original_init(config, init_rng, mesh, resume=resume)
        if resume:
            state = with_target_sharding(state, sharding)
        return state, sharding

    trainer_globals["init_train_state"] = init_train_state
    trainer_globals["_checkpoints"].restore_state = restore_state
