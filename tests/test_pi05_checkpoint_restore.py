"""Run with frozen trainer Python, JAX_PLATFORMS=cpu and four virtual CPU devices."""
import tempfile
import unittest
import os
from pathlib import Path
from types import SimpleNamespace

if os.environ.get("JAX_PLATFORMS") != "cpu":
    raise unittest.SkipTest("Explicit frozen trainer CPU run required; never allocate GPUs during collection")

import jax
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from openpi.training import checkpoints, utils
from openpi.shared import array_typing as at

from datafarm.training.checkpoint_restore import restore_state, with_target_sharding


class CheckpointRestoreTest(unittest.TestCase):
    @at.disable_typechecking()
    def test_two_to_four_preserves_parameters_optimizer_and_step(self):
        self.assertGreaterEqual(len(jax.devices()), 4)
        old_mesh = Mesh(np.array(jax.devices()[:2]), ("fsdp",))
        new_mesh = Mesh(np.array(jax.devices()[:4]), ("fsdp",))

        def sharding(mesh, value):
            return NamedSharding(mesh, PartitionSpec() if value.ndim == 0 else PartitionSpec("fsdp", None))

        for ema in (False, True):
            with self.subTest(ema=ema), tempfile.TemporaryDirectory() as directory:
                weight = np.arange(32, dtype=np.float32).reshape(8, 4)
                state = utils.TrainState(
                    step=np.array(7001, dtype=np.int32),
                    params=nnx.State({"weight": nnx.VariableState(nnx.Param, weight)}), model_def=None,
                    opt_state={"count": np.array(7001, dtype=np.int32), "moment": weight / 3},
                    tx=None, ema_decay=0.99 if ema else None,
                    ema_params=nnx.State({"weight": nnx.VariableState(nnx.Param, weight + 10)}) if ema else None,
                )
                state = jax.tree.map(lambda x: jax.device_put(x, sharding(old_mesh, x)), state)
                manager, _ = checkpoints.initialize_checkpoint_dir(
                    Path(directory) / "checkpoint", keep_period=None, overwrite=False, resume=False
                )
                loader = SimpleNamespace(data_config=lambda: SimpleNamespace(norm_stats=None, asset_id=None))
                try:
                    checkpoints.save_state(manager, state, loader, 7000)
                    manager.wait_until_finished()
                    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state)
                    targets = jax.tree.map(lambda x: sharding(new_mesh, x), state)
                    target = with_target_sharding(abstract, targets)
                    restored = restore_state(manager, target, loader, step=7000)
                    for before, after, expected in zip(
                        jax.tree.leaves(state), jax.tree.leaves(restored), jax.tree.leaves(targets), strict=True
                    ):
                        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
                        self.assertEqual(after.sharding, expected)
                    with self.assertRaisesRegex(ValueError, "explicit target sharding"):
                        restore_state(manager, abstract, loader, step=7000)
                finally:
                    manager.close()


if __name__ == "__main__":
    unittest.main()
