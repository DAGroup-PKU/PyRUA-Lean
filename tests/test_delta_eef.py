import numpy as np
import pytest
from datafarm.training.delta_eef import convert, ANGLES, wrap


def test_chunk_anchor_roundtrip_and_grippers():
    rng = np.random.default_rng(17)
    state = rng.normal(size=(3, 14)).astype(np.float32)
    actions = rng.normal(size=(3, 50, 14)).astype(np.float32)
    delta = convert(actions, state)
    actual = convert(delta, state, inverse=True)
    expected = actions.copy()
    expected[..., ANGLES] = wrap(expected[..., ANGLES])
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_array_equal(delta[..., (6, 13)], actions[..., (6, 13)])
    np.testing.assert_allclose(delta[:, -1, 2], actions[:, -1, 2] - state[:, 2])


def test_constant_right_bias_cancels_and_new_anchor_reconstructs():
    state = np.zeros(14, dtype=np.float32)
    actions = np.ones((50, 14), dtype=np.float32) * .2
    delta = convert(actions, state)
    state2, actions2 = state.copy(), actions.copy()
    state2[9] += .015
    actions2[:, 9] += .015
    np.testing.assert_allclose(convert(actions2, state2), delta, atol=1e-7)
    np.testing.assert_allclose(convert(delta, state2, inverse=True), actions2, atol=1e-6)


def test_euler_seam_and_padding():
    state = np.zeros(14)
    state[3] = np.pi - .01
    actions = np.zeros((50, 32))
    actions[:, 3] = -np.pi + .01
    delta = convert(actions, state)
    np.testing.assert_allclose(delta[:, 3], .02, atol=1e-6)
    np.testing.assert_array_equal(delta[:, 14:], 0)


def test_reject_missing_anchor_and_nan():
    with pytest.raises(ValueError):
        convert(np.zeros((50, 14)), np.zeros(13))
    with pytest.raises(ValueError):
        convert(np.zeros((2, 50, 14)), np.zeros(14))
    with pytest.raises(ValueError):
        convert(np.full((50, 14), np.nan), np.zeros(14))
