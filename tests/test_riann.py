"""Tests for the RIANN attitude-estimation model.

Invariant assertions covering the three inference modes: output shapes,
quaternion unit-norm, error handling, determinism, and hidden-state
preservation.
"""

import numpy as np
import pytest

from riann import RIANN  # exercises the top-level re-export


@pytest.fixture(scope="module")
def model():
    # One InferenceSession is enough; reset_state per-test isolates runs.
    return RIANN()


def _norm(q, axis=-1):
    return np.linalg.norm(q, axis=axis)


def test_predict_output_shape(model):
    acc = np.ones((100, 3))
    gyr = np.zeros((100, 3))
    q = model.predict(acc, gyr, fs=200)
    assert q.shape == (100, 4)


def test_predict_quaternions_unit_norm(model):
    acc = np.ones((100, 3))
    gyr = np.zeros((100, 3))
    q = model.predict(acc, gyr, fs=200)
    np.testing.assert_allclose(_norm(q), 1.0, atol=1e-2)


def test_predict_sequence_shapes(model):
    acc = np.ones((50, 3))
    gyr = np.zeros((50, 3))
    q, states = model.predict_sequence(acc, gyr, fs=100)
    assert q.shape == (50, 4)
    assert states.shape[0] == 50
    np.testing.assert_allclose(_norm(q), 1.0, atol=1e-2)


def test_predict_step_output_shape_and_norm(model):
    model.reset_state()
    model.set_sampling_rate(100)
    acc = np.array([0.0, 0.0, 9.81])
    gyr = np.array([0.0, 0.0, 0.0])
    q = model.predict_step(acc, gyr)
    assert q.shape == (4,)
    assert abs(_norm(q) - 1.0) < 1e-2


def test_predict_step_rejects_wrong_shape(model):
    with pytest.raises(ValueError):
        model.predict_step(np.zeros(4), np.zeros(3))  # acc wrong
    with pytest.raises(ValueError):
        model.predict_step(np.zeros(3), np.zeros(2))  # gyr wrong


def test_determinism_after_reset(model):
    acc = np.ones((30, 3))
    gyr = np.zeros((30, 3))

    model.reset_state()
    q1, _ = model.predict_sequence(acc, gyr, fs=100, reset_state=True)
    model.reset_state()
    q2, _ = model.predict_sequence(acc, gyr, fs=100, reset_state=True)

    np.testing.assert_array_equal(q1, q2)


def test_predict_does_not_mutate_external_state(model):
    """predict() saves/restores hidden_state, so a later step is unaffected."""
    model.reset_state()
    model.set_sampling_rate(200)
    before = model.hidden_state.copy()
    _ = model.predict(np.ones((10, 3)), np.zeros((10, 3)), fs=200)
    np.testing.assert_array_equal(model.hidden_state, before)


def test_set_sampling_rate_updates_dt(model):
    model.set_sampling_rate(100)
    assert model.fs == 100
    np.testing.assert_allclose(model.dt, 1.0 / 100)
    np.testing.assert_allclose(model.input_buffer[0, 0, 6], 1.0 / 100)
