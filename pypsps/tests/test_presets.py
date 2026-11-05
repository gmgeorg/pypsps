"""Equivalence of the model builders with the frozen v0.3.5 reference builders.

Until the presets land, the "new" side is `models.build_*` itself, so the harness is
trivially green; PRs 2 and 3 switch each builder to its preset and keep these cases.
"""

import numpy as np
import pytest
import tensorflow as tf

from ..keras import models
from . import _reference_builders as reference

N, K, F = 16, 3, 4

_BINARY = dict(n_states=K, n_features=F, predictive_state_hidden_layers=[(6, "relu")])
_CONTINUOUS = dict(
    n_states=K,
    n_features=F,
    predictive_state_hidden_layers=[(6, "relu")],
    outcome_hidden_layers=[(6, "relu")],
    log_scale_bounds=(-2.0, 4.0),
)

CASES = {
    "binary_normal-constant_scale-depth1": (
        "build_model_binary_normal",
        dict(_BINARY, outcome_hidden_layers=[(6, "tanh")], loc_layer=(5, "selu")),
    ),
    "binary_normal-scale_layer-depth2": (
        "build_model_binary_normal",
        dict(
            _BINARY,
            predictive_state_hidden_layers=[(6, "relu"), (5, "selu")],
            outcome_hidden_layers=[(6, "tanh"), (4, "relu")],
            loc_layer=(5, "selu"),
            scale_layer=(10, "tanh"),
        ),
    ),
    "binary_exponential": (
        "build_model_binary_exponential",
        dict(_BINARY, outcome_hidden_layers=[(6, "tanh")], log_rate_layer=(5, "selu")),
    ),
    "continuous_exponential": ("build_model_continuous_exponential", dict(_CONTINUOUS)),
    "continuous_weibull-shared_shape-init": (
        "build_model_continuous_weibull",
        dict(
            _CONTINUOUS,
            log_shape_bounds=(-1.0, 1.0),
            separate_shape_per_state=False,
            init_scale=5.0,
            treatment_init_values=[[0.0] * K, [1.0] * K],  # pypress: (n_params, n_states)
        ),
    ),
    "continuous_weibull-per_state_shape": (
        "build_model_continuous_weibull",
        dict(_CONTINUOUS, log_shape_bounds=(-1.0, 1.0)),
    ),
}


def _data(seed=0):
    """Inputs and a ``y_true`` whose width fits both binary and continuous models."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, F)).astype("float32")
    t = rng.integers(0, 2, size=(N, 1)).astype("float32")
    time = rng.uniform(0.5, 5.0, size=(N, 1))
    event = rng.integers(0, 2, size=(N, 1))
    return x, t, np.hstack([time, event, t]).astype("float32")


def _randomize(model, seed=1):
    """Random weights, so a swap of two same-shape layers changes the predictions."""
    rng = np.random.default_rng(seed)
    values = []
    for w in model.weights:
        value = 0.3 * rng.normal(size=w.shape)
        if w.path.endswith("moving_variance"):
            value = np.abs(value) + 0.5  # a negative variance makes every output NaN
        values.append(value.astype("float32"))
    model.set_weights(values)


def _loss_and_grads(model, x, t, y_true):
    """The compiled loss at ``training=False`` and its gradients."""
    with tf.GradientTape() as tape:
        loss = model.loss(_y_true_for(model, y_true), model([x, t], training=False))
    return loss, tape.gradient(loss, model.trainable_weights)


def _y_true_for(model, y_true):
    """``[outcome, treatment]`` for binary models, ``[time, event, treatment]`` otherwise."""
    if model.loss.layout.n_outcome_true_cols == 1:
        return y_true[:, 1:]  # event as a 0/1 "outcome" is fine for the equivalence check
    return y_true


@pytest.mark.parametrize("case", sorted(CASES))
def test_adapter_equivalence(case, tmp_path):
    """Same predictions, loss and gradients under a weight transfer; old weight files load."""
    name, kwargs = CASES[case]
    old = getattr(reference, name)(**kwargs)
    new = getattr(models, name)(**kwargs)
    _randomize(old)
    new.set_weights(old.get_weights())
    x, t, y_true = _data()

    expected = old([x, t], training=False)
    assert np.isfinite(expected).all()
    np.testing.assert_array_equal(new([x, t], training=False), expected)
    old_loss, old_grads = _loss_and_grads(old, x, t, y_true)
    new_loss, new_grads = _loss_and_grads(new, x, t, y_true)
    np.testing.assert_array_equal(new_loss, old_loss)
    for g_new, g_old in zip(new_grads, old_grads):
        np.testing.assert_array_equal(g_new, g_old)

    path = str(tmp_path / "old.weights.h5")
    old.save_weights(path)
    loaded = getattr(models, name)(**kwargs)
    loaded.load_weights(path)
    np.testing.assert_array_equal(loaded([x, t], training=False), old([x, t], training=False))
