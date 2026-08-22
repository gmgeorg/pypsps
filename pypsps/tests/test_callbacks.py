"""Test module for callbacks."""

import numpy as np
import pytest
import tensorflow as tf

from .. import datasets
from ..keras import callbacks, models


class _FakeLoss:
    """Minimal stand-in for a CausalLoss, exposing only `_alpha`."""

    def __init__(self, alpha):
        """Stores the alpha value."""
        self._alpha = alpha


class _FakeModel:
    """Minimal stand-in for a compiled keras model, exposing only `.loss`."""

    def __init__(self, alpha=None):
        """Wraps a `_FakeLoss` with the given alpha."""
        self.loss = _FakeLoss(alpha)


def test_constructor_validates_steepness():
    """Test that non-positive steepness raises ValueError."""
    with pytest.raises(ValueError, match="steepness"):
        callbacks.AlphaScheduleCallback(steepness=0.0)
    with pytest.raises(ValueError, match="steepness"):
        callbacks.AlphaScheduleCallback(steepness=-1.0)


def test_constructor_validates_half_life_epoch():
    """Test that a negative half_life_epoch raises ValueError."""
    with pytest.raises(ValueError, match="half_life_epoch"):
        callbacks.AlphaScheduleCallback(half_life_epoch=-1)


def test_constructor_validates_clamp_threshold():
    """Test that a negative clamp_threshold raises ValueError."""
    with pytest.raises(ValueError, match="clamp_threshold"):
        callbacks.AlphaScheduleCallback(clamp_threshold=-0.1)


def test_schedule_type_properties():
    """Test is_decay/is_increase/is_constant reflect alpha_start vs alpha_end."""
    decay = callbacks.AlphaScheduleCallback(alpha_start=10.0, alpha_end=1.0)
    assert decay.is_decay
    assert not decay.is_increase
    assert not decay.is_constant

    increase = callbacks.AlphaScheduleCallback(alpha_start=1.0, alpha_end=10.0)
    assert increase.is_increase
    assert not increase.is_decay
    assert not increase.is_constant

    constant = callbacks.AlphaScheduleCallback(alpha_start=5.0, alpha_end=5.0)
    assert constant.is_constant
    assert not constant.is_decay
    assert not constant.is_increase


def test_decay_schedule_starts_high_ends_low_and_monotonic():
    """Test a decay schedule moves from alpha_start down to alpha_end monotonically."""
    cb = callbacks.AlphaScheduleCallback(
        alpha_start=10.0, alpha_end=1.0, half_life_epoch=10, steepness=0.5, verbose=False
    )
    cb.set_model(_FakeModel())

    alphas = []
    for epoch in range(40):
        cb.on_epoch_begin(epoch)
        alphas.append(cb.model.loss._alpha)

    assert alphas[0] == pytest.approx(10.0, abs=0.5)
    assert alphas[-1] == pytest.approx(1.0, abs=1e-6)
    assert all(a >= b - 1e-9 for a, b in zip(alphas, alphas[1:]))
    assert cb.alpha_history == alphas


def test_increase_schedule_starts_low_ends_high_and_monotonic():
    """Test an increase schedule moves from alpha_start up to alpha_end monotonically."""
    cb = callbacks.AlphaScheduleCallback(
        alpha_start=1.0, alpha_end=10.0, half_life_epoch=10, steepness=0.5, verbose=False
    )
    cb.set_model(_FakeModel())

    alphas = []
    for epoch in range(40):
        cb.on_epoch_begin(epoch)
        alphas.append(cb.model.loss._alpha)

    assert alphas[0] == pytest.approx(1.0, abs=0.5)
    assert alphas[-1] == pytest.approx(10.0, abs=1e-6)
    assert all(a <= b + 1e-9 for a, b in zip(alphas, alphas[1:]))


def test_constant_schedule_keeps_alpha_fixed():
    """Test that alpha_start == alpha_end keeps alpha fixed across epochs."""
    cb = callbacks.AlphaScheduleCallback(alpha_start=5.0, alpha_end=5.0, verbose=False)
    cb.set_model(_FakeModel())

    for epoch in range(5):
        cb.on_epoch_begin(epoch)
        assert cb.model.loss._alpha == 5.0


def test_clamps_to_alpha_end_near_convergence():
    """Test alpha snaps exactly to alpha_end once within clamp_threshold."""
    cb = callbacks.AlphaScheduleCallback(
        alpha_start=10.0, alpha_end=1.0, half_life_epoch=5, steepness=1.0, verbose=False
    )
    cb.set_model(_FakeModel())

    cb.on_epoch_begin(100)
    assert cb.model.loss._alpha == cb.alpha_end


def test_raises_if_model_has_no_loss_with_alpha_attribute():
    """Test AttributeError is raised when the model's loss has no `_alpha`."""
    cb = callbacks.AlphaScheduleCallback(verbose=False)
    cb.set_model(_FakeModel.__new__(_FakeModel))  # no `.loss` attribute set at all
    with pytest.raises(AttributeError, match="_alpha"):
        cb.on_epoch_begin(0)


def test_get_default_callbacks_includes_alpha_schedule_with_requested_params():
    """Test get_default_callbacks wires alpha schedule params through unchanged."""
    cbs = callbacks.get_default_callbacks(
        monitor="val_causal_loss_metric",
        patience=10,
        alpha_start=8.0,
        alpha_end=2.0,
        half_life_epoch=4,
        steepness=0.2,
    )
    alpha_schedules = [c for c in cbs if isinstance(c, callbacks.AlphaScheduleCallback)]
    assert len(alpha_schedules) == 1
    alpha_schedule = alpha_schedules[0]
    assert alpha_schedule.alpha_start == 8.0
    assert alpha_schedule.alpha_end == 2.0
    assert alpha_schedule.half_life_epoch == 4
    assert alpha_schedule.steepness == 0.2


def test_alpha_schedule_updates_causal_loss_alpha_during_fit():
    """Test that alpha on a compiled CausalLoss actually moves during model.fit."""
    np.random.seed(10)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=200)
    inputs, outputs = ks_data.to_keras_inputs_outputs()

    tf.random.set_seed(10)
    model = models.build_toy_model(
        n_states=3, n_features=ks_data.n_features, compile=True, alpha=10.0
    )
    assert isinstance(model.loss._alpha, tf.Variable)
    assert float(model.loss._alpha.numpy()) == 10.0

    alpha_schedule = callbacks.AlphaScheduleCallback(
        alpha_start=10.0, alpha_end=1.0, half_life_epoch=1, steepness=1.0, verbose=False
    )
    model.fit(
        inputs,
        outputs,
        epochs=5,
        batch_size=64,
        verbose=0,
        callbacks=[alpha_schedule],
    )

    assert len(alpha_schedule.alpha_history) == 5
    assert alpha_schedule.alpha_history[0] < 10.0
    assert alpha_schedule.alpha_history[-1] < alpha_schedule.alpha_history[0]
    assert float(model.loss._alpha.numpy()) == alpha_schedule.alpha_history[-1]


def test_alpha_schedule_assigns_the_variable_in_place():
    """The callback must call .assign() on the tf.Variable, not rebind the attribute.

    Rebinding would replace the Variable with a plain float, breaking any already-traced
    training graph reading it via ReadVariableOp -- see
    test_losses.test_causal_loss_alpha_assign_reaches_traced_graph for the underlying mechanism.
    """
    model = models.build_toy_model(n_states=2, n_features=3, compile=True, alpha=10.0)
    alpha_var = model.loss._alpha
    assert isinstance(alpha_var, tf.Variable)

    alpha_schedule = callbacks.AlphaScheduleCallback(
        alpha_start=10.0, alpha_end=1.0, half_life_epoch=1, steepness=1.0, verbose=False
    )
    alpha_schedule.set_model(model)
    alpha_schedule.on_epoch_begin(5)

    assert model.loss._alpha is alpha_var, "callback rebound the attribute instead of assigning"
    assert isinstance(model.loss._alpha, tf.Variable)
    assert float(model.loss._alpha.numpy()) != 10.0
