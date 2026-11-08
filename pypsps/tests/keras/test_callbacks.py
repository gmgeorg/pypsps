"""Test module for callbacks."""

import warnings

import numpy as np
import pytest
import tensorflow as tf

from ... import datasets
from ...keras import callbacks, families, losses, mixture, models, presets


def _causal_loss(alpha=1.0) -> losses.CausalLoss:
    """A real CausalLoss (Normal outcome, binary treatment); the callbacks only use alpha."""
    return losses.CausalLoss.from_mixture(
        mixture.StateMixture(families.Normal(), families.Bernoulli(), n_states=3),
        alpha=alpha,
        reduction="sum_over_batch_size",
    )


class _FakeModel:
    """Minimal stand-in for a compiled keras model, exposing only `.loss`."""

    def __init__(self, alpha=1.0):
        """Wraps a CausalLoss with the given alpha."""
        self.loss = _causal_loss(alpha)


def test_schedule_is_assigned_to_the_loss_and_recorded():
    """The adapter writes the core schedule's alpha into the loss variable each epoch."""
    cb = callbacks.AlphaScheduleCallback(
        alpha_start=10.0, alpha_end=1.0, half_life_epoch=10, steepness=0.5, verbose=False
    )
    cb.set_model(_FakeModel())
    for epoch in range(40):
        cb.on_epoch_begin(epoch)
        assert float(cb.model.loss.alpha) == pytest.approx(cb.schedule.alpha(epoch), rel=1e-6)
    assert cb.alpha_history == [cb.schedule.alpha(e) for e in range(40)]
    assert (cb.alpha_start, cb.alpha_end, cb.half_life_epoch) == (10.0, 1.0, 10)
    assert (cb.steepness, cb.clamp_threshold) == (0.5, pytest.approx(0.9))
    assert cb.is_decay and not cb.is_increase and not cb.is_constant


def test_steep_schedule_does_not_overflow():
    """A late epoch of a steep schedule is alpha_end, with no warning."""
    cb = callbacks.AlphaScheduleCallback(
        alpha_start=10.0, alpha_end=1.0, half_life_epoch=10, steepness=10.0, verbose=False
    )
    cb.set_model(_FakeModel())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cb.on_epoch_begin(100)
    assert float(cb.model.loss.alpha) == 1.0


def test_raises_if_model_has_no_loss_with_alpha_attribute():
    """AttributeError, naming the loss type, when the model's loss isn't a CausalLoss."""
    cb = callbacks.AlphaScheduleCallback(verbose=False)
    cb.set_model(_FakeModel.__new__(_FakeModel))  # no `.loss` attribute set at all
    with pytest.raises(AttributeError, match="CausalLoss.*NoneType"):
        cb.on_epoch_begin(0)


def test_raises_for_a_non_causal_loss_with_a_float_alpha():
    """A loss with its own float `alpha` (eg Keras's focal loss) is not a CausalLoss."""
    model = _FakeModel()
    model.loss = tf.keras.losses.BinaryFocalCrossentropy(alpha=0.25)
    cb = callbacks.AlphaScheduleCallback(verbose=False)
    cb.set_model(model)
    with pytest.raises(AttributeError, match="BinaryFocalCrossentropy"):
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
    model = models.build_model(
        **presets.toy(n_states=3, n_features=ks_data.n_features, compile=True, alpha=10.0)
    )
    assert isinstance(model.loss.alpha, tf.Variable)
    assert float(model.loss.alpha.numpy()) == 10.0

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
    assert float(model.loss.alpha.numpy()) == alpha_schedule.alpha_history[-1]


def test_alpha_schedule_assigns_the_variable_in_place():
    """The callback must call .assign() on the tf.Variable, not rebind the attribute.

    Rebinding would replace the Variable with a plain float, breaking any already-traced
    training graph reading it via ReadVariableOp -- see
    test_losses.test_causal_loss_alpha_assign_reaches_traced_graph for the underlying mechanism.
    """
    model = models.build_model(**presets.toy(n_states=2, n_features=3, compile=True, alpha=10.0))
    alpha_var = model.loss.alpha
    assert isinstance(alpha_var, tf.Variable)

    alpha_schedule = callbacks.AlphaScheduleCallback(
        alpha_start=10.0, alpha_end=1.0, half_life_epoch=1, steepness=1.0, verbose=False
    )
    alpha_schedule.set_model(model)
    alpha_schedule.on_epoch_begin(5)

    assert model.loss.alpha is alpha_var, "callback rebound the attribute instead of assigning"
    assert isinstance(model.loss.alpha, tf.Variable)
    assert float(model.loss.alpha.numpy()) != 10.0


class _WeightsModel:
    """Stand-in model whose 'weights' are the epoch they were saved at."""

    def __init__(self):
        """Start before epoch 0 with alpha 1."""
        self.epoch = -1
        self.restored = None
        self.stop_training = False
        self.loss = _causal_loss(1.0)

    def get_weights(self):
        """Return the current epoch as the weights."""
        return [np.array(self.epoch)]

    def set_weights(self, w):
        """Record which epoch was restored."""
        self.restored = int(w[0])


def _drive(cb, series):
    """Feed a monitored series to a callback as epoch logs; return the stand-in model."""
    m = _WeightsModel()
    cb.set_model(m)
    cb.on_train_begin()
    for e, v in enumerate(series):
        m.epoch = e
        cb.on_epoch_end(e, {"val_m": v})
        if m.stop_training:
            break
    cb.on_train_end()
    return m


def _eligible(series, start=6, patience=50, min_delta=0.0):
    """Drive an EligibleBestCheckpoint over ``series``."""
    cb = callbacks.EligibleBestCheckpoint(
        "val_m", patience=patience, start_from_epoch=start, min_delta=min_delta
    )
    return cb, _drive(cb, series)


def test_eligible_checkpoint_ignores_epochs_before_start():
    """Better values before start_from_epoch are never selected."""
    cb, m = _eligible([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 5.0, 4.0, 3.5, 3.9])
    assert (cb.selected_epoch, cb.selected_value, m.restored) == (8, 3.5, 8)
    assert [t["eligible"] for t in cb.trace] == [False] * 6 + [True] * 4
    assert all(t["alpha"] == 1.0 for t in cb.trace)


def test_eligible_checkpoint_skips_nan_and_inf_and_ties_go_to_earliest():
    """A NaN at the first eligible epoch is not restored; ties keep the earlier epoch."""
    cb, m = _eligible([1, 1, 1, 1, 1, 1, float("nan"), 3.0, 2.0, 2.0, float("inf")])
    assert (cb.selected_epoch, m.restored) == (8, 8)


def test_eligible_checkpoint_min_delta_requires_a_real_improvement():
    """An improvement smaller than min_delta does not move the selection."""
    cb, _ = _eligible([9] * 6 + [3.0, 2.95, 2.5], min_delta=0.1)
    assert cb.selected_epoch == 8


def test_eligible_checkpoint_patience_counts_from_the_first_eligible_epoch():
    """Patience runs over eligible epochs only, and training stops when it runs out."""
    cb, m = _eligible([9] * 6 + [3.0, 4.0, 4.0, 4.0, 1.0], patience=3)
    assert cb.stopped_epoch == 9 and m.restored == 6


def test_eligible_checkpoint_raises_when_no_epoch_is_eligible():
    """No finite value at or after start_from_epoch raises instead of keeping the last weights."""
    with pytest.raises(callbacks.NoEligibleCheckpointError):
        _eligible([1.0] * 6 + [float("nan")] * 3)


def test_eligible_checkpoint_raises_on_missing_monitor():
    """A monitor that is not in the logs raises KeyError."""
    cb = callbacks.EligibleBestCheckpoint("val_missing", patience=2)
    with pytest.raises(KeyError, match="val_missing"):
        _drive(cb, [1.0])


def test_eligible_checkpoint_validates_arguments():
    """patience < 1 and a negative start_from_epoch raise ValueError."""
    with pytest.raises(ValueError, match="patience"):
        callbacks.EligibleBestCheckpoint("val_m", patience=0)
    with pytest.raises(ValueError, match="start_from_epoch"):
        callbacks.EligibleBestCheckpoint("val_m", patience=1, start_from_epoch=-1)


def test_lr_plateau_is_inactive_before_start_from_epoch():
    """No reduction during warmup even on a worsening monitor; the LR is still logged."""

    class _Opt:
        """Optimizer stand-in."""

        def __init__(self):
            """Hold a learning-rate variable."""
            self.learning_rate = tf.Variable(0.01)

    class _M:
        """Model stand-in."""

        stop_training = False

        def __init__(self):
            """Hold the optimizer."""
            self.optimizer = _Opt()

    model = _M()
    cb = callbacks.WarmupReduceLROnPlateau(
        monitor="val_m", factor=0.2, patience=1, min_lr=1e-6, mode="min", start_from_epoch=6
    )
    cb.set_model(model)
    cb.on_train_begin()
    for e in range(6):
        logs = {"val_m": float(e)}
        cb.on_epoch_end(e, logs)
        assert logs["learning_rate"] == pytest.approx(0.01)
    assert float(np.asarray(model.optimizer.learning_rate)) == pytest.approx(0.01)
    for e, v in zip(range(6, 9), [5.0, 6.0, 7.0]):
        cb.on_epoch_end(e, {"val_m": v})
    assert float(np.asarray(model.optimizer.learning_rate)) < 0.01


def test_gaussian_schedule_assigns_the_half_gaussian_values():
    """The callback assigns alpha_schedule.AlphaSchedule.alpha(e) to the loss each epoch."""
    cb = callbacks.GaussianAlphaSchedule(alpha_start=10.0, alpha_end=1.0, verbose=False)
    cb.set_model(_FakeModel(10.0))
    cb.on_train_begin()
    for epoch in range(8):
        cb.on_epoch_begin(epoch)
    want = [cb.schedule.alpha(e) for e in range(8)]
    assert cb.alpha_history == want
    assert float(cb.model.loss.alpha.numpy()) == pytest.approx(want[-1])
    assert cb.warmup_epochs == 6
    assert isinstance(cb, callbacks.AlphaScheduleCallback)


def test_gaussian_schedule_refuses_a_fit_that_does_not_start_at_zero():
    """Starting at epoch 3 raises MidFitResumeError."""
    cb = callbacks.GaussianAlphaSchedule(verbose=False)
    cb.set_model(_FakeModel(10.0))
    cb.on_train_begin()
    with pytest.raises(callbacks.MidFitResumeError):
        cb.on_epoch_begin(3)


def test_gaussian_schedule_rejects_invalid_parameters():
    """Validation comes from AlphaSchedule."""
    with pytest.raises(ValueError, match="decay_const"):
        callbacks.GaussianAlphaSchedule(decay_const=0.0)


def test_gaussian_schedule_and_eligible_checkpoint_in_a_fit():
    """Schedule, checkpoint and LR callbacks run together in model.fit."""
    np.random.seed(11)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=200)
    inputs, outputs = ks_data.to_keras_inputs_outputs()

    tf.random.set_seed(11)
    model = models.build_model(
        **presets.toy(n_states=3, n_features=ks_data.n_features, compile=True, alpha=10.0)
    )
    schedule = callbacks.GaussianAlphaSchedule(alpha_start=10.0, decay_const=0.5, verbose=False)
    checkpoint = callbacks.EligibleBestCheckpoint(
        "val_loss", patience=20, start_from_epoch=schedule.warmup_epochs
    )
    plateau = callbacks.WarmupReduceLROnPlateau(
        monitor="val_loss", patience=2, start_from_epoch=schedule.warmup_epochs
    )
    model.fit(
        inputs,
        outputs,
        epochs=8,
        batch_size=64,
        validation_split=0.25,
        verbose=0,
        callbacks=[schedule, checkpoint, plateau],
    )

    assert schedule.alpha_history == [schedule.schedule.alpha(e) for e in range(8)]
    assert float(model.loss.alpha.numpy()) == pytest.approx(schedule.alpha_history[-1])
    assert checkpoint.selected_epoch >= schedule.warmup_epochs == 4
    assert [t["alpha"] for t in checkpoint.trace] == pytest.approx(schedule.alpha_history)
