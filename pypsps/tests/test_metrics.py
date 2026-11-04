"""Tests for the pypsps metrics: family means, censoring, configs and the legacy forms."""

import math

import numpy as np
import pytest
import tensorflow as tf

from pypsps import utils
from pypsps.keras import families, losses, metrics, mixture, neglogliks
from pypsps.keras.layout import ColumnLayout


def _make_y_pred(outcome_blocks, weights, treatment_blocks):
    """Builds an interleaved y_pred tensor: [outcome params..., weights, treatment params...].

    outcome_blocks / treatment_blocks: lists of per-param arrays, each of shape
    (n_rows, n_states), eg for a Normal outcome with 2 states: [loc_by_state, scale_by_state].
    weights: (n_rows, n_states) prior state weights (rows should sum to 1).
    """
    blocks = [np.asarray(b, dtype=np.float32) for b in outcome_blocks]
    blocks.append(np.asarray(weights, dtype=np.float32))
    blocks.extend(np.asarray(b, dtype=np.float32) for b in treatment_blocks)
    return tf.constant(np.concatenate(blocks, axis=1))


_NORMAL_BERNOULLI = mixture.StateMixture(families.Normal(), families.Bernoulli(), n_states=2)
_WEIGHTS = np.array([[0.5, 0.5], [0.3, 0.7], [0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.2, 0.8]])
_PROPENSITY = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.8, 0.9]])


def _binary_fixture():
    """A Normal/Bernoulli K=2 batch: (y_true [y | a], y_pred, marginal propensity [N, 1])."""
    loc = np.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.0], [-2.0, 1.0], [0.0, 0.0], [1.5, 2.5]])
    y_pred = _make_y_pred([loc, np.ones_like(loc)], _WEIGHTS, [_PROPENSITY])
    y_true = tf.constant(
        [[1.0, 1.0], [0.0, 0.0], [2.0, 1.0], [-1.0, 0.0], [0.5, 0.0], [2.0, 1.0]], tf.float32
    )
    return y_true, y_pred, (_WEIGHTS * _PROPENSITY).sum(axis=1, keepdims=True)


def _metric_value(metric, y_true, y_pred, sample_weight=None) -> float:
    """One-shot value of a stateful metric: reset, update once, read."""
    metric.reset_state()
    metric.update_state(y_true, y_pred, sample_weight=sample_weight)
    return float(metric.result())


_PROPENSITY_METRICS = [
    (metrics.PropensityScoreBinaryCrossentropy, {}, tf.keras.metrics.BinaryCrossentropy),
    (metrics.PropensityScoreAUC, {"curve": "ROC"}, tf.keras.metrics.AUC),
    (metrics.PropensityScoreAUC, {"curve": "PR"}, tf.keras.metrics.AUC),
]


@pytest.mark.parametrize("cls,kwargs,reference", _PROPENSITY_METRICS)
def test_propensity_metrics_score_the_marginal_propensity(cls, kwargs, reference):
    """The propensity metrics compare the treatment with sum_k P(s_k | x) * p_k."""
    y_true, y_pred, marginal = _binary_fixture()
    expected = reference(**kwargs)
    expected.update_state(y_true[:, 1:2], marginal)
    np.testing.assert_allclose(
        _metric_value(cls(_NORMAL_BERNOULLI, **kwargs), y_true, y_pred),
        float(expected.result()),
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "cls,reference",
    [
        (metrics.PropensityScoreBinaryCrossentropy, tf.keras.metrics.BinaryCrossentropy),
        (metrics.PropensityScoreAUC, tf.keras.metrics.AUC),
    ],
)
def test_propensity_metrics_read_the_treatment_after_a_survival_outcome(cls, reference):
    """With a [time, event] outcome the treatment is column 2; trailing columns are ignored.

    Regression for 1a80698: reading y_true[:, 1:] would mix the event indicator in.
    """
    mix = mixture.StateMixture(families.Weibull(), families.Bernoulli(), n_states=1)
    propensity = np.array([[0.9], [0.1], [0.8], [0.3]])
    y_pred = _make_y_pred([np.zeros((4, 1)), np.zeros((4, 1))], np.ones((4, 1)), [propensity])
    # [time, event, treatment, trailing column]
    y_true = tf.constant(
        [[5.0, 0.0, 1.0, 7.0], [3.0, 1.0, 0.0, 7.0], [8.0, 1.0, 1.0, 7.0], [2.0, 0.0, 0.0, 7.0]]
    )
    expected = reference()
    expected.update_state(y_true[:, 2:3], propensity)
    np.testing.assert_allclose(
        _metric_value(cls(mix), y_true, y_pred), float(expected.result()), rtol=1e-5
    )


def test_propensity_metrics_reject_a_non_binary_treatment():
    """A propensity metric needs a binary treatment family."""
    mix = mixture.StateMixture(families.Weibull(), families.Normal(), n_states=2)
    with pytest.raises(ValueError, match="binary treatment"):
        metrics.PropensityScoreAUC(mix)


@pytest.mark.parametrize(
    "cls,error",
    [(metrics.TreatmentMeanSquaredError, np.square), (metrics.TreatmentMeanAbsoluteError, np.abs)],
)
def test_treatment_errors_mix_the_treatment_means(cls, error):
    """Bernoulli: the error of the marginal propensity. Normal: loc only, never the scale."""
    y_true, y_pred, marginal = _binary_fixture()
    expected = error(y_true.numpy()[:, 1:2] - marginal).mean()
    np.testing.assert_allclose(
        _metric_value(cls(_NORMAL_BERNOULLI), y_true, y_pred), expected, rtol=1e-5
    )

    # A Normal treatment with perfect per-state locs and scale 2 has zero error.
    mix = mixture.StateMixture(families.Weibull(), families.Normal(), n_states=2)
    dose = np.array([[1.0], [2.0], [-0.5]])
    y_pred = _make_y_pred(
        [np.zeros((3, 2)), np.zeros((3, 2))],
        [[0.3, 0.7]] * 3,
        [np.tile(dose, 2), np.full((3, 2), 2.0)],
    )
    y_true = tf.constant(np.column_stack([np.ones(3), np.ones(3), dose[:, 0]]), tf.float32)
    np.testing.assert_allclose(_metric_value(cls(mix), y_true, y_pred), 0.0, atol=1e-6)


@pytest.mark.parametrize(
    "cls,error",
    [(metrics.OutcomeMeanSquaredError, np.square), (metrics.OutcomeMeanAbsoluteError, np.abs)],
)
def test_outcome_errors_mix_family_means_not_raw_params(cls, error):
    """The MSE-bug regression. Normal K=2 with perfect locs and scale 2 scores 0 (the old
    raw-parameter mix also scored the scale column), and a Weibull outcome mixes
    exp(log_scale) * Gamma(1 + 1 / shape), not the log parameters."""
    y = np.array([[0.2], [1.5], [-0.3]])
    y_pred = _make_y_pred(
        [np.tile(y, 2), np.full((3, 2), 2.0)], [[0.4, 0.6]] * 3, [np.full((3, 2), 0.5)]
    )
    y_true = tf.constant(np.column_stack([y[:, 0], np.ones(3)]), tf.float32)
    np.testing.assert_allclose(
        _metric_value(cls(_NORMAL_BERNOULLI), y_true, y_pred), 0.0, atol=1e-6
    )
    raw_mix = utils.agg_outcome_pred(y_pred, n_outcome_pred_cols=2, n_treatment_pred_cols=1)
    assert error(y - np.asarray(raw_mix)).mean() > 0.5  # what the metric used to report

    mix = mixture.StateMixture(families.Weibull(), families.Bernoulli(), n_states=2)
    log_scale = np.array([[0.0, 1.0], [0.5, -0.5]])
    log_shape = np.array([[0.3, -0.2], [0.0, 0.4]])
    weights = np.array([[0.25, 0.75], [0.6, 0.4]])
    y_pred = _make_y_pred([log_scale, log_shape], weights, [np.full((2, 2), 0.5)])
    time = np.array([1.2, 0.7])
    y_true = tf.constant(np.column_stack([time, np.ones(2), np.zeros(2)]), tf.float32)
    gamma = np.vectorize(lambda s: math.gamma(1.0 + 1.0 / s))
    mean = (weights * np.exp(log_scale) * gamma(np.exp(log_shape))).sum(axis=1)
    np.testing.assert_allclose(
        _metric_value(cls(mix), y_true, y_pred), error(time - mean).mean(), rtol=1e-5
    )


# --- censored outcomes (events only, NaN with no events) ---

_SURVIVAL_FAMILIES = [families.Weibull(), families.Exponential()]
_OUTCOME_ERRORS = [
    (metrics.OutcomeMeanSquaredError, np.square),
    (metrics.OutcomeMeanAbsoluteError, np.abs),
]
_KERAS_ERROR = {
    np.square: tf.keras.metrics.MeanSquaredError,
    np.abs: tf.keras.metrics.MeanAbsoluteError,
}
_CENSORED_CASES = [
    pytest.param(f, cls, err, id=f"{type(f).__name__}-{cls.__name__}")
    for f in _SURVIVAL_FAMILIES
    for cls, err in _OUTCOME_ERRORS
]


def _survival_batch(family, event, seed=0, time_offset=0.0):
    """A K=2 survival batch: (y_true [time, event, a], y_pred, numpy mixed mean [N])."""
    rng = np.random.RandomState(seed)
    n = len(event)
    params = [rng.uniform(-0.5, 0.5, size=(n, 2)) for _ in family.param_names]
    weights = rng.dirichlet([1.0, 1.0], size=n)
    y_pred = _make_y_pred(params, weights, [np.full((n, 2), 0.5)])
    time = rng.uniform(0.5, 2.0, size=n) + time_offset
    y_true = tf.constant(np.column_stack([time, event, np.zeros(n)]), tf.float32)
    mix = mixture.StateMixture(family, families.Bernoulli(), n_states=2)
    mean = np.asarray(mix.outcome_mean(y_pred))[:, 0]
    return y_true, y_pred, time - mean


@pytest.mark.parametrize("family,cls,error", _CENSORED_CASES)
def test_outcome_error_censored_uses_events_only(family, cls, error):
    """Censored units, here with huge errors, contribute neither error nor weight."""
    event = np.array([1, 0, 1, 0, 1], dtype=float)
    y_true, y_pred, residual = _survival_batch(family, event)
    y_true = tf.tensor_scatter_nd_add(y_true, [[1, 0], [3, 0]], [1e4, 1e4])  # censored rows
    metric = cls(mixture.StateMixture(family, families.Bernoulli(), n_states=2))
    expected = error(residual[event == 1]).mean()
    np.testing.assert_allclose(_metric_value(metric, y_true, y_pred), expected, rtol=1e-5)


@pytest.mark.parametrize("family,cls,error", _CENSORED_CASES)
def test_zero_event_batch_leaves_state_unchanged(family, cls, error):
    """An all-censored batch adds 0 to total and to count."""
    metric = cls(mixture.StateMixture(family, families.Bernoulli(), n_states=2))
    y_true, y_pred, _ = _survival_batch(family, np.array([1, 0, 1.0]))
    metric.update_state(y_true, y_pred)
    before = (float(metric.result()), float(metric.total), float(metric.count))
    y_true, y_pred, _ = _survival_batch(family, np.zeros(4), seed=1)
    metric.update_state(y_true, y_pred)
    assert (float(metric.result()), float(metric.total), float(metric.count)) == before


@pytest.mark.parametrize("family,cls,error", _CENSORED_CASES)
def test_mixed_batches_equal_numpy_events_only_mean(family, cls, error):
    """Any order and split of the batches gives the events-only mean over all units."""
    rng = np.random.RandomState(3)
    event = (rng.uniform(size=24) < 0.5).astype(float)
    event[8:12] = 0.0  # one whole batch without events
    y_true, y_pred, residual = _survival_batch(family, event, seed=3)
    expected = error(residual[event == 1]).mean()
    metric = cls(mixture.StateMixture(family, families.Bernoulli(), n_states=2))
    for order in (np.arange(24), rng.permutation(24)):
        for size in (4, 5, 24):
            metric.reset_state()
            for start in range(0, 24, size):
                rows = order[start : start + size]
                metric.update_state(tf.gather(y_true, rows), tf.gather(y_pred, rows))
            np.testing.assert_allclose(float(metric.result()), expected, rtol=1e-5)


@pytest.mark.parametrize("family,cls,error", _CENSORED_CASES)
def test_no_events_reports_nan(family, cls, error):
    """NaN before any event, finite after one, NaN again after reset_state (decision N1)."""
    metric = cls(mixture.StateMixture(family, families.Bernoulli(), n_states=2))
    assert np.isnan(float(metric.result()))
    for seed in range(3):
        metric.update_state(*_survival_batch(family, np.zeros(4), seed=seed)[:2])
    assert np.isnan(float(metric.result()))
    metric.update_state(*_survival_batch(family, np.array([0, 1.0]))[:2])
    assert np.isfinite(float(metric.result()))
    metric.reset_state()
    assert np.isnan(float(metric.result()))


@pytest.mark.parametrize("cls,error", _OUTCOME_ERRORS)
def test_uncensored_metric_keeps_keras_result(cls, error):
    """A Normal outcome with all-zero sample_weight still reports Keras' 0.0, not NaN."""
    y_true, y_pred, _ = _binary_fixture()
    value = _metric_value(cls(_NORMAL_BERNOULLI), y_true, y_pred, sample_weight=np.zeros(6))
    assert value == 0.0


@pytest.mark.parametrize("family,cls,error", _CENSORED_CASES)
@pytest.mark.parametrize("weight_shape", [(5,), (5, 1)])
def test_censoring_mask_combines_with_sample_weight(family, cls, error, weight_shape):
    """The weights are sample_weight * event, for sample_weight of shape [N] or [N, 1]."""
    event = np.array([1, 0, 1, 1, 0], dtype=float)
    y_true, y_pred, residual = _survival_batch(family, event)
    sample_weight = np.array([0.5, 3.0, 2.0, 1.0, 4.0])
    weights = sample_weight * event
    expected = (weights * error(residual)).sum() / weights.sum()
    metric = cls(mixture.StateMixture(family, families.Bernoulli(), n_states=2))
    value = _metric_value(metric, y_true, y_pred, sample_weight.reshape(weight_shape))
    np.testing.assert_allclose(value, expected, rtol=1e-5)


@pytest.mark.parametrize("cls,error", _OUTCOME_ERRORS)
def test_uncensored_family_metric_unchanged_by_mask(cls, error):
    """For a Normal outcome the metric equals Keras' plain error of the mixed mean."""
    y_true, y_pred, _ = _binary_fixture()
    mean = _NORMAL_BERNOULLI.outcome_mean(y_pred)
    reference = _KERAS_ERROR[error]()
    reference.update_state(y_true[:, :1], mean, sample_weight=np.arange(6.0))
    np.testing.assert_allclose(
        _metric_value(cls(_NORMAL_BERNOULLI), y_true, y_pred, np.arange(6.0)),
        float(reference.result()),
        rtol=1e-6,
    )


def _censored_weibull_model():
    """The continuous Weibull model, recompiled with the outcome MSE, plus its data."""
    from .test_models import _continuous_model, _synthetic_continuous_survival

    inputs, y_true, _ = _synthetic_continuous_survival(n_samples=70)
    tf.random.set_seed(0)
    model = _continuous_model("weibull")
    model.compile(
        loss=model.loss,
        optimizer="adam",
        metrics=[metrics.OutcomeMeanSquaredError(model.loss.mixture)],
    )
    return model, inputs, y_true


def test_censored_metric_in_graph_mode_fit():
    """The final partial batch has no events: fit logs a finite value, and a graph-mode
    evaluate equals the eager events-only mean."""
    model, inputs, y_true = _censored_weibull_model()
    y_true[64:, 1] = 0.0  # batch_size 32: the last batch (6 units) is all censored
    assert y_true[:64, 1].sum() > 0
    history = model.fit(inputs, y_true, epochs=1, batch_size=32, shuffle=False, verbose=0)
    assert np.isfinite(history.history["outcome_mean_squared_error"][0])

    evaluated = model.evaluate(inputs, y_true, batch_size=32, verbose=0, return_dict=True)
    eager = metrics.OutcomeMeanSquaredError(model.loss.mixture)
    expected = _metric_value(
        eager, tf.constant(y_true), tf.constant(model.predict(inputs, verbose=0))
    )
    np.testing.assert_allclose(evaluated["outcome_mean_squared_error"], expected, rtol=1e-4)


def test_early_stopping_with_nan_first_epoch():
    """An all-censored validation set gives NaN every epoch. TerminateOnNaN (it reads only
    the loss) never fires; EarlyStopping keeps best = NaN and stops after patience + 1 epochs."""
    model, inputs, y_true = _censored_weibull_model()
    y_val = y_true.copy()
    y_val[:, 1] = 0.0
    history = model.fit(
        inputs,
        y_true,
        validation_data=(inputs, y_val),
        epochs=10,
        batch_size=32,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(monitor="val_outcome_mean_squared_error", patience=2),
        ],
    )
    assert np.all(np.isnan(history.history["val_outcome_mean_squared_error"]))
    assert np.all(np.isfinite(history.history["loss"]))
    assert len(history.history["loss"]) == 3


# --- the legacy three-int form and its shape rule (decision A1) ---


def test_shape_rule_maps_counts_onto_families():
    """(2 params, 1 col) is Normal, 1 param is the identity, anything else asks for mixture=."""
    normal = metrics.OutcomeMeanSquaredError(2, 1, 1)
    assert normal.mixture.outcome_family == families.Normal()
    y_true, y_pred, _ = _binary_fixture()
    np.testing.assert_allclose(
        _metric_value(normal, y_true, y_pred),
        _metric_value(metrics.OutcomeMeanSquaredError(_NORMAL_BERNOULLI), y_true, y_pred),
        rtol=1e-6,
    )

    identity = metrics.OutcomeMeanSquaredError(1, 1, 1)
    y_pred = tf.constant([[5.0, 1.0, 0.0], [10.0, 1.0, 0.0]])
    assert _metric_value(identity, tf.constant([[5.0, 0.0], [12.0, 1.0]]), y_pred) == 2.0

    for counts in [(2, 1, 2), (3, 1, 1)]:
        with pytest.raises(ValueError, match="mixture="):
            metrics.OutcomeMeanSquaredError(*counts)
    with pytest.raises(ValueError, match="mixture="):
        metrics.TreatmentMeanSquaredError(n_outcome_pred_cols=2, n_treatment_pred_cols=3,
                                          n_outcome_true_cols=1)  # fmt: skip


def test_legacy_propensity_metric_reads_the_last_column():
    """The two-int propensity form doesn't know C, so it keeps reading y_true[:, -1:]."""
    y_true, y_pred, marginal = _binary_fixture()
    expected = tf.keras.metrics.BinaryCrossentropy()
    expected.update_state(y_true[:, -1:], marginal)
    legacy = metrics.PropensityScoreBinaryCrossentropy(
        n_outcome_pred_cols=2, n_treatment_pred_cols=1
    )
    np.testing.assert_allclose(
        _metric_value(legacy, y_true, y_pred), float(expected.result()), rtol=1e-5
    )


def test_metric_constructor_errors():
    """A mixture and counts together, or neither, raise TypeError."""
    with pytest.raises(TypeError, match="not both"):
        metrics.OutcomeMeanSquaredError(_NORMAL_BERNOULLI, n_outcome_true_cols=1)
    with pytest.raises(TypeError, match="StateMixture"):
        metrics.OutcomeMeanSquaredError()
    with pytest.raises(TypeError, match="ColumnLayout"):
        metrics.uniform_entropy_gen()
    with pytest.raises(TypeError, match="not both"):
        metrics.uniform_entropy_gen(_NORMAL_BERNOULLI.layout, n_treatment_pred_cols=1)


def test_predictive_state_df_gen():
    """The state-weight generators take a layout; the legacy counts give the same value."""
    y_pred = _make_y_pred([np.zeros((3, 2))], _WEIGHTS[:3], [np.zeros((3, 2))])
    layout = ColumnLayout.from_counts(1, 1, 1, n_states=2)
    for gen in (metrics.predictive_state_df_gen, metrics.uniform_entropy_gen):
        value = float(gen(layout)(None, y_pred))
        np.testing.assert_allclose(float(gen(1, 1)(None, y_pred)), value, rtol=1e-7)
        np.testing.assert_allclose(
            float(gen(n_outcome_pred_cols=1, n_treatment_pred_cols=1)(None, y_pred)), value
        )


def _make_treatment_loss(**overrides):
    """Builds a plain (no-penalty) `TreatmentLoss` with sensible test defaults."""
    kwargs = dict(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction="sum_over_batch_size",
    )
    kwargs.update(overrides)
    return losses.TreatmentLoss(**kwargs)


def test_treatment_loss_penalty_free_is_identity_when_no_penalty_declared():
    """Base `TreatmentLoss.penalty_free()` has no penalty args to drop, so it must return an
    equivalent (but distinct) instance -- not the same object."""
    original = _make_treatment_loss()
    clean = original.penalty_free()

    assert clean is not original
    assert clean._n_outcome_true_cols == original._n_outcome_true_cols
    assert clean._n_outcome_pred_cols == original._n_outcome_pred_cols
    assert clean._n_treatment_pred_cols == original._n_treatment_pred_cols
    assert clean.get_config() == original.get_config()


class _BalancePenalizedTreatmentLoss(losses.TreatmentLoss):
    """Test double simulating a hypothetical within-state balance penalty folded directly
    into `TreatmentLoss.call` (`self._lambda_balance * balance_penalty`), the pattern
    `causal_loss_metric_gen` must be robust to -- without relying on the `lambda_balance`
    name, since `penalty_free()` is reconstructed via explicit constructor arguments, not
    attribute-name sniffing."""

    def __init__(self, *args, lambda_balance: float = 0.0, **kwargs):
        """Stores `lambda_balance`, the (test-only) penalty weight."""
        super().__init__(*args, **kwargs)
        self._lambda_balance = lambda_balance

    def call(self, y_true, y_pred):
        """Adds a constant `lambda_balance` penalty on top of the real treatment NLL."""
        return super().call(y_true, y_pred) + self._lambda_balance


def test_treatment_loss_penalty_free_zeros_declared_penalty_without_mutating_original():
    """A subclass that inherits `penalty_free()` without overriding it drops any constructor
    argument not explicitly forwarded by the base implementation -- here, `lambda_balance`
    falls back to its own "off" default (0.0) -- and the original, live instance must be
    left untouched."""
    original = _BalancePenalizedTreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        lambda_balance=5.0,
        reduction="sum_over_batch_size",
    )
    clean = original.penalty_free()

    assert clean is not original
    assert clean._lambda_balance == 0.0
    assert original._lambda_balance == 5.0


def test_treatment_loss_penalty_free_fails_loudly_for_undeclared_required_penalty_arg():
    """If a subclass adds a *required* penalty argument (no safe "off" default) and doesn't
    override `penalty_free()` to account for it, reconstruction must raise rather than
    silently guess a value -- forcing whoever adds the penalty to explicitly decide how
    `penalty_free()` should handle it."""

    class _RequiredPenaltyTreatmentLoss(losses.TreatmentLoss):
        """Test double whose penalty weight has no safe "off" default."""

        def __init__(self, *args, lambda_balance: float, **kwargs):
            """Stores the required `lambda_balance` penalty weight."""
            super().__init__(*args, **kwargs)
            self._lambda_balance = lambda_balance

    original = _RequiredPenaltyTreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        lambda_balance=5.0,
        reduction="sum_over_batch_size",
    )
    try:
        original.penalty_free()
        assert False, "expected TypeError: lambda_balance is required and not forwarded"
    except TypeError:
        pass


def test_causal_loss_metric_gen_strips_embedded_treatment_loss_penalty():
    """causal_loss_metric_gen must report the exact joint likelihood even when the passed-in
    treatment_loss instance carries an embedded penalty term -- the metric used for
    EarlyStopping/checkpoint selection/Optuna must never be a penalized proxy."""
    n_outcome_pred_cols, n_treatment_pred_cols, n_outcome_true_cols = 2, 1, 1
    weights = [[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]]
    y_pred = _make_y_pred(
        outcome_blocks=[np.zeros((3, 2)), np.ones((3, 2))],
        weights=weights,
        treatment_blocks=[[[0.9, 0.7], [0.5, 0.5], [0.2, 0.6]]],
    )
    y_true = tf.constant([[5.0, 1.0], [3.0, 0.0], [8.0, 1.0]], dtype=tf.float32)

    def _build(lambda_balance):
        """Builds a matching (outcome_loss, treatment_loss) pair for a given penalty weight."""
        outcome_loss = losses.OutcomeLoss(
            loss=neglogliks.NegloglikNormal(reduction="none"),
            treatment_loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
            n_outcome_true_cols=n_outcome_true_cols,
            n_outcome_pred_cols=n_outcome_pred_cols,
            n_treatment_pred_cols=n_treatment_pred_cols,
            reduction="sum_over_batch_size",
        )
        treatment_loss = _BalancePenalizedTreatmentLoss(
            loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
            n_outcome_true_cols=n_outcome_true_cols,
            n_outcome_pred_cols=n_outcome_pred_cols,
            n_treatment_pred_cols=n_treatment_pred_cols,
            lambda_balance=lambda_balance,
            reduction="sum_over_batch_size",
        )
        return outcome_loss, treatment_loss

    penalized_outcome_loss, penalized_treatment_loss = _build(lambda_balance=1000.0)
    clean_outcome_loss, clean_treatment_loss = _build(lambda_balance=0.0)

    metric = metrics.causal_loss_metric_gen(
        outcome_loss=penalized_outcome_loss, treatment_loss=penalized_treatment_loss
    )
    assert isinstance(metric, metrics.CausalLossMetric)
    metric.update_state(y_true, y_pred)
    metric_value = metric.result().numpy()

    expected_clean = (
        clean_outcome_loss(y_true, y_pred) + clean_treatment_loss(y_true, y_pred)
    ).numpy()
    expected_penalized = (
        penalized_outcome_loss(y_true, y_pred) + penalized_treatment_loss(y_true, y_pred)
    ).numpy()

    np.testing.assert_allclose(metric_value, expected_clean, rtol=1e-5)
    assert not np.isclose(metric_value, expected_penalized, rtol=1e-5)
    # The original, live training-time loss must be unmodified.
    assert penalized_treatment_loss._lambda_balance == 1000.0


def _normal_outcome_loss(cls=losses.OutcomeLoss, **extra):
    """An OutcomeLoss for a normal outcome (2 params per state) and binary treatment."""
    return cls(
        loss=neglogliks.NegloglikNormal(reduction="none"),
        treatment_loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction="sum_over_batch_size",
        **extra,
    )


def _normal_fixture():
    """y_true = [outcome | treatment] and y_pred for 2 states, normal outcome."""
    weights = [[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]]
    y_pred = _make_y_pred(
        outcome_blocks=[[[0.0, 1.0], [2.0, -1.0], [0.5, 0.5]], [[1.0, 2.0]] * 3],
        weights=weights,
        treatment_blocks=[[[0.8, 0.3], [0.4, 0.6], [0.5, 0.2]]],
    )
    y_true = tf.constant([[0.2, 1.0], [1.5, 0.0], [-0.3, 1.0]], dtype=tf.float32)
    return y_true, y_pred


class _PenalizedOutcomeLoss(losses.OutcomeLoss):
    """Test double: an OutcomeLoss whose call() adds a constant penalty."""

    def __init__(self, *args, lambda_penalty: float = 0.0, **kwargs):
        """Stores the penalty weight."""
        super().__init__(*args, **kwargs)
        self._lambda_penalty = lambda_penalty

    def call(self, y_true, y_pred):
        """Adds the penalty to the outcome negative log-likelihood."""
        return super().call(y_true, y_pred) + self._lambda_penalty


def test_outcome_nll_metric_equals_the_outcome_loss():
    """The metric is the outcome loss itself, logged as outcome_nll_metric."""
    outcome = _normal_outcome_loss()
    y_true, y_pred = _normal_fixture()
    metric = metrics.outcome_nll_metric_gen(outcome)
    assert metric.name == "outcome_nll_metric"
    np.testing.assert_allclose(
        float(metric(y_true, y_pred)), float(outcome(y_true, y_pred)), rtol=1e-7
    )


def test_outcome_nll_metric_is_penalty_free_and_leaves_the_loss_untouched():
    """A penalty in the training loss does not reach the metric, nor is it removed from the loss."""
    y_true, y_pred = _normal_fixture()
    penalized = _normal_outcome_loss(_PenalizedOutcomeLoss, lambda_penalty=5.0)
    clean = float(metrics.outcome_nll_metric_gen(_normal_outcome_loss())(y_true, y_pred))
    np.testing.assert_allclose(
        float(metrics.outcome_nll_metric_gen(penalized)(y_true, y_pred)), clean, rtol=1e-7
    )
    assert penalized._lambda_penalty == 5.0
    np.testing.assert_allclose(float(penalized(y_true, y_pred)), clean + 5.0, rtol=1e-6)


def test_outcome_nll_metric_does_not_move_with_alpha():
    """Changing the training loss's alpha leaves the metric unchanged."""
    from pypsps import datasets
    from pypsps.keras import models

    np.random.seed(3)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=128)
    inputs, outputs = ks_data.to_keras_inputs_outputs()
    tf.random.set_seed(3)
    model = models.build_toy_model(n_states=2, n_features=ks_data.n_features, alpha=1.0)
    metric = metrics.outcome_nll_metric_gen(model.loss._outcome_loss)
    model.compile(loss=model.loss, optimizer="adam", metrics=[metric])

    def evaluate():
        """Return (loss, outcome_nll_metric) on the sample."""
        res = model.evaluate(inputs, outputs, verbose=0, return_dict=True)
        return res["loss"], res["outcome_nll_metric"]

    loss_1, nll_1 = evaluate()
    model.loss.alpha = 10.0
    loss_10, nll_10 = evaluate()
    assert loss_10 > loss_1
    np.testing.assert_allclose(nll_10, nll_1, rtol=1e-6)


def test_causal_loss_metric_ignores_the_training_alpha():
    """The metric is the alpha-1 joint NLL, whatever alpha the training loss holds."""
    outcome, treatment = _normal_outcome_loss(), _make_treatment_loss()
    y_true, y_pred = _normal_fixture()
    training = losses.CausalLoss(outcome, treatment, alpha=1.0)
    metric = metrics.CausalLossMetric(outcome, treatment)

    at_one = _metric_value(metric, y_true, y_pred)
    np.testing.assert_allclose(at_one, float(training(y_true, y_pred)), rtol=1e-6)
    training.alpha = 5.0
    assert float(training(y_true, y_pred)) > at_one
    np.testing.assert_allclose(_metric_value(metric, y_true, y_pred), at_one, rtol=1e-7)


def test_causal_loss_metric_weights_the_outcome_nll():
    """value = outcome_loss_weight * outcome NLL + treatment NLL."""
    outcome, treatment = _normal_outcome_loss(), _make_treatment_loss()
    y_true, y_pred = _normal_fixture()
    nll_y, nll_a = float(outcome(y_true, y_pred)), float(treatment(y_true, y_pred))
    for weight in (1.0, 2.5):
        metric = metrics.CausalLossMetric(outcome, treatment, outcome_loss_weight=weight)
        np.testing.assert_allclose(
            _metric_value(metric, y_true, y_pred), weight * nll_y + nll_a, rtol=1e-6
        )


@pytest.mark.parametrize("weight", [1.0, 2.5, 0.0])
def test_causal_loss_metric_config_round_trips(weight):
    """The config holds both losses and the weight, and no alpha; the rebuilt metric agrees."""
    import json

    y_true, y_pred = _normal_fixture()
    metric = metrics.CausalLossMetric(
        _normal_outcome_loss(), _make_treatment_loss(), outcome_loss_weight=weight
    )
    config = metric.get_config()
    json.dumps(config)
    assert "alpha" not in config
    assert config["outcome_loss_weight"] == weight
    assert {"outcome_loss", "treatment_loss"} <= set(config)

    rebuilt = metrics.CausalLossMetric.from_config(config)
    assert rebuilt._outcome_loss_weight == weight
    np.testing.assert_allclose(
        _metric_value(rebuilt, y_true, y_pred), _metric_value(metric, y_true, y_pred), rtol=1e-6
    )


def test_causal_loss_metric_allows_zero_outcome_loss_weight():
    """outcome_loss_weight=0 is legal: the metric is the treatment NLL alone."""
    treatment = _make_treatment_loss()
    y_true, y_pred = _normal_fixture()
    metric = metrics.CausalLossMetric(_normal_outcome_loss(), treatment, outcome_loss_weight=0)
    np.testing.assert_allclose(
        _metric_value(metric, y_true, y_pred), float(treatment(y_true, y_pred)), rtol=1e-6
    )


def test_causal_loss_metric_rejects_alpha_and_wrong_loss_types():
    """There is no alpha parameter; the two losses must be pypsps outcome/treatment losses."""
    outcome, treatment = _normal_outcome_loss(), _make_treatment_loss()
    with pytest.raises(TypeError):
        metrics.CausalLossMetric(outcome, treatment, alpha=1.0)
    with pytest.raises(TypeError, match="OutcomeLoss"):
        metrics.CausalLossMetric(treatment, treatment)
    with pytest.raises(TypeError, match="TreatmentLoss"):
        metrics.CausalLossMetric(outcome, outcome)


def test_causal_loss_metric_gen_alpha_is_deprecated_and_ignored():
    """A passed alpha warns (FutureWarning, naming v0.5.0) and doesn't change the value."""
    import warnings

    outcome, treatment = _normal_outcome_loss(), _make_treatment_loss()
    y_true, y_pred = _normal_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain = metrics.causal_loss_metric_gen(outcome, treatment)
        weighted = metrics.causal_loss_metric_gen(outcome, treatment, outcome_loss_weight=2.0)
    with pytest.warns(FutureWarning, match="v0.5.0"):
        with_alpha = metrics.causal_loss_metric_gen(outcome, treatment, alpha=5.0)

    value = _metric_value(plain, y_true, y_pred)
    np.testing.assert_allclose(_metric_value(with_alpha, y_true, y_pred), value, rtol=1e-7)
    assert _metric_value(weighted, y_true, y_pred) != pytest.approx(value)


def test_metric_generators_return_serializable_metrics():
    """The three MeanMetricWrapper generators round-trip through Keras serialization."""
    y_true, y_pred = _normal_fixture()
    for metric in (
        metrics.outcome_nll_metric_gen(_normal_outcome_loss()),
        metrics.predictive_state_df_gen(n_outcome_pred_cols=2, n_treatment_pred_cols=1),
        metrics.uniform_entropy_gen(n_outcome_pred_cols=2, n_treatment_pred_cols=1),
    ):
        rebuilt = tf.keras.utils.deserialize_keras_object(
            tf.keras.utils.serialize_keras_object(metric)
        )
        assert rebuilt.name == metric.name
        np.testing.assert_allclose(
            _metric_value(rebuilt, y_true, y_pred),
            _metric_value(metric, y_true, y_pred),
            rtol=1e-6,
        )


def test_penalty_free_drops_a_penalty_even_if_get_config_serializes_it():
    """penalty_free() rebuilds from the base class's config, so a subclass that serializes
    its penalty in its own get_config still gets the "off" default in the copy."""

    class _SerializedPenaltyTreatmentLoss(_BalancePenalizedTreatmentLoss):
        """Serializes its penalty, as a well-behaved Keras subclass would."""

        def get_config(self):
            """Adds lambda_balance to the config."""
            return {**super().get_config(), "lambda_balance": self._lambda_balance}

    original = _SerializedPenaltyTreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        lambda_balance=5.0,
        reduction="sum_over_batch_size",
    )
    assert original.get_config()["lambda_balance"] == 5.0
    assert original.penalty_free()._lambda_balance == 0.0


def test_outcome_nll_metric_keeps_outcome_likelihood():
    """The outcome metric uses the training loss's outcome_likelihood."""
    y_true, y_pred = _normal_fixture()
    interventional = _normal_outcome_loss(outcome_likelihood="interventional")
    observational = _normal_outcome_loss(outcome_likelihood="observational")
    for chosen, other in ((interventional, observational), (observational, interventional)):
        got = float(metrics.outcome_nll_metric_gen(chosen)(y_true, y_pred))
        np.testing.assert_allclose(got, float(chosen(y_true, y_pred)), rtol=1e-7)
        assert got != pytest.approx(float(other(y_true, y_pred)))


_METRIC_CLASSES = [
    (metrics.PropensityScoreBinaryCrossentropy, "propensity_score_binary_crossentropy"),
    (metrics.PropensityScoreAUC, "propensity_score_auc"),
    (metrics.TreatmentMeanSquaredError, "treatment_mean_squared_error"),
    (metrics.TreatmentMeanAbsoluteError, "treatment_mean_absolute_error"),
    (metrics.OutcomeMeanSquaredError, "outcome_mean_squared_error"),
    (metrics.OutcomeMeanAbsoluteError, "outcome_mean_absolute_error"),
]
_METRIC_IDS = [cls.__name__ for cls, _ in _METRIC_CLASSES]


def _legacy_counts(cls) -> dict:
    """The legacy constructor counts for the Normal/Bernoulli fixture."""
    counts = {"n_outcome_pred_cols": 2, "n_treatment_pred_cols": 1}
    if not issubclass(cls, metrics._PropensityMetric):
        counts["n_outcome_true_cols"] = 1
    return counts


@pytest.mark.parametrize("cls,default_name", _METRIC_CLASSES, ids=_METRIC_IDS)
def test_metrics_have_explicit_default_names(cls, default_name):
    """The default name is explicit, so Keras doesn't add a per-session `_N` suffix."""
    assert cls(_NORMAL_BERNOULLI).name == default_name
    assert cls(_NORMAL_BERNOULLI).name == default_name
    assert cls(**_legacy_counts(cls)).name == default_name


@pytest.mark.parametrize("cls,default_name", _METRIC_CLASSES, ids=_METRIC_IDS)
def test_metrics_forward_name_and_kwargs(cls, default_name):
    """`name` and other keyword arguments reach the Keras base metric."""
    metric = cls(_NORMAL_BERNOULLI, name="custom", dtype="float64")
    assert metric.name == "custom"
    assert metric.dtype == "float64"


@pytest.mark.parametrize("cls,default_name", _METRIC_CLASSES, ids=_METRIC_IDS)
@pytest.mark.parametrize("form", ["mixture", "legacy"])
def test_metric_config_round_trips(cls, default_name, form):
    """`from_config(get_config())` rebuilds the metric with the same result. A mixture metric
    stores the mixture; a legacy one keeps its counts."""
    import json

    y_true, y_pred, _ = _binary_fixture()
    if form == "mixture":
        metric = cls(_NORMAL_BERNOULLI, name="custom")
    else:
        metric = cls(**_legacy_counts(cls), name="custom")
    config = metric.get_config()
    json.dumps(tf.keras.utils.serialize_keras_object(metric))
    if form == "mixture":
        assert config["mixture"]["registered_name"] == "pypsps>StateMixture"
        assert "n_outcome_pred_cols" not in config
    else:
        assert "mixture" not in config
        assert {k: config[k] for k in _legacy_counts(cls)} == _legacy_counts(cls)

    rebuilt = tf.keras.utils.deserialize_keras_object(
        json.loads(json.dumps(tf.keras.utils.serialize_keras_object(metric)))
    )
    assert type(rebuilt) is cls and rebuilt.name == "custom"
    assert rebuilt.mixture == metric.mixture
    np.testing.assert_allclose(
        _metric_value(rebuilt, y_true, y_pred), _metric_value(metric, y_true, y_pred), rtol=1e-6
    )


def test_censored_metric_survives_a_config_round_trip():
    """The rebuilt metric is still censored: NaN with no events."""
    mix = mixture.StateMixture(families.Weibull((-2.0, 5.0), (-1.0, 1.0)), families.Normal(), 2)
    rebuilt = metrics.OutcomeMeanAbsoluteError.from_config(
        metrics.OutcomeMeanAbsoluteError(mix).get_config()
    )
    assert rebuilt.mixture == mix
    assert np.isnan(float(rebuilt.result()))


def test_propensity_score_auc_config_keeps_curve():
    """The AUC's own arguments (`curve`, `num_thresholds`) survive a config round trip."""
    metric = metrics.PropensityScoreAUC(_NORMAL_BERNOULLI, curve="PR", num_thresholds=50)
    rebuilt = metrics.PropensityScoreAUC.from_config(metric.get_config())
    assert rebuilt.curve.name == "PR"
    assert rebuilt.num_thresholds == 50


def test_propensity_score_auc_honors_curve():
    """`curve="PR"` computes the PR AUC; it used to be dropped, which gave the ROC AUC."""
    y_true, y_pred, marginal = _binary_fixture()
    expected = {}
    for curve in ("ROC", "PR"):
        reference = tf.keras.metrics.AUC(curve=curve)
        reference.update_state(y_true[:, 1:2], marginal)
        expected[curve] = float(reference.result())
    # The fixture must tell the two curves apart, or the test proves nothing.
    assert abs(expected["PR"] - expected["ROC"]) > 1e-3
    for curve in ("ROC", "PR"):
        metric = metrics.PropensityScoreAUC(_NORMAL_BERNOULLI, curve=curve)
        np.testing.assert_allclose(
            _metric_value(metric, y_true, y_pred), expected[curve], rtol=1e-5
        )


def test_generator_metrics_serialize_layout():
    """predictive_state_df_gen(layout) and uniform_entropy_gen(layout) round-trip through JSON
    with the layout kwarg; the legacy int kwargs still round-trip too."""
    import json

    y_true, y_pred, _ = _binary_fixture()
    layout = _NORMAL_BERNOULLI.layout
    for gen in (metrics.predictive_state_df_gen, metrics.uniform_entropy_gen):
        for metric in (gen(layout), gen(2, 1)):
            config = json.loads(json.dumps(tf.keras.utils.serialize_keras_object(metric)))
            rebuilt = tf.keras.utils.deserialize_keras_object(config)
            assert rebuilt.name == metric.name
            np.testing.assert_allclose(
                _metric_value(rebuilt, y_true, y_pred),
                _metric_value(metric, y_true, y_pred),
                rtol=1e-6,
            )
        assert gen(layout).get_config()["layout"]["registered_name"] == "pypsps>ColumnLayout"
