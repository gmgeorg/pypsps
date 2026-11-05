"""Losses and loss metrics weight each unit by its ``sample_weight``."""

import numpy as np
import pytest
import tensorflow as tf

from ..keras import families, losses, metrics, mixture, models

N, K = 8, 3
_MIX = mixture.StateMixture(families.Normal(), families.Bernoulli(), n_states=K)
_W = np.array([0.5, 3.0, 0.0, 1.0, 2.0, 0.25, 1.5, 4.0], dtype="float32")


def _data(seed=0, n=N):
    """``(y_true [n, 2], y_pred [n, K * 4])`` for a Normal outcome and a binary treatment."""
    rng = np.random.default_rng(seed)
    y_pred = np.hstack(
        [
            rng.normal(size=(n, K)),
            rng.uniform(0.5, 2.0, size=(n, K)),
            rng.dirichlet(np.ones(K), size=n),
            rng.uniform(0.1, 0.9, size=(n, K)),
        ]
    )
    y_true = np.hstack([rng.normal(size=(n, 1)), rng.integers(0, 2, size=(n, 1))])
    return y_true.astype("float32"), y_pred.astype("float32")


def _per_unit(y_true, y_pred, alpha=1.0, outcome_loss_weight=1.0):
    """The causal loss of each unit, from the mixture directly."""
    outcome = _MIX.outcome_nll(y_true, y_pred).numpy()
    treatment = _MIX.treatment_nll(y_true, y_pred).numpy()
    return outcome_loss_weight * outcome + alpha * treatment


def _reduce(values, reduction):
    return {"none": values, "sum": values.sum(), "sum_over_batch_size": values.mean()}[reduction]


@pytest.mark.parametrize("reduction", ["none", "sum", "sum_over_batch_size"])
@pytest.mark.parametrize("weight_shape", [(N,), (N, 1)])
def test_outcome_and_treatment_losses_weight_each_unit(reduction, weight_shape):
    y_true, y_pred = _data()
    w = _W.reshape(weight_shape)
    outcome = losses.OutcomeLoss(_MIX, reduction=reduction)(y_true, y_pred, sample_weight=w)
    treatment = losses.TreatmentLoss(_MIX, reduction=reduction)(y_true, y_pred, sample_weight=w)
    expected_o = _reduce(_W * _MIX.outcome_nll(y_true, y_pred).numpy(), reduction)
    expected_t = _reduce(_W * _MIX.treatment_nll(y_true, y_pred).numpy(), reduction)
    # Keras matches the rank of an [N, 1] weight, so "none" then returns [N, 1].
    np.testing.assert_allclose(np.ravel(outcome), np.ravel(expected_o), rtol=1e-5)
    np.testing.assert_allclose(np.ravel(treatment), np.ravel(expected_t), rtol=1e-5)


@pytest.mark.parametrize("reduction", ["none", "sum", "sum_over_batch_size"])
def test_causal_loss_weights_each_unit(reduction):
    y_true, y_pred = _data(seed=1)
    loss = losses.CausalLoss.from_mixture(
        _MIX, alpha=2.0, outcome_loss_weight=0.5, reduction=reduction
    )
    value = loss(y_true, y_pred, sample_weight=_W).numpy()
    per_unit = _per_unit(y_true, y_pred, alpha=2.0, outcome_loss_weight=0.5)
    np.testing.assert_allclose(value, _reduce(_W * per_unit, reduction), rtol=1e-5)


def test_causal_loss_no_longer_scales_the_batch_mean_by_the_mean_weight():
    """The v0.3.4 value was mean(w) * mean(loss); that is not the weighted mean."""
    y_true, y_pred = _data(seed=2)
    loss = losses.CausalLoss.from_mixture(_MIX)
    per_unit = _per_unit(y_true, y_pred)
    old = _W.mean() * per_unit.mean()
    new = float(loss(y_true, y_pred, sample_weight=_W))
    assert not np.isclose(new, old, rtol=1e-3)
    np.testing.assert_allclose(new, (_W * per_unit).mean(), rtol=1e-5)


def test_integer_weights_equal_repeating_units():
    """Weight 2 counts a unit twice, weight 0 drops it."""
    y_true, y_pred = _data(seed=3)
    counts = np.array([2, 0, 1, 3, 1, 0, 1, 2])
    loss = losses.CausalLoss.from_mixture(_MIX, reduction="sum")
    weighted = float(loss(y_true, y_pred, sample_weight=counts.astype("float32")))
    repeated = float(loss(np.repeat(y_true, counts, axis=0), np.repeat(y_pred, counts, axis=0)))
    np.testing.assert_allclose(weighted, repeated, rtol=1e-5)


@pytest.mark.parametrize("reduction", ["none", "sum", "sum_over_batch_size"])
def test_regularizer_is_added_once_and_not_weighted(reduction):
    """The predictive-state regularizer is a batch penalty: added after the reduction."""
    y_true, y_pred = _data(seed=4)
    reg = tf.keras.regularizers.L2(0.1)
    penalty = float(reg(tf.constant(_MIX.layout.decode_pred(y_pred).state_weights)))
    with_reg = losses.CausalLoss.from_mixture(
        _MIX, predictive_states_regularizer=reg, reduction=reduction
    )
    without = losses.CausalLoss.from_mixture(_MIX, reduction=reduction)
    for w in (None, _W, 10 * _W):
        diff = with_reg(y_true, y_pred, sample_weight=w) - without(y_true, y_pred, sample_weight=w)
        np.testing.assert_allclose(diff.numpy(), np.full(np.shape(diff), penalty), rtol=1e-4)


def test_causal_loss_metric_is_the_weighted_mean_over_units():
    """Across batches of different sizes it is sum(w * loss) / sum(w), not a mean of batches."""
    y_true, y_pred = _data(seed=5)
    metric = metrics.CausalLossMetric(losses.OutcomeLoss(_MIX), losses.TreatmentLoss(_MIX))
    for rows in (slice(0, 6), slice(6, N)):
        metric.update_state(y_true[rows], y_pred[rows], sample_weight=_W[rows])
    per_unit = _per_unit(y_true, y_pred)
    np.testing.assert_allclose(float(metric.result()), (_W * per_unit).sum() / _W.sum(), rtol=1e-5)

    unweighted = metrics.CausalLossMetric(losses.OutcomeLoss(_MIX), losses.TreatmentLoss(_MIX))
    for rows in (slice(0, 6), slice(6, N)):
        unweighted.update_state(y_true[rows], y_pred[rows])
    np.testing.assert_allclose(float(unweighted.result()), per_unit.mean(), rtol=1e-5)


def test_outcome_nll_metric_is_the_weighted_mean_over_units():
    y_true, y_pred = _data(seed=6)
    metric = metrics.outcome_nll_metric_gen(losses.OutcomeLoss(_MIX))
    for rows in (slice(0, 3), slice(3, N)):
        metric.update_state(y_true[rows], y_pred[rows], sample_weight=_W[rows])
    nll = _MIX.outcome_nll(y_true, y_pred).numpy()
    np.testing.assert_allclose(float(metric.result()), (_W * nll).sum() / _W.sum(), rtol=1e-5)


def test_metrics_leave_the_training_loss_reduction_alone():
    outcome, treatment = losses.OutcomeLoss(_MIX), losses.TreatmentLoss(_MIX)
    metrics.CausalLossMetric(outcome, treatment)
    metrics.outcome_nll_metric_gen(outcome)
    assert outcome.reduction == treatment.reduction == "sum_over_batch_size"


def test_evaluate_with_sample_weight_matches_the_loss():
    """model.evaluate passes sample_weight through to the per-unit causal loss."""
    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(N, 4)).astype("float32")
    t = (np.arange(N) % 2).astype("float32")[:, None]
    y = np.hstack([rng.normal(size=(N, 1)), t]).astype("float32")
    model = models.build_toy_model(n_states=K, n_features=4)
    evaluated = model.evaluate([x, t], y, sample_weight=_W, batch_size=N, verbose=0)
    if isinstance(evaluated, list):
        evaluated = evaluated[0]
    direct = model.loss(y, model.predict([x, t], verbose=0), sample_weight=_W)
    np.testing.assert_allclose(evaluated, float(direct), rtol=1e-4)
