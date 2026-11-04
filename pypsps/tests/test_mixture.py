"""Tests for the state mixture."""

import math

import numpy as np
import pytest
import tensorflow as tf

from ..keras import families, mixture
from ..keras.layout import ColumnLayout


def _normal_pdf(y, loc, scale):
    """Gaussian density."""
    return np.exp(-0.5 * ((y - loc) / scale) ** 2) / (np.sqrt(2 * np.pi) * scale)


def _bernoulli_pmf(a, p):
    """Bernoulli mass."""
    return p**a * (1 - p) ** (1 - a)


def _normal_bernoulli(weights, seed=0):
    """(mixture, y_true, y_pred, parts) for a Normal outcome and Bernoulli treatment."""
    weights = np.asarray(weights, dtype="float64")
    n_rows, n_states = weights.shape
    rng = np.random.RandomState(seed)
    locs = rng.uniform(-2.0, 2.0, (n_rows, n_states))
    scales = rng.uniform(0.5, 2.0, (n_rows, n_states))
    probs = rng.uniform(0.05, 0.95, (n_rows, n_states))
    y = rng.normal(size=(n_rows, 1))
    a = rng.randint(0, 2, (n_rows, 1)).astype("float64")
    mix = mixture.StateMixture(families.Normal(), families.Bernoulli(), n_states=n_states)
    y_pred = mix.layout.encode(
        {"loc": locs.astype("f4"), "scale": scales.astype("f4")},
        weights.astype("f4"),
        {"prob": probs.astype("f4")},
    )
    y_true = np.concatenate([y, a], axis=1).astype("f4")
    return mix, tf.constant(y_true), tf.constant(y_pred), (weights, locs, scales, probs, y, a)


_WEIGHTS = {
    1: [[1.0]],
    2: [[0.5, 0.5], [0.999, 0.001], [0.001, 0.999], [0.1, 0.9]],
    3: [[1.0 / 3, 1.0 / 3, 1.0 / 3], [0.98, 0.01, 0.01], [0.2, 0.3, 0.5]],
}


@pytest.mark.parametrize("n_states", [1, 2, 3])
def test_outcome_and_treatment_nll_telescope_to_joint_nll(n_states):
    """-log p(a, y | x) = treatment_nll + observational outcome_nll, also near-degenerate."""
    mix, y_true, y_pred, (w, locs, scales, probs, y, a) = _normal_bernoulli(_WEIGHTS[n_states])
    joint = -np.log((w * _bernoulli_pmf(a, probs) * _normal_pdf(y, locs, scales)).sum(axis=1))
    total = mix.treatment_nll(y_true, y_pred) + mix.outcome_nll(y_true, y_pred, "observational")
    np.testing.assert_allclose(total.numpy(), joint, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        mix.causal_nll(y_true, y_pred, "observational").numpy(), total.numpy(), rtol=1e-6
    )


def test_interventional_matches_closed_form():
    """ "interventional" is -log sum_k P(s_k | x) p(y | s_k, a, x)."""
    rng = np.random.RandomState(1)
    mix, y_true, y_pred, (w, locs, scales, _, y, _) = _normal_bernoulli(
        rng.dirichlet(np.ones(3), size=16), seed=1
    )
    want = -np.log((w * _normal_pdf(y, locs, scales)).sum(axis=1))
    got = mix.outcome_nll(y_true, y_pred)  # the default
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-4, atol=1e-5)


def test_interventional_ignores_treatment_head():
    """The interventional outcome NLL has zero gradient into the treatment block."""
    rng = np.random.RandomState(2)
    mix, y_true, y_pred, _ = _normal_bernoulli(rng.dirichlet(np.ones(3), size=8), seed=2)
    treatment_cols = slice(3 * 3, None)
    grads = {}
    for name in ("interventional", "observational"):
        with tf.GradientTape() as tape:
            tape.watch(y_pred)
            value = tf.reduce_sum(mix.outcome_nll(y_true, y_pred, name))
        grads[name] = tape.gradient(value, y_pred).numpy()[:, treatment_cols]
    assert np.all(grads["interventional"] == 0.0)
    assert np.any(grads["observational"] != 0.0)
    with pytest.raises(ValueError, match="prior"):
        mix.outcome_nll(y_true, y_pred, "prior")


def test_state_posterior_is_bayes_rule_and_rows_sum_to_one():
    """P(s | x, a) is proportional to P(s | x) p(a | s)."""
    rng = np.random.RandomState(3)
    mix, y_true, y_pred, (w, _, _, probs, _, a) = _normal_bernoulli(
        rng.dirichlet(np.ones(3), size=8), seed=3
    )
    unnorm = w * _bernoulli_pmf(a, probs)
    got = mix.state_posterior(y_true, y_pred).numpy()
    np.testing.assert_allclose(got, unnorm / unnorm.sum(1, keepdims=True), rtol=1e-5)
    np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=1e-6)
    nll = mix.nll_per_state(y_true, y_pred)
    assert nll.outcome.shape == nll.treatment.shape == (8, 3)


def test_state_posterior_equals_state_weights_when_one_state():
    """With K=1 the posterior is the (unit) weight."""
    mix, y_true, y_pred, _ = _normal_bernoulli(np.ones((5, 1)))
    np.testing.assert_array_equal(mix.state_posterior(y_true, y_pred).numpy(), np.ones((5, 1)))


def test_outcome_mean_mixes_family_means_not_raw_params():
    """The MSE-bug regression: mix E[Y | s_k], never the raw parameters."""
    # Normal, K=2, perfect loc and scale 2: the mean is exactly the truth.
    truth = np.array([[1.0], [-2.0], [0.5]], dtype="f4")
    mix = mixture.StateMixture(families.Normal(), families.Bernoulli(), n_states=2)
    y_pred = mix.layout.encode(
        {"loc": np.tile(truth, (1, 2)), "scale": np.full((3, 2), 2.0, "f4")},
        np.array([[0.3, 0.7]] * 3, "f4"),
        {"prob": np.full((3, 2), 0.5, "f4")},
    )
    np.testing.assert_allclose(mix.outcome_mean(y_pred).numpy(), truth, rtol=1e-6)

    # Weibull: sum_k w_k * lambda_k * Gamma(1 + 1/k_k), not the mean from mixed log-params.
    mix = mixture.StateMixture(families.Weibull(), families.Normal(), n_states=2)
    w = np.array([[0.25, 0.75]], "f4")
    log_scale, log_shape = np.array([[0.0, 1.0]], "f4"), np.array([[0.5, -0.5]], "f4")
    y_pred = mix.layout.encode(
        {"log_scale": log_scale, "log_shape": log_shape},
        w,
        {"loc": np.zeros((1, 2), "f4"), "scale": np.ones((1, 2), "f4")},
    )
    per_state = [np.exp(s) * math.gamma(1 + np.exp(-k)) for s, k in zip(log_scale[0], log_shape[0])]
    want = float((w[0] * per_state).sum())
    np.testing.assert_allclose(mix.outcome_mean(y_pred).numpy(), [[want]], rtol=1e-5)
    np.testing.assert_allclose(
        mix.outcome_mean(y_pred, per_state=True).numpy(), [per_state], rtol=1e-5
    )
    mixed_params = np.exp((w * log_scale).sum()) * math.gamma(1 + np.exp(-(w * log_shape).sum()))
    assert abs(want - mixed_params) > 0.1


def test_treatment_mean_is_marginal_propensity():
    """Bernoulli: sum_k P(s_k | x) * prob_k."""
    mix, _, y_pred, (w, _, _, probs, _, _) = _normal_bernoulli(_WEIGHTS[3])
    want = (w * probs).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(mix.treatment_mean(y_pred).numpy(), want, rtol=1e-5)


def test_treatment_mean_normal_uses_loc_only():
    """A Normal treatment mixes loc, not (loc, scale)."""
    mix = mixture.StateMixture(families.Exponential(), families.Normal(), n_states=2)
    y_pred = mix.layout.encode(
        {"log_scale": np.zeros((1, 2), "f4")},
        np.array([[0.5, 0.5]], "f4"),
        {"loc": np.array([[1.0, 3.0]], "f4"), "scale": np.array([[10.0, 10.0]], "f4")},
    )
    np.testing.assert_allclose(mix.treatment_mean(y_pred).numpy(), [[2.0]])


def test_reduce_modes():
    """none / sum / sum_over_batch_size (and legacy auto); anything else raises."""
    values = tf.constant([1.0, 2.0, 6.0])
    np.testing.assert_array_equal(mixture.reduce(values, "none"), values)
    np.testing.assert_array_equal(mixture.reduce(values, None), values)
    assert float(mixture.reduce(values, "sum")) == 9.0
    assert float(mixture.reduce(values, "sum_over_batch_size")) == 3.0
    assert float(mixture.reduce(values, "auto")) == 3.0
    with pytest.raises(ValueError, match="reduction must be one of"):
        mixture.reduce(values, "mean_with_sample_weight")


def test_causal_nll_outcome_loss_weight():
    """causal_nll = w * outcome_nll + treatment_nll; weight 0 cuts the outcome block."""
    mix, y_true, y_pred, _ = _normal_bernoulli(_WEIGHTS[2])
    o, t = mix.outcome_nll(y_true, y_pred), mix.treatment_nll(y_true, y_pred)
    for weight in (0.5, 1.0, 2.0):
        np.testing.assert_allclose(
            mix.causal_nll(y_true, y_pred, outcome_loss_weight=weight).numpy(),
            (weight * o + t).numpy(),
            rtol=1e-6,
        )
    with tf.GradientTape() as tape:
        tape.watch(y_pred)
        value = mix.causal_nll(y_true, y_pred, outcome_loss_weight=0.0)
        total = tf.reduce_sum(value)
    np.testing.assert_array_equal(value.numpy(), t.numpy())
    grad = tape.gradient(total, y_pred).numpy()
    assert np.all(grad[:, : 2 * 2] == 0.0)
    with pytest.raises(ValueError, match=">= 0"):
        mix.causal_nll(y_true, y_pred, outcome_loss_weight=-1.0)


@pytest.mark.parametrize("n_states", [None, 3])
def test_mixture_config_round_trip_and_equality(n_states):
    """The mixture serializes its families and K, and rebuilds equal."""
    mix = mixture.StateMixture(
        families.Weibull((-2.0, 5.0), (-1.0, 1.5)), families.Normal(), n_states
    )
    config = tf.keras.utils.serialize_keras_object(mix)
    assert config["config"]["outcome_family"]["registered_name"] == "pypsps>Weibull"
    rebuilt = tf.keras.utils.deserialize_keras_object(config)
    assert rebuilt == mix and rebuilt is not mix
    assert (
        rebuilt.layout
        == mix.layout
        == ColumnLayout.from_families(n_states, families.Weibull(), families.Normal())
    )
    assert mix != mixture.StateMixture(families.Weibull(), families.Normal(), n_states)


def test_treatment_family_must_have_one_true_col():
    """A survival family can't be the treatment."""
    with pytest.raises(ValueError, match="exactly one y_true column"):
        mixture.StateMixture(families.Normal(), families.Weibull(), n_states=2)
    with pytest.raises(TypeError, match="Family"):
        mixture.StateMixture("normal", families.Bernoulli())
