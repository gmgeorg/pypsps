"""Test module for loss functions."""

import random
import warnings

import numpy as np
import pytest
import tensorflow as tf

from .. import datasets, utils
from ..keras import families, losses, mixture, models, neglogliks

tfk = tf.keras


def test_psps_model_build_and_predict():
    """Test build & predict"""
    tf.random.set_seed(0)
    random.seed(0)
    np.random.seed(0)

    ks_data = datasets.KangSchafer(true_ate=10, seed=10).sample(n_samples=1000)

    inputs, outputs = ks_data.to_keras_inputs_outputs()
    assert outputs.shape == (1000, 2)

    tf.random.set_seed(10)
    model = models.build_toy_model(n_states=3, n_features=ks_data.n_features, compile=True)
    preds = model.predict(inputs)
    outcome_params_pred, weights, propensity_score = utils.split_y_pred(preds, 2, 1)

    assert outcome_params_pred.shape == (1000, 3 * 2)  # (obs, states * 2) for (loc, scale)
    assert propensity_score.shape[0] == 1000
    assert weights.shape == (1000, 3)


@pytest.mark.parametrize(
    "reduction,expected_shape",
    [("sum", ()), ("sum_over_batch_size", ()), ("none", (5,))],  # ("auto", 1),
)
def test_psps_causal_loss(reduction, expected_shape):
    """Test psps causal loss"""
    tf.random.set_seed(0)
    random.seed(0)
    np.random.seed(0)

    pypsps_outcome_loss = losses.OutcomeLoss(
        loss=neglogliks.NegloglikNormal(reduction="none"),
        treatment_loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction=reduction,
    )

    pypsps_treat_loss = losses.TreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction=reduction,
    )
    pypsps_causal_loss = losses.CausalLoss(
        outcome_loss=pypsps_outcome_loss,
        treatment_loss=pypsps_treat_loss,
        alpha=1.0,
        outcome_loss_weight=1.0,
        predictive_states_regularizer=tf.keras.regularizers.l2(0.1),
        reduction=reduction,
    )

    ks_data = datasets.KangSchafer(true_ate=10, seed=10).sample(n_samples=5)

    inputs, outputs = ks_data.to_keras_inputs_outputs()
    assert outputs.shape == (5, 2)

    tf.random.set_seed(10)
    model = models.build_toy_model(n_states=3, n_features=ks_data.n_features, compile=True)
    preds = model.predict(inputs)

    causal_loss = pypsps_causal_loss(outputs, preds)
    print(reduction, causal_loss)
    assert causal_loss.shape == expected_shape


def test_causal_loss_alpha_is_a_variable_assigned_not_rebound():
    """CausalLoss.alpha must be a tf.Variable, and .assign() must reach an already-traced
    (tf.function-compiled) call -- the way model.fit() actually evaluates the loss
    (run_eagerly=False by default).

    A plain-float `_alpha` gets baked into the graph as a constant at the first trace; a
    schedule rebinding a plain-float alpha afterward (e.g. AlphaScheduleCallback) would then only be
    updating an eager Python attribute the compiled training function never reads again -- the
    schedule would silently never reach the optimizer. This is the regression guard for that.
    """
    outcome_loss = losses.OutcomeLoss(
        loss=neglogliks.NegloglikNormal(reduction="none"),
        treatment_loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction="sum_over_batch_size",
    )
    treatment_loss = losses.TreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction="sum_over_batch_size",
    )
    causal_loss = losses.CausalLoss(
        outcome_loss=outcome_loss,
        treatment_loss=treatment_loss,
        alpha=1.0,
        reduction="sum_over_batch_size",
    )
    assert isinstance(causal_loss.alpha, tf.Variable)

    rng = np.random.RandomState(0)
    n_rows, n_states = 8, 2
    weights = tf.nn.softmax(rng.uniform(size=(n_rows, n_states)), axis=1).numpy()
    locs = rng.uniform(-2.0, 2.0, size=(n_rows, n_states))
    scales = rng.uniform(0.5, 2.0, size=(n_rows, n_states))
    treat_probs = rng.uniform(0.05, 0.95, size=(n_rows, n_states))
    y_pred = _build_synthetic_preds(weights, locs, scales, treat_probs)
    y_true = tf.constant(
        np.concatenate([rng.normal(size=(n_rows, 1)), rng.randint(0, 2, size=(n_rows, 1))], axis=1),
        dtype=tf.float32,
    )

    traced = tf.function(lambda yt, yp: causal_loss(yt, yp))
    causal_loss.alpha.assign(1.0)
    loss_alpha_1 = traced(y_true, y_pred).numpy()
    causal_loss.alpha.assign(100.0)
    loss_alpha_100 = traced(y_true, y_pred).numpy()

    assert not np.isclose(loss_alpha_1, loss_alpha_100), (
        "assigning a new alpha did not change the traced graph's output -- an alpha schedule "
        "would silently fail to reach the optimizer during model.fit()"
    )


def _build_synthetic_preds(weights, locs, scales, treat_probs):
    """Builds a synthetic interleaved y_pred tensor for OutcomeLoss/TreatmentLoss.

    All inputs have shape (n_rows, n_states). Layout matches utils.split_y_pred with
    n_outcome_pred_cols=2 (Normal loc, scale) and n_treatment_pred_cols=1 (Bernoulli prob).
    """
    return tf.constant(
        np.concatenate([locs, scales, weights, treat_probs], axis=1), dtype=tf.float32
    )


def _outcome_likelihood_fixture(n_rows=16, n_states=3, seed=0):
    """Random normal-outcome, binary-treatment (y_true, y_pred) and the parts they are built from."""
    rng = np.random.RandomState(seed)
    weights = rng.dirichlet(np.ones(n_states), size=n_rows)
    locs = rng.uniform(-2.0, 2.0, size=(n_rows, n_states))
    scales = rng.uniform(0.5, 2.0, size=(n_rows, n_states))
    treat_probs = rng.uniform(0.05, 0.95, size=(n_rows, n_states))
    outcome_true = rng.normal(size=(n_rows, 1))
    treatment_true = rng.randint(0, 2, size=(n_rows, 1)).astype(np.float64)
    y_pred = _build_synthetic_preds(weights, locs, scales, treat_probs)
    y_true = tf.constant(np.concatenate([outcome_true, treatment_true], axis=1), dtype=tf.float32)
    return y_true, y_pred, (weights, locs, scales, outcome_true)


def _normal_outcome_loss(reduction="none", **kwargs):
    """OutcomeLoss for a normal outcome and a binary treatment."""
    return losses.OutcomeLoss(
        loss=neglogliks.NegloglikNormal(reduction="none"),
        treatment_loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction=reduction,
        **kwargs,
    )


def test_outcome_likelihood_accepts_strings_and_rejects_unknown_values():
    """Strings convert to the enum; an unknown value raises ValueError."""
    assert _normal_outcome_loss(outcome_likelihood="interventional")._outcome_likelihood is (
        losses.OutcomeLikelihood.INTERVENTIONAL
    )
    assert _normal_outcome_loss(outcome_likelihood="observational")._outcome_likelihood is (
        losses.OutcomeLikelihood.OBSERVATIONAL
    )
    with pytest.raises(ValueError, match="prior"):
        _normal_outcome_loss(outcome_likelihood="prior")


def test_default_outcome_likelihood_is_interventional():
    """Without the argument, OutcomeLoss evaluates the interventional likelihood."""
    assert _normal_outcome_loss()._outcome_likelihood is losses.OutcomeLikelihood.INTERVENTIONAL
    y_true, y_pred, _ = _outcome_likelihood_fixture(seed=4)
    np.testing.assert_array_equal(
        _normal_outcome_loss()(y_true, y_pred).numpy(),
        _normal_outcome_loss(outcome_likelihood="interventional")(y_true, y_pred).numpy(),
    )


def test_penalty_free_keeps_outcome_likelihood():
    """Metrics built from penalty_free() use the same likelihood as the training loss."""
    clean = _normal_outcome_loss(outcome_likelihood="observational").penalty_free()
    assert clean._outcome_likelihood is losses.OutcomeLikelihood.OBSERVATIONAL
    y_true, y_pred, _ = _outcome_likelihood_fixture(seed=3)
    np.testing.assert_array_equal(
        clean(y_true, y_pred).numpy(),
        _normal_outcome_loss(outcome_likelihood="observational")(y_true, y_pred).numpy(),
    )


def test_causal_loss_shares_an_alpha_variable():
    """A float32 tf.Variable alpha is used as is; other dtypes are refused."""
    treatment = losses.TreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction="sum_over_batch_size",
    )
    alpha = tf.Variable(3.0, trainable=False, dtype=tf.float32)
    causal = losses.CausalLoss(_normal_outcome_loss("sum_over_batch_size"), treatment, alpha=alpha)
    assert causal.alpha is alpha
    with pytest.raises(ValueError, match="float32"):
        losses.CausalLoss(
            _normal_outcome_loss("sum_over_batch_size"),
            treatment,
            alpha=tf.Variable(1.0, dtype=tf.float64),
        )


def _normal_bernoulli_mixture(n_states=None):
    """The mixture the legacy Normal/BCE constructors map to."""
    return mixture.StateMixture(families.Normal(), families.Bernoulli(), n_states=n_states)


@pytest.mark.parametrize("likelihood", ["interventional", "observational"])
@pytest.mark.parametrize("reduction", ["none", "sum", "sum_over_batch_size"])
def test_mixture_constructor_matches_legacy_constructor(likelihood, reduction):
    """OutcomeLoss(mixture) / TreatmentLoss(mixture) equal the three-int constructors."""
    y_true, y_pred, _ = _outcome_likelihood_fixture()
    mix = _normal_bernoulli_mixture(n_states=3)
    legacy = _normal_outcome_loss(reduction, outcome_likelihood=likelihood)
    assert legacy.mixture == _normal_bernoulli_mixture()
    new = losses.OutcomeLoss(mix, likelihood, reduction=reduction)  # positional likelihood
    assert new.outcome_likelihood is losses.OutcomeLikelihood(likelihood)
    np.testing.assert_array_equal(new(y_true, y_pred).numpy(), legacy(y_true, y_pred).numpy())

    legacy_t = losses.TreatmentLoss(
        tf.keras.losses.BinaryCrossentropy(reduction="none"), 1, 2, 1, reduction=reduction
    )
    new_t = losses.TreatmentLoss(mix, reduction=reduction)
    np.testing.assert_array_equal(new_t(y_true, y_pred).numpy(), legacy_t(y_true, y_pred).numpy())


def test_legacy_loss_keeps_its_nlls_and_counts():
    """The legacy view metrics, utils and inference still read until they take the mixture."""
    loss = _normal_outcome_loss()
    assert isinstance(loss._loss, neglogliks.NegloglikNormal)
    assert isinstance(loss._treatment_loss, tf.keras.losses.BinaryCrossentropy)
    assert (loss._n_outcome_true_cols, loss._n_outcome_pred_cols, loss._n_treatment_pred_cols) == (
        1,
        2,
        1,
    )
    new = losses.OutcomeLoss(_normal_bernoulli_mixture(3))
    assert not hasattr(new, "_loss") and new._n_outcome_pred_cols == 2


def test_constructor_errors():
    """Mixture and legacy arguments don't mix; the legacy form needs every count."""
    mix = _normal_bernoulli_mixture(3)
    bce = tf.keras.losses.BinaryCrossentropy(reduction="none")
    with pytest.raises(TypeError, match="not both"):
        losses.OutcomeLoss(mix, n_outcome_true_cols=1)
    with pytest.raises(TypeError, match="not both"):
        losses.TreatmentLoss(mix, loss=bce)
    with pytest.raises(TypeError, match="n_treatment_pred_cols"):
        losses.TreatmentLoss(bce, 1, 2)
    with pytest.raises(ValueError, match="reduction='none'"):
        losses.TreatmentLoss(tf.keras.losses.BinaryCrossentropy(), 1, 2, 1)
    with pytest.raises(ValueError, match="maps to Normal"):
        losses.OutcomeLoss(neglogliks.NegloglikNormal(reduction="none"), bce, 1, 1, 1)


def test_causal_loss_shares_one_mixture_and_rejects_a_mismatch():
    """On the mixture path the two losses must agree, and end up sharing one object."""
    a, b = _normal_bernoulli_mixture(3), _normal_bernoulli_mixture(3)
    causal = losses.CausalLoss(losses.OutcomeLoss(a), losses.TreatmentLoss(b))
    assert causal.mixture is a and causal._treatment_loss.mixture is a
    assert causal.layout == a.layout
    other = mixture.StateMixture(families.Normal(), families.Normal(), n_states=3)
    with pytest.raises(ValueError, match="share one StateMixture"):
        losses.CausalLoss(losses.OutcomeLoss(a), losses.TreatmentLoss(other))


def test_legacy_causal_loss_warns_on_a_treatment_mismatch():
    """Decision A6: the legacy path warns (raises from v0.5.0); a match links the mixtures."""
    treatment = losses.TreatmentLoss(
        neglogliks.NegloglikNormal(reduction="none"), 1, 2, 2, reduction="none"
    )
    outcome = losses.OutcomeLoss(
        neglogliks.NegloglikNormal(reduction="none"),
        tf.keras.losses.BinaryCrossentropy(reduction="none"),
        1,
        2,
        1,
        reduction="none",
    )
    with pytest.warns(FutureWarning, match="v0.5.0"):
        losses.CausalLoss(outcome, treatment, reduction="none")
    assert treatment.mixture is not outcome.mixture

    matching = losses.TreatmentLoss(
        tf.keras.losses.BinaryCrossentropy(reduction="none"), 1, 2, 1, reduction="none"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        causal = losses.CausalLoss(outcome, matching, reduction="none")
    assert causal._treatment_loss.mixture is outcome.mixture


def test_from_mixture_equals_the_legacy_causal_loss():
    """CausalLoss.from_mixture gives the legacy value, regularizer included."""
    y_true, y_pred, _ = _outcome_likelihood_fixture(seed=5)
    reg = tf.keras.regularizers.L2(0.1)
    new = losses.CausalLoss.from_mixture(
        _normal_bernoulli_mixture(3),
        alpha=2.0,
        outcome_likelihood="observational",
        outcome_loss_weight=0.5,
        predictive_states_regularizer=reg,
    )
    legacy = losses.CausalLoss(
        _normal_outcome_loss("sum_over_batch_size", outcome_likelihood="observational"),
        losses.TreatmentLoss(
            tf.keras.losses.BinaryCrossentropy(reduction="none"),
            1,
            2,
            1,
            reduction="sum_over_batch_size",
        ),
        alpha=2.0,
        outcome_loss_weight=0.5,
        predictive_states_regularizer=reg,
    )
    np.testing.assert_allclose(float(new(y_true, y_pred)), float(legacy(y_true, y_pred)), rtol=1e-6)
