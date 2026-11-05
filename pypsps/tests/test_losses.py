"""Test module for loss functions."""

import random

import numpy as np
import pytest
import tensorflow as tf

from .. import datasets
from ..keras import families, losses, mixture, models, neglogliks, presets
from ..keras.layout import ColumnLayout

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
    model = models.build_model(
        **presets.toy(n_states=3, n_features=ks_data.n_features, compile=True)
    )
    parts = ColumnLayout.from_model(model).decode_pred(model.predict(inputs, verbose=0))

    assert {k: v.shape for k, v in parts.outcome.items()} == {"loc": (1000, 3), "scale": (1000, 3)}
    assert parts.treatment["prob"].shape == (1000, 3)
    assert parts.state_weights.shape == (1000, 3)


@pytest.mark.parametrize(
    "reduction,expected_shape",
    [("sum", ()), ("sum_over_batch_size", ()), ("none", (5,))],  # ("auto", 1),
)
def test_psps_causal_loss(reduction, expected_shape):
    """Test psps causal loss"""
    tf.random.set_seed(0)
    random.seed(0)
    np.random.seed(0)

    pypsps_outcome_loss = _normal_outcome_loss(reduction)
    pypsps_treat_loss = _treatment_loss(reduction)
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
    model = models.build_model(
        **presets.toy(n_states=3, n_features=ks_data.n_features, compile=True)
    )
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
    outcome_loss = _normal_outcome_loss("sum_over_batch_size", n_states=2)
    treatment_loss = _treatment_loss("sum_over_batch_size", n_states=2)
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

    All inputs have shape (n_rows, n_states). The layout is Normal (loc, scale) outcome and
    Bernoulli (prob) treatment.
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


def _normal_bernoulli_mixture(n_states=3):
    """A Normal outcome and Bernoulli treatment mixture."""
    return mixture.StateMixture(families.Normal(), families.Bernoulli(), n_states=n_states)


def _normal_outcome_loss(reduction="none", n_states=3, **kwargs):
    """OutcomeLoss for a normal outcome and a binary treatment."""
    return losses.OutcomeLoss(_normal_bernoulli_mixture(n_states), reduction=reduction, **kwargs)


def _treatment_loss(reduction="none", n_states=3):
    """TreatmentLoss for a normal outcome and a binary treatment."""
    return losses.TreatmentLoss(_normal_bernoulli_mixture(n_states), reduction=reduction)


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
    treatment = _treatment_loss("sum_over_batch_size")
    alpha = tf.Variable(3.0, trainable=False, dtype=tf.float32)
    causal = losses.CausalLoss(_normal_outcome_loss("sum_over_batch_size"), treatment, alpha=alpha)
    assert causal.alpha is alpha
    with pytest.raises(ValueError, match="float32"):
        losses.CausalLoss(
            _normal_outcome_loss("sum_over_batch_size"),
            treatment,
            alpha=tf.Variable(1.0, dtype=tf.float64),
        )


@pytest.mark.parametrize("likelihood", ["interventional", "observational"])
def test_outcome_likelihood_is_the_second_positional_argument(likelihood):
    """OutcomeLoss(mixture, likelihood) equals the keyword form."""
    y_true, y_pred, _ = _outcome_likelihood_fixture()
    positional = losses.OutcomeLoss(_normal_bernoulli_mixture(), likelihood, reduction="none")
    assert positional.outcome_likelihood is losses.OutcomeLikelihood(likelihood)
    np.testing.assert_array_equal(
        positional(y_true, y_pred).numpy(),
        _normal_outcome_loss(outcome_likelihood=likelihood)(y_true, y_pred).numpy(),
    )


def test_constructors_take_only_a_mixture():
    """The three-int constructors are gone (v0.5.0): anything but a StateMixture raises."""
    nll = neglogliks.NegloglikNormal(reduction="none")
    bce = tf.keras.losses.BinaryCrossentropy(reduction="none")
    with pytest.raises(TypeError, match="takes a StateMixture"):
        losses.OutcomeLoss(nll, bce)
    with pytest.raises(TypeError, match="takes a StateMixture"):
        losses.TreatmentLoss(bce)
    with pytest.raises(TypeError):
        losses.OutcomeLoss(loss=nll, treatment_loss=bce, n_outcome_true_cols=1)


@pytest.mark.parametrize("cls", [losses.OutcomeLoss, losses.TreatmentLoss])
def test_a_legacy_config_raises(cls):
    """A v0.3.0-v0.3.3 config (column counts, no mixture) raises with the migration path."""
    config = {
        "name": "loss",
        "reduction": "sum_over_batch_size",
        "loss": tf.keras.utils.serialize_keras_object(
            tf.keras.losses.BinaryCrossentropy(reduction="none")
        ),
        "n_outcome_true_cols": 1,
        "n_outcome_pred_cols": 2,
        "n_treatment_pred_cols": 1,
    }
    with pytest.raises(ValueError, match="no 'mixture'.*compile=False"):
        cls.from_config(config)


def test_causal_loss_shares_one_mixture_and_rejects_a_mismatch():
    """On the mixture path the two losses must agree, and end up sharing one object."""
    a, b = _normal_bernoulli_mixture(3), _normal_bernoulli_mixture(3)
    causal = losses.CausalLoss(losses.OutcomeLoss(a), losses.TreatmentLoss(b))
    assert causal.mixture is a and causal._treatment_loss.mixture is a
    assert causal.layout == a.layout
    other = mixture.StateMixture(families.Normal(), families.Normal(), n_states=3)
    with pytest.raises(ValueError, match="share one StateMixture"):
        losses.CausalLoss(losses.OutcomeLoss(a), losses.TreatmentLoss(other))


def test_from_mixture_equals_the_composed_causal_loss():
    """CausalLoss.from_mixture equals CausalLoss of the two losses, regularizer included."""
    y_true, y_pred, _ = _outcome_likelihood_fixture(seed=5)
    reg = tf.keras.regularizers.L2(0.1)
    new = losses.CausalLoss.from_mixture(
        _normal_bernoulli_mixture(3),
        alpha=2.0,
        outcome_likelihood="observational",
        outcome_loss_weight=0.5,
        predictive_states_regularizer=reg,
    )
    composed = losses.CausalLoss(
        _normal_outcome_loss("sum_over_batch_size", outcome_likelihood="observational"),
        _treatment_loss("sum_over_batch_size"),
        alpha=2.0,
        outcome_loss_weight=0.5,
        predictive_states_regularizer=reg,
    )
    np.testing.assert_allclose(
        float(new(y_true, y_pred)), float(composed(y_true, y_pred)), rtol=1e-6
    )
