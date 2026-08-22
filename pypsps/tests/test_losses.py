"""Test module for loss functions."""

import random

import numpy as np
import pytest
import tensorflow as tf

from .. import datasets, utils
from ..keras import losses, models, neglogliks

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
    """CausalLoss._alpha must be a tf.Variable, and .assign() must reach an already-traced
    (tf.function-compiled) call -- the way model.fit() actually evaluates the loss
    (run_eagerly=False by default).

    A plain-float `_alpha` gets baked into the graph as a constant at the first trace; a
    schedule rebinding `loss._alpha` afterward (e.g. AlphaScheduleCallback) would then only be
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
    assert isinstance(causal_loss._alpha, tf.Variable)

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
    causal_loss._alpha.assign(1.0)
    loss_alpha_1 = traced(y_true, y_pred).numpy()
    causal_loss._alpha.assign(100.0)
    loss_alpha_100 = traced(y_true, y_pred).numpy()

    assert not np.isclose(loss_alpha_1, loss_alpha_100), (
        "assigning a new alpha did not change the traced graph's output -- an alpha schedule "
        "would silently fail to reach the optimizer during model.fit()"
    )


def _normal_pdf(y: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * ((y - loc) / scale) ** 2) / (scale * np.sqrt(2.0 * np.pi))


def _bernoulli_pmf(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.where(a == 1, p, 1.0 - p)


def _build_synthetic_preds(weights, locs, scales, treat_probs):
    """Builds a synthetic interleaved y_pred tensor for OutcomeLoss/TreatmentLoss.

    All inputs have shape (n_rows, n_states). Layout matches utils.split_y_pred with
    n_outcome_pred_cols=2 (Normal loc, scale) and n_treatment_pred_cols=1 (Bernoulli prob).
    """
    return tf.constant(
        np.concatenate([locs, scales, weights, treat_probs], axis=1), dtype=tf.float32
    )


@pytest.mark.parametrize(
    "n_states,weights",
    [
        (1, [[1.0]]),
        (2, [[0.5, 0.5], [0.999, 0.001], [0.001, 0.999], [0.1, 0.9]]),
        (3, [[1.0 / 3, 1.0 / 3, 1.0 / 3], [0.98, 0.01, 0.01], [0.2, 0.3, 0.5]]),
    ],
)
def test_outcome_and_treatment_loss_telescope_to_joint_nll(n_states, weights):
    """-log p(a, y | x) = TreatmentLoss (dose, prior-weighted) + OutcomeLoss (posterior-weighted).

    Verifies the identity against a closed-form joint mixture likelihood computed directly
    (without logsumexp), for K=1,2,3 states including near-degenerate mixture weights.
    """
    weights = np.array(weights, dtype=np.float64)
    n_rows = weights.shape[0]

    rng = np.random.RandomState(0)
    locs = rng.uniform(-2.0, 2.0, size=(n_rows, n_states))
    scales = rng.uniform(0.5, 2.0, size=(n_rows, n_states))
    treat_probs = rng.uniform(0.05, 0.95, size=(n_rows, n_states))

    outcome_true = rng.normal(size=(n_rows, 1))
    treatment_true = rng.randint(0, 2, size=(n_rows, 1)).astype(np.float64)

    # Closed-form joint mixture likelihood (no logsumexp): p(a, y | x) = sum_k w_k * p(a|s_k) * p(y|s_k,a).
    joint_components = (
        weights
        * _bernoulli_pmf(treatment_true, treat_probs)
        * _normal_pdf(outcome_true, locs, scales)
    )
    closed_form_nll = -np.log(joint_components.sum(axis=1))

    y_pred = _build_synthetic_preds(weights, locs, scales, treat_probs)
    y_true = tf.constant(np.concatenate([outcome_true, treatment_true], axis=1), dtype=tf.float32)

    treatment_nll = tf.keras.losses.BinaryCrossentropy(reduction="none")
    outcome_loss = losses.OutcomeLoss(
        loss=neglogliks.NegloglikNormal(reduction="none"),
        treatment_loss=treatment_nll,
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction="none",
    )
    treatment_loss = losses.TreatmentLoss(
        loss=treatment_nll,
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction="none",
    )

    dose_term = treatment_loss(y_true, y_pred).numpy()
    outcome_term = outcome_loss(y_true, y_pred).numpy()

    np.testing.assert_allclose(dose_term + outcome_term, closed_form_nll, rtol=1e-4, atol=1e-4)
