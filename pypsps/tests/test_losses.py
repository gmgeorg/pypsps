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
    """-log p(a, y | x) = TreatmentLoss (prior-weighted) + observational OutcomeLoss.

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
        outcome_likelihood="observational",
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


def _v0_1_10_outcome_nll(outcome_loss, y_true, y_pred):
    """Frozen copy of `OutcomeLoss.call` as of v0.1.10 (posterior weights only).

    The reference that pins the observational path to be bit-identical.
    """
    self = outcome_loss
    n_states = utils.get_n_states(y_pred, self._n_outcome_pred_cols, self._n_treatment_pred_cols)
    outcome_params_pred, weights, treatment_pred = utils.split_y_pred(
        y_pred, self._n_outcome_pred_cols, self._n_treatment_pred_cols
    )
    outcome_true, treatment_true = utils.split_y_true(y_true, self._n_outcome_true_cols)
    gamma = losses.prob_state_given_treatment_features(
        weights=weights,
        treatment_true=treatment_true,
        treatment_pred=treatment_pred,
        treatment_nll_loss=self._treatment_loss,
    )
    log_weights = utils.safe_log(gamma)
    log_components = []
    for j in range(n_states):
        cols_to_select = losses._gen_col_selectors(j, n_states, self._n_outcome_pred_cols)
        outcome_pred_state_j = tf.gather(outcome_params_pred, cols_to_select, axis=1)
        nll_state_j = self._loss(y_true=outcome_true, y_pred=outcome_pred_state_j)
        log_components.append(tf.expand_dims(log_weights[:, j] + (-nll_state_j), axis=1))
    marginal_nll = -tf.math.reduce_logsumexp(tf.concat(log_components, axis=1), axis=1)
    if self.reduction == tf.keras.losses.Reduction.NONE:
        return marginal_nll
    weighted_loss_sum = tf.reduce_sum(marginal_nll)
    if self.reduction in (tf.keras.losses.Reduction.SUM,):
        return weighted_loss_sum
    return weighted_loss_sum / tf.cast(tf.shape(y_true)[0], y_true.dtype)


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


@pytest.mark.parametrize("reduction", ["none", "sum", "sum_over_batch_size"])
@pytest.mark.parametrize(
    "outcome_likelihood", ["observational", losses.OutcomeLikelihood.OBSERVATIONAL]
)
def test_observational_likelihood_is_bit_identical_to_v0_1_10(reduction, outcome_likelihood):
    """The observational path returns exactly the v0.1.10 default's values and gradients."""
    y_true, y_pred, _ = _outcome_likelihood_fixture()
    loss = _normal_outcome_loss(reduction, outcome_likelihood=outcome_likelihood)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(y_pred)
        new = loss(y_true, y_pred)
        old = _v0_1_10_outcome_nll(loss, y_true, y_pred)
    np.testing.assert_array_equal(new.numpy(), old.numpy())
    np.testing.assert_array_equal(
        tape.gradient(new, y_pred).numpy(), tape.gradient(old, y_pred).numpy()
    )


def test_interventional_likelihood_matches_closed_form():
    """ "interventional" is -log sum_k P(s_k | x) p(y | s_k, a, x)."""
    y_true, y_pred, (weights, locs, scales, outcome_true) = _outcome_likelihood_fixture(seed=1)
    loss = _normal_outcome_loss(outcome_likelihood="interventional")
    want = -np.log((weights * _normal_pdf(outcome_true, locs, scales)).sum(axis=1))
    np.testing.assert_allclose(loss(y_true, y_pred).numpy(), want, rtol=1e-4, atol=1e-5)


def test_interventional_likelihood_ignores_the_treatment_head():
    """The interventional outcome loss has no gradient into the treatment predictions."""
    y_true, y_pred, _ = _outcome_likelihood_fixture(seed=2)
    treatment_cols = slice(3 * 3, None)  # [locs | scales | weights | treatment probs]
    grads = {}
    for name in ("interventional", "observational"):
        with tf.GradientTape() as tape:
            tape.watch(y_pred)
            value = tf.reduce_sum(_normal_outcome_loss(outcome_likelihood=name)(y_true, y_pred))
        grads[name] = tape.gradient(value, y_pred).numpy()[:, treatment_cols]
    assert np.all(grads["interventional"] == 0.0)
    assert np.any(grads["observational"] != 0.0)


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
    assert causal._alpha is alpha
    with pytest.raises(ValueError, match="float32"):
        losses.CausalLoss(
            _normal_outcome_loss("sum_over_batch_size"),
            treatment,
            alpha=tf.Variable(1.0, dtype=tf.float64),
        )
