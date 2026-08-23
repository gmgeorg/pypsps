import numpy as np
import tensorflow as tf

from pypsps.keras import losses, metrics, neglogliks


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


def test_propensity_score_binary_crossentropy():
    """PropensityScoreBinaryCrossentropy must compare treatment_true against the
    weighted (prior) average of the state-conditional propensity predictions."""
    weights = [[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]]
    treatment_by_state = [[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]]  # p(a=1 | s_k, x)

    y_pred = _make_y_pred(
        outcome_blocks=[np.zeros((3, 2))],  # n_outcome_pred_cols=1, unused by this metric
        weights=weights,
        treatment_blocks=[treatment_by_state],  # n_treatment_pred_cols=1
    )
    y_true = tf.constant([[0.0, 1.0], [0.0, 0.0], [0.0, 1.0]], dtype=tf.float32)

    metric = metrics.PropensityScoreBinaryCrossentropy(
        n_outcome_pred_cols=1, n_treatment_pred_cols=1
    )
    metric.update_state(y_true, y_pred)

    marginal_propensity = (np.asarray(weights) * np.asarray(treatment_by_state)).sum(axis=1)
    expected = tf.keras.metrics.BinaryCrossentropy()
    expected.update_state(y_true=y_true[:, -1:], y_pred=marginal_propensity[:, None])

    np.testing.assert_allclose(metric.result().numpy(), expected.result().numpy(), rtol=1e-5)


def test_propensity_score_auc():
    """Test propensity score AUC uses the weighted (prior) marginal propensity."""
    weights = [[0.5, 0.5], [0.3, 0.7], [0.9, 0.1], [0.1, 0.9]]
    treatment_by_state = [[0.9, 0.1], [0.5, 0.5], [0.2, 0.8], [0.7, 0.3]]

    y_pred = _make_y_pred(
        outcome_blocks=[np.zeros((4, 2))],
        weights=weights,
        treatment_blocks=[treatment_by_state],
    )
    y_true = tf.constant([[0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=tf.float32)

    metric = metrics.PropensityScoreAUC(n_outcome_pred_cols=1, n_treatment_pred_cols=1)
    metric.update_state(y_true, y_pred)

    marginal_propensity = (np.asarray(weights) * np.asarray(treatment_by_state)).sum(axis=1)
    expected = tf.keras.metrics.AUC()
    expected.update_state(y_true=y_true[:, -1:], y_pred=marginal_propensity[:, None])

    np.testing.assert_allclose(metric.result().numpy(), expected.result().numpy(), rtol=1e-5)


def test_propensity_score_binary_crossentropy_treatment_is_last_column():
    """Regression test: propensity metrics must read treatment from the LAST column of
    y_true, not from y_true[:, 1:]. Outcome data can have more than one column (e.g. a
    survival outcome of [event_time, event_indicator]), in which case y_true[:, 1:]
    would silently leak non-treatment columns into the propensity metric. Treatment is
    always appended as the final column by utils.prepare_keras_inputs_outputs, so
    y_true[:, -1:] must be used regardless of how many outcome columns precede it.
    """
    weights = [[1.0], [1.0], [1.0]]  # n_states=1: no mixing, marginal == state-0 prediction
    treatment_by_state = [[0.9], [0.1], [0.8]]

    y_pred = _make_y_pred(
        outcome_blocks=[np.zeros((3, 1)), np.zeros((3, 1))],  # n_outcome_pred_cols=2
        weights=weights,
        treatment_blocks=[treatment_by_state],
    )
    # 2 outcome-like columns followed by the treatment column (last).
    y_true = tf.constant([[5.0, 0.0, 1.0], [3.0, 1.0, 0.0], [8.0, 1.0, 1.0]], dtype=tf.float32)

    metric = metrics.PropensityScoreBinaryCrossentropy(
        n_outcome_pred_cols=2, n_treatment_pred_cols=1
    )
    metric.update_state(y_true, y_pred)

    expected = tf.keras.metrics.BinaryCrossentropy()
    expected.update_state(y_true=y_true[:, -1:], y_pred=np.asarray(treatment_by_state))

    np.testing.assert_allclose(metric.result().numpy(), expected.result().numpy(), rtol=1e-5)


def test_propensity_score_auc_treatment_is_last_column():
    """Regression test: see test_propensity_score_binary_crossentropy_treatment_is_last_column."""
    weights = [[1.0], [1.0], [1.0]]
    treatment_by_state = [[0.9], [0.1], [0.8]]

    y_pred = _make_y_pred(
        outcome_blocks=[np.zeros((3, 1)), np.zeros((3, 1))],
        weights=weights,
        treatment_blocks=[treatment_by_state],
    )
    y_true = tf.constant([[5.0, 0.0, 1.0], [3.0, 1.0, 0.0], [8.0, 1.0, 1.0]], dtype=tf.float32)

    metric = metrics.PropensityScoreAUC(n_outcome_pred_cols=2, n_treatment_pred_cols=1)
    metric.update_state(y_true, y_pred)

    expected = tf.keras.metrics.AUC()
    expected.update_state(y_true=y_true[:, -1:], y_pred=np.asarray(treatment_by_state))

    np.testing.assert_allclose(metric.result().numpy(), expected.result().numpy(), rtol=1e-5)


def test_treatment_mean_squared_error():
    """test for treatment MSE"""
    weights = [[1.0], [1.0], [1.0]]  # n_states=1: agg_treatment_pred passes values through
    treat_pred = [[2.1], [3.9], [6.2]]
    y_pred = _make_y_pred(
        outcome_blocks=[np.array([[9.0], [19.0], [31.0]])],
        weights=weights,
        treatment_blocks=[treat_pred],
    )
    y_true = tf.constant([[10, 2], [20, 4], [30, 6]], dtype=tf.float32)
    metric = metrics.TreatmentMeanSquaredError(
        n_outcome_pred_cols=1, n_treatment_pred_cols=1, n_outcome_true_cols=1
    )
    metric.update_state(y_true, y_pred)
    mse = metric.result().numpy()
    # Since predictions are close, MSE should be low.
    assert mse < 1.0


def test_treatment_mean_absolute_error():
    """test for MAE treatment"""
    weights = [[1.0], [1.0], [1.0]]
    treat_pred = [[2.0], [4.0], [6.0]]
    y_pred = _make_y_pred(
        outcome_blocks=[np.array([[9.0], [20.0], [30.0]])],
        weights=weights,
        treatment_blocks=[treat_pred],
    )
    y_true = tf.constant([[10, 2], [20, 4], [30, 6]], dtype=tf.float32)
    metric = metrics.TreatmentMeanAbsoluteError(
        n_outcome_pred_cols=1, n_treatment_pred_cols=1, n_outcome_true_cols=1
    )
    metric.update_state(y_true, y_pred)
    mae = metric.result().numpy()
    # Expect near zero error
    np.testing.assert_allclose(mae, 0.0, atol=1e-6)


def test_outcome_mean_squared_error():
    """Tests for MSE"""
    y_true = tf.constant(
        [
            [5, 100],  # outcome_true=5, treatment_true=100 (ignored)
            [10, 200],
            [15, 300],
        ],
        dtype=tf.float32,
    )
    # n_states=1: [outcome, weight=1.0, treatment]. Weight of 1.0 means agg_outcome_pred
    # passes the outcome prediction through unweighted.
    y_pred = tf.constant([[5, 1.0, 0.0], [10, 1.0, 0.0], [15, 1.0, 0.0]], dtype=tf.float32)
    metric = metrics.OutcomeMeanSquaredError(
        n_outcome_pred_cols=1, n_treatment_pred_cols=1, n_outcome_true_cols=1
    )
    metric.update_state(y_true, y_pred)
    mse = metric.result().numpy()
    np.testing.assert_allclose(mse, 0.0, atol=1e-6)


def test_outcome_mean_absolute_error():
    """tests for outcome MAE."""
    y_true = tf.constant([[5, 100], [10, 200], [15, 300]], dtype=tf.float32)
    y_pred = tf.constant([[5, 1.0, 0.0], [10, 1.0, 0.0], [15, 1.0, 0.0]], dtype=tf.float32)
    metric = metrics.OutcomeMeanAbsoluteError(
        n_outcome_pred_cols=1, n_treatment_pred_cols=1, n_outcome_true_cols=1
    )
    metric.update_state(y_true, y_pred)
    mae = metric.result().numpy()
    np.testing.assert_allclose(mae, 0.0, atol=1e-6)


def test_predictive_state_df_gen():
    """Test predictive state gen"""
    n_outcome_pred_cols = 1
    n_treatment_pred_cols = 1
    weights = [[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]]
    y_pred = _make_y_pred(
        outcome_blocks=[np.zeros((3, 2))],
        weights=weights,
        treatment_blocks=[np.zeros((3, 2))],
    )
    func = metrics.predictive_state_df_gen(n_outcome_pred_cols, n_treatment_pred_cols)
    result = func(None, y_pred)
    # Check that result is a scalar tensor.
    assert result.shape.ndims == 0 or (result.shape.ndims == 1 and result.shape[0] == 1)


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
    assert clean._loss is original._loss


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

    metric_fn = metrics.causal_loss_metric_gen(
        outcome_loss=penalized_outcome_loss, treatment_loss=penalized_treatment_loss
    )
    metric_value = metric_fn(y_true, y_pred).numpy()

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
