import numpy as np
import tensorflow as tf

from pypsps.keras import metrics


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
