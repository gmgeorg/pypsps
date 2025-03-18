import numpy as np
import pytest
import tensorflow as tf

# Import the metrics module from your package.
# Adjust the import path as needed.
from pypsps.keras import metrics

# --- Dummy utility implementations for testing ---


def dummy_split_y_pred(y_pred, n_outcome_pred_cols, n_treatment_pred_cols):
    """
    Splits y_pred into a tuple of three tensors:
      outcome_pred: first n_outcome_pred_cols columns,
      treatment_pred: next n_treatment_pred_cols columns,
      propensity_score: remaining columns.
    """
    outcome_pred = y_pred[:, :n_outcome_pred_cols]
    treatment_pred = y_pred[:, n_outcome_pred_cols : n_outcome_pred_cols + n_treatment_pred_cols]
    propensity_score = y_pred[:, n_outcome_pred_cols + n_treatment_pred_cols :]
    return outcome_pred, treatment_pred, propensity_score


def dummy_split_y_true(y_true, n_outcome_true_cols):
    """
    Splits y_true into a tuple of two tensors:
      outcome_true: first n_outcome_true_cols columns,
      treatment_true: remaining columns.
    """
    outcome_true = y_true[:, :n_outcome_true_cols]
    treatment_true = y_true[:, n_outcome_true_cols:]
    return outcome_true, treatment_true


def dummy_agg_outcome_pred(y_pred, n_outcome_pred_cols, n_treatment_pred_cols):
    """
    Aggregates outcome predictions by taking the mean over the outcome columns.
    (For simplicity, since there's only one outcome column in our test,
    this just returns that column.)
    """
    outcome_pred = y_pred[:, :n_outcome_pred_cols]
    # Here, if outcome_pred has one column, the mean is the column itself.
    return outcome_pred


# Monkey-patch the utility functions used by the metrics.
@pytest.fixture(autouse=True)
def patch_utils(monkeypatch):
    from pypsps import utils as p_utils

    monkeypatch.setattr(p_utils, "split_y_pred", dummy_split_y_pred)
    monkeypatch.setattr(p_utils, "split_y_true", dummy_split_y_true)
    monkeypatch.setattr(p_utils, "agg_outcome_pred", dummy_agg_outcome_pred)
    # For predictive_state_df_gen, we rely on pypress.utils.tr_kernel.
    # Here we assume the existing implementation is sufficient.
    # If desired, you could also override tr_kernel with a dummy function.
    yield


def test_propensity_score_binary_crossentropy():
    # Create dummy y_true and y_pred.
    # For this metric, update_state extracts:
    #   treatment_true = y_true[:, 1:]
    #   propensity_score = split_y_pred(y_pred, n_outcome_pred_cols=2, n_treatment_pred_cols=1)[2]
    # Let's assume y_true shape is (batch, 2) and y_pred shape is (batch, 4)
    # so that:
    #   outcome_pred = y_pred[:, :2]
    #   treatment_pred = y_pred[:, 2:3]
    #   propensity_score = y_pred[:, 3:]
    y_true = tf.constant([[0, 1], [1, 0], [1, 1]], dtype=tf.float32)
    # Construct y_pred so that propensity_score (last column) are:
    # [[0.4], [0.8], [1.2]]
    y_pred = tf.constant(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], dtype=tf.float32
    )

    metric = metrics.PropensityScoreBinaryCrossentropy()
    # Call update_state; under the hood, this will pass treatment_true and propensity_score
    metric.update_state(y_true, y_pred)
    result = metric.result().numpy()
    # For a basic test, simply ensure that a result was computed (value can vary)
    assert isinstance(result, np.float32) or isinstance(result, float)


def test_propensity_score_auc():
    """Test propensity score AUC"""
    y_true = tf.constant([[0, 1], [1, 0], [1, 1]], dtype=tf.float32)
    y_pred = tf.constant(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]], dtype=tf.float32
    )

    metric = metrics.PropensityScoreAUC()
    metric.update_state(y_true, y_pred)
    auc = metric.result().numpy()
    # AUC should be between 0 and 1.
    assert 0.0 <= auc <= 1.0


def test_treatment_mean_squared_error():
    """test for treatment MSE"""
    # For TreatmentMeanSquaredError:
    #   treat_pred = split_y_pred(y_pred, n_outcome_pred_cols=1, n_treatment_pred_cols=2)[2]
    #   treat_true = split_y_true(y_true, n_outcome_true_cols=1)[1]
    # We'll supply y_true with 2 columns and y_pred with 4 columns.
    # Let treat_true be the second column of y_true.
    y_true = tf.constant([[10, 2], [20, 4], [30, 6]], dtype=tf.float32)
    # For y_pred, split as:
    # outcome_pred: first column, treatment_pred: next two columns, treat_pred: last column.
    # Let treat_pred be close to treat_true so MSE is small.
    y_pred = tf.constant([[9, 0, 0, 2.1], [19, 0, 0, 3.9], [31, 0, 0, 6.2]], dtype=tf.float32)
    metric = metrics.TreatmentMeanSquaredError()
    metric.update_state(y_true, y_pred)
    mse = metric.result().numpy()
    # Since predictions are close, MSE should be low.
    assert mse < 1.0


def test_treatment_mean_absolute_error():
    """ttest for MAE treatment"""
    y_true = tf.constant([[10, 2], [20, 4], [30, 6]], dtype=tf.float32)
    y_pred = tf.constant([[9, 0, 0, 2.0], [20, 0, 0, 4.0], [30, 0, 0, 6.0]], dtype=tf.float32)
    metric = metrics.TreatmentMeanAbsoluteError()
    metric.update_state(y_true, y_pred)
    mae = metric.result().numpy()
    # Expect near zero error
    np.testing.assert_allclose(mae, 0.0, atol=1e-6)


# --- Tests for OutcomeMeanSquaredError ---


def test_outcome_mean_squared_error():
    """Tests for MSE"""
    # OutcomeMeanSquaredError uses agg_outcome_pred to compute a prediction
    # and outcome_true = split_y_true(y_true, n_outcome_true_cols=1)[0].
    # We'll set outcome_true equal to the outcome_pred.
    y_true = tf.constant(
        [
            [5, 100],  # outcome_true=5, treatment_true=100 (ignored)
            [10, 200],
            [15, 300],
        ],
        dtype=tf.float32,
    )
    # For y_pred, we need at least 2 outcome prediction columns.
    # Let outcome_pred be first 2 columns. For simplicity, set both equal.
    # And set a dummy treatment part afterward.
    y_pred = tf.constant(
        [[5, 5, 0.0, 0.0], [10, 10, 0.0, 0.0], [15, 15, 0.0, 0.0]], dtype=tf.float32
    )
    metric = metrics.OutcomeMeanSquaredError()
    metric.update_state(y_true, y_pred)
    mse = metric.result().numpy()
    np.testing.assert_allclose(mse, 0.0, atol=1e-6)


def test_outcome_mean_absolute_error():
    """tests for outcome MAE."""
    y_true = tf.constant([[5, 100], [10, 200], [15, 300]], dtype=tf.float32)
    y_pred = tf.constant(
        [[5, 5, 0.0, 0.0], [10, 10, 0.0, 0.0], [15, 15, 0.0, 0.0]], dtype=tf.float32
    )
    metric = metrics.OutcomeMeanAbsoluteError()
    metric.update_state(y_true, y_pred)
    mae = metric.result().numpy()
    np.testing.assert_allclose(mae, 0.0, atol=1e-6)


# --- Test for predictive_state_df_gen ---
def test_predictive_state_df_gen():
    # For predictive_state_df_gen, the returned function extracts weights from split_y_pred.
    # Our dummy split_y_pred returns:
    #   outcome_pred = y_pred[:, :n_outcome_pred_cols],
    #   treatment_pred = y_pred[:, n_outcome_pred_cols:n_outcome_pred_cols+n_treatment_pred_cols],
    #   propensity_score = y_pred[:, n_outcome_pred_cols+n_treatment_pred_cols:].
    # Let’s simulate a y_pred where the "weights" (i.e. the second element) are known.
    n_outcome_pred_cols = 2
    n_treatment_pred_cols = 1
    # Let y_pred be of shape (3, 4): first 2 for outcome, next 1 for weights, last 1 for propensity.
    y_pred = tf.constant([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], dtype=tf.float32)
    # We define a dummy tr_kernel that simply returns the sum of weights.
    # But here we rely on the actual pypress.utils.tr_kernel; if that's not easily predictable,
    # we can simply test that predictive_state_df returns a scalar tensor.
    func = metrics.predictive_state_df_gen(n_outcome_pred_cols, n_treatment_pred_cols)
    result = func(None, y_pred)
    # Check that result is a scalar tensor.
    assert result.shape.ndims == 0 or (result.shape.ndims == 1 and result.shape[0] == 1)
