"""Module for metrics from pypsps predictions."""

import tensorflow as tf
import pypress.utils
from .. import utils


@tf.keras.utils.register_keras_serializable(package="pypsps")
class PropensityScoreBinaryCrossentropy(tf.keras.metrics.BinaryCrossentropy):
    """Computes cross entropy for the propensity score. Used as a metric in pypsps model."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state."""
        _, _, propensity_score = utils.split_y_pred(y_pred, n_outcome_pred_cols=2, n_treatment_pred_cols=1)
        treatment_true = y_true[:, 1:]
        super().update_state(
            y_true=treatment_true, y_pred=propensity_score, sample_weight=sample_weight
        )


@tf.keras.utils.register_keras_serializable(package="pypsps")
class PropensityScoreAUC(tf.keras.metrics.AUC):
    """AUC computed on the ouptut for propensity part."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        _, _, propensity_score = utils.split_y_pred(y_pred, n_outcome_pred_cols=2, n_treatment_pred_cols=1)
        treatment_true = y_true[:, 1:]
        super().update_state(
            y_true=treatment_true, y_pred=propensity_score, sample_weight=sample_weight
        )


@tf.keras.utils.register_keras_serializable(package="pypsps")
class OutcomeMeanSquaredError(tf.keras.metrics.MeanSquaredError):
    """MSE computed on the ouptut for weighted average outcome prediction."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        avg_outcome = utils.agg_outcome_pred(y_pred, n_outcome_pred_cols=2, n_treatment_pred_cols=1)
        outcome_true = utils.split_y_true(y_true, n_outcome_cols=1)[0]
        super().update_state(
            y_true=outcome_true, y_pred=avg_outcome, sample_weight=sample_weight
        )

# TODO: add unit test for Outcome* metrics
@tf.keras.utils.register_keras_serializable(package="pypsps")
class OutcomeMeanAbsoluteError(tf.keras.metrics.MeanAbsoluteError):
    """MSE computed on the ouptut for weighted average outcome prediction."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        avg_outcome = utils.agg_outcome_pred(y_pred, n_outcome_pred_cols=2, n_treatment_pred_cols=1)
        outcome_true = utils.split_y_true(y_true, n_outcome_cols=1)[0]
        super().update_state(
            y_true=outcome_true, y_pred=avg_outcome, sample_weight=sample_weight
        )

# TODO: make public in pypress and move this as a metric to pypress/metrics.py

def _tr_kernel(weights: tf.Tensor) -> tf.Tensor:
    """Computes trace of kernel matrix implied by PRESS tensor."""
    return tf.reduce_sum(
        tf.linalg.diag_part(
            tf.matmul(tf.transpose(pypress.utils.tf_col_normalize(weights)), weights)
        )
    )

def predictive_state_df(y_true, y_pred) -> tf.Tensor:
    """Computes degrees of freedom of predictive state weights."""
    del y_true
    _, weights, _ = utils.split_y_pred(y_pred, 
                                       n_outcome_pred_cols=1,
                                       n_treatment_pred_cols=2)
    return _tr_kernel(weights)