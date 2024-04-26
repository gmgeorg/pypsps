"""Module for metrics from pypsps predictions."""

import tensorflow as tf
from .. import utils


@tf.keras.utils.register_keras_serializable(package="pypsps")
class PropensityScoreBinaryCrossentropy(tf.keras.metrics.BinaryCrossentropy):
    """Computes cross entropy for the propensity score. Used as a metric in pypsps model."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state."""
        _, _, _, propensity_score = utils.split_y_pred(y_pred)
        treatment_true = y_true[:, 1:]
        super().update_state(
            y_true=treatment_true, y_pred=propensity_score, sample_weight=sample_weight
        )


@tf.keras.utils.register_keras_serializable(package="pypsps")
class PropensityScoreAUC(tf.keras.metrics.AUC):
    """AUC computed on the ouptut for propensity part."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        _, _, _, propensity_score = utils.split_y_pred(y_pred)
        treatment_true = y_true[:, 1:]
        super().update_state(
            y_true=treatment_true, y_pred=propensity_score, sample_weight=sample_weight
        )


@tf.keras.utils.register_keras_serializable(package="pypsps")
class OutcomeMeanSquaredError(tf.keras.metrics.MeanSquaredError):
    """MSE computed on the ouptut for weighted average outcome prediction."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        avg_outcome = utils.agg_outcome_pred(y_pred)
        outcome_true = y_true[:, 0:1]
        super().update_state(
            y_true=outcome_true, y_pred=avg_outcome, sample_weight=sample_weight
        )


@tf.keras.utils.register_keras_serializable(package="pypsps")
class OutcomeMeanAbsoluteError(tf.keras.metrics.MeanAbsoluteError):
    """MSE computed on the ouptut for weighted average outcome prediction."""

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        avg_outcome = utils.agg_outcome_pred(y_pred)
        outcome_true = y_true[:, 0:1]
        super().update_state(
            y_true=outcome_true, y_pred=avg_outcome, sample_weight=sample_weight
        )
