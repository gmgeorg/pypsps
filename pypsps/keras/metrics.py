"""Module for metrics from pypsps predictions."""

import tensorflow as tf
from pypsps import utils


@tf.keras.utils.register_keras_serializable(package="psps")
def propensity_score_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor):
    """Computes cross entropy for the propensity score. Used as a metric in pypsps model."""
    _, _, propensity_score, _ = utils.split_y_pred(y_pred)
    treatment_true = y_true[:, 1]
    return tf.keras.metrics.binary_crossentropy(
        y_true=treatment_true, y_pred=propensity_score
    )
