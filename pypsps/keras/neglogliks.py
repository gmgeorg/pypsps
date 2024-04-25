"""Module to implement distributions and log-likelihood loss fcts."""

import tensorflow as tf
import math
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions


@tf.keras.utils.register_keras_serializable(package="pypsps")
class NegloglikLoss(tf.keras.losses.Loss):
    """Computes the negative log-likelihood of y ~ Distribution.

    This is a general purpose class for any (!) tfd.Distribution.
    """

    def __init__(self, distribution_constructor: tfd.Distribution, **kwargs):
        self._distribution_constructor = distribution_constructor
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """Implements the loss function call."""
        if isinstance(y_pred, np.ndarray):
            n_params = y_pred.shape[1]
        else:
            n_params = y_pred.get_shape().as_list()[1]

        y_pred_cols = [tf.squeeze(c) for c in tf.split(y_pred, n_params, axis=1)]
        distr = self._distribution_constructor(*y_pred_cols)
        losses = -distr.log_prob(y_true)

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return losses
        if self.reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_sum(losses)
        if self.reduction in (
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            tf.keras.losses.Reduction.AUTO,
        ):
            return tf.reduce_mean(losses)
        raise NotImplementedError("reduction='%s' is not implemented", self.reduction)


def _negloglik(y: tf.Tensor, mu: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    """Computes negative log-likelihood of data y ~ Normal(mu, sigma)."""
    negloglik_element = tf.math.log(2.0 * math.pi) / 2.0 + tf.math.log(sigma)
    negloglik_element += 0.5 * tf.square((y - mu) / sigma)
    return negloglik_element


@tf.keras.utils.register_keras_serializable(package="pypsps")
class NegloglikNormal(tf.keras.losses.Loss):
    """Computes the negative log-likelihood of y ~ N(mu, sigma^2)."""

    def call(self, y_true, y_pred):
        """Implements the loss function call."""
        y_pred_mu = y_pred[:, 0]
        y_pred_scale = y_pred[:, 1]

        losses = _negloglik(y_true, y_pred_mu, y_pred_scale)
        if self.reduction == tf.keras.losses.Reduction.NONE:
            return losses
        if self.reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_sum(losses)
        if self.reduction in (
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            tf.keras.losses.Reduction.AUTO,
        ):
            return tf.reduce_mean(losses)
        raise NotImplementedError("reduction='%s' is not implemented", self.reduction)
