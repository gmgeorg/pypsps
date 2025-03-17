"""Module to implement distributions and log-likelihood loss fcts."""

import math

import tensorflow as tf
import tensorflow_probability as tfp

import pypsps.utils

tfd = tfp.distributions


_EPS = 1e-6


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
        n_params = pypsps.utils.get_n_cols(y_pred)

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


def _negloglik_normal(y: tf.Tensor, loc: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    """Computes negative log-likelihood of data y ~ Normal(mu, sigma)."""
    negloglik_element = tf.math.log(2.0 * math.pi) / 2.0 + tf.math.log(scale + _EPS)
    negloglik_element += 0.5 * tf.square((y - loc) / (scale + _EPS))
    return tf.squeeze(negloglik_element)


@tf.keras.utils.register_keras_serializable(package="pypsps")
class NegloglikNormal(tf.keras.losses.Loss):
    """Computes the negative log-likelihood of y ~ N(mu, sigma^2)."""

    def call(self, y_true, y_pred):
        """Implements the loss function call."""
        y_true = tf.squeeze(y_true)
        loc_pred = y_pred[:, 0]
        scale_pred = y_pred[:, 1]
        losses = _negloglik_normal(y_true, loc_pred, scale_pred)
        if self.reduction == tf.keras.losses.Reduction.NONE:
            return losses
        if self.reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_sum(losses, axis=-1)
        if self.reduction in (
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            tf.keras.losses.Reduction.AUTO,
        ):
            return tf.reduce_mean(losses, axis=-1)
        raise NotImplementedError("reduction='%s' is not implemented", self.reduction)
