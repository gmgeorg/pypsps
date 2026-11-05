"""Module to implement distributions and log-likelihood loss fcts."""

import math
from typing import TYPE_CHECKING, Any

import tensorflow as tf

import pypsps.utils

if TYPE_CHECKING:  # annotation only: pypsps never imports TensorFlow Probability at runtime
    import tensorflow_probability as tfp


_EPS = 1e-6


@tf.keras.utils.register_keras_serializable(package="pypsps")
class NegloglikLoss(tf.keras.losses.Loss):
    """Computes the negative log-likelihood of y ~ Distribution.

    This is a general purpose class for any (!) ``tfp.distributions.Distribution``. The
    caller passes the constructor, so the caller needs TensorFlow Probability installed;
    pypsps itself does not depend on it.
    """

    def __init__(self, distribution_constructor: "tfp.distributions.Distribution", **kwargs: Any):
        """Stores the distribution constructor used to build the loss."""
        self._distribution_constructor = distribution_constructor
        super().__init__(**kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Implements the loss function call."""
        n_params = pypsps.utils.get_n_cols(y_pred)

        y_pred_cols = [tf.squeeze(c, axis=-1) for c in tf.split(y_pred, n_params, axis=1)]
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
    return negloglik_element


@tf.keras.utils.register_keras_serializable(package="pypsps")
class NegloglikNormal(tf.keras.losses.Loss):
    """Computes the negative log-likelihood of y ~ N(mu, sigma^2)."""

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Implements the loss function call."""
        if y_true.shape.rank == 2 and y_true.shape[-1] == 1:
            y_true = tf.squeeze(y_true, axis=-1)
        loc_pred = y_pred[:, 0]
        scale_pred = y_pred[:, 1]
        losses = _negloglik_normal(y=y_true, loc=loc_pred, scale=scale_pred)
        losses = tf.ensure_shape(losses, [None])
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


def _negloglik_exponential(
    event_time: tf.Tensor, event_indicator: tf.Tensor, rate: tf.Tensor
) -> tf.Tensor:
    """
    Computes the negative log-likelihood for an exponential distribution with censoring.

    For each observation i:
      - If an event occurs (event_indicator[i] == 1):
            log-likelihood = log(rate[i]) - rate[i] * event_time[i]
      - If censored (event_indicator[i] == 0):
            log-likelihood = - rate[i] * event_time[i]

    Therefore, the negative log-likelihood for observation i is:
      loss_i = rate[i] * event_time[i] - event_indicator[i] * log(rate[i])

    Parameters
    ----------
    event_time : tf.Tensor, shape (n,)
        The observed event or censoring times.
    event_indicator : tf.Tensor, shape (n,)
        Binary indicator (1 if event occurred, 0 if censored).
    rate : tf.Tensor, shape (n,)
        The predicted rate (λ) of the exponential distribution.

    Returns
    -------
    tf.Tensor
        A tensor of shape (n,) containing the negative log-likelihood for each observation.
    """
    rate = tf.cast(rate, tf.float32)
    log_rate = tf.math.log(rate + _EPS)
    # Ensure inputs are float32
    event_time = tf.cast(event_time, tf.float32)
    event_indicator = tf.cast(event_indicator, tf.float32)

    # Compute the negative log likelihood per observation
    nll = rate * event_time - event_indicator * log_rate
    return nll


@tf.keras.utils.register_keras_serializable(package="pypsps")
class NegloglikExponential(tf.keras.losses.Loss):
    """Computes the negative log-likelihood of an Exponential survival model with censorship."""

    def __init__(
        self,
        reduction: str = tf.keras.losses.Reduction.AUTO,
        log_rate: bool = False,
        name: str = "negloglik_exponential",
    ):
        """Stores whether y_pred is the rate itself or its log (log_rate)."""
        super().__init__(reduction=reduction, name=name)
        self._log_rate = log_rate

    def get_config(self):
        """Includes `log_rate` so save/load round-trips this constructor arg."""
        config = super().get_config()
        config.update({"log_rate": self._log_rate})
        return config

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Implements the loss function call."""
        event_time = y_true[:, 0]
        event_indicator = y_true[:, 1]

        if self._log_rate:
            y_pred = tf.exp(y_pred)

        # y_pred is the rate; may be [N] or [N, 1] depending on caller
        rate = y_pred[:, 0] if y_pred.shape.rank == 2 else y_pred

        losses = _negloglik_exponential(
            event_time=event_time, event_indicator=event_indicator, rate=rate
        )
        losses = tf.ensure_shape(losses, [None])

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return losses
        if self.reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_sum(losses, axis=-1)
        if self.reduction in (
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            tf.keras.losses.Reduction.AUTO,
        ):
            return tf.reduce_mean(losses, axis=-1)
        raise NotImplementedError(f"reduction='{self.reduction}' is not implemented")


def _negloglik_exponential_scale(
    event_time: tf.Tensor, event_indicator: tf.Tensor, log_scale: tf.Tensor
) -> tf.Tensor:
    """
    Computes the negative log-likelihood for an exponential distribution with censoring.

    This version uses the SCALE (mean) parameterization instead of rate, which is more
    numerically stable for survival models where we directly predict log(mean_survival_time).

    For exponential distribution with scale μ (mean survival time):
      - PDF: f(t) = (1/μ) * exp(-t/μ)
      - Survival: S(t) = exp(-t/μ)
      - Hazard: h(t) = 1/μ (constant)

    For each observation i:
      - If an event occurs (event_indicator[i] == 1):
            log-likelihood = -log(μ) - t/μ = -log_scale - t*exp(-log_scale)
      - If censored (event_indicator[i] == 0):
            log-likelihood = -t/μ = -t*exp(-log_scale)

    Therefore, the negative log-likelihood for observation i is:
      loss_i = t * exp(-log_scale) + event_indicator * log_scale

    Parameters
    ----------
    event_time : tf.Tensor, shape (n,)
        The observed event or censoring times.
    event_indicator : tf.Tensor, shape (n,)
        Binary indicator (1 if event occurred, 0 if censored).
    log_scale : tf.Tensor, shape (n,)
        The predicted log of the scale parameter (log of mean survival time).

    Returns
    -------
    tf.Tensor
        A tensor of shape (n,) containing the negative log-likelihood for each observation.
    """
    log_scale = tf.cast(log_scale, tf.float32)
    event_time = tf.cast(event_time, tf.float32)
    event_indicator = tf.cast(event_indicator, tf.float32)

    # Compute the negative log likelihood per observation
    # NLL = t/μ + δ*log(μ) = t*exp(-log_scale) + δ*log_scale
    nll = event_time * tf.exp(-log_scale) + event_indicator * log_scale
    return nll


@tf.keras.utils.register_keras_serializable(package="pypsps")
class NegloglikExponentialScale(tf.keras.losses.Loss):
    """Computes the negative log-likelihood of an Exponential survival model with censorship.

    This version uses the SCALE (mean) parameterization: the model directly predicts
    log(mean_survival_time) instead of log(hazard_rate). This is more numerically stable
    because:
    1. The gradient flows more naturally when predicting the quantity we care about (mean time)
    2. Small errors in log-space don't get amplified by the inversion (1/rate)
    3. The bounds on log_scale are more intuitive (log of reasonable survival times)

    For exponential distribution:
      - scale = μ = E[T] = mean survival time
      - rate = λ = 1/μ = hazard rate
      - log_scale = log(μ) = -log(λ)
    """

    def __init__(
        self,
        reduction: str = tf.keras.losses.Reduction.AUTO,
        name: str = "negloglik_exponential_scale",
    ):
        """Standard Loss constructor; no extra state beyond reduction/name."""
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Implements the loss function call.

        Args:
            y_true: Tensor of shape [N, 2] with columns [event_time, event_indicator]
            y_pred: Tensor of shape [N, 1] or [N] with log_scale predictions

        Returns:
            Loss tensor
        """
        event_time = y_true[:, 0]
        event_indicator = y_true[:, 1]

        # y_pred is log_scale (log of mean survival time); may be [N] or [N, 1]
        log_scale = y_pred[:, 0] if y_pred.shape.rank == 2 else y_pred

        losses = _negloglik_exponential_scale(
            event_time=event_time, event_indicator=event_indicator, log_scale=log_scale
        )
        losses = tf.ensure_shape(losses, [None])

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return losses
        if self.reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_sum(losses, axis=-1)
        if self.reduction in (
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            tf.keras.losses.Reduction.AUTO,
        ):
            return tf.reduce_mean(losses, axis=-1)
        raise NotImplementedError(f"reduction='{self.reduction}' is not implemented")


def _negloglik_weibull(
    event_time: tf.Tensor,
    event_indicator: tf.Tensor,
    log_scale: tf.Tensor,
    log_shape: tf.Tensor,
) -> tf.Tensor:
    """
    Computes the negative log-likelihood for a Weibull distribution with censoring.

    Uses the (log_scale, log_shape) parameterization for numerical stability.

    For Weibull distribution with scale λ and shape k:
      - PDF: f(t) = (k/λ) * (t/λ)^(k-1) * exp(-(t/λ)^k)
      - Survival: S(t) = exp(-(t/λ)^k)
      - Hazard: h(t) = (k/λ) * (t/λ)^(k-1)

    Log-likelihood for observation i:
      - If event (δ=1): log(k) - k*log(λ) + (k-1)*log(t) - (t/λ)^k
      - If censored (δ=0): -(t/λ)^k

    Therefore, the negative log-likelihood is:
      NLL = (t/λ)^k - δ * [log(k) - k*log(λ) + (k-1)*log(t)]
          = exp(k * (log(t) - log_scale)) - δ * [log_shape - k*log_scale + (k-1)*log(t)]

    Parameters
    ----------
    event_time : tf.Tensor, shape (n,)
        The observed event or censoring times (must be > 0).
    event_indicator : tf.Tensor, shape (n,)
        Binary indicator (1 if event occurred, 0 if censored).
    log_scale : tf.Tensor, shape (n,)
        The predicted log of the scale parameter λ.
    log_shape : tf.Tensor, shape (n,)
        The predicted log of the shape parameter k.

    Returns
    -------
    tf.Tensor
        A tensor of shape (n,) containing the negative log-likelihood for each observation.
    """
    log_scale = tf.cast(log_scale, tf.float32)
    log_shape = tf.cast(log_shape, tf.float32)
    event_time = tf.cast(event_time, tf.float32)
    event_indicator = tf.cast(event_indicator, tf.float32)

    # Ensure event_time > 0 for log
    log_time = tf.math.log(event_time + _EPS)

    # shape k = exp(log_shape)
    k = tf.exp(log_shape)

    # Compute (t/λ)^k = exp(k * (log(t) - log_scale))
    z = tf.exp(k * (log_time - log_scale))

    # Log-likelihood terms for events:
    # log(k) - k*log(λ) + (k-1)*log(t) = log_shape - k*log_scale + (k-1)*log_time
    log_pdf_term = log_shape - k * log_scale + (k - 1.0) * log_time

    # NLL = z - δ * log_pdf_term
    nll = z - event_indicator * log_pdf_term

    return nll


@tf.keras.utils.register_keras_serializable(package="pypsps")
class NegloglikWeibull(tf.keras.losses.Loss):
    """Computes the negative log-likelihood of a Weibull survival model with censorship.

    Uses the (log_scale, log_shape) parameterization where:
      - scale λ = exp(log_scale): characteristic life / scale parameter
      - shape k = exp(log_shape): controls hazard behavior
        - k < 1: decreasing hazard (infant mortality)
        - k = 1: constant hazard (exponential distribution)
        - k > 1: increasing hazard (aging/wear-out)

    The model predicts 2 parameters per observation: [log_scale, log_shape].

    For Weibull distribution:
      - Mean: E[T] = λ * Γ(1 + 1/k)
      - Survival: S(t) = exp(-(t/λ)^k)
    """

    def __init__(
        self,
        reduction: str = tf.keras.losses.Reduction.AUTO,
        name: str = "negloglik_weibull",
    ):
        """Standard Loss constructor; no extra state beyond reduction/name."""
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Implements the loss function call.

        Args:
            y_true: Tensor of shape [N, 2] with columns [event_time, event_indicator]
            y_pred: Tensor of shape [N, 2] with columns [log_scale, log_shape]

        Returns:
            Loss tensor
        """
        event_time = y_true[:, 0]
        event_indicator = y_true[:, 1]

        # y_pred has 2 columns: [log_scale, log_shape]
        log_scale = y_pred[:, 0]
        log_shape = y_pred[:, 1]

        losses = _negloglik_weibull(
            event_time=event_time,
            event_indicator=event_indicator,
            log_scale=log_scale,
            log_shape=log_shape,
        )
        losses = tf.ensure_shape(losses, [None])

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return losses
        if self.reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_sum(losses, axis=-1)
        if self.reduction in (
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            tf.keras.losses.Reduction.AUTO,
        ):
            return tf.reduce_mean(losses, axis=-1)
        raise NotImplementedError(f"reduction='{self.reduction}' is not implemented")
