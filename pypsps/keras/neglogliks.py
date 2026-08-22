"""Module to implement distributions and log-likelihood loss fcts."""

import math
from typing import Callable, Union

import tensorflow as tf
import tensorflow_probability as tfp

import pypsps.utils

tfd = tfp.distributions


_EPS = 1e-6

LossLike = Union[tf.keras.losses.Loss, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]


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


class NegloglikExponential(tf.keras.losses.Loss):
    """Computes the negative log-likelihood of an Exponential survival model with censorship."""

    def __init__(
        self,
        reduction=tf.keras.losses.Reduction.AUTO,
        log_rate: bool = False,
        name="negloglik_exponential",
    ):
        super().__init__(reduction=reduction, name=name)
        self._log_rate = log_rate

    def call(self, y_true, y_pred):
        """Implements the loss function call."""
        event_time = y_true[:, 0]
        event_indicator = y_true[:, 1]

        if self._log_rate:
            y_pred = tf.exp(y_pred)

        # y_pred is the rate
        losses = _negloglik_exponential(
            tf.squeeze(event_time), tf.squeeze(event_indicator), rate=tf.squeeze(y_pred)
        )

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


def negloglik_per_state(
    negloglik: LossLike,
    y_true: tf.Tensor,  # [N, C]
    y_pred: tf.Tensor,  # [N, K * P] (Interleaved parameters)
    n_states: int,  # Pass as Python int to ensure loop unrolling
) -> tf.Tensor:
    """
    Returns -log p(y | state) as [N, K] using an elementwise NLL,
    avoiding tf.reshape to prevent Rank/Shape inference errors.

    Args:
        negloglik: Loss function with reduction=NONE
        y_true: True labels [N, C]
        y_pred: Predicted parameters [N, K * P] in interleaved format
        n_states: Number of states (K)

    Returns:
        negative loglikelihood per row per state [N, K]
    """
    y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    if y_true.shape.rank == 1:
        y_true = y_true[:, None]

    # Check NLL reduction if it's a Keras Loss
    if hasattr(negloglik, "reduction") and negloglik.reduction != tf.keras.losses.Reduction.NONE:
        raise ValueError("negloglik Loss must have reduction=NONE.")

    n_cols = pypsps.utils.get_n_cols(y_pred)
    n_params_per_state = int(n_cols / n_states)

    nll_list = []

    # Loop over states: Each iteration creates a concrete slice of the graph
    for j in range(n_states):
        # 1. Identify which columns belong to state j
        # Assuming: [state0_p0, state1_p0, ..., stateK_p0, state0_p1, ...]
        indices = pypsps.utils.get_state_column_indices(j, n_states, n_params_per_state)
        # 2. Extract parameters for state j: [N, P]
        params_state_j = tf.gather(y_pred, indices, axis=1)
        # 3. Compute NLL for this state: [N]
        if isinstance(negloglik, tf.keras.losses.Loss):
            nll_j = negloglik(y_true=y_true, y_pred=params_state_j)
        else:
            nll_j = negloglik(y_true, params_state_j)

        nll_list.append(nll_j)

    # Stack results back into [N, K]
    nll_per_state = tf.stack(nll_list, axis=1, name="negloglik_per_state")

    return nll_per_state


def posterior_from_negloglik_per_state(
    weights: tf.Tensor,  # [N,K] = P(S | X)
    nll: tf.Tensor,  # [N,K] = negative log-likelihood of A given S_k
) -> tf.Tensor:
    """Compute posterior gamma = P(S | X, T) = softmax(log P(S|X) + log p(T|S,·)).

    Computes posterior responsibilities:
        gamma = P(S | X, A)

    using Bayes' rule in log-space:
        gamma_k ∝ P(S_k | X) * exp(-nll_k)

    Second equality follows because T is independent of X given S.

    Args:
        weights:
            Prior state probabilities P(S | X), shape [N,K].
        nll:
            Per-state negative log-likelihoods, shape [N,K].

    Returns:
        gamma:
            Posterior state probabilities P(S | X, A), shape [N,K].
    """
    weights = tf.convert_to_tensor(weights)
    nll = tf.convert_to_tensor(nll)

    log_w = pypsps.utils.safe_log(weights, name="log_weights")  # [N,K]
    gamma = tf.nn.softmax(log_w - nll, axis=1)  # posterior weights
    return gamma
