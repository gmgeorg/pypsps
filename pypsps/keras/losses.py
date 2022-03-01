"""Module for pypsps losses."""


from typing import Callable, Optional, Tuple

import warnings

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import math
from pypsps import utils


"""
import tensorflow_probability as tfp
distrs = []

for i in range(3):
    distrs.append(tfp.distributions.Normal(
    np.float32(y_pred_j[i, 0]), np.float32(y_pred_j[i, 1]), validate_args=False, allow_nan_stats=True, name='Normal'
))
    

def negloglik(y, rv_y):
    print(y, rv_y.parameters)
    return -rv_y.log_prob(y)
    
# Compare to tfp loglik
[negloglik(y_true_j[i], d) for i, d in enumerate(distrs)]
"""


def _negloglik(y, mu, sigma):
    negloglik_element = tf.math.log(2.0 * math.pi) / 2.0 + tf.math.log(sigma)
    negloglik_element += 0.5 * tf.square((y - mu) / sigma)
    return negloglik_element


@tf.keras.utils.register_keras_serializable(package="psps")
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
            tf.keras.losses.Reduction.AUTO,
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        ):
            return tf.reduce_mean(losses)
        raise NotImplementedError("reduction='%s' is not implemented", self.reduction)


def negloglik_normal_each(y_true, y_pred):
    """Compute negative log-likelihood for y ~ Normal(mu, sigma^2)."""
    y_pred_mu = y_pred[:, 0]
    y_pred_scale = y_pred[:, 1]

    return _negloglik(y_true, y_pred_mu, y_pred_scale)


def negloglik_normal(y_true, y_pred):
    return tf.reduce_sum(negloglik_normal_each(y_true, y_pred))


@tf.keras.utils.register_keras_serializable(package="psps")
class OutcomeLoss(tf.keras.losses.Loss):
    """Computes outcome loss for a pypsps model with multi-output predictions.

    The outcome loss for pypsps model is a weighted sum of state-level outcome losses,
    where weights are the predictive states embeddings from the propensity model.
    """

    def __init__(self, loss: tf.keras.losses.Loss, **kwargs):
        """Initializes the outcome loss.

        Args:
          outcome_loss: a keras loss function with NONE reduction (ie element-wise).
            This is a requirement to properly computed the weighted loss across states.
        """
        super().__init__(**kwargs)
        assert isinstance(loss, tf.keras.losses.Loss)
        assert loss.reduction == tf.keras.losses.Reduction.NONE
        self._loss = loss

    def call(self, y_true, y_pred):
        """Evaluates Causal Loss on (y_true, y_pred) for binary loss and Normal outcomes.

        y_pred is a combination of
          * mean predictions per state (mu_j | X, T) [ N x J ]
          * scale predictions per state (scale_j | X, T)  [ N x J]
          * propensity score (P(treatment | X) [ N x 1 ]
          * predictive state weights (P(state j | X)  [ N x J ]
        """
        n_states = utils.get_n_states(y_pred)
        outcome_pred, const_scale, _, weights = utils.split_y_pred(y_pred)

        outcome_true = y_true[:, 0]

        weighted_loss = 0.0
        for j in range(n_states):
            weighted_loss += weights[:, j] * self._loss(
                outcome_true,
                tf.stack([outcome_pred[:, j], const_scale[:, j]], axis=1),
            )

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return weighted_loss

        weighted_loss = tf.reduce_sum(weighted_loss)
        if self.reduction in (tf.keras.losses.Reduction.SUM,):
            return weighted_loss

        # Divide by batch sample size; note that sum of all weights = n_samples
        # since weights are softmax per row.
        if self.reduction in (
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            tf.keras.losses.Reduction.AUTO,
        ):
            weighted_loss /= tf.reduce_sum(weights)
            return weighted_loss

        raise NotImplementedError(
            "self.reduction='%s' is not implemented", self.reduction
        )


@tf.keras.utils.register_keras_serializable(package="psps")
class TreatmentLoss(tf.keras.losses.Loss):
    """Implements treatment loss for output of pypsps predictions."""

    def __init__(self, loss: tf.keras.losses.Loss, **kwargs):
        super().__init__(**kwargs)
        self._loss = loss

    def call(self, y_true, y_pred):
        """Evaluates loss on treatment label and predicted treatment of y_pred (propensity score)."""
        return self._loss(y_true[:, 1], utils.split_y_pred(y_pred)[2])


@tf.keras.utils.register_keras_serializable(package="psps")
class CausalLoss(tf.keras.losses.Loss):
    """PSPS causal loss is the sum of outcome loss + treatment loss."""

    def __init__(
        self,
        outcome_loss: OutcomeLoss,
        treatment_loss: TreatmentLoss,
        alpha: float = 1.0,
        outcome_loss_weight: float = 1.0,
        predictive_states_regularizer: Optional[
            tf.keras.regularizers.Regularizer
        ] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert isinstance(outcome_loss, OutcomeLoss)
        assert isinstance(treatment_loss, TreatmentLoss)

        self._outcome_loss = outcome_loss
        self._treatment_loss = treatment_loss
        self._alpha = alpha
        self._outcome_loss_weight = outcome_loss_weight
        self._predictive_states_regularizer = predictive_states_regularizer
        self._update_loss_reduction()

    def _update_loss_reduction(self):
        """Updates loss reduction of outcome & treatment according to causal reduction loss."""
        if self._treatment_loss.reduction != self.reduction:
            warnings.warn(
                "Setting 'reduction' of treatment loss to user-specified reduction: '%s'."
                % self.reduction,
            )
            self._treatment_loss.reduction = self.reduction

        if self._outcome_loss.reduction != self.reduction:
            warnings.warn(
                "Setting 'reduction' of outcome loss to user-specified reduction: '%s'."
                % self.reduction,
            )
            self._outcome_loss.reduction = self.reduction

    def call(self, y_true, y_pred):
        """Computes the causal loss from y_true and multi-output predictions."""
        loss_outcome = self._outcome_loss(y_true, y_pred)
        loss_treatment = self._treatment_loss(y_true, y_pred)

        #  print(loss_treatment, loss_outcome)
        total_loss = (
            self._outcome_loss_weight * loss_outcome + self._alpha * loss_treatment
        )
        if self._predictive_states_regularizer is not None:
            weights = utils.split_y_pred(y_pred)[3]
            total_loss += self._predictive_states_regularizer(weights)

        return total_loss
