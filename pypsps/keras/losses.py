"""Module for pypsps losses."""

import warnings
from typing import Optional

import tensorflow as tf

from .. import utils


def _gen_col_selectors(x: int, n: int, k: int) -> list:
    return [x + i * n for i in range(k)]


@tf.keras.utils.register_keras_serializable(package="pypsps")
class OutcomeLoss(tf.keras.losses.Loss):
    """Computes outcome loss for a pypsps model with multi-output predictions.

    The outcome loss for pypsps model is a weighted sum of state-level outcome losses,
    where weights are the predictive states embeddings from the propensity model.
    """

    def __init__(
        self,
        loss: tf.keras.losses.Loss,
        n_outcome_true_cols: int,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        **kwargs,
    ):
        """Initializes the outcome loss.

        Args:
          loss: a keras loss function with NONE reduction (ie element-wise).  This is a requirement to properly computed the
            weighted loss across states.
          n_outcome_true_cols: number of outcome columns in y_true.  Used to split outcome_true and treatment_true.
          n_outcome_pred_cols: number of outcome columns in y_pred.
          n_treatment_pred_cols: number of treatment columns in y_pred.
          **kwargs: additional arguments passed to keras Loss class.
        """
        super().__init__(**kwargs)
        assert isinstance(loss, tf.keras.losses.Loss)
        assert loss.reduction == tf.keras.losses.Reduction.NONE
        self._loss = loss
        self._n_outcome_true_cols = n_outcome_true_cols
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def call(self, y_true, y_pred):
        """Evaluates Causal Loss on (y_true, y_pred) for binary loss and Normal outcomes.

        y_pred is a combination of
          * outcome parameter predictions per state (params_j | X, T) [ N x J ]
          * predictive state weights (P(state j | X)  [ N x J ]
          * propensity score (P(treatment | X) [ N x 1 ]
        """
        n_states = utils.get_n_states(
            y_pred, self._n_outcome_pred_cols, self._n_treatment_pred_cols
        )
        outcome_params_pred, weights, _ = utils.split_y_pred(
            y_pred, self._n_outcome_pred_cols, self._n_treatment_pred_cols
        )

        outcome_true = utils.split_y_true(y_true, self._n_outcome_true_cols)[0]
        weighted_loss = 0.0
        for j in range(n_states):
            cols_to_select = _gen_col_selectors(j, n_states, self._n_outcome_pred_cols)
            outcome_pred_state_j = tf.gather(outcome_params_pred, cols_to_select, axis=1)
            loss_state_j = self._loss(
                y_true=outcome_true,
                y_pred=outcome_pred_state_j,
            )
            weighted_loss += weights[:, j] * loss_state_j

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return weighted_loss

        weighted_loss_sum = tf.reduce_sum(weighted_loss)
        if self.reduction in (tf.keras.losses.Reduction.SUM,):
            return weighted_loss_sum

        # Divide by batch sample size; note that sum of all weights = n_samples
        # since weights are softmax per row.
        if self.reduction in (
            tf.keras.losses.Reduction.AUTO,
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        ):
            weighted_loss_avg = weighted_loss_sum / tf.reduce_sum(weights)
            return weighted_loss_avg

        raise NotImplementedError("self.reduction='%s' is not implemented", self.reduction)


@tf.keras.utils.register_keras_serializable(package="pypsps")
class TreatmentLoss(tf.keras.losses.Loss):
    """Implements treatment loss for output of pypsps predictions."""

    def __init__(
        self,
        loss: tf.keras.losses.Loss,
        n_outcome_true_cols: int,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        **kwargs,
    ):
        """Initializes class.

        Args:
          loss: a keras loss function with NONE reduction (ie element-wise).
          n_outcome_true_cols: number of outcome columns in y_true.  Used to split outcome_true and treatment_true.
          n_outcome_pred_cols: number of outcome columns in y_pred.
          n_treatment_pred_cols: number of treatment columns in y_pred.
        """

        super().__init__(**kwargs)
        self._loss = loss
        self._n_outcome_true_cols = n_outcome_true_cols
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def call(self, y_true, y_pred):
        """Evaluates loss on treatment label and predicted treatment of y_pred (propensity score)."""
        _, treat_true = utils.split_y_true(y_true, n_outcome_true_cols=self._n_outcome_true_cols)
        _, _, treat_preds = utils.split_y_pred(
            y_pred, self._n_outcome_pred_cols, self._n_treatment_pred_cols
        )
        loss = self._loss(
            y_true=treat_true,
            y_pred=treat_preds,
        )
        return loss


@tf.keras.utils.register_keras_serializable(package="pypsps")
class CausalLoss(tf.keras.losses.Loss):
    """PSPS causal loss is the sum of outcome loss + treatment loss.

    Causal loss from PSPS is based on the joint distribution P(outcome, treatment | features)
    which decomposes into

        Pr(Y, T | X) = Pr(Y | T, X) * Pr(T | X)

    which in log-likelihood terms is

        loglik(Y, T; X) = loglik(Y; T, X) + alpha * loglik(T; X)

    where alpha = 1 (by default). See Eq (10) in
    https://proceedings.mlr.press/v177/kelly22a/kelly22a.pdf
    for details (in paper lambda == alpha).
    """

    def __init__(
        self,
        outcome_loss: OutcomeLoss,
        treatment_loss: TreatmentLoss,
        alpha: float = 1.0,
        outcome_loss_weight: float = 1.0,
        predictive_states_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        **kwargs,
    ):
        """Initializes the causal loss class.

        Args:
            outcome_loss: instance of an outcome loss; defaults to a Normal log-likelihood.
            treatment_loss: instance of a treatment loss; defaults to binary treatment loss
              (ie binary cross entropy).
            alpha: penalty parameter for the treatment loss. Defaults to 1.0 so
              that total causal loss equals the joint log-likelihood.
            outcome_loss_weight: weight of outcome loss; defaults to 1.0.
            predictive_states_regularizer: optional; user can define a predictive
              state regularizer.
        """
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

        total_loss = self._outcome_loss_weight * loss_outcome + self._alpha * loss_treatment
        if self._predictive_states_regularizer is not None:
            weights = utils.split_y_pred(
                y_pred=y_pred,
                n_outcome_pred_cols=self._outcome_loss._n_outcome_pred_cols,
                n_treatment_pred_cols=self._outcome_loss._n_treatment_pred_cols,
            )[1]
            total_loss += self._predictive_states_regularizer(weights)

        return total_loss
