"""Module for pypsps losses."""

import warnings
from typing import Optional

import tensorflow as tf

from .. import utils
from . import neglogliks


def _gen_col_selectors(x: int, n: int, k: int) -> list:
    return utils.get_state_column_indices(x, n, k)


def prob_state_given_treatment_features(
    weights: tf.Tensor,  # [N,K] = P(S|X)
    treatment_true: tf.Tensor,  # [N,1] (or [N])
    treatment_pred: tf.Tensor,  # [N,K*P] interleaved per-state treatment params
    treatment_nll_loss: tf.keras.losses.Loss,
) -> tf.Tensor:
    """Computes gamma = P(S | X, A) using any proper elementwise NLL treatment loss.

    Applies Bayes' rule: gamma_k(a, x) ∝ P(s_k | x) * p(a | s_k, x), where the prior
    P(s_k | x) is `weights` and p(a | s_k, x) is derived from `treatment_nll_loss` evaluated
    on the state-conditional treatment predictions.

    Requirements:
      - treatment_pred has shape [N, K*P] interleaved per-state and matches what
        treatment_nll_loss expects as y_pred (P params per state).
      - treatment_nll_loss.reduction == NONE and returns elementwise NLL.

    Args:
        weights: Prior state probabilities [N,K]
        treatment_true: True treatment values [N,1] or [N]
        treatment_pred: Predicted treatment parameters [N, K*P] in interleaved format
        treatment_nll_loss: NLL loss function with reduction=NONE

    Returns:
      gamma: [N,K]
    """
    weights = tf.convert_to_tensor(weights)
    treatment_true = tf.convert_to_tensor(treatment_true)
    treatment_pred = tf.convert_to_tensor(treatment_pred)

    if not isinstance(treatment_nll_loss, tf.keras.losses.Loss):
        raise TypeError("treatment_nll_loss must be a tf.keras.losses.Loss instance.")
    if treatment_nll_loss.reduction != tf.keras.losses.Reduction.NONE:
        raise ValueError("treatment_nll_loss must have reduction=NONE.")

    # Ensure treatment_true is [N,1]
    if treatment_true.shape.rank == 1:
        treatment_true = treatment_true[:, None]

    n_states = utils.get_n_cols(weights)

    # Per-state NLL: [N,K]
    nll = neglogliks.negloglik_per_state(
        negloglik=treatment_nll_loss,
        y_true=treatment_true,
        y_pred=treatment_pred,
        n_states=n_states,
    )

    gamma = neglogliks.posterior_from_negloglik_per_state(weights, nll)
    return gamma


@tf.keras.utils.register_keras_serializable(package="pypsps")
class OutcomeLoss(tf.keras.losses.Loss):
    """Computes outcome loss for a pypsps model with multi-output predictions.

    The outcome loss for pypsps model is evaluated as the negative marginal log-likelihood
    of a mixture of experts over states

        -log p(y | a, x) = -log sum_k gamma_k(a, x) * p(y | s_k, a, x)

    where gamma_k(a, x) = P(s_k | x, a) is the *posterior* state probability, ie the prior
    predictive state weights P(s_k | x) updated with the evidence from the observed
    treatment `a` via Bayes' rule (see `prob_state_given_treatment_features`). This is what
    makes -log p(a, y | x) telescope exactly into TreatmentLoss (-log p(a | x), using the
    prior) plus OutcomeLoss (-log p(y | a, x), using the posterior).
    """

    def __init__(
        self,
        loss: tf.keras.losses.Loss,
        treatment_loss: tf.keras.losses.Loss,
        n_outcome_true_cols: int,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        **kwargs,
    ):
        """Initializes the outcome loss.

        Args:
          loss: a keras loss function with NONE reduction (ie element-wise). This is a requirement to properly computed the
            mixture loss across states.
          treatment_loss: a keras loss function with NONE reduction, used to convert the prior
            predictive state weights into the posterior P(state | X, treatment) via Bayes' rule.
            Should be the same elementwise NLL used by the corresponding `TreatmentLoss`.
          n_outcome_true_cols: number of outcome columns in y_true. Used to split outcome_true and treatment_true.
          n_outcome_pred_cols: number of outcome prediction params per state in y_pred.
          n_treatment_pred_cols: number of treatment prediction params per state in y_pred.
          **kwargs: additional arguments passed to keras Loss class.
        """
        super().__init__(**kwargs)
        assert isinstance(loss, tf.keras.losses.Loss)
        assert loss.reduction == tf.keras.losses.Reduction.NONE
        assert isinstance(treatment_loss, tf.keras.losses.Loss)
        assert treatment_loss.reduction == tf.keras.losses.Reduction.NONE
        self._loss = loss
        self._treatment_loss = treatment_loss
        self._n_outcome_true_cols = n_outcome_true_cols
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def call(self, y_true, y_pred):
        """Evaluates Causal Loss on (y_true, y_pred) for binary loss and Normal outcomes.

        y_pred is a combination of
          * outcome parameter predictions per state (params_j | X, T) [ N x J ]
          * predictive state weights (P(state j | X) [ N x J ]
          * state-conditional treatment predictions (params_j | X) [ N x J ]
        """
        n_states = utils.get_n_states(
            y_pred, self._n_outcome_pred_cols, self._n_treatment_pred_cols
        )
        outcome_params_pred, weights, treatment_pred = utils.split_y_pred(
            y_pred, self._n_outcome_pred_cols, self._n_treatment_pred_cols
        )

        outcome_true, treatment_true = utils.split_y_true(y_true, self._n_outcome_true_cols)

        # Posterior state weights: gamma_k(a, x) = P(s_k | x, a), replacing the prior weights
        # so that the outcome mixture conditions on the observed treatment.
        gamma = prob_state_given_treatment_features(
            weights=weights,
            treatment_true=treatment_true,
            treatment_pred=treatment_pred,
            treatment_nll_loss=self._treatment_loss,
        )

        log_weights = utils.safe_log(gamma)
        log_components = []

        for j in range(n_states):
            cols_to_select = _gen_col_selectors(j, n_states, self._n_outcome_pred_cols)
            outcome_pred_state_j = tf.gather(outcome_params_pred, cols_to_select, axis=1)

            # self._loss returns state-specific NLL: -log p(Y | A, X, s_j)
            nll_state_j = self._loss(
                y_true=outcome_true,
                y_pred=outcome_pred_state_j,
            )

            # Convert NLL to log probability and add log weight
            log_p_j = -nll_state_j
            log_joint_j = log_weights[:, j] + log_p_j
            log_components.append(tf.expand_dims(log_joint_j, axis=1))

        # Stack log components -> shape: (batch_size, n_states)
        stacked_log_mix = tf.concat(log_components, axis=1)

        # Compute exact negative marginal log-likelihood per sample
        marginal_nll = -tf.math.reduce_logsumexp(stacked_log_mix, axis=1)

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return marginal_nll

        weighted_loss_sum = tf.reduce_sum(marginal_nll)
        if self.reduction in (tf.keras.losses.Reduction.SUM,):
            return weighted_loss_sum

        # Divide by batch sample size; note that sum of all weights = n_samples
        # since weights are softmax per row.
        if self.reduction in (
            tf.keras.losses.Reduction.AUTO,
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        ):
            weighted_loss_avg = weighted_loss_sum / tf.cast(tf.shape(y_true)[0], y_true.dtype)
            return weighted_loss_avg

        raise NotImplementedError("self.reduction='%s' is not implemented", self.reduction)


@tf.keras.utils.register_keras_serializable(package="pypsps")
class TreatmentLoss(tf.keras.losses.Loss):
    """Implements treatment (dose) loss for output of pypsps predictions.

    The treatment predictions in y_pred are state-conditional (the "dose head" is itself a
    mixture over predictive states), so the treatment loss is evaluated as the negative
    marginal log-likelihood of a mixture of experts over states, using the *prior* predictive
    state weights P(s_k | x):

        -log p(a | x) = -log sum_k P(s_k | x) * p(a | s_k, x)

    This is the term that telescopes with OutcomeLoss's posterior-weighted mixture into the
    exact joint -log p(a, y | x).
    """

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
          n_outcome_pred_cols: number of outcome prediction params per state in y_pred.
          n_treatment_pred_cols: number of treatment prediction params per state in y_pred.
        """

        super().__init__(**kwargs)
        assert isinstance(loss, tf.keras.losses.Loss)
        assert loss.reduction == tf.keras.losses.Reduction.NONE
        self._loss = loss
        self._n_outcome_true_cols = n_outcome_true_cols
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def call(self, y_true, y_pred):
        """Evaluates the marginal treatment (dose) loss -log p(a | x)."""
        n_states = utils.get_n_states(
            y_pred, self._n_outcome_pred_cols, self._n_treatment_pred_cols
        )
        _, weights, treatment_pred = utils.split_y_pred(
            y_pred=y_pred,
            n_outcome_pred_cols=self._n_outcome_pred_cols,
            n_treatment_pred_cols=self._n_treatment_pred_cols,
        )
        _, treatment_true = utils.split_y_true(
            y_true, n_outcome_true_cols=self._n_outcome_true_cols
        )

        # Per-state NLL: -log p(a | s_k, x), shape [N, K]
        nll_per_state = neglogliks.negloglik_per_state(
            negloglik=self._loss,
            y_true=treatment_true,
            y_pred=treatment_pred,
            n_states=n_states,
        )

        log_weights = utils.safe_log(weights)
        marginal_nll = -tf.math.reduce_logsumexp(log_weights - nll_per_state, axis=1)

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return marginal_nll

        loss_sum = tf.reduce_sum(marginal_nll)
        if self.reduction in (tf.keras.losses.Reduction.SUM,):
            return loss_sum

        if self.reduction in (
            tf.keras.losses.Reduction.AUTO,
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        ):
            return loss_sum / tf.cast(tf.shape(y_true)[0], y_true.dtype)

        raise NotImplementedError("self.reduction='%s' is not implemented", self.reduction)


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

    Note on alpha=0: this zeroes only the explicit `loglik(T; X)` term (`treatment_loss`).
    It is "dose-likelihood-off", not "dose-head-off": `outcome_loss` mixes per-state outcome
    predictions with the posterior gamma = P(state | X, T) (see `OutcomeLoss`,
    `prob_state_given_treatment_features`), which is a differentiable function of the
    treatment/dose head's own predictions. So even at alpha=0, gradients from the outcome loss
    still reach the dose head through gamma -- setting alpha=0 does not isolate the outcome model
    from the dose head. Do not describe alpha=0 as an "outcome-only" ablation.
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
        # A tf.Variable, not a plain float: `call()` is traced into model.fit()'s compiled
        # training function (run_eagerly=False by default). A plain float gets baked into that
        # graph as a constant at the first trace; a callback rebinding `loss._alpha` afterward
        # (e.g. AlphaScheduleCallback) would then be updating an eager Python attribute that the
        # traced graph never reads again. A tf.Variable's value is re-read via ReadVariableOp on
        # every graph execution, so `loss._alpha.assign(...)` between epochs actually reaches the
        # optimizer. Callers must use `.assign(...)`, not `=`, to update it after construction.
        self._alpha = tf.Variable(float(alpha), trainable=False, dtype=tf.float32, name="alpha")
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
