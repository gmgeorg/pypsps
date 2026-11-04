"""Frozen copies of `OutcomeLoss.call` and `TreatmentLoss.call` as of v0.3.1.

The reference the state-mixture losses are checked against. Only the legacy per-state loop
(`neglogliks.negloglik_per_state`) and plain slicing are used, so later refactors of the
losses can't change it. Delete together with the three-int adapters (v0.5.0).
"""

import tensorflow as tf

from .. import utils
from ..keras import neglogliks


def _split(y_true, y_pred, n_outcome_true_cols, n_outcome_pred_cols, n_treatment_pred_cols):
    """Today's column split: (outcome params, weights, treatment params, y_out, y_treat, K)."""
    n_states = int(y_pred.shape[1]) // (n_outcome_pred_cols + 1 + n_treatment_pred_cols)
    w0 = n_outcome_pred_cols * n_states
    c = n_outcome_true_cols
    return (
        y_pred[:, :w0],
        y_pred[:, w0 : w0 + n_states],
        y_pred[:, w0 + n_states :],
        y_true[:, :c],
        y_true[:, c : c + 1],
        n_states,
    )


def _reduce(values, reduction, y_true):
    """Today's reduction switch."""
    if reduction == "none":
        return values
    total = tf.reduce_sum(values)
    if reduction == "sum":
        return total
    return total / tf.cast(tf.shape(y_true)[0], y_true.dtype)


def outcome_nll(
    loss,
    treatment_loss,
    y_true,
    y_pred,
    n_outcome_true_cols,
    n_outcome_pred_cols,
    n_treatment_pred_cols,
    likelihood="interventional",
    reduction="none",
):
    """Frozen `OutcomeLoss.call`."""
    params, weights, treat_pred, y_out, y_treat, n_states = _split(
        y_true, y_pred, n_outcome_true_cols, n_outcome_pred_cols, n_treatment_pred_cols
    )
    if likelihood == "interventional":
        gamma = weights
    else:
        nll_a = neglogliks.negloglik_per_state(treatment_loss, y_treat, treat_pred, n_states)
        gamma = neglogliks.posterior_from_negloglik_per_state(weights, nll_a)
    log_weights = utils.safe_log(gamma)
    components = []
    for j in range(n_states):
        cols = [j + i * n_states for i in range(n_outcome_pred_cols)]
        nll_j = loss(y_true=y_out, y_pred=tf.gather(params, cols, axis=1))
        components.append(tf.expand_dims(log_weights[:, j] - nll_j, axis=1))
    marginal = -tf.math.reduce_logsumexp(tf.concat(components, axis=1), axis=1)
    return _reduce(marginal, reduction, y_true)


def treatment_nll(
    loss,
    y_true,
    y_pred,
    n_outcome_true_cols,
    n_outcome_pred_cols,
    n_treatment_pred_cols,
    reduction="none",
):
    """Frozen `TreatmentLoss.call`."""
    _, weights, treat_pred, _, y_treat, n_states = _split(
        y_true, y_pred, n_outcome_true_cols, n_outcome_pred_cols, n_treatment_pred_cols
    )
    nll = neglogliks.negloglik_per_state(loss, y_treat, treat_pred, n_states)
    marginal = -tf.math.reduce_logsumexp(utils.safe_log(weights) - nll, axis=1)
    return _reduce(marginal, reduction, y_true)
