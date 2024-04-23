"""Module for general utilities."""

from typing import Tuple, Union
import numpy as np
import tensorflow as tf

_Y_PRED_DTYPE = Union[np.ndarray, tf.Tensor]


def get_n_states(y_pred: _Y_PRED_DTYPE) -> int:
    """Determines number of states based on `y_pred` tensor."""
    if isinstance(y_pred, np.ndarray):
        n_cols = y_pred.shape[1]
    else:
        n_cols = y_pred.get_shape().as_list()[1]

    # y_pred is a concatenation of [means predictions, scale preds, propensity score, predictive state weights]
    # which are of dimension [n_states, n_states, 1, n_states] --> n_states = (n_cols - 1) / 3
    n_states = int((n_cols - 1) / 3)
    return n_states


def split_y_pred(y_pred: _Y_PRED_DTYPE) -> Tuple:
    """Splits y_pred into a tuple of (propensity score, means, scale, predictive state weights)."""

    n_states = get_n_states(y_pred)
    outcome_pred = y_pred[:, :n_states]
    weights = y_pred[:, -(n_states + 1) : -1]
    scale_pred = y_pred[:, (n_states) : (2 * n_states)]
    prop_score = y_pred[:, -1:]
    return outcome_pred, scale_pred, weights, prop_score


def agg_outcome_pred(y_pred: _Y_PRED_DTYPE) -> np.ndarray:
    """Aggregates state-level outcome predictions to aggregate the outcome prediction.

    Does this by a weighted average of outcome predictions per state, where weight
    of outcome prediction in state j equals the state level weight of the causal
    state simplex predictions.
    """
    _, outcome_pred, _, weights = split_y_pred(y_pred)

    weighted_outcome = (weights * outcome_pred).sum(axis=1)[:, np.newaxis]
    return weighted_outcome
