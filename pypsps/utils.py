"""Module for general utilities."""

from typing import Tuple
import numpy as np


def get_n_states(y_pred) -> int:
    """Determines number of states based on `y_pred` tensor."""
    if isinstance(y_pred, np.ndarray):
        n_cols = y_pred.shape[1]
    else:
        n_cols = y_pred.get_shape().as_list()[1]

    # y_pred is a concatenation of [means predictions, scale preds, propensity score, predictive state weights]
    # which are of dimension [n_states, n_states, 1, n_states] --> n_states = (n_cols - 1) / 3
    n_states = int((n_cols - 1) / 3)
    return n_states


def split_y_pred(y_pred) -> Tuple:
    """Splits y_pred into a tuple of (means, scale, propensity score, predictive state weights)."""

    n_states = get_n_states(y_pred)
    outcome_pred = y_pred[:, :n_states]
    weights = y_pred[:, -n_states:]
    const_scale = y_pred[:, n_states : (2 * n_states)]
    prop_score = y_pred[:, (2 * n_states + 1) : (2 * n_states + 2)]

    return outcome_pred, const_scale, prop_score, weights
