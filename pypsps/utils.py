"""Module for general utilities."""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

_Y_PRED_DTYPE = Union[np.ndarray, tf.Tensor]
_DATA_DTYPE = Union[np.ndarray, pd.DataFrame]


def get_n_cols(y: _Y_PRED_DTYPE) -> int:
    """Gets the number of columns of a np array or TF tensor."""
    if isinstance(y, np.ndarray):
        n_cols = y.shape[1]
    else:
        n_cols = y.get_shape().as_list()[1]
    return n_cols


def get_n_states(
    y_pred: _Y_PRED_DTYPE, n_outcome_pred_cols: int, n_treatment_pred_cols: int
) -> int:
    """Determines number of states based on `y_pred` tensor.

    Number of states is equal to (n_outcome_preds * n_states + n_states + n_treatment_pred_cols) = n_cols.

    Args:
      y_pred: Tensor with all predictions.
      treatment_pred_cols: number of prediction cols to represent the treatment. Defaults to 1 for a
        propensity score model (Bernoulli probability).

    Returns:
        Number of states.
    """
    n_cols = get_n_cols(y_pred)

    # y_pred is a concatenation of [means predictions, scale preds, predictive state weights, propensity score]
    # which are of dimension [n_states * outcome pred cols, <treament prediction cols>] --> n_states = (n_cols - <treatment pred cols>) / 3
    n_states = int((n_cols - n_treatment_pred_cols) / (n_outcome_pred_cols + 1))
    return n_states


def split_y_pred(
    y_pred: _Y_PRED_DTYPE,
    n_outcome_pred_cols: int,
    n_treatment_pred_cols: int,
) -> Tuple[_Y_PRED_DTYPE, _Y_PRED_DTYPE, _Y_PRED_DTYPE, _Y_PRED_DTYPE]:
    """Splits y_pred into a tuple of (outcome preds, predictive state weights, treatment preds)."""

    n_states = get_n_states(y_pred, n_outcome_pred_cols, n_treatment_pred_cols)

    outcome_params_pred = y_pred[:, : (n_outcome_pred_cols * n_states)]
    weights = y_pred[:, (n_outcome_pred_cols * n_states) : ((n_outcome_pred_cols + 1) * n_states)]
    treatment_pred = y_pred[:, -n_treatment_pred_cols:]

    return outcome_params_pred, weights, treatment_pred


def split_outcome_pred(
    outcome_pred: _Y_PRED_DTYPE, n_outcome_pred_cols: int
) -> List[_Y_PRED_DTYPE]:
    """Splits the outcome parameter predictions per state into separate params per state tensors."""
    if isinstance(outcome_pred, np.ndarray):
        return np.split(outcome_pred, n_outcome_pred_cols, axis=1)
    else:
        return tf.split(outcome_pred, n_outcome_pred_cols, axis=1)


def split_y_true(
    y_true: _Y_PRED_DTYPE, n_outcome_true_cols: int
) -> Tuple[_Y_PRED_DTYPE, _Y_PRED_DTYPE]:
    """Splits y_true = (outcome, treatment) into separate tensors."""
    outcome_true = y_true[:, :n_outcome_true_cols]
    treatment_true = y_true[:, n_outcome_true_cols:]
    return outcome_true, treatment_true


def agg_outcome_pred(
    y_pred: _Y_PRED_DTYPE, n_outcome_pred_cols: int, n_treatment_pred_cols: int
) -> np.ndarray:
    """Aggregates state-level outcome predictions to aggregate the outcome prediction.

    Does this by a weighted average of outcome predictions per state, where weight
    of outcome prediction in state j equals the state level weight of the causal
    state simplex predictions.
    """
    outcome_pred, weights, _ = split_y_pred(
        y_pred, n_outcome_pred_cols=n_outcome_pred_cols, n_treatment_pred_cols=n_treatment_pred_cols
    )

    outcome_pred_list = split_outcome_pred(outcome_pred, n_outcome_pred_cols=n_outcome_pred_cols)

    is_np = isinstance(outcome_pred, np.ndarray)
    weighted_outcomes = []
    for param_pred in outcome_pred_list:
        if isinstance(weights, np.ndarray):
            weighted_outcome = (weights * param_pred).sum(axis=1)[:, np.newaxis]
        else:
            weighted_outcome = tf.reduce_sum(weights * param_pred, axis=1)[:, tf.newaxis]
        weighted_outcomes.append(weighted_outcome)

    if is_np:
        return np.concatenate(weighted_outcomes, axis=1)
    else:
        return tf.concat(weighted_outcomes, axis=1)


def prepare_keras_inputs_outputs(
    features: _DATA_DTYPE, treatments: _DATA_DTYPE, outcomes: _DATA_DTYPE
) -> Tuple[Tuple[np.ndarray], np.ndarray]:
    """Prepares inputs/outputs for the keras model training and prediction interface."""
    if isinstance(features, pd.DataFrame):
        features = features.values
    if isinstance(treatments, pd.DataFrame):
        treatments = treatments.values
    if outcomes is not None:
        if isinstance(outcomes, pd.DataFrame):
            outcomes = outcomes.values

    input_data = [features.astype("float32"), treatments]
    if outcomes is None:
        output_data = None
    else:
        output_data = np.hstack([outcomes.astype("float32"), treatments])

    return (
        input_data,
        output_data,
    )
