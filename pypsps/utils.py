"""Module for general utilities."""

import dataclasses
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


def get_state_column_indices(state_idx: int, n_states: int, n_params_per_state: int) -> List[int]:
    """Returns column indices for `state_idx` in an interleaved per-state prediction block.

    Prediction blocks (outcome params, treatment params) are laid out interleaved across
    states, ie [state_0_param_0, state_1_param_0, ..., state_0_param_1, state_1_param_1, ...].
    This returns the column indices belonging to `state_idx` across all `n_params_per_state`
    parameters.
    """
    return [state_idx + i * n_states for i in range(n_params_per_state)]


def get_n_states(
    y_pred: _Y_PRED_DTYPE, n_outcome_pred_cols: int, n_treatment_pred_cols: int
) -> int:
    """Determines number of states based on `y_pred` tensor.

    y_pred is a concatenation of [outcome params, predictive state weights, treatment params],
    where outcome params and treatment params are both interleaved per-state blocks of width
    (n_states * n_outcome_pred_cols) and (n_states * n_treatment_pred_cols) respectively, and
    predictive state weights are of width n_states. So:

        n_cols = n_states * (n_outcome_pred_cols + 1 + n_treatment_pred_cols)

    Args:
      y_pred: Tensor with all predictions.
      n_outcome_pred_cols: number of outcome prediction params per state.
      n_treatment_pred_cols: number of treatment prediction params per state.

    Returns:
        Number of states.
    """
    n_cols = get_n_cols(y_pred)
    n_states = int(n_cols / (n_outcome_pred_cols + n_treatment_pred_cols + 1))
    return n_states


def split_y_pred(
    y_pred: _Y_PRED_DTYPE,
    n_outcome_pred_cols: int,
    n_treatment_pred_cols: int,
) -> Tuple[_Y_PRED_DTYPE, _Y_PRED_DTYPE, _Y_PRED_DTYPE]:
    """Splits y_pred into a tuple of (outcome preds, predictive state weights, treatment preds).

    Both outcome preds and treatment preds are per-state interleaved blocks (state-conditional
    predictions, not marginalized), of width (n_states * n_outcome_pred_cols) and
    (n_states * n_treatment_pred_cols) respectively.
    """

    n_states = get_n_states(y_pred, n_outcome_pred_cols, n_treatment_pred_cols)

    outcome_params_pred = y_pred[:, : (n_outcome_pred_cols * n_states)]
    weights = y_pred[:, (n_outcome_pred_cols * n_states) : ((n_outcome_pred_cols + 1) * n_states)]
    treatment_pred = y_pred[:, ((n_outcome_pred_cols + 1) * n_states) :]

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


def _weighted_state_average(
    pred_block: _Y_PRED_DTYPE, weights: _Y_PRED_DTYPE, n_params_per_state: int
) -> _Y_PRED_DTYPE:
    """Weighted average of an interleaved per-state prediction block using state weights."""
    pred_list = split_outcome_pred(pred_block, n_outcome_pred_cols=n_params_per_state)

    is_np = isinstance(pred_block, np.ndarray)
    weighted_preds = []
    for param_pred in pred_list:
        if is_np:
            weighted_pred = (weights * param_pred).sum(axis=1)[:, np.newaxis]
        else:
            weighted_pred = tf.reduce_sum(weights * param_pred, axis=1)[:, tf.newaxis]
        weighted_preds.append(weighted_pred)

    if is_np:
        return np.concatenate(weighted_preds, axis=1)
    else:
        return tf.concat(weighted_preds, axis=1)


def agg_outcome_pred(
    y_pred: _Y_PRED_DTYPE, n_outcome_pred_cols: int, n_treatment_pred_cols: int
) -> _Y_PRED_DTYPE:
    """Aggregates state-level outcome predictions to aggregate the outcome prediction.

    Does this by a weighted average of outcome predictions per state, where weight
    of outcome prediction in state j equals the state level weight of the causal
    state simplex predictions.
    """
    outcome_pred, weights, _ = split_y_pred(
        y_pred, n_outcome_pred_cols=n_outcome_pred_cols, n_treatment_pred_cols=n_treatment_pred_cols
    )
    return _weighted_state_average(outcome_pred, weights, n_outcome_pred_cols)


def agg_treatment_pred(
    y_pred: _Y_PRED_DTYPE, n_outcome_pred_cols: int, n_treatment_pred_cols: int
) -> _Y_PRED_DTYPE:
    """Aggregates state-conditional treatment predictions into the marginal P(A | X).

    Does this by a weighted average of treatment predictions per state, where weight
    of the treatment prediction in state j equals the prior state weight P(state_j | X).
    """
    _, weights, treatment_pred = split_y_pred(
        y_pred, n_outcome_pred_cols=n_outcome_pred_cols, n_treatment_pred_cols=n_treatment_pred_cols
    )
    return _weighted_state_average(treatment_pred, weights, n_treatment_pred_cols)


@dataclasses.dataclass
class ColumnLayout:
    """Column layout of a compiled pypsps model's y_pred / y_true tensors.

    These three ints are what `split_y_pred`, `split_y_true`, `agg_outcome_pred`,
    `agg_treatment_pred`, and the `pypsps.keras.metrics` constructors need to know how to slice
    a model's (interleaved, per-state) prediction tensor; see `get_column_layout`.
    """

    n_outcome_pred_cols: int
    n_treatment_pred_cols: int
    n_outcome_true_cols: int


def get_column_layout(model: tf.keras.Model) -> ColumnLayout:
    """Reads a compiled pypsps model's column layout off its loss, instead of hardcoding it.

    A `pypsps.keras.losses.CausalLoss` already stores `n_outcome_pred_cols`,
    `n_treatment_pred_cols`, and `n_outcome_true_cols` on its `_outcome_loss`/`_treatment_loss`
    (they were required constructor args to build the model in the first place). Re-typing
    these ints at every downstream call site is a copy/paste hazard once a model's outcome
    distribution changes shape (e.g. Normal's loc+scale = 2 outcome params/state vs.
    exponential's log_rate = 1) -- exactly the bug this helper is meant to prevent.

    Requires `model.loss` to be a `pypsps.keras.losses.CausalLoss` instance, i.e. the model
    must already be compiled via one of this package's `build_*` functions (or equivalent).
    """
    outcome_loss = model.loss._outcome_loss
    treatment_loss = model.loss._treatment_loss
    return ColumnLayout(
        n_outcome_pred_cols=outcome_loss._n_outcome_pred_cols,
        n_treatment_pred_cols=treatment_loss._n_treatment_pred_cols,
        n_outcome_true_cols=outcome_loss._n_outcome_true_cols,
    )


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


def safe_log(p: tf.Tensor, neg_inf: float = -1e9, name: str | None = None) -> tf.Tensor:
    """Numerically stable log(p) for p in [0, 1], e.g. softmax/simplex outputs.

    Equivalent in spirit to computing log-weights via `log_softmax(logits)` directly
    (no artificial floor on small-but-nonzero values), for cases where only the
    already-softmaxed probabilities are available and not the underlying logits.
    `log(p + eps)` imposes a hard floor at `log(eps)` (e.g. ~ -13.8 for eps=1e-6),
    which distorts sharp/near-degenerate states whose true log-probability is far
    below that floor. This instead only substitutes `neg_inf` for exact zeros
    (float32 softmax underflow), and does so via `tf.where` so no NaN/inf gradient
    flows back through `tf.math.log` at p == 0.
    """
    p = tf.convert_to_tensor(p)
    is_positive = p > 0
    safe_p = tf.where(is_positive, p, tf.ones_like(p))
    return tf.where(
        is_positive,
        tf.math.log(safe_p),
        tf.fill(tf.shape(p), tf.constant(neg_inf, p.dtype)),
        name=name,
    )
