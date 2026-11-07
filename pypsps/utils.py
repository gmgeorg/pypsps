"""Module for general utilities.

This module imports neither TensorFlow nor PyTorch.
"""

from typing import Any, Union

import numpy as np
import pandas as pd

_Y_PRED_DTYPE = Any  # a np.ndarray or a TensorFlow / PyTorch tensor
_DATA_DTYPE = Union[np.ndarray, pd.DataFrame]


def get_n_cols(y: _Y_PRED_DTYPE) -> int:
    """Gets the number of columns of a np array or a tensor."""
    n_cols = y.shape[1]
    return n_cols


def prepare_keras_inputs_outputs(
    features: _DATA_DTYPE, treatments: _DATA_DTYPE, outcomes: _DATA_DTYPE
) -> tuple[tuple[np.ndarray], np.ndarray]:
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


def __getattr__(name: str) -> Any:
    """`safe_log` moved to `pypsps.keras.utils`; it is loaded lazily so this module stays TF-free."""
    if name == "safe_log":
        from .keras import utils

        return utils.safe_log
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
