"""Module for computing causal estimates (UTE & ATE) based on model predictions.

Inference for now only supports binary treatment (i.e., switch between 0 / 1 to obtain
counterfactual predictions).
"""

from typing import Any, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from . import utils


def predict_counterfactual(
    model: tf.keras.Model, features: Any, treatment: Any
) -> Union[pd.Series, np.ndarray]:
    """Predicts counterfactual estimates given features and model.

    Counterfactual predictions can be obtained by setting treatment values
    to user-specified values (rather than using observed data), i.e.,

        Pr(Y | X, do(T))

    E.g., for binary treatment, the pypsps model can be used to switch between
    factual (observed) and counter-factual (unobserved) predictions.  This usually
    gets accomplished by setting treatment == 0 (or == 1) for all observations.

    Args:
      model: a trained pypsps model.
      features: features (X) for the causal model. Often a
        pd.DataFrame/np.ndarray, but can also be non-standard
        data structure as long as the pypsps model can use it as
        input to model.predict([features, ...]).
      treatment: user-specified values for the treatment variables.

    Returns:
      Model predictions for Pr(Y | X, do(T))
    """
    return model.predict([features, treatment])


def predict_ute(model: tf.keras.Model, features: Any) -> Union[pd.Series, np.ndarray]:
    """Predicts unit-level treatment effect for binary treatment.

    Args:
      model: a trained pypsps model.
      features: features (X) for the causal model. Often a
        pd.DataFrame/np.ndarray, but can also be non-standard
        data structure as long as the pypsps model can use it as
        input to model.predict([features, ...]).

    Returns:
      A pd.Series (if features is a DataFrame) or a np.ndarray of same number of
      rows as features.
    """
    y_pred0 = predict_counterfactual(model, features, np.zeros(shape=features.shape[0]))
    y_pred1 = predict_counterfactual(model, features, np.ones(shape=features.shape[0]))

    outcome_pred0, _, weights, _ = utils.split_y_pred(y_pred0)
    outcome_pred1 = utils.split_y_pred(y_pred1)[0]

    utes = outcome_pred1 - outcome_pred0
    weighted_ute = (weights * utes).sum(axis=1)
    if isinstance(features, pd.DataFrame):
        return pd.Series(weighted_ute, index=features.index, name="ute")
    return weighted_ute


def predict_ate(model: tf.keras.Model, features: pd.DataFrame) -> float:
    """Computes average treatment effect as averaging UTE estimates."""
    return predict_ute(model, features).mean()
