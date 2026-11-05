"""Module for computing causal estimates (UTE & ATE) based on model predictions.

Inference supports

* binary treatment (i.e., switch between 0 / 1)
* continuous treatment along a grid (compared to baseline treatment)

to obtain counterfactual predictions).

For uncertainty estimates see the bootstrap.py module.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

from .keras.mixture import StateMixture


def _state_means_and_weights(
    mixture: StateMixture, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Each state's outcome mean E[Y | a, x, s_k] and the state weights, both ``[N, K]``."""
    means = np.asarray(mixture.outcome_mean(y_pred, per_state=True))
    weights = np.asarray(mixture.layout.decode_pred(y_pred).state_weights)
    return means, weights


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
    return model.predict([features, treatment], verbose=0)


def predict_ute_binary(model: tf.keras.Model, features: Any) -> Union[pd.Series, np.ndarray]:
    """Predicts unit-level treatment effect for binary treatment.

    The outcome family comes from the model (`StateMixture.from_model`), so this also works
    on a model loaded with ``compile=False``. An outcome family without a mean (a custom loss
    wrapped by the legacy constructors) raises `NotImplementedError`.

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

    mixture = StateMixture.from_model(model)
    outcome_mean_pred0, weights = _state_means_and_weights(mixture, y_pred0)
    outcome_mean_pred1, _ = _state_means_and_weights(mixture, y_pred1)

    utes = outcome_mean_pred1 - outcome_mean_pred0
    weighted_ute = (weights * utes).sum(axis=1)
    if isinstance(features, pd.DataFrame):
        return pd.Series(weighted_ute, index=features.index, name="ute")
    return weighted_ute


def predict_ate_binary(model: tf.keras.Model, features: pd.DataFrame) -> float:
    """Computes average treatment effect as averaging UTE estimates."""
    return predict_ute_binary(model, features).mean()


def predict_ute_continuous(
    model: tf.keras.Model,
    features: Any,
    treatment_grid: List[float],
    baseline_treatment: Optional[float] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    """Predicts unit-level treatment effect for continuous treatment.

    The outcome family comes from the model (`StateMixture.from_model`), so this also works
    on a model loaded with ``compile=False``. An outcome family without a mean (a custom loss
    wrapped by the legacy constructors) raises `NotImplementedError`.

    Args:
      model: a trained pypsps model.
      features: features (X) for the causal model. Often a
        pd.DataFrame/np.ndarray, but can also be non-standard
        data structure as long as the pypsps model can use it as
        input to model.predict([features, ...]).
      treatment_grid: grid of treatment values to get UTE for.  This will be the number of columns
        in the resulting array / dataframe
      baseline_treatment: what's the baseline treatment value.  If None, uses the avg of the treatment_grid.

    Returns:
      A pd.DataFrame (if features is a DataFrame) or a np.ndarray of same number of
      rows as features; number of columns is the number of treatment values on the grid.
    """

    if baseline_treatment is None:
        baseline_treatment = np.mean(np.array(treatment_grid))

    n_samples = features.shape[0]
    base_treatment = np.full((n_samples, 1), baseline_treatment)

    mixture = StateMixture.from_model(model)
    base_preds = predict_counterfactual(model, features, base_treatment)
    mean_base, weights_base = _state_means_and_weights(mixture, base_preds)

    utes = []
    for c in tqdm.tqdm(treatment_grid):
        c_treatment = np.full((n_samples, 1), c)
        c_preds = predict_counterfactual(model, features, c_treatment)
        mean_c, _ = _state_means_and_weights(mixture, c_preds)
        c_ute = mean_c - mean_base
        # Weighted average across states.
        c_ute_weighted = (weights_base * c_ute).sum(axis=1)
        utes.append(c_ute_weighted)

    utes = np.array(utes).transpose()
    if isinstance(features, pd.DataFrame):
        return pd.DataFrame(utes, index=features.index, columns=treatment_grid)
    return utes


def predict_ate_continuous(
    model: tf.keras.Model,
    features: Any,
    treatment_grid: List[float],
    baseline_treatment: Optional[float] = None,
) -> Union[pd.Series, np.ndarray]:
    """Computes the average treatment effect (ATE) for continuous treatment.

    Args:
      model: a trained pypsps model.
      features: features (X) for the causal model. Often a
        pd.DataFrame/np.ndarray, but can also be non-standard
        data structure as long as the pypsps model can use it as
        input to model.predict([features, ...]).
      treatment_grid: grid of treatment values to get UTE for.  This will be the number of columns
        in the resulting array / dataframe
      baseline_treatment: what's the baseline treatment value.  If None, uses the avg of the treatment_grid.

    Returns:
      A pd.Series (if features is a DataFrame) or a np.ndarray of same number of
      rows as the number of values in treatment grid.
    """
    ute_cont = predict_ute_continuous(model, features, treatment_grid, baseline_treatment)
    ate_cont = ute_cont.mean(axis=0)
    if isinstance(ate_cont, pd.Series):
        ate_cont.name = "ate"
    return ate_cont
