"""Module for computing causal estimates (UTE & ATE) based on model predictions.

Inference works on any pypsps model: a Keras model from `pypsps.keras.models` or a PyTorch
model from `pypsps.torch.models`. It needs neither framework installed beyond the model's own.

Inference supports

* binary treatment (i.e., switch between 0 / 1)
* continuous treatment along a grid (compared to baseline treatment)

to obtain counterfactual predictions).

For uncertainty estimates see the bootstrap.py module.
"""

from typing import Any

import numpy as np
import pandas as pd
import tqdm

from . import _backend


def predict_counterfactual(model: Any, features: Any, treatment: Any) -> pd.Series | np.ndarray:
    """Predicts counterfactual estimates given features and model.

    Counterfactual predictions can be obtained by setting treatment values
    to user-specified values (rather than using observed data), i.e.,

        Pr(Y | X, do(T))

    E.g., for binary treatment, the pypsps model can be used to switch between
    factual (observed) and counter-factual (unobserved) predictions.  This usually
    gets accomplished by setting treatment == 0 (or == 1) for all observations.

    Args:
      model: a trained pypsps model (Keras or PyTorch).
      features: features (X) for the causal model. Often a
        pd.DataFrame/np.ndarray, but can also be non-standard
        data structure as long as the pypsps model can use it as
        input to model.predict([features, ...]).
      treatment: user-specified values for the treatment variables.

    Returns:
      Model predictions for Pr(Y | X, do(T))
    """
    return _backend.predict(model, features, treatment)


def predict_ute_binary(model: Any, features: Any) -> pd.Series | np.ndarray:
    """Predicts unit-level treatment effect for binary treatment.

    The outcome family comes from the model (`StateMixture.from_model`), so this also works
    on a model loaded with ``compile=False``. An outcome family without a mean (a
    `families.LossFamily`) raises `NotImplementedError`.

    Args:
      model: a trained pypsps model (Keras or PyTorch).
      features: features (X) for the causal model. Often a
        pd.DataFrame/np.ndarray, but can also be non-standard
        data structure as long as the pypsps model can use it as
        input to model.predict([features, ...]).

    Returns:
      A pd.Series (if features is a DataFrame) or a np.ndarray of same number of
      rows as features.
    """
    outcome_mean_pred0, weights = _backend.state_means_and_weights(
        model, features, np.zeros(shape=features.shape[0])
    )
    outcome_mean_pred1, _ = _backend.state_means_and_weights(
        model, features, np.ones(shape=features.shape[0])
    )

    utes = outcome_mean_pred1 - outcome_mean_pred0
    weighted_ute = (weights * utes).sum(axis=1)
    if isinstance(features, pd.DataFrame):
        return pd.Series(weighted_ute, index=features.index, name="ute")
    return weighted_ute


def predict_ate_binary(model: Any, features: pd.DataFrame) -> float:
    """Computes average treatment effect as averaging UTE estimates."""
    return predict_ute_binary(model, features).mean()


def predict_ute_continuous(
    model: Any,
    features: Any,
    treatment_grid: list[float],
    baseline_treatment: float | None = None,
    weights: str = "prior",
) -> pd.DataFrame | np.ndarray:
    """Predicts unit-level treatment effect for continuous treatment.

    The outcome family comes from the model (`StateMixture.from_model`), so this also works
    on a model loaded with ``compile=False``. An outcome family without a mean (a
    `families.LossFamily`) raises `NotImplementedError`.

    Args:
      model: a trained pypsps model (Keras or PyTorch).
      features: features (X) for the causal model. Often a
        pd.DataFrame/np.ndarray, but can also be non-standard
        data structure as long as the pypsps model can use it as
        input to model.predict([features, ...]).
      treatment_grid: grid of treatment values to get UTE for.  This will be the number of columns
        in the resulting array / dataframe
      baseline_treatment: what's the baseline treatment value.  If None, uses the avg of the treatment_grid.
      weights: which state weights mix the per-state outcome means.

        * ``"prior"`` (default): P(S | X) = P(S | X, do(A)), which does not depend on the
          treatment. The unit effect is ``sum_k P(s_k | x) (m_k(a, x) - m_k(a_0, x))`` with
          ``m_k = E[Y | a, x, s_k]``. These are the weights the ``"interventional"`` outcome
          likelihood mixes with in training.
        * ``"posterior"``: P(S | X, A = a), by Bayes' rule from the per-state treatment density.
          The effect is ``sum_k P(s_k | x, a) m_k(a, x) - sum_k P(s_k | x, a_0) m_k(a_0, x)``,
          the difference of the model's conditional means E[Y | a, x]. These are the weights the
          ``"observational"`` outcome likelihood mixes with in training. Averaged over units it
          is the g-formula for E[Y | a, X], which is the do(a) effect only if X alone removes
          the confounding.

    Returns:
      A pd.DataFrame (if features is a DataFrame) or a np.ndarray of same number of
      rows as features; number of columns is the number of treatment values on the grid.

    Raises:
      ValueError: if ``weights`` is not ``"prior"`` or ``"posterior"``.
    """
    _backend.check_weights(weights)

    if baseline_treatment is None:
        baseline_treatment = np.mean(np.array(treatment_grid))

    n_samples = features.shape[0]
    base_treatment = np.full((n_samples, 1), baseline_treatment)

    mean_base, weights_base = _backend.state_means_and_weights(
        model, features, base_treatment, weights
    )
    base_mixed = (weights_base * mean_base).sum(axis=1)

    utes = []
    for c in tqdm.tqdm(treatment_grid):
        c_treatment = np.full((n_samples, 1), c)
        mean_c, weights_c = _backend.state_means_and_weights(model, features, c_treatment, weights)
        if weights == "prior":
            # The prior weights do not depend on the treatment: weight the state differences.
            c_ute_weighted = (weights_base * (mean_c - mean_base)).sum(axis=1)
        else:
            c_ute_weighted = (weights_c * mean_c).sum(axis=1) - base_mixed
        utes.append(c_ute_weighted)

    utes = np.array(utes).transpose()
    if isinstance(features, pd.DataFrame):
        return pd.DataFrame(utes, index=features.index, columns=treatment_grid)
    return utes


def predict_ate_continuous(
    model: Any,
    features: Any,
    treatment_grid: list[float],
    baseline_treatment: float | None = None,
    weights: str = "prior",
) -> pd.Series | np.ndarray:
    """Computes the average treatment effect (ATE) for continuous treatment.

    Args:
      model: a trained pypsps model (Keras or PyTorch).
      features: features (X) for the causal model. Often a
        pd.DataFrame/np.ndarray, but can also be non-standard
        data structure as long as the pypsps model can use it as
        input to model.predict([features, ...]).
      treatment_grid: grid of treatment values to get UTE for.  This will be the number of columns
        in the resulting array / dataframe
      baseline_treatment: what's the baseline treatment value.  If None, uses the avg of the treatment_grid.
      weights: ``"prior"`` (default) or ``"posterior"`` state weights; see
        `predict_ute_continuous`.

    Returns:
      A pd.Series (if features is a DataFrame) or a np.ndarray of same number of
      rows as the number of values in treatment grid.
    """
    ute_cont = predict_ute_continuous(
        model, features, treatment_grid, baseline_treatment, weights=weights
    )
    ate_cont = ute_cont.mean(axis=0)
    if isinstance(ate_cont, pd.Series):
        ate_cont.name = "ate"
    return ate_cont
