"""Module for bootstrapping causal effect estiamtes (UTE & ATE)."""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from . import inference


def bootstrap_ute_binary(
    model: tf.keras.Model,
    features: pd.DataFrame,
    n_bootstraps: int = 100,
) -> pd.DataFrame:
    """
    Generate bootstrap samples of unit-level treatment effects for binary treatment.

    Args:
      model: Trained pypsps model supporting binary treatment.
      features: Feature matrix (DataFrame). Typically inputs[0].
      n_bootstraps: Number of bootstrap replicates.

    Returns:
      DataFrame of shape (n_bootstraps, n_units) with each row a bootstrap replicate of UTEs.
    """
    n_samples = features.shape[0]
    rng = np.random.RandomState(0)
    ute_samples = []

    for i in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        if isinstance(features, pd.DataFrame):
            feat_bs = features.iloc[idx]
        else:
            feat_bs = features[idx]
        ute_bs = inference.predict_ute_binary(model, feat_bs)
        ute_samples.append(ute_bs)

    df = pd.DataFrame(ute_samples)
    df.index.name = "bootstrap_id"
    return df


def bootstrap_ate_binary(
    model: tf.keras.Model,
    features: Union[pd.DataFrame, np.ndarray],
    n_bootstraps: int = 100,
) -> pd.Series:
    """
    Estimate confidence intervals for the ATE of a binary treatment via bootstrapping.

    Args:
      model: Trained pypsps model supporting binary treatment.
      features: Feature matrix (DataFrame or ndarray).  Usually inputs[0].
      n_bootstraps: Number of bootstrap samples.

    Returns:
       pd.Series: the bootstrap estimates of the ATE. Number of rows = n_bootstraps.
    """
    n_samples = features.shape[0]
    ate_samples = []

    rng = np.random.RandomState(n_samples)
    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        if isinstance(features, pd.DataFrame):
            feat_bs = features.iloc[idx]
        else:
            feat_bs = features[idx]
        ate_bs = inference.predict_ate_binary(model, feat_bs)
        ate_samples.append(ate_bs)

    out = pd.Series(ate_samples, name="ate")
    out.index.name = "bootstrap_id"
    return out


def bootstrap_ate_continuous(
    model: tf.keras.Model,
    features: Union[pd.DataFrame, np.ndarray],
    treatment_grid: List[float],
    baseline_treatment: Optional[float] = None,
    n_bootstraps: int = 100,
) -> pd.DataFrame:
    """
    Estimate confidence intervals for ATE over a treatment grid via bootstrapping.

    Args:
      model: Trained pypsps model supporting continuous treatment.
      features: Feature matrix (DataFrame or ndarray).
      treatment_grid: Treatment values to compute ATE for.
      n_bootstrap: Number of bootstrap samples.
      alpha: Significance level for two-sided intervals.
      baseline_treatment: Baseline value for comparison (defaults to grid mean).

    Returns:
      ci_df: DataFrame with index=treatment_grid and columns ["lower", "upper"].
    """
    n_samples = features.shape[0]
    rng = np.random.RandomState(n_samples)

    ate_samples = []
    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        if isinstance(features, pd.DataFrame):
            feat_bs = features.iloc[idx]
        else:
            feat_bs = features[idx]
        ate_bs = inference.predict_ate_continuous(
            model, feat_bs, treatment_grid, baseline_treatment
        )
        ate_samples.append(ate_bs)
    ate_arr = np.array(ate_samples)  # shape (n_bootstrap, len(treatment_grid))
    out = pd.DataFrame(ate_arr, columns=treatment_grid)
    out.index.name = "bootstrap_id"
    return out
