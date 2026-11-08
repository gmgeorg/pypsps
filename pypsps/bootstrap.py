"""Module for bootstrapping causal effect estiamtes (UTE & ATE)."""

from typing import Any

import numpy as np
import pandas as pd

from . import inference


def bootstrap_ute_binary(
    model: Any,
    features: pd.DataFrame,
    n_bootstraps: int = 100,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Generate bootstrap samples of unit-level treatment effects for binary treatment.

    Args:
      model: Trained pypsps model supporting binary treatment.
      features: Feature matrix (DataFrame). Typically inputs[0].
      n_bootstraps: Number of bootstrap replicates.
      random_state: seed for the resampling RNG.

    Returns:
      DataFrame of shape (n_bootstraps, n_units) with each row a bootstrap replicate of UTEs.
    """
    n_samples = features.shape[0]
    rng = np.random.RandomState(random_state)
    ute_samples = []

    for i in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        if isinstance(features, pd.DataFrame):
            feat_bs = features.iloc[idx]
        else:
            feat_bs = features[idx]
        ute_bs = inference.predict_ute_binary(model, feat_bs)
        # Drop the row labels: a resample repeats them, and each replicate is one row.
        ute_samples.append(np.asarray(ute_bs))

    df = pd.DataFrame(ute_samples)
    df.index.name = "bootstrap_id"
    return df


def bootstrap_ate_binary(
    model: Any,
    features: pd.DataFrame | np.ndarray,
    n_bootstraps: int = 100,
    random_state: int = 0,
) -> pd.Series:
    """
    Estimate confidence intervals for the ATE of a binary treatment via bootstrapping.

    Args:
      model: Trained pypsps model supporting binary treatment.
      features: Feature matrix (DataFrame or ndarray).  Usually inputs[0].
      n_bootstraps: Number of bootstrap samples.
      random_state: seed for the resampling RNG.

    Returns:
       pd.Series: the bootstrap estimates of the ATE. Number of rows = n_bootstraps.
    """
    n_samples = features.shape[0]
    ate_samples = []

    rng = np.random.RandomState(random_state)
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
    model: Any,
    features: pd.DataFrame | np.ndarray,
    treatment_grid: list[float],
    baseline_treatment: float | None = None,
    n_bootstraps: int = 100,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Bootstrap the ATE of a continuous treatment over a treatment grid.

    Each replicate resamples the rows of ``features`` with replacement and computes the ATE at
    every grid value relative to the baseline. The raw replicates are returned, so take
    quantiles of each column for confidence intervals.

    Args:
      model: Trained pypsps model supporting continuous treatment.
      features: Feature matrix (DataFrame or ndarray).
      treatment_grid: Treatment values to compute ATE for.
      baseline_treatment: Baseline value for comparison (defaults to grid mean).
      n_bootstraps: Number of bootstrap replicates.
      random_state: seed for the resampling RNG.

    Returns:
      DataFrame of shape (n_bootstraps, len(treatment_grid)): one row per bootstrap draw
      (index named ``bootstrap_id``) and one column per treatment grid value.
    """
    n_samples = features.shape[0]
    rng = np.random.RandomState(random_state)

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
