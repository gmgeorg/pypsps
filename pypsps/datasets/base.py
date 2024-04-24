"""Base module for all (simulated) datasets."""

from typing import Optional, Tuple

import abc
import pandas as pd
import numpy as np


def expit(x):
    """Expit function (inverse of logit)."""
    return 1.0 / (1.0 + np.exp(-x))


class CausalDataset(object):
    """A class holing a causal dataset."""

    def __init__(
        self,
        outcomes: pd.DataFrame,
        treatments: pd.DataFrame,
        features: pd.DataFrame,
        latent_features: pd.DataFrame = None,
        true_ate: Optional[float] = None,
        true_ute: Optional[pd.DataFrame] = None,
        true_propensity_score: Optional[pd.DataFrame] = None,
    ):
        """Initializes the class."""
        if isinstance(treatments, pd.Series):
            treatments = treatments.to_frame()
        if isinstance(features, pd.Series):
            features = features.to_frame()
        if isinstance(outcomes, pd.Series):
            outcomes = outcomes.to_frame()

        self.treatments = treatments
        self.features = features
        self.outcomes = outcomes
        self.latent_features = latent_features
        self.true_ate = true_ate
        self.true_ute = true_ute
        self.true_propensity_score = true_propensity_score

    def to_data_frame(self) -> pd.DataFrame:
        """Returns all data as a concatenated DataFrame."""
        list_dfs = [self.outcomes, self.treatments, self.features]
        if self.latent_features is not None:
            list_dfs.append(self.latent_features)
        return pd.concat(list_dfs, axis=1)

    def to_keras_inputs_outputs(
        self,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Returns (inputs, outputs) for a .fit() method for keras model."""
        input_data = [self.features.values, self.treatments.values]
        output_data = np.hstack([self.outcomes.values, self.treatments.values])
        return (
            input_data,
            output_data,
        )

    @property
    def n_features(self) -> int:
        """Number of features in data."""
        return self.features.shape[1]

    @property
    def n_treatments(self) -> int:
        """Number of treatments in data."""
        return self.treatments.shape[1]

    @property
    def n_outcomes(self) -> int:
        """Number of outcomes in data."""
        return self.outcomes.shape[1]

    @property
    def n_samples(self) -> int:
        """Number of samples in data."""
        return self.treatments.shape[0]

    def naive_ate(self) -> float:
        """Computes a naive ATE estimate using difference in differences (for binary treatment)."""
        assert (
            self.treatments.shape[1] == 1
        ), "naive_ate() requires a univariate treatment"

        treat_series = self.treatments.iloc[:, 0]
        assert treat_series.nunique() == 2, "naive_ate() requires binary treatment."
        avg_by_treatment = self.outcomes.groupby(treat_series).mean().sort_index()

        return (avg_by_treatment.iloc[1] - avg_by_treatment.iloc[0]).iloc[0]

    def naive_ute(self) -> pd.Series:
        """Computes naive UTE as the same (naive) ATE for each row."""
        return pd.Series(
            self.naive_ate(), index=self.treatments.index, name="naive_ute"
        )


class BaseSimulator(abc.ABC):
    """Base class for simulating causal datasets."""

    def __init__(self, **kwargs):
        """Initializes the class."""
        super().__init__(**kwargs)

    @abc.abstractmethod
    def sample(self, n_samples: int, **kwargs) -> CausalDataset:
        """Implements the simulation and adds ._causal_dataset.

        Args:
            n_samples: sample size (control + treated).

        Returns:
             A CausalDataset with n_samples observations.
        """
        raise NotImplementedError("sample() needs to be implemented.")
