"""Survival simulation to evaluate observational causal algorithms."""

import numpy as np
import pandas as pd

from . import base


class SurvivalData(base.BaseSimulator):
    """Implements survival data simulation."""

    def __init__(self, treatment_multiplier: float, **kwargs):
        super().__init__(**kwargs)
        self._treatment_multiplier = treatment_multiplier
        self._rng = np.random.RandomState(self._seed)

    def sample(self, n_samples: int, **kwargs) -> base.CausalDataset:
        """Implements the simulation."""

        # covariate generation
        data = pd.DataFrame(
            {
                "age": np.clip(self._rng.normal(65, 5, n_samples), 40, 90),
                "sex": self._rng.binomial(1, 0.45, n_samples),
                "stage": self._rng.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.3, 0.4, 0.1]),
                "hpv": self._rng.binomial(1, 0.65, n_samples),
                "smoking_pack_years": self._rng.gamma(shape=2, scale=5, size=n_samples),
                "comorbidities": self._rng.poisson(0.7, n_samples),
                "ecog_performance": self._rng.choice(
                    [0, 1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.08, 0.02]
                ),
                "tumor_volume": np.abs(self._rng.normal(3, 1, n_samples)),
            }
        )

        # Continuous treatment assignment (Cause)
        true_treatment_assignment = (
            -0.1 * (data["age"] - 65)
            + 0.5 * data["stage"] ** 2
            - 2.5 * data["hpv"]
            + 0.3 * data["smoking_pack_years"]
        )

        treatment_model = true_treatment_assignment + self._rng.normal(0, 3, n_samples)
        data["cause"] = np.clip(treatment_model + 65, 55, 75)

        # Survival time generation
        true_ute = (
            # Linear term + Quadratic term
            self._treatment_multiplier * (data["cause"] - 65) / 10
            + 0.5 * self._treatment_multiplier * ((data["cause"] - 65) / 10) ** 2
        )

        hazard_base = 0.1 * np.exp(
            0.05 * (data["age"] - 65)
            + 0.3 * data["stage"]
            - 0.8 * data["hpv"]
            + 0.02 * data["smoking_pack_years"]
            + 0.1 * data["tumor_volume"]
            + true_ute
        )

        # simulated survival times with Weibull distribution
        shape = 1.5  # Increasing risk over time
        scale = 1 / (hazard_base * shape)
        true_times = self._rng.weibull(shape, n_samples) * scale

        # Introduce censoring based on covariates
        censoring_hazard = 0.5 * np.exp(
            0.03 * (data["age"] - 65) - 0.1 * data["hpv"] + 0.01 * data["tumor_volume"]
        )
        censoring_times = self._rng.exponential(1 / censoring_hazard)

        data["time"] = np.minimum(true_times, censoring_times)
        data["event"] = (true_times <= censoring_times).astype(int)

        # Time discretization to months
        data["time"] = np.round(data["time"] * 12, 1)

        return base.CausalDataset(
            treatments=pd.Series(data["cause"], name="treatment"),
            features=data[sorted(data.columns)].drop(["time", "event", "cause"], axis=1),
            outcomes=data[["time", "event"]],
            true_ate=true_ute.mean(),
            true_ute=pd.Series(true_ute, name="true_ute"),
            true_propensity_score=true_treatment_assignment,
        )
