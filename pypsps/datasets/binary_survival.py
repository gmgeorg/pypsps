"""Toy example survival of a cancer treatment to survival times."""

import math

import numpy as np
import pandas as pd
from scipy.special import expit  # logistic sigmoid

from . import base

_FEAT_COLS = ["gender", "age", "comorbidity", "cancer_severity"]


def _simple_custom_uuid(val):
    """
    Returns a simple custom UUID string for the given value.
    This function uses the built-in hash() and converts the absolute hash value
    to a base-36 string (digits and lowercase letters).
    """
    # Get absolute hash value to avoid negative numbers.
    h = abs(hash(val))
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    if h == 0:
        return alphabet[0]
    s = []
    while h:
        s.append(alphabet[h % 36])
        h //= 36
    return "".join(reversed(s))


class CancerSurvivalSimulator(base.BaseSimulator):
    """Cancer survival simulation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rng = np.random.RandomState(self._seed)

    def sample(self, n_samples: int):
        """Samples example dataset."""

        # 1. Generate features
        # Gender: randomly assign Male/Female
        genders = self._rng.choice(["male", "female"], size=n_samples)

        # Age: uniformly distributed from 30 to 80.
        ages = self._rng.uniform(30, 80, size=n_samples)

        # Comorbidity: categorical; probabilities: Low (50%), Medium (30%), High (20%)
        comorbidity = self._rng.choice(["low", "medium", "high"], size=n_samples, p=[0.5, 0.3, 0.2])

        # Cancer severity: uniformly from 1 to 10.
        cancer_severity = self._rng.uniform(1, 10, size=n_samples)

        # 2. Treatment assignment (chemotherapy)
        # Logistic model: more likely if cancer_severity is high and age is low.
        # We'll use: lp = -0.6 + 0.1 * cancer_severity - 0.015 * age.
        lp = 1.0 + 0.5 * cancer_severity - 0.05 * ages
        p_chemo = expit(lp)
        chemo = self._rng.binomial(1, p_chemo, size=n_samples)

        # 3. Simulate true recovery time from exponential distribution.
        # For untreated: median = 365 days => scale = 365/ln2
        # For treated: median = 365/2 days => scale = (365/2)/ln2
        # np.random.exponential uses "scale" parameter = 1/lambda = mean.
        scale_untreated = 365 / math.log(2)  # ~527 days
        scale_treated = (100.0) / math.log(2)  # ~263.5 days

        # For each patient, choose scale based on treatment.
        scales = np.where(chemo == 1, scale_treated, scale_untreated)
        # Simulate recovery time from exponential distribution.
        true_recovery_time = self._rng.exponential(scale=scales)

        # 4. Impose study follow-up: end at 730 days.
        # Observed time is min(true_recovery_time, 730)
        observed_time = np.minimum(true_recovery_time, 540)
        # Natural event indicator: 1 if true_recovery_time <= 730, else 0.
        event_indicator = (true_recovery_time <= 540).astype(int)

        # 6. Assemble DataFrame
        df = pd.DataFrame(
            {
                "gender": genders,
                "age": ages,
                "comorbidity": comorbidity,
                "cancer_severity": cancer_severity,
                "chemotherapy": chemo,
                "true_recovery_time": true_recovery_time,
                "event_time": observed_time,
                "event_indicator": event_indicator,
                "prob_chemotherapy": p_chemo,
            }
        )
        df = df.sort_values("prob_chemotherapy", ascending=False)
        df["patient_id"] = (df.index.to_series() + 1e6).apply(_simple_custom_uuid)
        df = df.set_index("patient_id", verify_integrity=True)

        df["gender"] = df["gender"].map({"male": 0, "female": 1})
        df["comorbidity"] = df["comorbidity"].map({"low": 0, "medium": 1, "high": 2})

        true_ute = pd.Series((365.0 / 2.0) / math.log(2.0), index=df.index, name="true_ute")

        return base.CausalDataset(
            treatments=df["chemotherapy"],
            outcomes=df[["event_time", "event_indicator"]],
            features=df[_FEAT_COLS],
            true_ate=true_ute.mean(),
            true_ute=true_ute,
            true_propensity_score=df["prob_chemotherapy"],
            true_outcomes=df["true_recovery_time"],
        )
