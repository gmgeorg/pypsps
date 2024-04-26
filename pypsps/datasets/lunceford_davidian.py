"""Lunceford-Davidian simulation to evaluate observational causal algorithms."""

import numpy as np
import pandas as pd
import enum

from . import base


class Association(enum.Enum):
    NONE = "none"
    MODERATE = "moderate"
    STRONG = "strong"


BETA_LOOKUP = {
    Association.NONE: np.array([0.0, 0.0, 0.0, 0.0]),
    Association.MODERATE: np.array([0.0, 0.3, -0.3, 0.3]),
    Association.STRONG: np.array([0.0, 0.6, -0.6, 0.6]),
}

XI_LOOKUP = {
    Association.NONE: np.array([0.0, 0.0, 0.0]),
    Association.MODERATE: np.array([-0.5, 0.5, 0.5]),
    Association.STRONG: np.array([-1.0, 1.0, 1.0]),
}

NU = np.array([0.0, -1.0, 1.0, -1.0, 2.0])


B_COV = np.array(
    [
        [
            1.0,
            0.5,
            -0.5,
            -0.5,
        ],
        [0.5, 1.0, -0.5, -0.5],
        [-0.5, -0.5, 1.0, 0.5],
        [-0.5, -0.5, 0.5, 1],
    ]
)

A0 = np.array([-1.0, -1.0, 1, 1])
A1 = np.array([1, 1, -1.0, -1.0])


class LuncefordDavidian(base.BaseSimulator):
    """Implements Lunceford Davidian simulation."""

    def __init__(
        self,
        association: Association,
    ):
        super().__init__()
        if isinstance(association, str):
            association = Association(association)
        self._association = association

    def sample(self, n_samples: int, **kwargs) -> base.CausalDataset:
        """Implements the Lunceford Davidian simulation."""
        x3 = pd.Series(np.random.binomial(n=1, p=0.25, size=(n_samples,)), name="x3")
        prob_z3 = 0.75 * x3 + 0.25 * (1 - x3)
        z3 = pd.Series(np.random.binomial(n=1, p=prob_z3), name="z3")

        x1x2z1z2 = np.zeros(shape=(n_samples, 4))
        x3_eq_0 = x3 == 0
        x3_eq_1 = x3 == 1
        x1x2z1z2[x3_eq_0, :] = np.random.multivariate_normal(
            A0, B_COV, size=(x3_eq_0.sum())
        )
        x1x2z1z2[x3_eq_1, :] = np.random.multivariate_normal(
            A0, B_COV, size=(x3_eq_1.sum())
        )
        x1x2z1z2 = pd.DataFrame(x1x2z1z2, columns=["x1", "x2", "z1", "z2"])

        xz = pd.concat([x1x2z1z2, x3, z3], axis=1)

        beta_tmp = BETA_LOOKUP[self._association]
        propensity_score = base.expit(
            beta_tmp[0]
            + np.dot(
                xz[["x1", "x2", "x3"]],
                beta_tmp[1:],
            )
        )
        treatment = np.random.binomial(n=1, p=propensity_score)
        noise = np.random.normal(size=(n_samples,))

        xi_tmp = XI_LOOKUP[self._association]
        outcome = (
            NU[0]
            + NU[1] * xz["x1"]
            + NU[2] * xz["x2"]
            + NU[3] * xz["x3"]
            + NU[4] * treatment
            + xi_tmp[0] * xz["z1"]
            + xi_tmp[1] * xz["z2"]
            + xi_tmp[2] * xz["z3"]
            + noise
        )

        propensity_score = pd.Series(propensity_score, name="propensity_score")

        return base.CausalDataset(
            treatments=pd.Series(treatment, name="treatment"),
            features=xz[sorted(xz.columns)],
            outcomes=pd.Series(outcome, name="outcome"),
            # True ATE = 2. (in expectation) for Lunceford-Davidian simulation.
            true_ate=2.0,
            true_ute=NU[4] * pd.Series(treatment, name="true_ute"),
            true_propensity_score=propensity_score,
        )
