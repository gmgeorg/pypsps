"""Module for the KangSchafer simulation."""

import numpy as np

import pandas as pd
from . import base


class KangSchafer(base.BaseSimulator):
    """Kang-Schafer simulated dataset.

    Kang-Schafer illustrates selection bias of outcome under non-informative
    response.  Originally the treatment effect is 0, but in this simulation we
    allow user to set it a pre-defined value.
    """

    def __init__(
        self,
        true_ate: float = 0.0,
        observe_transformed_features: bool = False,
    ):
        """Initializes a `BaseSimulator` instance that returns a `CausalDataset` at .run().

        Args:
          true_ate: true ATE. Defaults to 0. as in KangSchafer but allows to specify
            non-zero treatment effect for purpose of simulation.
          observe_transformed_features: if True, it uses the non-linear transformed features
            as the observed features.  This will make a linear model unspecified
            in terms of 'features'.
        """
        super().__init__()
        self._true_ate = true_ate
        self._observe_transformed_features = observe_transformed_features

    def sample(self, n_samples: int, **kwargs) -> base.CausalDataset:
        """Implements the Kang-Schafer simulation."""

        z_arr = pd.DataFrame(
            np.random.normal(size=(n_samples, 4)),
            columns=["z" + str(i) for i in range(1, 5)],
        )

        propensity_score = base.expit(np.dot(z_arr, np.array([-1.0, 0.5, -0.25, -0.1])))

        treatment = pd.Series(np.random.binomial(1, propensity_score), name="treatment")

        outcome = pd.Series(
            210.0
            + np.dot(z_arr, np.array([27.4, 13.7, 13.7, 13.7]))
            + self._true_ate * treatment
            + np.random.normal(size=(n_samples,)),
            name="outcome",
        )

        # Non-linear transformation of original features (linear wrt propensity score).
        z1, z2, z3, z4 = np.hsplit(z_arr.values, 4)
        x_arr = pd.DataFrame(
            np.hstack(
                [
                    np.exp(z1 / 2.0),
                    z2 / (1 + np.exp(z1)) + 10.0,
                    np.power(z1 * z3 / 25 + 0.6, 3),
                    np.square(z2 + z4 + 20.0),
                ]
            ),
            columns=["x" + str(i) for i in range(1, 5)],
        )

        propensity_score = pd.Series(propensity_score, name="propensity_score")
        if self._observe_transformed_features:
            # Use transformed X variables as observed features (and hence original
            # features are latent).
            return base.CausalDataset(
                treatments=treatment,
                outcomes=outcome,
                features=x_arr,
                latent_features=z_arr,
                true_ate=self._true_ate,
                true_ute=pd.Series(
                    self._true_ate, index=treatment.index, name="true_ute"
                ),
                true_propensity_score=propensity_score,
            )

        # Return the original features, such that outcome/treatment is linear in the features.
        return base.CausalDataset(
            treatments=treatment,
            outcomes=outcome,
            features=z_arr,
            latent_features=x_arr,
            true_ate=self._true_ate,
            true_ute=pd.Series(self._true_ate, index=treatment.index, name="true_ute"),
            true_propensity_score=propensity_score,
        )
