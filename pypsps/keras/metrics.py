"""Module for metrics from pypsps predictions."""

import pypress.utils
import tensorflow as tf

from .. import utils
from . import losses


@tf.keras.utils.register_keras_serializable(package="pypsps")
class PropensityScoreBinaryCrossentropy(tf.keras.metrics.BinaryCrossentropy):
    """Computes cross entropy for the propensity score. Used as a metric in pypsps model."""

    def __init__(
        self,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        name="propensity_score_binary_crossentropy",
        **kwargs,
    ):
        super().__init__(name=name)
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state."""
        _, _, propensity_score = utils.split_y_pred(
            y_pred,
            n_outcome_pred_cols=self._n_outcome_pred_cols,
            n_treatment_pred_cols=self._n_treatment_pred_cols,
        )
        treatment_true = y_true[:, 1:]
        super().update_state(
            y_true=treatment_true, y_pred=propensity_score, sample_weight=sample_weight
        )


@tf.keras.utils.register_keras_serializable(package="pypsps")
class PropensityScoreAUC(tf.keras.metrics.AUC):
    """AUC computed on the ouptut for propensity part."""

    def __init__(self, n_outcome_pred_cols: int, n_treatment_pred_cols: int, **kwargs):
        super().__init__()
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        _, _, propensity_score = utils.split_y_pred(
            y_pred,
            n_outcome_pred_cols=self._n_outcome_pred_cols,
            n_treatment_pred_cols=self._n_treatment_pred_cols,
        )
        treatment_true = y_true[:, 1:]
        super().update_state(
            y_true=treatment_true, y_pred=propensity_score, sample_weight=sample_weight
        )


@tf.keras.utils.register_keras_serializable(package="pypsps")
class TreatmentMeanSquaredError(tf.keras.metrics.MeanSquaredError):
    """MSE computed on continuous treatment prediction."""

    def __init__(
        self,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        n_outcome_true_cols: int,
        **kwargs,
    ):
        super().__init__()
        self._n_outcome_true_cols = n_outcome_true_cols
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        treat_pred = utils.split_y_pred(
            y_pred,
            n_outcome_pred_cols=self._n_outcome_pred_cols,
            n_treatment_pred_cols=self._n_treatment_pred_cols,
        )[2]
        treat_true = utils.split_y_true(y_true, n_outcome_true_cols=self._n_outcome_true_cols)[1]
        super().update_state(y_true=treat_true, y_pred=treat_pred, sample_weight=sample_weight)


# TODO: add unit test for Outcome* metrics
@tf.keras.utils.register_keras_serializable(package="pypsps")
class TreatmentMeanAbsoluteError(tf.keras.metrics.MeanAbsoluteError):
    """MSE computed on the ouptut for weighted average outcome prediction."""

    def __init__(
        self,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        n_outcome_true_cols: int,
        **kwargs,
    ):
        super().__init__()
        self._n_outcome_true_cols = n_outcome_true_cols
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        treat_pred = utils.split_y_pred(
            y_pred,
            n_outcome_pred_cols=self._n_outcome_pred_cols,
            n_treatment_pred_cols=self._n_treatment_pred_cols,
        )[2]
        treat_true = utils.split_y_true(y_true, n_outcome_true_cols=self._n_treatment_pred_cols)[1]
        super().update_state(y_true=treat_true, y_pred=treat_pred, sample_weight=sample_weight)


@tf.keras.utils.register_keras_serializable(package="pypsps")
class OutcomeMeanSquaredError(tf.keras.metrics.MeanSquaredError):
    """MSE computed on the ouptut for weighted average outcome prediction."""

    def __init__(
        self,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        n_outcome_true_cols: int,
        **kwargs,
    ):
        super().__init__()
        self._n_outcome_true_cols = n_outcome_true_cols
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        avg_outcome = utils.agg_outcome_pred(
            y_pred,
            n_outcome_pred_cols=self._n_outcome_pred_cols,
            n_treatment_pred_cols=self._n_treatment_pred_cols,
        )
        outcome_true = utils.split_y_true(y_true, n_outcome_true_cols=self._n_outcome_true_cols)[0]
        super().update_state(y_true=outcome_true, y_pred=avg_outcome, sample_weight=sample_weight)


# TODO: add unit test for Outcome* metrics
@tf.keras.utils.register_keras_serializable(package="pypsps")
class OutcomeMeanAbsoluteError(tf.keras.metrics.MeanAbsoluteError):
    """MSE computed on the ouptut for weighted average outcome prediction."""

    def __init__(
        self,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        n_outcome_true_cols: int,
        **kwargs,
    ):
        super().__init__()
        self._n_outcome_true_cols = n_outcome_true_cols
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        avg_outcome = utils.agg_outcome_pred(
            y_pred,
            n_outcome_pred_cols=self._n_outcome_pred_cols,
            n_treatment_pred_cols=self._n_treatment_pred_cols,
        )
        outcome_true = utils.split_y_true(y_true, n_outcome_true_cols=self._n_outcome_true_cols)[0]
        super().update_state(y_true=outcome_true, y_pred=avg_outcome, sample_weight=sample_weight)


def predictive_state_df_gen(n_outcome_pred_cols: int, n_treatment_pred_cols: int):
    """Metric for degrees of freedom of predictive states."""

    def predictive_state_df(y_true, y_pred) -> tf.Tensor:
        """Computes degrees of freedom of predictive state weights."""
        del y_true
        _, weights, _ = utils.split_y_pred(
            y_pred,
            n_outcome_pred_cols=n_outcome_pred_cols,
            n_treatment_pred_cols=n_treatment_pred_cols,
        )
        return pypress.utils.tr_kernel(weights)

    return predictive_state_df


def causal_loss_metric_gen(
    outcome_loss: losses.OutcomeLoss,
    treatment_loss: losses.TreatmentLoss,
    alpha: float = 1.0,
    outcome_loss_weight: float = 1.0,
):
    """
    Function wrapper that returns a metric function computing the causal loss.

    The returned function takes (y_true, y_pred) as inputs and computes:

        causal_loss = outcome_loss_weight * outcome_loss(y_true, y_pred)
                      + alpha * treatment_loss(y_true, y_pred)

    This metric function can be passed to model.compile(metrics=[...]).

    Parameters
    ----------
    outcome_loss : OutcomeLoss
        Instance of an outcome loss (e.g. Normal log-likelihood loss).
    treatment_loss : TreatmentLoss
        Instance of a treatment loss (e.g. binary cross-entropy for treatment prediction).
    alpha : float, default=1.0
        Penalty parameter for treatment loss.
    outcome_loss_weight : float, default=1.0
        Weight for the outcome loss.

    Returns
    -------
    function
        A function metric that takes (y_true, y_pred) and returns the causal loss as a float value (can be passed as metric).
    """
    # Construct an instance of CausalLoss with the given parameters.
    causal_loss_obj = losses.CausalLoss(
        outcome_loss=outcome_loss,
        treatment_loss=treatment_loss,
        alpha=alpha,
        outcome_loss_weight=outcome_loss_weight,
    )

    def causal_loss_metric(y_true, y_pred) -> tf.Tensor:
        """Metric function computing the causal loss."""
        # Call the causal loss object to compute the loss per example.
        # Here we assume causal_loss_obj returns per-example loss.
        return causal_loss_obj(y_true, y_pred)

    causal_loss_metric.__name__ = "causal_loss"
    return causal_loss_metric
