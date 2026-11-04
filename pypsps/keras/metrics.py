"""Module for metrics from pypsps predictions."""

import warnings

import pypress.utils
import tensorflow as tf

from .. import utils
from . import losses


class _ColumnArgsMixin:
    """Stores the y_pred / y_true column counts and adds them to ``get_config``.

    Subclasses set ``_COLUMN_ARGS`` to the constructor arguments they store as ``_<name>``.
    Every other constructor argument (``name``, ``dtype``, ``curve``, ...) goes to the Keras
    base metric, whose ``get_config`` returns it, so ``cls.from_config(m.get_config())``
    rebuilds the metric.
    """

    _COLUMN_ARGS: tuple[str, ...] = ("n_outcome_pred_cols", "n_treatment_pred_cols")

    def get_config(self) -> dict:
        """Returns the base metric's config plus the column counts."""
        config = super().get_config()
        config.update({k: getattr(self, f"_{k}") for k in self._COLUMN_ARGS})
        return config


_WITH_OUTCOME_TRUE_COLS = ("n_outcome_pred_cols", "n_treatment_pred_cols", "n_outcome_true_cols")


@tf.keras.utils.register_keras_serializable(package="pypsps")
class PropensityScoreBinaryCrossentropy(_ColumnArgsMixin, tf.keras.metrics.BinaryCrossentropy):
    """Computes cross entropy for the propensity score. Used as a metric in pypsps model."""

    def __init__(
        self,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        name="propensity_score_binary_crossentropy",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state."""
        propensity_score = utils.agg_treatment_pred(
            y_pred,
            n_outcome_pred_cols=self._n_outcome_pred_cols,
            n_treatment_pred_cols=self._n_treatment_pred_cols,
        )
        # TODO: make it robust with utils.split_y_true() to handle multi-output y_true
        treatment_true = y_true[:, -1:]
        super().update_state(
            y_true=treatment_true, y_pred=propensity_score, sample_weight=sample_weight
        )


@tf.keras.utils.register_keras_serializable(package="pypsps")
class PropensityScoreAUC(_ColumnArgsMixin, tf.keras.metrics.AUC):
    """AUC computed on the output for propensity part.

    Keyword arguments (``curve``, ``num_thresholds``, ...) go to ``tf.keras.metrics.AUC``.
    """

    def __init__(
        self,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        name="propensity_score_auc",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        propensity_score = utils.agg_treatment_pred(
            y_pred,
            n_outcome_pred_cols=self._n_outcome_pred_cols,
            n_treatment_pred_cols=self._n_treatment_pred_cols,
        )
        # Pick the last column of y_true as the treatment label (binary).
        # TODO: make it robust with utils.split_y_true() to handle multi-output y_true
        treatment_true = y_true[:, -1:]
        super().update_state(
            y_true=treatment_true, y_pred=propensity_score, sample_weight=sample_weight
        )


@tf.keras.utils.register_keras_serializable(package="pypsps")
class TreatmentMeanSquaredError(_ColumnArgsMixin, tf.keras.metrics.MeanSquaredError):
    """MSE computed on continuous treatment prediction."""

    _COLUMN_ARGS = _WITH_OUTCOME_TRUE_COLS

    def __init__(
        self,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        n_outcome_true_cols: int,
        name="treatment_mean_squared_error",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._n_outcome_true_cols = n_outcome_true_cols
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        treat_pred = utils.agg_treatment_pred(
            y_pred,
            n_outcome_pred_cols=self._n_outcome_pred_cols,
            n_treatment_pred_cols=self._n_treatment_pred_cols,
        )
        treat_true = utils.split_y_true(y_true, n_outcome_true_cols=self._n_outcome_true_cols)[1]
        super().update_state(y_true=treat_true, y_pred=treat_pred, sample_weight=sample_weight)


@tf.keras.utils.register_keras_serializable(package="pypsps")
class TreatmentMeanAbsoluteError(_ColumnArgsMixin, tf.keras.metrics.MeanAbsoluteError):
    """MAE computed on continuous treatment prediction."""

    _COLUMN_ARGS = _WITH_OUTCOME_TRUE_COLS

    def __init__(
        self,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        n_outcome_true_cols: int,
        name="treatment_mean_absolute_error",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._n_outcome_true_cols = n_outcome_true_cols
        self._n_outcome_pred_cols = n_outcome_pred_cols
        self._n_treatment_pred_cols = n_treatment_pred_cols

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates state"""
        treat_pred = utils.agg_treatment_pred(
            y_pred,
            n_outcome_pred_cols=self._n_outcome_pred_cols,
            n_treatment_pred_cols=self._n_treatment_pred_cols,
        )
        treat_true = utils.split_y_true(y_true, n_outcome_true_cols=self._n_outcome_true_cols)[1]
        super().update_state(y_true=treat_true, y_pred=treat_pred, sample_weight=sample_weight)


@tf.keras.utils.register_keras_serializable(package="pypsps")
class OutcomeMeanSquaredError(_ColumnArgsMixin, tf.keras.metrics.MeanSquaredError):
    """MSE computed on the output for weighted average outcome prediction."""

    _COLUMN_ARGS = _WITH_OUTCOME_TRUE_COLS

    def __init__(
        self,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        n_outcome_true_cols: int,
        name="outcome_mean_squared_error",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
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


@tf.keras.utils.register_keras_serializable(package="pypsps")
class OutcomeMeanAbsoluteError(_ColumnArgsMixin, tf.keras.metrics.MeanAbsoluteError):
    """MAE computed on the output for weighted average outcome prediction."""

    _COLUMN_ARGS = _WITH_OUTCOME_TRUE_COLS

    def __init__(
        self,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        n_outcome_true_cols: int,
        name="outcome_mean_absolute_error",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
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


@tf.keras.utils.register_keras_serializable(package="pypsps")
def predictive_state_df(
    y_true, y_pred, n_outcome_pred_cols: int, n_treatment_pred_cols: int
) -> tf.Tensor:
    """Degrees of freedom of the predictive state weights (`pypress.utils.tr_kernel`)."""
    del y_true
    _, weights, _ = utils.split_y_pred(
        y_pred,
        n_outcome_pred_cols=n_outcome_pred_cols,
        n_treatment_pred_cols=n_treatment_pred_cols,
    )
    return pypress.utils.tr_kernel(weights)


def predictive_state_df_gen(
    n_outcome_pred_cols: int, n_treatment_pred_cols: int
) -> tf.keras.metrics.MeanMetricWrapper:
    """Metric for the degrees of freedom of the predictive states, logged as
    ``predictive_state_df``. Serializable: it wraps the registered `predictive_state_df`."""
    return tf.keras.metrics.MeanMetricWrapper(
        fn=predictive_state_df,
        name="predictive_state_df",
        n_outcome_pred_cols=n_outcome_pred_cols,
        n_treatment_pred_cols=n_treatment_pred_cols,
    )


@tf.keras.utils.register_keras_serializable(package="pypsps")
class CausalLossMetric(tf.keras.metrics.Mean):
    """The joint NLL: ``outcome_loss_weight * outcome NLL + 1 * treatment NLL``.

    Unlike the training `losses.CausalLoss`, it has no predictive-state regularizer, no
    penalty embedded in the outcome or treatment loss (it holds their `penalty_free()`
    copies), and alpha fixed at 1, whatever alpha the training loss uses. Its values are
    therefore comparable across epochs and alpha schedules, which makes it a sound
    early-stopping and checkpoint criterion. ``outcome_loss_weight=0`` is allowed (unusual,
    not recommended): the metric is then the treatment NLL alone.

    There is deliberately no ``alpha`` argument; passing one raises `TypeError`.
    """

    def __init__(
        self,
        outcome_loss: losses.OutcomeLoss,
        treatment_loss: losses.TreatmentLoss,
        outcome_loss_weight: float = 1.0,
        name="causal_loss_metric",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if not isinstance(outcome_loss, losses.OutcomeLoss):
            raise TypeError(f"outcome_loss must be an OutcomeLoss; got {type(outcome_loss)}")
        if not isinstance(treatment_loss, losses.TreatmentLoss):
            raise TypeError(f"treatment_loss must be a TreatmentLoss; got {type(treatment_loss)}")
        self._outcome = outcome_loss.penalty_free()
        self._treatment = treatment_loss.penalty_free()
        # No range check and no `or 1.0`: 0 is a legal weight.
        self._outcome_loss_weight = float(outcome_loss_weight)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the per-batch joint NLL."""
        value = self._outcome_loss_weight * self._outcome(y_true, y_pred) + self._treatment(
            y_true, y_pred
        )
        return super().update_state(value, sample_weight=sample_weight)

    def get_config(self) -> dict:
        """Returns the config; the two losses are serialized Keras objects."""
        return {
            **super().get_config(),
            "outcome_loss": tf.keras.utils.serialize_keras_object(self._outcome),
            "treatment_loss": tf.keras.utils.serialize_keras_object(self._treatment),
            "outcome_loss_weight": self._outcome_loss_weight,
        }

    @classmethod
    def from_config(cls, config: dict) -> "CausalLossMetric":
        """Rebuilds the metric, deserializing the two losses."""
        config = dict(config)
        config["outcome_loss"] = tf.keras.utils.deserialize_keras_object(config["outcome_loss"])
        config["treatment_loss"] = tf.keras.utils.deserialize_keras_object(config["treatment_loss"])
        return cls(**config)


_UNSET = object()


def causal_loss_metric_gen(
    outcome_loss: losses.OutcomeLoss,
    treatment_loss: losses.TreatmentLoss,
    alpha=_UNSET,
    outcome_loss_weight: float = 1.0,
) -> CausalLossMetric:
    """Returns a `CausalLossMetric`, logged as ``causal_loss_metric``.

    Parameters
    ----------
    outcome_loss : OutcomeLoss
        The outcome loss; the metric uses its penalty-free copy.
    treatment_loss : TreatmentLoss
        The treatment loss; the metric uses its penalty-free copy.
    alpha : deprecated
        Ignored, with a `FutureWarning`: the metric always uses alpha = 1. Removed in v0.5.0.
    outcome_loss_weight : float, default=1.0
        Weight of the outcome NLL. Pass the training loss's weight; 0 is allowed.
    """
    if alpha is not _UNSET:
        warnings.warn(
            "causal_loss_metric_gen: alpha is ignored; causal_loss_metric uses alpha = 1. "
            "The argument is removed in v0.5.0",
            FutureWarning,
            stacklevel=2,
        )
    return CausalLossMetric(outcome_loss, treatment_loss, outcome_loss_weight=outcome_loss_weight)


def outcome_nll_metric_gen(outcome_loss: losses.OutcomeLoss) -> tf.keras.metrics.MeanMetricWrapper:
    """Metric of the outcome negative log-likelihood alone, without penalties or alpha.

    Suitable as an early-stopping and checkpoint criterion that does not depend on the alpha
    schedule. Pass to ``model.compile(metrics=[...])``; it is logged as ``outcome_nll_metric``.
    """
    return tf.keras.metrics.MeanMetricWrapper(
        fn=outcome_loss.penalty_free(), name="outcome_nll_metric"
    )


@tf.keras.utils.register_keras_serializable(package="pypsps")
def uniform_entropy(
    y_true, y_pred, n_outcome_pred_cols: int, n_treatment_pred_cols: int
) -> tf.Tensor:
    """Mean Shannon entropy of the predictive state weights (higher = more uniform)."""
    del y_true
    _, weights, _ = utils.split_y_pred(
        y_pred,
        n_outcome_pred_cols=n_outcome_pred_cols,
        n_treatment_pred_cols=n_treatment_pred_cols,
    )
    # Shannon entropy per row: -sum(p_i * log(p_i)), with 0 * log(0) taken as 0.
    # safe_xlogx computes that exactly; an epsilon floor (log(p_i + eps)) would distort
    # the entropy of sharp, near-degenerate state weights.
    entropy_per_row = -1 * tf.math.reduce_sum(pypress.utils.safe_xlogx(weights), axis=1)
    return tf.math.reduce_mean(entropy_per_row)


def uniform_entropy_gen(
    n_outcome_pred_cols: int, n_treatment_pred_cols: int
) -> tf.keras.metrics.MeanMetricWrapper:
    """
    Metric of the average Shannon entropy of the predictive state weights, logged as
    ``uniform_entropy``.

    The entropy is computed as:
        mean_entropy = mean(-sum(p_i * log(p_i)))

    where p_i are the predictive state weights for each state. Higher entropy indicates more
    uniform state distributions.

    Parameters
    ----------
    n_outcome_pred_cols : int
        Number of outcome prediction columns in y_pred.
    n_treatment_pred_cols : int
        Number of treatment prediction columns in y_pred.

    Returns
    -------
    tf.keras.metrics.MeanMetricWrapper
        A serializable metric wrapping the registered `uniform_entropy`.
    """
    return tf.keras.metrics.MeanMetricWrapper(
        fn=uniform_entropy,
        name="uniform_entropy",
        n_outcome_pred_cols=n_outcome_pred_cols,
        n_treatment_pred_cols=n_treatment_pred_cols,
    )
