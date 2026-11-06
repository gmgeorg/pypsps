"""Module for metrics from pypsps predictions.

The propensity, treatment and outcome metrics take the model's `StateMixture`
(``metric(mixture)``): they mix each state's family mean with the state weights, and read
``y_true`` through the mixture's column layout. The state-weight metrics
(`predictive_state_df_gen`, `uniform_entropy_gen`) only need the `ColumnLayout`.
"""

from typing import Any, Optional

import pypress.keras.statistics
import tensorflow as tf

from . import families, losses
from . import mixture as mixture_lib
from .layout import Array, ColumnLayout

_ser = tf.keras.utils.serialize_keras_object
_deser = tf.keras.utils.deserialize_keras_object


class _MixtureMetricMixin:
    """Stores the mixture and serializes it."""

    _DEFAULT_NAME: str

    def _set_mixture(self, mixture: mixture_lib.StateMixture) -> None:
        """Takes the mixture; TypeError unless it is a `StateMixture`."""
        if not isinstance(mixture, mixture_lib.StateMixture):
            raise TypeError(
                f"{type(self).__name__} takes a StateMixture (e.g. model.loss.mixture), got "
                f"{type(mixture).__name__}; the column-count constructor was removed in v0.5.0"
            )
        self._check_mixture(mixture)
        self._mixture = mixture

    def _check_mixture(self, mixture: mixture_lib.StateMixture) -> None:
        """Subclasses may reject a mixture they can't score."""

    @property
    def mixture(self) -> mixture_lib.StateMixture:
        """The state mixture the metric reads."""
        return self._mixture

    def get_config(self) -> dict:
        """The base metric's config plus the serialized mixture."""
        return {**super().get_config(), "mixture": _ser(self._mixture)}

    @classmethod
    def from_config(cls, config: dict):
        """Rebuilds the metric.

        Raises:
          ValueError: for a config without a mixture (saved by v0.3.0-v0.3.3).
        """
        if "mixture" not in config:
            raise ValueError(losses._LEGACY_CONFIG.format(cls=cls.__name__))
        config = dict(config)
        config["mixture"] = _deser(config["mixture"])
        return cls(**config)


class _PropensityMetric(_MixtureMetricMixin):
    """Scores the marginal propensity ``sum_k P(s_k | x) * p_k`` against the treatment."""

    def __init__(
        self, mixture: mixture_lib.StateMixture, name: Optional[str] = None, **kwargs: Any
    ):
        """Initializes the metric.

        Args:
          mixture: the model's `StateMixture`; its treatment family must be binary.
          name: the metric name; defaults to the class's name.
          **kwargs: passed to the Keras base metric (``dtype``, ``curve``, ...).
        """
        super().__init__(name=name or self._DEFAULT_NAME, **kwargs)
        self._set_mixture(mixture)

    def _check_mixture(self, mixture: mixture_lib.StateMixture):
        """Raises unless the treatment is binary."""
        if not mixture.treatment_family.is_binary:
            raise ValueError(
                f"{type(self).__name__} needs a binary treatment family, got "
                f"{mixture.treatment_family!r}"
            )

    def update_state(self, y_true: Array, y_pred: Array, sample_weight: Optional[Array] = None):
        """Updates the base metric with (treatment, marginal propensity)."""
        return super().update_state(
            y_true=self._mixture.layout.decode_true(y_true).treatment,
            y_pred=self._mixture.treatment_mean(y_pred),
            sample_weight=sample_weight,
        )


@tf.keras.utils.register_keras_serializable(package="pypsps")
class PropensityScoreBinaryCrossentropy(_PropensityMetric, tf.keras.metrics.BinaryCrossentropy):
    """Cross entropy of the marginal propensity score P(A = 1 | x)."""

    _DEFAULT_NAME = "propensity_score_binary_crossentropy"


@tf.keras.utils.register_keras_serializable(package="pypsps")
class PropensityScoreAUC(_PropensityMetric, tf.keras.metrics.AUC):
    """AUC of the marginal propensity score P(A = 1 | x).

    Keyword arguments (``curve``, ``num_thresholds``, ...) go to ``tf.keras.metrics.AUC``.
    """

    _DEFAULT_NAME = "propensity_score_auc"


class _ErrorMetric(_MixtureMetricMixin):
    """Shared constructor of the treatment and outcome MSE/MAE."""

    def __init__(
        self, mixture: mixture_lib.StateMixture, name: Optional[str] = None, **kwargs: Any
    ):
        """Initializes the metric.

        Args:
          mixture: the model's `StateMixture`.
          name: the metric name; defaults to the class's name.
          **kwargs: passed to the Keras base metric (``dtype``).
        """
        super().__init__(name=name or self._DEFAULT_NAME, **kwargs)
        self._set_mixture(mixture)


class _TreatmentErrorMetric(_ErrorMetric):
    """Scores the mixed treatment mean ``sum_k P(s_k | x) * E[A | s_k]``."""

    def update_state(self, y_true: Array, y_pred: Array, sample_weight: Optional[Array] = None):
        """Updates the base metric with (treatment, mixed treatment mean)."""
        treatment = self._mixture.layout.decode_true(y_true).treatment
        return super().update_state(
            y_true=tf.cast(treatment, tf.float32),
            y_pred=self._mixture.treatment_mean(y_pred),
            sample_weight=sample_weight,
        )


@tf.keras.utils.register_keras_serializable(package="pypsps")
class TreatmentMeanSquaredError(_TreatmentErrorMetric, tf.keras.metrics.MeanSquaredError):
    """MSE of the mixed treatment mean against the observed treatment."""

    _DEFAULT_NAME = "treatment_mean_squared_error"


@tf.keras.utils.register_keras_serializable(package="pypsps")
class TreatmentMeanAbsoluteError(_TreatmentErrorMetric, tf.keras.metrics.MeanAbsoluteError):
    """MAE of the mixed treatment mean against the observed treatment."""

    _DEFAULT_NAME = "treatment_mean_absolute_error"


def _has_censoring(family: families.Family) -> bool:
    """True if the family masks units by an event indicator."""
    return family.event_indicator(tf.zeros([1, family.n_true_cols])) is not None


class _OutcomeErrorMetric(_ErrorMetric):
    """Scores the mixed outcome mean ``sum_k P(s_k | x) * E[Y | a, x, s_k]``.

    For a censored (survival) family, only units with an observed event count: the event
    indicator multiplies ``sample_weight``. A batch with no events then adds nothing, and
    `result` is NaN (not 0) while no event has been seen since the last reset.
    """

    def _set_mixture(self, mixture: mixture_lib.StateMixture) -> None:
        """Also fixes whether the outcome family is censored."""
        super()._set_mixture(mixture)
        self._censored = _has_censoring(self._mixture.outcome_family)

    def update_state(self, y_true: Array, y_pred: Array, sample_weight: Optional[Array] = None):
        """Updates the base metric with (response, mixed outcome mean), masking censored units."""
        family = self._mixture.outcome_family
        outcome = self._mixture.layout.decode_true(y_true).outcome
        event = family.event_indicator(outcome)
        if event is not None:
            event = tf.reshape(tf.cast(event, self.dtype), [-1])
            if sample_weight is None:
                sample_weight = event
            else:
                sample_weight = event * tf.reshape(tf.cast(sample_weight, self.dtype), [-1])
        return super().update_state(
            y_true=tf.cast(family.response(outcome), tf.float32),
            y_pred=self._mixture.outcome_mean(y_pred),
            sample_weight=sample_weight,
        )

    def result(self):
        """The weighted mean error; NaN for a censored family with zero event weight."""
        if not self._censored:
            return super().result()
        return tf.where(
            self.count > 0,
            tf.math.divide_no_nan(self.total, self.count),
            tf.constant(float("nan"), self.dtype),
        )


@tf.keras.utils.register_keras_serializable(package="pypsps")
class OutcomeMeanSquaredError(_OutcomeErrorMetric, tf.keras.metrics.MeanSquaredError):
    """MSE of the mixed outcome mean (events only for censored outcomes)."""

    _DEFAULT_NAME = "outcome_mean_squared_error"


@tf.keras.utils.register_keras_serializable(package="pypsps")
class OutcomeMeanAbsoluteError(_OutcomeErrorMetric, tf.keras.metrics.MeanAbsoluteError):
    """MAE of the mixed outcome mean (events only for censored outcomes)."""

    _DEFAULT_NAME = "outcome_mean_absolute_error"


def _layout_kwargs(name: str, layout: ColumnLayout) -> dict:
    """The ``fn`` kwargs of a state-weight metric; TypeError unless ``layout`` is a layout."""
    if not isinstance(layout, ColumnLayout):
        raise TypeError(
            f"{name} takes a ColumnLayout (e.g. mixture.layout), got {type(layout).__name__}; "
            "the column-count form was removed in v0.5.0"
        )
    return {"layout": layout}


@tf.keras.utils.register_keras_serializable(package="pypsps")
def predictive_state_df(y_true: Array, y_pred: Array, layout: ColumnLayout) -> tf.Tensor:
    """Degrees of freedom, trace(H), of the predictive state weights (pypress)."""
    del y_true
    weights = layout.decode_pred(y_pred).state_weights
    stats = pypress.keras.statistics
    return stats.degrees_of_freedom(
        stats.compute("trace_numerator", weights), stats.compute("trace_denominator", weights)
    )


def predictive_state_df_gen(layout: ColumnLayout) -> tf.keras.metrics.MeanMetricWrapper:
    """Metric for the degrees of freedom of the predictive states, logged as
    ``predictive_state_df``. Serializable: it wraps the registered `predictive_state_df`.

    Pass the model's `ColumnLayout` (``mixture.layout``).
    """
    return tf.keras.metrics.MeanMetricWrapper(
        fn=predictive_state_df,
        name="predictive_state_df",
        **_layout_kwargs("predictive_state_df_gen", layout),
    )


def _per_unit(loss: tf.keras.losses.Loss) -> tf.keras.losses.Loss:
    """Sets ``reduction="none"``, so a `Mean` metric weights units, not batches."""
    loss.reduction = "none"
    return loss


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
        name: str = "causal_loss_metric",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs)
        if not isinstance(outcome_loss, losses.OutcomeLoss):
            raise TypeError(f"outcome_loss must be an OutcomeLoss; got {type(outcome_loss)}")
        if not isinstance(treatment_loss, losses.TreatmentLoss):
            raise TypeError(f"treatment_loss must be a TreatmentLoss; got {type(treatment_loss)}")
        self._outcome = _per_unit(outcome_loss.penalty_free())
        self._treatment = _per_unit(treatment_loss.penalty_free())
        # No range check and no `or 1.0`: 0 is a legal weight.
        self._outcome_loss_weight = float(outcome_loss_weight)

    def update_state(self, y_true: Array, y_pred: Array, sample_weight: Optional[Array] = None):
        """Accumulates the joint NLL of each unit, weighted by ``sample_weight``."""
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


def causal_loss_metric_gen(
    outcome_loss: losses.OutcomeLoss,
    treatment_loss: losses.TreatmentLoss,
    *,
    outcome_loss_weight: float = 1.0,
) -> CausalLossMetric:
    """Returns a `CausalLossMetric`, logged as ``causal_loss_metric``.

    Parameters
    ----------
    outcome_loss : OutcomeLoss
        The outcome loss; the metric uses its penalty-free copy.
    treatment_loss : TreatmentLoss
        The treatment loss; the metric uses its penalty-free copy.
    outcome_loss_weight : float, default=1.0
        Weight of the outcome NLL. Pass the training loss's weight; 0 is allowed. The
        metric always uses alpha = 1.
    """
    return CausalLossMetric(outcome_loss, treatment_loss, outcome_loss_weight=outcome_loss_weight)


def outcome_nll_metric_gen(outcome_loss: losses.OutcomeLoss) -> tf.keras.metrics.MeanMetricWrapper:
    """Metric of the outcome negative log-likelihood alone, without penalties or alpha.

    Suitable as an early-stopping and checkpoint criterion that does not depend on the alpha
    schedule. Pass to ``model.compile(metrics=[...])``; it is logged as ``outcome_nll_metric``.
    """
    return tf.keras.metrics.MeanMetricWrapper(
        fn=_per_unit(outcome_loss.penalty_free()), name="outcome_nll_metric"
    )


@tf.keras.utils.register_keras_serializable(package="pypsps")
def uniform_entropy(y_true: Array, y_pred: Array, layout: ColumnLayout) -> tf.Tensor:
    """Mean Shannon entropy of the predictive state weights (higher = more uniform)."""
    del y_true
    weights = layout.decode_pred(y_pred).state_weights
    # Shannon entropy per row: -sum(p_i * log(p_i)), with 0 * log(0) taken as 0.
    # safe_xlogx computes that exactly; an epsilon floor (log(p_i + eps)) would distort
    # the entropy of sharp, near-degenerate state weights.
    entropy_per_row = -1 * tf.math.reduce_sum(pypress.keras.statistics.safe_xlogx(weights), axis=1)
    return tf.math.reduce_mean(entropy_per_row)


def uniform_entropy_gen(layout: ColumnLayout) -> tf.keras.metrics.MeanMetricWrapper:
    """
    Metric of the average Shannon entropy of the predictive state weights, logged as
    ``uniform_entropy``.

    The entropy is computed as:
        mean_entropy = mean(-sum(p_i * log(p_i)))

    where p_i are the predictive state weights for each state. Higher entropy indicates more
    uniform state distributions.

    Parameters
    ----------
    layout : ColumnLayout
        The model's column layout (``mixture.layout``).

    Returns
    -------
    tf.keras.metrics.MeanMetricWrapper
        A serializable metric wrapping the registered `uniform_entropy`.
    """
    return tf.keras.metrics.MeanMetricWrapper(
        fn=uniform_entropy,
        name="uniform_entropy",
        **_layout_kwargs("uniform_entropy_gen", layout),
    )
