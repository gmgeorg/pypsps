"""Module for metrics from pypsps predictions.

The propensity, treatment and outcome metrics take the model's `StateMixture`
(``metric(mixture)``): they mix each state's family mean with the state weights, and read
``y_true`` through the mixture's column layout. The state-weight metrics
(`predictive_state_df_gen`, `uniform_entropy_gen`) only need the `ColumnLayout`.

The legacy three-int constructors (``n_outcome_pred_cols``, ``n_treatment_pred_cols``,
``n_outcome_true_cols``) still work. They map the counts onto families with a shape rule: 2
parameters and 1 ``y_true`` column is `Normal`, 1 parameter is its own mean (identity), and
anything else raises, asking for a mixture.
"""

import dataclasses
import warnings
from typing import Optional

import pypress.utils
import tensorflow as tf

from . import families, losses
from . import mixture as mixture_lib
from .layout import ColumnLayout

_ser = tf.keras.utils.serialize_keras_object
_deser = tf.keras.utils.deserialize_keras_object


@dataclasses.dataclass(frozen=True)
class _Identity(families.Family):
    """Legacy shape rule only: one parameter per state that is its own mean.

    Not registered: metrics built from ints serialize the ints, never this family.
    """

    n_true_cols: int = 1

    @property
    def params(self) -> tuple[families.ParamSpec, ...]:
        """One linear parameter."""
        return (families.ParamSpec("value"),)

    def nll(self, y_true, params):
        """Raises: the identity has no likelihood."""
        raise NotImplementedError("the legacy identity family has no likelihood")

    def mean(self, params):
        """The parameter itself."""
        return tf.cast(params["value"], tf.float32)


def _shape_rule(n_params: int, n_true_cols: int, role: str) -> families.Family:
    """Maps legacy counts onto a family: (2, 1) is Normal, 1 parameter is the identity."""
    n_params, n_true_cols = int(n_params), int(n_true_cols)
    if n_params == 2 and n_true_cols == 1:
        return families.Normal()
    if n_params == 1:
        return _Identity(n_true_cols=n_true_cols)
    raise ValueError(
        f"the {role} mean is ambiguous for {n_params} parameters per state and {n_true_cols} "
        f"y_true columns; pass the model's StateMixture as mixture= (e.g. model.loss.mixture)"
    )


def _placeholder(n_params: int, n_true_cols: int = 1) -> families.LossFamily:
    """A family the metric never evaluates; it only fixes the column count."""
    return families.LossFamily(None, int(n_params), int(n_true_cols))


class _MixtureMetricMixin:
    """Stores the mixture and serializes it; the legacy three-int form is kept as is.

    ``_LEGACY_COUNTS`` names the three-int arguments a subclass accepts. A metric built from
    ints keeps them in ``_legacy_counts`` and writes them (not the mixture) to its config,
    so it reloads through the same shape rule.
    """

    _DEFAULT_NAME: str
    _LEGACY_COUNTS: tuple[str, ...]

    def _set_mixture(self, mixture, **counts) -> None:
        """Takes the mixture, or builds one from the legacy counts."""
        if isinstance(mixture, mixture_lib.StateMixture):
            if any(v is not None for v in counts.values()):
                raise TypeError("pass either a StateMixture or the legacy column counts, not both")
            self._mixture = mixture
            self._legacy_counts = None
            self._check_mixture(mixture)
            return
        if mixture is not None:  # legacy positional: the first count
            if counts["n_outcome_pred_cols"] is not None:
                raise TypeError("n_outcome_pred_cols given twice")
            counts["n_outcome_pred_cols"] = mixture
        missing = [k for k in self._LEGACY_COUNTS if counts.get(k) is None]
        if missing:
            raise TypeError(
                f"{type(self).__name__} needs a StateMixture (or the legacy "
                f"{', '.join(self._LEGACY_COUNTS)}); missing {', '.join(missing)}"
            )
        self._legacy_counts = {k: int(counts[k]) for k in self._LEGACY_COUNTS}
        self._mixture = self._legacy_mixture(**self._legacy_counts)

    def _check_mixture(self, mixture: mixture_lib.StateMixture) -> None:
        """Subclasses may reject a mixture they can't score."""

    @property
    def mixture(self) -> mixture_lib.StateMixture:
        """The state mixture the metric reads."""
        return self._mixture

    def get_config(self) -> dict:
        """The base metric's config plus the serialized mixture (or the legacy counts)."""
        config = super().get_config()
        if self._legacy_counts is not None:
            config.update(self._legacy_counts)
        else:
            config["mixture"] = _ser(self._mixture)
        return config

    @classmethod
    def from_config(cls, config: dict):
        """Rebuilds the metric; a config without ``"mixture"`` goes through the shape rule."""
        config = dict(config)
        if "mixture" in config:
            config["mixture"] = _deser(config["mixture"])
        return cls(**config)


class _PropensityMetric(_MixtureMetricMixin):
    """Scores the marginal propensity ``sum_k P(s_k | x) * p_k`` against the treatment."""

    _LEGACY_COUNTS = ("n_outcome_pred_cols", "n_treatment_pred_cols")

    def __init__(
        self,
        mixture: Optional[mixture_lib.StateMixture] = None,
        n_treatment_pred_cols: Optional[int] = None,
        name: Optional[str] = None,
        *,
        n_outcome_pred_cols: Optional[int] = None,
        **kwargs,
    ):
        """Initializes the metric.

        Args:
          mixture: the model's `StateMixture`; its treatment family must be binary.
          n_treatment_pred_cols: legacy form only: treatment parameters per state.
          name: the metric name; defaults to the class's name.
          n_outcome_pred_cols: legacy form only: outcome parameters per state (or pass it as
            the first argument).
          **kwargs: passed to the Keras base metric (``dtype``, ``curve``, ...).
        """
        super().__init__(name=name or self._DEFAULT_NAME, **kwargs)
        self._set_mixture(
            mixture,
            n_outcome_pred_cols=n_outcome_pred_cols,
            n_treatment_pred_cols=n_treatment_pred_cols,
        )

    @staticmethod
    def _legacy_mixture(n_outcome_pred_cols, n_treatment_pred_cols):
        """Placeholder outcome; the treatment family from the shape rule."""
        return mixture_lib.StateMixture(
            _placeholder(n_outcome_pred_cols),
            _shape_rule(n_treatment_pred_cols, 1, "treatment"),
        )

    def _check_mixture(self, mixture):
        """Raises unless the treatment is binary."""
        if not mixture.treatment_family.is_binary:
            raise ValueError(
                f"{type(self).__name__} needs a binary treatment family, got "
                f"{mixture.treatment_family!r}"
            )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates the base metric with (treatment, marginal propensity)."""
        if self._legacy_counts is not None:
            treatment = y_true[:, -1:]  # the legacy form doesn't know the outcome columns
        else:
            treatment = self._mixture.layout.decode_true(y_true).treatment
        return super().update_state(
            y_true=treatment,
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

    _LEGACY_COUNTS = ("n_outcome_pred_cols", "n_treatment_pred_cols", "n_outcome_true_cols")

    def __init__(
        self,
        mixture: Optional[mixture_lib.StateMixture] = None,
        n_treatment_pred_cols: Optional[int] = None,
        n_outcome_true_cols: Optional[int] = None,
        name: Optional[str] = None,
        *,
        n_outcome_pred_cols: Optional[int] = None,
        **kwargs,
    ):
        """Initializes the metric.

        Args:
          mixture: the model's `StateMixture`.
          n_treatment_pred_cols: legacy form only: treatment parameters per state.
          n_outcome_true_cols: legacy form only: outcome columns in y_true.
          name: the metric name; defaults to the class's name.
          n_outcome_pred_cols: legacy form only: outcome parameters per state (or pass it as
            the first argument).
          **kwargs: passed to the Keras base metric (``dtype``).
        """
        super().__init__(name=name or self._DEFAULT_NAME, **kwargs)
        self._set_mixture(
            mixture,
            n_outcome_pred_cols=n_outcome_pred_cols,
            n_treatment_pred_cols=n_treatment_pred_cols,
            n_outcome_true_cols=n_outcome_true_cols,
        )


class _TreatmentErrorMetric(_ErrorMetric):
    """Scores the mixed treatment mean ``sum_k P(s_k | x) * E[A | s_k]``."""

    @staticmethod
    def _legacy_mixture(n_outcome_pred_cols, n_treatment_pred_cols, n_outcome_true_cols):
        """Placeholder outcome; the treatment family from the shape rule."""
        return mixture_lib.StateMixture(
            _placeholder(n_outcome_pred_cols, n_outcome_true_cols),
            _shape_rule(n_treatment_pred_cols, 1, "treatment"),
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
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

    def _set_mixture(self, mixture, **counts) -> None:
        """Also fixes whether the outcome family is censored."""
        super()._set_mixture(mixture, **counts)
        self._censored = _has_censoring(self._mixture.outcome_family)

    @staticmethod
    def _legacy_mixture(n_outcome_pred_cols, n_treatment_pred_cols, n_outcome_true_cols):
        """The outcome family from the shape rule; placeholder treatment."""
        return mixture_lib.StateMixture(
            _shape_rule(n_outcome_pred_cols, n_outcome_true_cols, "outcome"),
            _placeholder(n_treatment_pred_cols),
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
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


def _state_weights(y_pred, layout, n_outcome_pred_cols, n_treatment_pred_cols) -> tf.Tensor:
    """The ``[N, K]`` state weights, via ``layout`` or (legacy) the two counts."""
    if layout is None:
        layout = ColumnLayout.from_counts(n_outcome_pred_cols, n_treatment_pred_cols, 1)
    return layout.decode_pred(y_pred).state_weights


def _layout_kwargs(name, layout, n_outcome_pred_cols, n_treatment_pred_cols) -> dict:
    """The ``fn`` kwargs of a state-weight metric: ``layout=``, or the legacy two ints."""
    if isinstance(layout, ColumnLayout):
        if n_outcome_pred_cols is not None or n_treatment_pred_cols is not None:
            raise TypeError(f"{name}: pass either a ColumnLayout or the column counts, not both")
        return {"layout": layout}
    if layout is not None:  # legacy positional: the first count
        if n_outcome_pred_cols is not None:
            raise TypeError(f"{name}: n_outcome_pred_cols given twice")
        n_outcome_pred_cols = layout
    if n_outcome_pred_cols is None or n_treatment_pred_cols is None:
        raise TypeError(f"{name} needs a ColumnLayout (e.g. mixture.layout)")
    return {
        "n_outcome_pred_cols": int(n_outcome_pred_cols),
        "n_treatment_pred_cols": int(n_treatment_pred_cols),
    }


@tf.keras.utils.register_keras_serializable(package="pypsps")
def predictive_state_df(
    y_true,
    y_pred,
    layout: Optional[ColumnLayout] = None,
    n_outcome_pred_cols: Optional[int] = None,
    n_treatment_pred_cols: Optional[int] = None,
) -> tf.Tensor:
    """Degrees of freedom of the predictive state weights (`pypress.utils.tr_kernel`).

    Pass ``layout``; the two counts are the legacy form.
    """
    del y_true
    weights = _state_weights(y_pred, layout, n_outcome_pred_cols, n_treatment_pred_cols)
    return pypress.utils.tr_kernel(weights)


def predictive_state_df_gen(
    layout: Optional[ColumnLayout] = None,
    n_treatment_pred_cols: Optional[int] = None,
    *,
    n_outcome_pred_cols: Optional[int] = None,
) -> tf.keras.metrics.MeanMetricWrapper:
    """Metric for the degrees of freedom of the predictive states, logged as
    ``predictive_state_df``. Serializable: it wraps the registered `predictive_state_df`.

    Pass the model's `ColumnLayout` (``mixture.layout``). The legacy form
    ``predictive_state_df_gen(n_outcome_pred_cols, n_treatment_pred_cols)`` still works.
    """
    return tf.keras.metrics.MeanMetricWrapper(
        fn=predictive_state_df,
        name="predictive_state_df",
        **_layout_kwargs(
            "predictive_state_df_gen", layout, n_outcome_pred_cols, n_treatment_pred_cols
        ),
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
    y_true,
    y_pred,
    layout: Optional[ColumnLayout] = None,
    n_outcome_pred_cols: Optional[int] = None,
    n_treatment_pred_cols: Optional[int] = None,
) -> tf.Tensor:
    """Mean Shannon entropy of the predictive state weights (higher = more uniform).

    Pass ``layout``; the two counts are the legacy form.
    """
    del y_true
    weights = _state_weights(y_pred, layout, n_outcome_pred_cols, n_treatment_pred_cols)
    # Shannon entropy per row: -sum(p_i * log(p_i)), with 0 * log(0) taken as 0.
    # safe_xlogx computes that exactly; an epsilon floor (log(p_i + eps)) would distort
    # the entropy of sharp, near-degenerate state weights.
    entropy_per_row = -1 * tf.math.reduce_sum(pypress.utils.safe_xlogx(weights), axis=1)
    return tf.math.reduce_mean(entropy_per_row)


def uniform_entropy_gen(
    layout: Optional[ColumnLayout] = None,
    n_treatment_pred_cols: Optional[int] = None,
    *,
    n_outcome_pred_cols: Optional[int] = None,
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
    layout : ColumnLayout
        The model's column layout (``mixture.layout``). The legacy form passes
        ``n_outcome_pred_cols`` and ``n_treatment_pred_cols`` instead.

    Returns
    -------
    tf.keras.metrics.MeanMetricWrapper
        A serializable metric wrapping the registered `uniform_entropy`.
    """
    return tf.keras.metrics.MeanMetricWrapper(
        fn=uniform_entropy,
        name="uniform_entropy",
        **_layout_kwargs("uniform_entropy_gen", layout, n_outcome_pred_cols, n_treatment_pred_cols),
    )
