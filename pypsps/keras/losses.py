"""Module for pypsps losses."""

import warnings
from typing import Any, Callable, Optional, Union

import tensorflow as tf

from . import mixture as mixture_lib
from .layout import ColumnLayout
from .mixture import OutcomeLikelihood  # re-exported; defined in mixture

_ser = tf.keras.utils.serialize_keras_object
_deser = tf.keras.utils.deserialize_keras_object

_LEGACY_CONFIG = (
    "{cls} config has no 'mixture': it was saved by pypsps v0.3.0-v0.3.3 with the column-count "
    "constructor, which v0.5.0 removed. Load the model with compile=False, rebuild it with "
    "pypsps.keras.models.build_model and copy the weights (binary and toy models need "
    "models.copy_legacy_weights of pypsps v0.4.x); see 'Models saved before v0.5.0' in "
    "docs/model_building.md."
)


def _require_mixture(mixture: Any, cls: str) -> mixture_lib.StateMixture:
    """Returns ``mixture``, or raises TypeError unless it is a `StateMixture`."""
    if not isinstance(mixture, mixture_lib.StateMixture):
        raise TypeError(
            f"{cls} takes a StateMixture, got {type(mixture).__name__}; the column-count "
            "constructor was removed in v0.5.0"
        )
    return mixture


def _mixture_from_config(config: dict, cls: str) -> dict:
    """A copy of ``config`` with the mixture deserialized; ValueError for a legacy config."""
    if "mixture" not in config:
        raise ValueError(_LEGACY_CONFIG.format(cls=cls))
    config = dict(config)
    config["mixture"] = _deser(config["mixture"])
    return config


class _MixtureLossMixin:
    """Shared state of `OutcomeLoss` and `TreatmentLoss`: the mixture and its layout."""

    @property
    def mixture(self) -> mixture_lib.StateMixture:
        """The state mixture the loss evaluates."""
        return self._mixture

    @property
    def layout(self) -> ColumnLayout:
        """The column layout of ``y_pred`` and ``y_true``."""
        return self._mixture.layout


@tf.keras.utils.register_keras_serializable(package="pypsps")
class OutcomeLoss(_MixtureLossMixin, tf.keras.losses.Loss):
    """Outcome loss of a pypsps model: the negative log-likelihood of a mixture over states.

        -log sum_k gamma_k * p(y | s_k, a, x)

    By default (``outcome_likelihood="interventional"``) gamma_k = P(s_k | x), the prior
    predictive state weights, which equal P(s_k | x, do(a)) since the states precede
    treatment. The loss is then the interventional likelihood -log p(y | do(a), x).

    With ``outcome_likelihood="observational"``, gamma_k = P(s_k | x, a) is the *posterior*
    state probability, ie the prior weights updated with the evidence from the observed
    treatment `a` via Bayes' rule. The loss is then -log p(y | a, x), and -log p(a, y | x)
    telescopes exactly into TreatmentLoss (-log p(a | x), using the prior) plus OutcomeLoss.
    See `OutcomeLikelihood` and `pypsps.keras.mixture.StateMixture.outcome_nll`.
    """

    def __init__(
        self,
        mixture: mixture_lib.StateMixture,
        outcome_likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
        **kwargs: Any,
    ):
        """Initializes the outcome loss.

        Args:
          mixture: the `StateMixture` to evaluate.
          outcome_likelihood: `OutcomeLikelihood` member or its string value
            ("interventional", the default, or "observational").
          **kwargs: passed to the Keras Loss (``name``, ``reduction``).

        Raises:
          TypeError: if ``mixture`` is not a `StateMixture`.
        """
        super().__init__(**kwargs)
        self._mixture = _require_mixture(mixture, "OutcomeLoss")
        self._outcome_likelihood = OutcomeLikelihood(outcome_likelihood)

    @property
    def outcome_likelihood(self) -> OutcomeLikelihood:
        """Which outcome likelihood the loss evaluates."""
        return self._outcome_likelihood

    def get_config(self) -> dict:
        """Returns the config: the serialized mixture and the outcome likelihood."""
        return {
            **super().get_config(),
            "mixture": _ser(self._mixture),
            "outcome_likelihood": self._outcome_likelihood.value,
        }

    @classmethod
    def from_config(cls, config: dict) -> "OutcomeLoss":
        """Rebuilds the loss.

        Raises:
          ValueError: for a config without a mixture (saved by v0.3.0-v0.3.3).
        """
        return cls(**_mixture_from_config(config, cls.__name__))

    def penalty_free(self) -> "OutcomeLoss":
        """Returns a fresh instance of this exact class with no penalty terms applied.

        Rebuilds `type(self)` from `OutcomeLoss.get_config(self)`, the *base* class's config,
        never from `self.get_config()` or by scanning attributes by naming convention. That is
        the contract for stripping penalties out of a metric like `CausalLossMetric`: a
        subclass whose `call()` adds a penalty (e.g. a within-state balance term) may
        serialize it in its own `get_config`, and it still won't reach the penalty-free copy.
        A penalty argument with an "off" default falls back to off; a required one raises. A
        subclass that adds a *non-penalty* constructor argument must override this method.
        The copy keeps the mixture.
        """
        return type(self).from_config(OutcomeLoss.get_config(self))

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """The outcome mixture NLL per unit, ``[N]``; see `StateMixture.outcome_nll`.

        Keras then weights each unit by its ``sample_weight`` and applies ``reduction``.
        """
        return self._mixture.outcome_nll(y_true, y_pred, self._outcome_likelihood)


@tf.keras.utils.register_keras_serializable(package="pypsps")
class TreatmentLoss(_MixtureLossMixin, tf.keras.losses.Loss):
    """Treatment (dose) loss of a pypsps model: -log p(a | x).

    The treatment predictions in y_pred are state-conditional, so the treatment loss is the
    negative log-likelihood of a mixture over states with the *prior* state weights:

        -log p(a | x) = -log sum_k P(s_k | x) * p(a | s_k, x)

    This is the term that telescopes with OutcomeLoss's posterior-weighted mixture into the
    exact joint -log p(a, y | x).
    """

    def __init__(self, mixture: mixture_lib.StateMixture, **kwargs: Any):
        """Initializes the treatment loss.

        Args:
          mixture: the `StateMixture` to evaluate.
          **kwargs: passed to the Keras Loss (``name``, ``reduction``).

        Raises:
          TypeError: if ``mixture`` is not a `StateMixture`.
        """
        super().__init__(**kwargs)
        self._mixture = _require_mixture(mixture, "TreatmentLoss")

    def get_config(self) -> dict:
        """Returns the config: the serialized mixture."""
        return {**super().get_config(), "mixture": _ser(self._mixture)}

    @classmethod
    def from_config(cls, config: dict) -> "TreatmentLoss":
        """Rebuilds the loss.

        Raises:
          ValueError: for a config without a mixture (saved by v0.3.0-v0.3.3).
        """
        return cls(**_mixture_from_config(config, cls.__name__))

    def penalty_free(self) -> "TreatmentLoss":
        """Returns a fresh instance of this exact class with no penalty terms applied.

        Rebuilds `type(self)` from `TreatmentLoss.get_config(self)`, the base class's config.
        See `OutcomeLoss.penalty_free` for the contract.
        """
        return type(self).from_config(TreatmentLoss.get_config(self))

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """The treatment mixture NLL per unit, ``[N]``; see `StateMixture.treatment_nll`.

        Keras then weights each unit by its ``sample_weight`` and applies ``reduction``.
        """
        return self._mixture.treatment_nll(y_true, y_pred)


def _link_mixtures(outcome_loss: "OutcomeLoss", treatment_loss: "TreatmentLoss") -> None:
    """Checks the two losses have equal mixtures, then makes them share the outcome one.

    Raises:
      ValueError: if the mixtures differ: the outcome loss's state posterior would then use a
        different treatment likelihood than the treatment loss.
    """
    o_mix, t_mix = outcome_loss.mixture, treatment_loss.mixture
    if o_mix != t_mix:
        raise ValueError(
            f"outcome_loss and treatment_loss must share one StateMixture; got {o_mix!r} "
            f"and {t_mix!r}. Build both from one mixture (CausalLoss.from_mixture)."
        )
    treatment_loss._mixture = o_mix


def _state_penalty(regularizer: Callable, weights: tf.Tensor) -> tf.Tensor:
    """The amount ``regularizer`` adds to the loss for a batch of state weights.

    pypress regularizers (>= 0.5.2) are activity regularizers: calling one returns its penalty
    times the batch size, to cancel the division Keras applies to activity losses. `CausalLoss`
    adds the penalty to the loss itself, with no such division, so it uses ``penalty()``.
    Any other regularizer (e.g. a built-in Keras one) is called.
    """
    penalty = getattr(regularizer, "penalty", None)
    return penalty(weights) if callable(penalty) else regularizer(weights)


@tf.keras.utils.register_keras_serializable(package="pypsps")
class CausalLoss(tf.keras.losses.Loss):
    """PSPS causal loss is the sum of outcome loss + treatment loss.

    Causal loss from PSPS is based on the joint distribution P(outcome, treatment | features)
    which decomposes into

        Pr(Y, T | X) = Pr(Y | T, X) * Pr(T | X)

    which in log-likelihood terms is

        loglik(Y, T; X) = loglik(Y; T, X) + alpha * loglik(T; X)

    where alpha = 1 (by default). See Eq (10) in
    https://proceedings.mlr.press/v177/kelly22a/kelly22a.pdf
    for details (in paper lambda == alpha).

    Note on alpha=0: this zeroes only the explicit `loglik(T; X)` term (`treatment_loss`).
    The outcome loss still mixes with the posterior state weights under
    `outcome_likelihood="observational"`, which depend on the treatment/dose head (see
    `OutcomeLoss`, `StateMixture.state_posterior`), so alpha=0 is not an "outcome-only"
    ablation.
    """

    def __init__(
        self,
        outcome_loss: OutcomeLoss,
        treatment_loss: TreatmentLoss,
        alpha: Union[float, tf.Variable] = 1.0,
        outcome_loss_weight: float = 1.0,
        predictive_states_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ):
        """Initializes the causal loss class.

        Args:
            outcome_loss: instance of an outcome loss; defaults to a Normal log-likelihood.
            treatment_loss: instance of a treatment loss; defaults to binary treatment loss
              (ie binary cross entropy).
            alpha: weight of the treatment log-likelihood in the causal loss (λ in Eq. 10
              of the paper). Defaults to 1.0 so that total causal loss equals the joint
              log-likelihood. Values above 1 act as a penalty that prioritizes fitting
              treatment over fitting outcome: training then learns the propensity and the
              predictive states first, and refines the outcome models afterwards. A float32
              `tf.Variable` is used as is (shared, not copied).
            outcome_loss_weight: weight of outcome loss; defaults to 1.0.
            predictive_states_regularizer: optional; user can define a predictive
              state regularizer.
        """
        super().__init__(**kwargs)
        if not isinstance(outcome_loss, OutcomeLoss):
            raise TypeError(f"outcome_loss must be an OutcomeLoss, got {type(outcome_loss)}")
        if not isinstance(treatment_loss, TreatmentLoss):
            raise TypeError(f"treatment_loss must be a TreatmentLoss, got {type(treatment_loss)}")
        _link_mixtures(outcome_loss, treatment_loss)

        self._outcome_loss = outcome_loss
        self._treatment_loss = treatment_loss
        # A tf.Variable, not a plain float: `call()` is traced into model.fit()'s compiled
        # training function (run_eagerly=False by default). A plain float gets baked into that
        # graph as a constant at the first trace; a callback rebinding it afterward (e.g.
        # AlphaScheduleCallback) would then be updating an eager Python attribute that the
        # traced graph never reads again. A tf.Variable's value is re-read via ReadVariableOp on
        # every graph execution, so assigning it between epochs actually reaches the optimizer.
        # The `alpha` property's setter assigns; it never rebinds.
        if isinstance(alpha, tf.Variable):
            if alpha.dtype != tf.float32:
                raise ValueError(f"alpha variable must be float32, got {alpha.dtype}")
            self._alpha_var = alpha
        else:
            self._alpha_var = tf.Variable(
                float(alpha), trainable=False, dtype=tf.float32, name="alpha"
            )
        self._outcome_loss_weight = outcome_loss_weight
        self._predictive_states_regularizer = predictive_states_regularizer
        self._update_loss_reduction()

    @classmethod
    def from_mixture(
        cls,
        mixture: mixture_lib.StateMixture,
        alpha: Union[float, tf.Variable] = 1.0,
        outcome_likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
        outcome_loss_weight: float = 1.0,
        predictive_states_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        reduction: str = "sum_over_batch_size",
        **kwargs: Any,
    ) -> "CausalLoss":
        """A causal loss whose outcome and treatment losses share one `StateMixture`."""
        return cls(
            outcome_loss=OutcomeLoss(
                mixture, outcome_likelihood=outcome_likelihood, reduction=reduction
            ),
            treatment_loss=TreatmentLoss(mixture, reduction=reduction),
            alpha=alpha,
            outcome_loss_weight=outcome_loss_weight,
            predictive_states_regularizer=predictive_states_regularizer,
            reduction=reduction,
            **kwargs,
        )

    @property
    def mixture(self) -> mixture_lib.StateMixture:
        """The state mixture of the outcome loss (shared with the treatment loss)."""
        return self._outcome_loss.mixture

    @property
    def layout(self) -> ColumnLayout:
        """The column layout of ``y_pred`` and ``y_true``."""
        return self.mixture.layout

    @property
    def outcome_loss(self) -> OutcomeLoss:
        """The outcome loss (read-only)."""
        return self._outcome_loss

    @property
    def treatment_loss(self) -> TreatmentLoss:
        """The treatment loss (read-only)."""
        return self._treatment_loss

    @property
    def outcome_loss_weight(self) -> float:
        """The weight of the outcome loss (read-only)."""
        return self._outcome_loss_weight

    @property
    def alpha(self) -> tf.Variable:
        """The float32 variable holding alpha.

        Read it with ``float(loss.alpha)``; share it by passing it to another `CausalLoss`;
        change it with ``loss.alpha = x`` or ``loss.alpha.assign(x)``.
        """
        return self._alpha_var

    @alpha.setter
    def alpha(self, value: float) -> None:
        if isinstance(value, tf.Variable):
            raise TypeError(
                "Pass a shared alpha variable to the constructor, CausalLoss(alpha=var); "
                "assigning one later would leave the traced training graph reading the old one."
            )
        self._alpha_var.assign(float(value))

    def get_config(self) -> dict:
        """Returns the config, with the *current* alpha value and serialized nested objects."""
        reg = self._predictive_states_regularizer
        return {
            **super().get_config(),
            "outcome_loss": _ser(self._outcome_loss),
            "treatment_loss": _ser(self._treatment_loss),
            "alpha": float(self._alpha_var.numpy()),
            "outcome_loss_weight": float(self._outcome_loss_weight),
            "predictive_states_regularizer": None if reg is None else _ser(reg),
        }

    @classmethod
    def from_config(cls, config: dict) -> "CausalLoss":
        """Rebuilds the loss; alpha becomes a fresh float32 variable with the saved value."""
        config = dict(config)
        for key in ("outcome_loss", "treatment_loss", "predictive_states_regularizer"):
            if config.get(key) is not None:
                config[key] = _deser(config[key])
        return cls(**config)

    def _update_loss_reduction(self):
        """Updates loss reduction of outcome & treatment according to causal reduction loss."""
        if self._treatment_loss.reduction != self.reduction:
            warnings.warn(
                "Setting 'reduction' of treatment loss to user-specified reduction: '%s'."
                % self.reduction,
            )
            self._treatment_loss.reduction = self.reduction

        if self._outcome_loss.reduction != self.reduction:
            warnings.warn(
                "Setting 'reduction' of outcome loss to user-specified reduction: '%s'."
                % self.reduction,
            )
            self._outcome_loss.reduction = self.reduction

    def __call__(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """The reduced causal loss, plus the predictive-state regularizer.

        Keras weights each unit's loss from `call` by ``sample_weight`` and reduces it. The
        regularizer is a penalty on the batch's state weights, not a per-unit loss, so it is
        added once, after the reduction, and ``sample_weight`` does not scale it. Under
        ``reduction="none"`` it is added to every unit.
        """
        loss = super().__call__(y_true, y_pred, sample_weight=sample_weight)
        if self._predictive_states_regularizer is None:
            return loss
        weights = self.layout.decode_pred(tf.convert_to_tensor(y_pred, loss.dtype)).state_weights
        return loss + tf.cast(
            _state_penalty(self._predictive_states_regularizer, weights), loss.dtype
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """The causal loss per unit, ``[N]``, without the regularizer (see `__call__`)."""
        loss_outcome = self._outcome_loss.call(y_true, y_pred)
        loss_treatment = self._treatment_loss.call(y_true, y_pred)
        return self._outcome_loss_weight * loss_outcome + self._alpha_var * loss_treatment
