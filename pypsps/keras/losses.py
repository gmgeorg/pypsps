"""Module for pypsps losses."""

import warnings
from typing import Any, Optional, Union

import tensorflow as tf

from .. import utils
from . import families
from . import mixture as mixture_lib
from . import neglogliks
from .layout import ColumnLayout
from .mixture import OutcomeLikelihood  # re-exported; defined in mixture

_ser = tf.keras.utils.serialize_keras_object
_deser = tf.keras.utils.deserialize_keras_object


def _gen_col_selectors(x: int, n: int, k: int) -> list:
    return utils.get_state_column_indices(x, n, k)


def prob_state_given_treatment_features(
    weights: tf.Tensor,  # [N,K] = P(S|X)
    treatment_true: tf.Tensor,  # [N,1] (or [N])
    treatment_pred: tf.Tensor,  # [N,K*P] interleaved per-state treatment params
    treatment_nll_loss: tf.keras.losses.Loss,
) -> tf.Tensor:
    """Computes gamma = P(S | X, A) using any proper elementwise NLL treatment loss.

    Applies Bayes' rule: gamma_k(a, x) ∝ P(s_k | x) * p(a | s_k, x), where the prior
    P(s_k | x) is `weights` and p(a | s_k, x) is derived from `treatment_nll_loss` evaluated
    on the state-conditional treatment predictions.

    Requirements:
      - treatment_pred has shape [N, K*P] interleaved per-state and matches what
        treatment_nll_loss expects as y_pred (P params per state).
      - treatment_nll_loss.reduction == NONE and returns elementwise NLL.

    Args:
        weights: Prior state probabilities [N,K]
        treatment_true: True treatment values [N,1] or [N]
        treatment_pred: Predicted treatment parameters [N, K*P] in interleaved format
        treatment_nll_loss: NLL loss function with reduction=NONE

    Returns:
      gamma: [N,K]
    """
    weights = tf.convert_to_tensor(weights)
    treatment_true = tf.convert_to_tensor(treatment_true)
    treatment_pred = tf.convert_to_tensor(treatment_pred)

    if not isinstance(treatment_nll_loss, tf.keras.losses.Loss):
        raise TypeError("treatment_nll_loss must be a tf.keras.losses.Loss instance.")
    if treatment_nll_loss.reduction != tf.keras.losses.Reduction.NONE:
        raise ValueError("treatment_nll_loss must have reduction=NONE.")

    # Ensure treatment_true is [N,1]
    if treatment_true.shape.rank == 1:
        treatment_true = treatment_true[:, None]

    n_states = utils.get_n_cols(weights)

    # Per-state NLL: [N,K]
    nll = neglogliks.negloglik_per_state(
        negloglik=treatment_nll_loss,
        y_true=treatment_true,
        y_pred=treatment_pred,
        n_states=n_states,
    )

    gamma = neglogliks.posterior_from_negloglik_per_state(weights, nll)
    return gamma


_ALPHA_DEPRECATION = (
    "CausalLoss._alpha is deprecated since v0.3.0 and will be removed in v0.4.0; "
    "use CausalLoss.alpha"
)


def _check_elementwise(loss: Any, what: str) -> None:
    """Raises unless ``loss`` is a Keras loss with ``reduction="none"``."""
    if not isinstance(loss, tf.keras.losses.Loss):
        raise TypeError(f"{what} must be a tf.keras.losses.Loss, got {type(loss).__name__}")
    if loss.reduction != tf.keras.losses.Reduction.NONE:
        raise ValueError(f"{what} must have reduction='none', got {loss.reduction!r}")


def _require_counts(**counts: Optional[int]) -> None:
    """Raises if any of the legacy three-int arguments is missing."""
    missing = [name for name, value in counts.items() if value is None]
    if missing:
        raise TypeError(
            f"the legacy loss constructor needs {', '.join(missing)}; or pass a StateMixture"
        )


class _MixtureLossMixin:
    """Shared state of `OutcomeLoss` and `TreatmentLoss`: the mixture, and the legacy view.

    ``_legacy`` is True for a loss built with the three-int constructor. Such a loss also
    keeps the elementwise NLLs it was given as ``_loss`` (and ``_treatment_loss``).
    """

    @property
    def mixture(self) -> mixture_lib.StateMixture:
        """The state mixture the loss evaluates."""
        return self._mixture

    @property
    def layout(self) -> ColumnLayout:
        """The column layout of ``y_pred`` and ``y_true``."""
        return self._mixture.layout

    # Legacy three-int view, read by metrics and utils until they take the mixture.
    @property
    def _n_outcome_true_cols(self) -> int:
        """C."""
        return self._mixture.layout.n_outcome_true_cols

    @property
    def _n_outcome_pred_cols(self) -> int:
        """Outcome parameters per state."""
        return self._mixture.layout.n_outcome_pred_cols

    @property
    def _n_treatment_pred_cols(self) -> int:
        """Treatment parameters per state."""
        return self._mixture.layout.n_treatment_pred_cols


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

    Construct it from a mixture, ``OutcomeLoss(mixture, outcome_likelihood=...)``. The legacy
    form ``OutcomeLoss(loss, treatment_loss, n_outcome_true_cols, n_outcome_pred_cols,
    n_treatment_pred_cols)`` still works: it maps the elementwise NLLs onto families with
    `families.from_negloglik`.
    """

    def __init__(
        self,
        mixture: Optional[mixture_lib.StateMixture] = None,
        treatment_loss: Optional[tf.keras.losses.Loss] = None,
        n_outcome_true_cols: Optional[int] = None,
        n_outcome_pred_cols: Optional[int] = None,
        n_treatment_pred_cols: Optional[int] = None,
        outcome_likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
        *,
        loss: Optional[tf.keras.losses.Loss] = None,
        **kwargs: Any,
    ):
        """Initializes the outcome loss.

        Args:
          mixture: the `StateMixture` to evaluate. In the legacy form, the elementwise outcome
            NLL instead (or pass it as ``loss=``).
          treatment_loss: legacy form only: the elementwise treatment NLL (``reduction="none"``),
            used for the state posterior.
          n_outcome_true_cols: legacy form only: outcome columns in y_true.
          n_outcome_pred_cols: legacy form only: outcome parameters per state.
          n_treatment_pred_cols: legacy form only: treatment parameters per state.
          outcome_likelihood: `OutcomeLikelihood` member or its string value
            ("interventional", the default, or "observational").
          loss: legacy form only: the elementwise outcome NLL (``reduction="none"``).
          **kwargs: passed to the Keras Loss (``name``, ``reduction``).
        """
        if isinstance(mixture, mixture_lib.StateMixture) and isinstance(
            treatment_loss, (str, OutcomeLikelihood)
        ):  # OutcomeLoss(mixture, "observational")
            outcome_likelihood, treatment_loss = treatment_loss, None
        super().__init__(**kwargs)
        self._outcome_likelihood = OutcomeLikelihood(outcome_likelihood)
        legacy = (loss, treatment_loss, n_outcome_true_cols, n_outcome_pred_cols,
                  n_treatment_pred_cols)  # fmt: skip
        if isinstance(mixture, mixture_lib.StateMixture):
            if any(v is not None for v in legacy):
                raise TypeError("pass either a StateMixture or the legacy loss arguments, not both")
            self._mixture = mixture
            self._legacy = False
            return
        loss = mixture if loss is None else loss
        _require_counts(
            loss=loss,
            treatment_loss=treatment_loss,
            n_outcome_true_cols=n_outcome_true_cols,
            n_outcome_pred_cols=n_outcome_pred_cols,
            n_treatment_pred_cols=n_treatment_pred_cols,
        )
        _check_elementwise(loss, "loss")
        _check_elementwise(treatment_loss, "treatment_loss")
        self._mixture = mixture_lib.StateMixture(
            families.from_negloglik(loss, n_outcome_pred_cols, n_outcome_true_cols),
            families.from_negloglik(treatment_loss, n_treatment_pred_cols, 1),
        )
        self._legacy = True
        self._loss = loss
        self._treatment_loss = treatment_loss

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
        """Rebuilds the loss; a v0.3.0 three-int config goes through the legacy constructor."""
        config = dict(config)
        if "mixture" in config:
            config["mixture"] = _deser(config["mixture"])
        else:
            config["loss"] = _deser(config["loss"])
            config["treatment_loss"] = _deser(config["treatment_loss"])
        return cls(**config)

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

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """The reduced outcome mixture NLL; see `StateMixture.outcome_nll`."""
        nll = self._mixture.outcome_nll(y_true, y_pred, self._outcome_likelihood)
        return mixture_lib.reduce(nll, self.reduction)


@tf.keras.utils.register_keras_serializable(package="pypsps")
class TreatmentLoss(_MixtureLossMixin, tf.keras.losses.Loss):
    """Treatment (dose) loss of a pypsps model: -log p(a | x).

    The treatment predictions in y_pred are state-conditional, so the treatment loss is the
    negative log-likelihood of a mixture over states with the *prior* state weights:

        -log p(a | x) = -log sum_k P(s_k | x) * p(a | s_k, x)

    This is the term that telescopes with OutcomeLoss's posterior-weighted mixture into the
    exact joint -log p(a, y | x).

    Construct it from a mixture, ``TreatmentLoss(mixture)``. The legacy form
    ``TreatmentLoss(loss, n_outcome_true_cols, n_outcome_pred_cols, n_treatment_pred_cols)``
    still works; its outcome family is a placeholder that is never evaluated.
    """

    def __init__(
        self,
        mixture: Optional[mixture_lib.StateMixture] = None,
        n_outcome_true_cols: Optional[int] = None,
        n_outcome_pred_cols: Optional[int] = None,
        n_treatment_pred_cols: Optional[int] = None,
        *,
        loss: Optional[tf.keras.losses.Loss] = None,
        **kwargs: Any,
    ):
        """Initializes the treatment loss.

        Args:
          mixture: the `StateMixture` to evaluate. In the legacy form, the elementwise
            treatment NLL instead (or pass it as ``loss=``).
          n_outcome_true_cols: legacy form only: outcome columns in y_true.
          n_outcome_pred_cols: legacy form only: outcome parameters per state.
          n_treatment_pred_cols: legacy form only: treatment parameters per state.
          loss: legacy form only: the elementwise treatment NLL (``reduction="none"``).
          **kwargs: passed to the Keras Loss (``name``, ``reduction``).
        """
        super().__init__(**kwargs)
        legacy = (loss, n_outcome_true_cols, n_outcome_pred_cols, n_treatment_pred_cols)
        if isinstance(mixture, mixture_lib.StateMixture):
            if any(v is not None for v in legacy):
                raise TypeError("pass either a StateMixture or the legacy loss arguments, not both")
            self._mixture = mixture
            self._legacy = False
            return
        loss = mixture if loss is None else loss
        _require_counts(
            loss=loss,
            n_outcome_true_cols=n_outcome_true_cols,
            n_outcome_pred_cols=n_outcome_pred_cols,
            n_treatment_pred_cols=n_treatment_pred_cols,
        )
        _check_elementwise(loss, "loss")
        self._mixture = mixture_lib.StateMixture(
            families.LossFamily(None, n_outcome_pred_cols, n_outcome_true_cols),  # placeholder
            families.from_negloglik(loss, n_treatment_pred_cols, 1),
        )
        self._legacy = True
        self._loss = loss

    def get_config(self) -> dict:
        """Returns the config: the serialized mixture."""
        return {**super().get_config(), "mixture": _ser(self._mixture)}

    @classmethod
    def from_config(cls, config: dict) -> "TreatmentLoss":
        """Rebuilds the loss; a v0.3.0 three-int config goes through the legacy constructor."""
        config = dict(config)
        if "mixture" in config:
            config["mixture"] = _deser(config["mixture"])
        else:
            config["loss"] = _deser(config["loss"])
        return cls(**config)

    def penalty_free(self) -> "TreatmentLoss":
        """Returns a fresh instance of this exact class with no penalty terms applied.

        Rebuilds `type(self)` from `TreatmentLoss.get_config(self)`, the base class's config.
        See `OutcomeLoss.penalty_free` for the contract.
        """
        return type(self).from_config(TreatmentLoss.get_config(self))

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """The reduced treatment mixture NLL; see `StateMixture.treatment_nll`."""
        return mixture_lib.reduce(self._mixture.treatment_nll(y_true, y_pred), self.reduction)


_MISMATCH_WARNING = (
    "CausalLoss: the outcome and treatment losses disagree on the treatment model or the "
    "column counts ({detail}). The outcome loss's state posterior then uses a different "
    "treatment likelihood than the treatment loss. This warns for the legacy three-int "
    "constructors and will raise from v0.5.0; build both from one StateMixture "
    "(CausalLoss.from_mixture)."
)


def _link_mixtures(outcome_loss: "OutcomeLoss", treatment_loss: "TreatmentLoss") -> None:
    """Checks the two losses agree, then makes the treatment loss share the outcome mixture.

    Losses built from a mixture must have equal mixtures (else ValueError). Legacy three-int
    losses only need the same treatment family and column counts; a mismatch warns
    (decision A6) and leaves each loss with its own mixture, as before.
    """
    o_mix, t_mix = outcome_loss.mixture, treatment_loss.mixture
    if not (outcome_loss._legacy or treatment_loss._legacy):
        if o_mix != t_mix:
            raise ValueError(
                f"outcome_loss and treatment_loss must share one StateMixture; got {o_mix!r} "
                f"and {t_mix!r}"
            )
        treatment_loss._mixture = o_mix
        return
    counts = lambda m: (  # noqa: E731
        m.layout.n_outcome_pred_cols,
        m.layout.n_treatment_pred_cols,
        m.layout.n_outcome_true_cols,
    )
    if o_mix.treatment_family != t_mix.treatment_family:
        detail = f"{o_mix.treatment_family!r} vs {t_mix.treatment_family!r}"
    elif counts(o_mix) != counts(t_mix):
        detail = f"counts {counts(o_mix)} vs {counts(t_mix)}"
    else:
        treatment_loss._mixture = o_mix
        return
    warnings.warn(_MISMATCH_WARNING.format(detail=detail), FutureWarning, stacklevel=3)


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
    It is "dose-likelihood-off", not "dose-head-off": `outcome_loss` mixes per-state outcome
    predictions with the posterior gamma = P(state | X, T) (see `OutcomeLoss`,
    `prob_state_given_treatment_features`), which is a differentiable function of the
    treatment/dose head's own predictions. So even at alpha=0, gradients from the outcome loss
    still reach the dose head through gamma -- setting alpha=0 does not isolate the outcome model
    from the dose head. Do not describe alpha=0 as an "outcome-only" ablation.
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

    @property
    def _alpha(self) -> tf.Variable:
        """Deprecated alias of `alpha`."""
        warnings.warn(_ALPHA_DEPRECATION, FutureWarning, stacklevel=2)
        return self._alpha_var

    @_alpha.setter
    def _alpha(self, value: float) -> None:
        warnings.warn(_ALPHA_DEPRECATION, FutureWarning, stacklevel=2)
        self.alpha = value

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
        outcome_loss, treatment_loss = config["outcome_loss"], config["treatment_loss"]
        if (
            isinstance(outcome_loss, OutcomeLoss)
            and isinstance(treatment_loss, TreatmentLoss)
            and outcome_loss.mixture != treatment_loss.mixture
        ):
            # Only a legacy-path loss saves two different mixtures; load it the same way.
            treatment_loss._legacy = True
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

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Computes the causal loss from y_true and multi-output predictions."""
        loss_outcome = self._outcome_loss(y_true, y_pred)
        loss_treatment = self._treatment_loss(y_true, y_pred)

        total_loss = self._outcome_loss_weight * loss_outcome + self._alpha_var * loss_treatment
        if self._predictive_states_regularizer is not None:
            weights = self.layout.decode_pred(y_pred).state_weights
            total_loss += self._predictive_states_regularizer(weights)

        return total_loss
