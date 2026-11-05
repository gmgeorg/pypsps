"""The state mixture: everything that mixes over predictive states.

A pypsps model predicts, for each unit, state weights P(s_k | x) and per-state outcome and
treatment parameters. `StateMixture` holds the outcome and treatment families plus the
column layout, and computes from ``(y_true, y_pred)``:

* the per-state NLL matrices ``[N, K]`` of outcome and treatment;
* the state posterior P(s_k | x, a), by Bayes' rule from the treatment likelihood;
* the outcome mixture NLL ``-log sum_k gamma_k p(y | a, x, s_k)``, with gamma the state
  weights (interventional) or the state posterior (observational);
* the treatment mixture NLL ``-log sum_k P(s_k | x) p(a | s_k)``;
* mixed means, which mix each state's family mean (never raw parameters).

The losses in `pypsps.keras.losses` are thin wrappers that call it and reduce.
"""

import enum
from typing import NamedTuple, Optional, Union

import tensorflow as tf

from .. import utils
from . import families as families_lib
from .layout import Array, ColumnLayout, PredParts

_ser = tf.keras.utils.serialize_keras_object
_deser = tf.keras.utils.deserialize_keras_object


class OutcomeLikelihood(str, enum.Enum):
    """Which outcome likelihood the outcome mixture NLL evaluates.

    ``INTERVENTIONAL`` (the default in `OutcomeLoss`): -log p(Y | do(A), X), mixing the
    per-state outcome likelihoods with the prior state weights P(S | X) = P(S | X, do(A)),
    since the states precede treatment; these are the same weights the ATE estimates use.
    ``OBSERVATIONAL``: -log p(Y | A, X), mixing the per-state outcome likelihoods with the
    posterior state weights P(S | X, A); together with `TreatmentLoss` this is the exact
    joint likelihood of (Y, A | X).
    """

    OBSERVATIONAL = "observational"
    INTERVENTIONAL = "interventional"


class PerStateNLL(NamedTuple):
    """Per-state negative log-likelihoods, each ``[N, K]``."""

    outcome: tf.Tensor  # -log p(y | a, x, s_k)
    treatment: tf.Tensor  # -log p(a | s_k)


_REDUCTIONS = ("none", "sum", "sum_over_batch_size")


def reduce(values: tf.Tensor, reduction: Optional[str]) -> tf.Tensor:
    """Reduces per-unit ``[N]`` values: ``"none"``, ``"sum"`` or ``"sum_over_batch_size"``.

    ``None`` means ``"none"``; the legacy ``"auto"`` means ``"sum_over_batch_size"``.
    """
    key = "none" if reduction is None else str(getattr(reduction, "value", reduction)).lower()
    if key == "auto":
        key = "sum_over_batch_size"
    if key == "none":
        return values
    if key not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
    total = tf.reduce_sum(values)
    if key == "sum":
        return total
    return total / tf.cast(tf.shape(values)[0], values.dtype)


def posterior(state_weights: tf.Tensor, nll_per_state: tf.Tensor) -> tf.Tensor:
    """P(s_k | x, a) = softmax(log P(s_k | x) - nll_k): Bayes' rule in log space, ``[N, K]``."""
    log_w = utils.safe_log(tf.convert_to_tensor(state_weights), name="log_weights")
    return tf.nn.softmax(log_w - tf.convert_to_tensor(nll_per_state), axis=1)


def _mixture_nll(gamma: tf.Tensor, nll_per_state: tf.Tensor) -> tf.Tensor:
    """``-log sum_k gamma_k exp(-nll_k)``, per unit ``[N]``."""
    return -tf.math.reduce_logsumexp(utils.safe_log(gamma) - nll_per_state, axis=1)


@tf.keras.utils.register_keras_serializable(package="pypsps")
class StateMixture:
    """The outcome and treatment families of a pypsps model, mixed over predictive states.

    Args:
      outcome_family: distribution of the outcome within a state.
      treatment_family: distribution of the treatment within a state; must read exactly one
        ``y_true`` column.
      n_states: K, the number of predictive states.
    """

    def __init__(
        self,
        outcome_family: families_lib.Family,
        treatment_family: families_lib.Family,
        n_states: int,
    ):
        """Validates the families and derives the column layout."""
        for name, family in (
            ("outcome_family", outcome_family),
            ("treatment_family", treatment_family),
        ):
            if not isinstance(family, families_lib.Family):
                raise TypeError(f"{name} must be a pypsps Family, got {type(family).__name__}")
        if treatment_family.n_true_cols != 1:
            raise ValueError(
                "treatment_family must read exactly one y_true column, got "
                f"{type(treatment_family).__name__} with n_true_cols={treatment_family.n_true_cols}"
            )
        self._outcome_family = outcome_family
        self._treatment_family = treatment_family
        self._layout = ColumnLayout.from_families(n_states, outcome_family, treatment_family)

    @property
    def outcome_family(self) -> families_lib.Family:
        """The outcome family."""
        return self._outcome_family

    @property
    def treatment_family(self) -> families_lib.Family:
        """The treatment family."""
        return self._treatment_family

    @property
    def n_states(self) -> int:
        """K, the number of predictive states."""
        return self._layout.n_states

    @property
    def layout(self) -> ColumnLayout:
        """The column layout of ``y_pred`` and ``y_true``."""
        return self._layout

    # --- per-state quantities ---

    def _outcome_nll_per_state(self, y_true: Array, pred: PredParts) -> tf.Tensor:
        """``[N, K]`` outcome NLL from decoded parts."""
        return self._outcome_family.nll(self._layout.decode_true(y_true).outcome, pred.outcome)

    def _treatment_nll_per_state(self, y_true: Array, pred: PredParts) -> tf.Tensor:
        """``[N, K]`` treatment NLL from decoded parts."""
        treatment = self._layout.decode_true(y_true).treatment
        return self._treatment_family.nll(treatment, pred.treatment)

    def nll_per_state(self, y_true: Array, y_pred: Array) -> PerStateNLL:
        """Outcome and treatment NLL per unit and state, each ``[N, K]``."""
        pred = self._layout.decode_pred(y_pred)
        return PerStateNLL(
            outcome=self._outcome_nll_per_state(y_true, pred),
            treatment=self._treatment_nll_per_state(y_true, pred),
        )

    def state_posterior(self, y_true: Array, y_pred: Array) -> tf.Tensor:
        """P(s_k | x, a), ``[N, K]``; rows sum to 1."""
        pred = self._layout.decode_pred(y_pred)
        return posterior(pred.state_weights, self._treatment_nll_per_state(y_true, pred))

    # --- mixture NLLs ---

    def outcome_nll(
        self,
        y_true: Array,
        y_pred: Array,
        likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
    ) -> tf.Tensor:
        """``-log sum_k gamma_k p(y | a, x, s_k)`` per unit, ``[N]``.

        gamma is the state weights P(s | x) for ``"interventional"``, and the state posterior
        P(s | x, a) for ``"observational"``.
        """
        likelihood = OutcomeLikelihood(likelihood)
        pred = self._layout.decode_pred(y_pred)
        nll_y = self._outcome_nll_per_state(y_true, pred)
        if likelihood is OutcomeLikelihood.INTERVENTIONAL:
            gamma = pred.state_weights  # P(s | x) = P(s | x, do(a))
        else:
            gamma = posterior(pred.state_weights, self._treatment_nll_per_state(y_true, pred))
        return _mixture_nll(gamma, nll_y)

    def treatment_nll(self, y_true: Array, y_pred: Array) -> tf.Tensor:
        """``-log sum_k P(s_k | x) p(a | s_k)`` per unit, ``[N]``: the propensity likelihood."""
        pred = self._layout.decode_pred(y_pred)
        return _mixture_nll(pred.state_weights, self._treatment_nll_per_state(y_true, pred))

    def causal_nll(
        self,
        y_true: Array,
        y_pred: Array,
        likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
        outcome_loss_weight: float = 1.0,
    ) -> tf.Tensor:
        """``outcome_loss_weight * outcome_nll + treatment_nll`` per unit, ``[N]``.

        Alpha is fixed at 1 and there are no penalties. With weight 1 and
        ``likelihood="observational"`` this is the exact joint -log p(y, a | x).
        """
        if outcome_loss_weight < 0:
            raise ValueError(f"outcome_loss_weight must be >= 0, got {outcome_loss_weight}")
        treatment = self.treatment_nll(y_true, y_pred)
        if outcome_loss_weight == 0:
            return treatment
        return outcome_loss_weight * self.outcome_nll(y_true, y_pred, likelihood) + treatment

    # --- means ---

    def outcome_mean(self, y_pred: Array, per_state: bool = False) -> tf.Tensor:
        """E[Y | a, x]: ``[N, 1]`` mixing the family means; ``per_state=True`` gives ``[N, K]``."""
        pred = self._layout.decode_pred(y_pred)
        means = self._outcome_family.mean(pred.outcome)
        if per_state:
            return means
        return tf.reduce_sum(pred.state_weights * means, axis=1, keepdims=True)

    def treatment_mean(self, y_pred: Array) -> tf.Tensor:
        """E[A | x] = ``sum_k P(s_k | x) * mean_k``, ``[N, 1]``; the propensity for Bernoulli."""
        pred = self._layout.decode_pred(y_pred)
        means = self._treatment_family.mean(pred.treatment)
        return tf.reduce_sum(pred.state_weights * means, axis=1, keepdims=True)

    # --- lookup on a model ---

    @classmethod
    def from_model(cls, model: tf.keras.Model) -> "StateMixture":
        """The state mixture of a pypsps model, compiled or not.

        It is read from the model's `pypsps.keras.layers.CausalOutput` layer, which every
        builder adds since v0.3.4, so a model loaded with ``compile=False`` works too. A model
        without that layer (built before v0.3.4) falls back to the mixture of its compiled
        `pypsps.keras.losses.CausalLoss`.

        Raises:
          ValueError: if the output layer and the compiled loss disagree on the families or K.
          RuntimeError: if the model has neither a `CausalOutput` layer nor a pypsps loss.
        """
        layer_mixtures = [
            m
            for m in (getattr(layer, "mixture", None) for layer in model.layers)
            if isinstance(m, StateMixture)
        ]
        if len(layer_mixtures) > 1:
            raise ValueError(f"model has {len(layer_mixtures)} CausalOutput layers; expected one")
        layer_mixture = layer_mixtures[0] if layer_mixtures else None
        loss_mixture = getattr(getattr(model, "loss", None), "mixture", None)
        if not isinstance(loss_mixture, StateMixture):
            loss_mixture = None

        if layer_mixture is None:
            if loss_mixture is None:
                raise RuntimeError(
                    "Cannot find the model's state mixture: it has no CausalOutput layer and "
                    f"model.loss is {type(getattr(model, 'loss', None)).__name__}, not a pypsps "
                    "CausalLoss. Compile it with a CausalLoss, or rebuild it with pypsps >= "
                    "0.3.4 and copy the weights with `new.set_weights(old.get_weights())`."
                )
            return loss_mixture
        if loss_mixture is not None and layer_mixture != loss_mixture:
            raise ValueError(
                f"The model's CausalOutput layer holds {layer_mixture!r}, but its loss holds "
                f"{loss_mixture!r}"
            )
        return layer_mixture

    # --- serialization and equality ---

    def get_config(self) -> dict:
        """The two serialized families and ``n_states``."""
        return {
            "n_states": self.n_states,
            "outcome_family": _ser(self._outcome_family),
            "treatment_family": _ser(self._treatment_family),
        }

    @classmethod
    def from_config(cls, config: dict) -> "StateMixture":
        """Rebuilds the mixture and its families.

        Raises:
          ValueError: if ``n_states`` is missing; only the column-count loss and metric
            constructors removed in v0.5.0 saved such a mixture.
        """
        if config.get("n_states") is None:
            raise ValueError(
                "StateMixture config has no n_states: it was saved by a loss or metric built "
                "with the column-count constructor, which v0.5.0 removed. Load the model with "
                "compile=False and compile it with pypsps.keras.models.compile_model."
            )
        return cls(
            outcome_family=_deser(config["outcome_family"]),
            treatment_family=_deser(config["treatment_family"]),
            n_states=config["n_states"],
        )

    def __eq__(self, other: object) -> bool:
        """Equal families and ``n_states``."""
        return (
            isinstance(other, StateMixture)
            and self._outcome_family == other._outcome_family
            and self._treatment_family == other._treatment_family
            and self.n_states == other.n_states
        )

    def __hash__(self):
        """Consistent with `__eq__`."""
        return hash((self._outcome_family, self._treatment_family, self.n_states))

    def __repr__(self) -> str:
        """Shows the families and K."""
        return (
            f"StateMixture({self._outcome_family!r}, {self._treatment_family!r}, "
            f"n_states={self.n_states})"
        )
