"""The state mixture on PyTorch: everything that mixes over predictive states.

The same quantities and the same formulas as `pypsps.keras.mixture`; see there. The state
mixture holds the outcome and treatment families plus the column layout, and computes from
``(y_true, y_pred)`` the per-state NLL matrices, the state posterior, the outcome / treatment
/ causal mixture NLLs and the mixed means.
"""

from typing import NamedTuple, Optional, Union

import torch

from ..core import mixture as core_mixture
from ..core.layout import ColumnLayout, PredParts
from ..core.likelihood import OutcomeLikelihood
from . import families as families_lib

Array = torch.Tensor


class PerStateNLL(NamedTuple):
    """Per-state negative log-likelihoods, each ``[N, K]``."""

    outcome: torch.Tensor  # -log p(y | a, x, s_k)
    treatment: torch.Tensor  # -log p(a | s_k)


def safe_log(p: torch.Tensor, neg_inf: float = -1e9) -> torch.Tensor:
    """Numerically stable log(p) for p in [0, 1], e.g. softmax/simplex outputs.

    Substitutes ``neg_inf`` for exact zeros only (no epsilon floor), through ``torch.where`` on
    a safe argument so no NaN gradient flows back through ``log`` at ``p == 0``.
    """
    p = torch.as_tensor(p)
    is_positive = p > 0
    safe_p = torch.where(is_positive, p, torch.ones_like(p))
    return torch.where(is_positive, torch.log(safe_p), torch.full_like(p, neg_inf))


def reduce(values: torch.Tensor, reduction: Optional[str]) -> torch.Tensor:
    """Reduces per-unit ``[N]`` values: ``"none"``, ``"sum"`` or ``"sum_over_batch_size"``.

    ``None`` means ``"none"``; the legacy ``"auto"`` means ``"sum_over_batch_size"``.
    """
    key = core_mixture.reduction_key(reduction)
    if key == "none":
        return values
    total = values.sum()
    if key == "sum":
        return total
    return total / values.shape[0]


def posterior(state_weights: torch.Tensor, nll_per_state: torch.Tensor) -> torch.Tensor:
    """P(s_k | x, a) = softmax(log P(s_k | x) - nll_k): Bayes' rule in log space, ``[N, K]``."""
    return torch.softmax(safe_log(state_weights) - nll_per_state, dim=1)


def _mixture_nll(gamma: torch.Tensor, nll_per_state: torch.Tensor) -> torch.Tensor:
    """``-log sum_k gamma_k exp(-nll_k)``, per unit ``[N]``."""
    return -torch.logsumexp(safe_log(gamma) - nll_per_state, dim=1)


class StateMixture(core_mixture.MixtureSpec):
    """The outcome and treatment families of a pypsps model, mixed over predictive states.

    Args:
      outcome_family: distribution of the outcome within a state.
      treatment_family: distribution of the treatment within a state; must read exactly one
        ``y_true`` column.
      n_states: K, the number of predictive states.
    """

    _family_base = families_lib.Family
    _layout_cls = ColumnLayout

    # --- per-state quantities ---

    def _outcome_nll_per_state(self, y_true: Array, pred: PredParts) -> torch.Tensor:
        """``[N, K]`` outcome NLL from decoded parts."""
        return self._outcome_family.nll(self._layout.decode_true(y_true).outcome, pred.outcome)

    def _treatment_nll_per_state(self, y_true: Array, pred: PredParts) -> torch.Tensor:
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

    def state_posterior(self, y_true: Array, y_pred: Array) -> torch.Tensor:
        """P(s_k | x, a), ``[N, K]``; rows sum to 1."""
        pred = self._layout.decode_pred(y_pred)
        return posterior(pred.state_weights, self._treatment_nll_per_state(y_true, pred))

    # --- mixture NLLs ---

    def outcome_nll(
        self,
        y_true: Array,
        y_pred: Array,
        likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
    ) -> torch.Tensor:
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

    def treatment_nll(self, y_true: Array, y_pred: Array) -> torch.Tensor:
        """``-log sum_k P(s_k | x) p(a | s_k)`` per unit, ``[N]``: the propensity likelihood."""
        pred = self._layout.decode_pred(y_pred)
        return _mixture_nll(pred.state_weights, self._treatment_nll_per_state(y_true, pred))

    def causal_nll(
        self,
        y_true: Array,
        y_pred: Array,
        likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
        outcome_loss_weight: float = 1.0,
    ) -> torch.Tensor:
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

    def outcome_mean(self, y_pred: Array, per_state: bool = False) -> torch.Tensor:
        """E[Y | a, x]: ``[N, 1]`` mixing the family means; ``per_state=True`` gives ``[N, K]``."""
        pred = self._layout.decode_pred(y_pred)
        means = self._outcome_family.mean(pred.outcome)
        if per_state:
            return means
        return (pred.state_weights * means).sum(dim=1, keepdim=True)

    def treatment_mean(self, y_pred: Array) -> torch.Tensor:
        """E[A | x] = ``sum_k P(s_k | x) * mean_k``, ``[N, 1]``; the propensity for Bernoulli."""
        pred = self._layout.decode_pred(y_pred)
        means = self._treatment_family.mean(pred.treatment)
        return (pred.state_weights * means).sum(dim=1, keepdim=True)

    # --- lookup on a model ---

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> "StateMixture":
        """The state mixture of a pypsps model (its ``mixture`` attribute).

        Raises:
          RuntimeError: if the model has none, i.e. it is not a `pypsps.torch.models.PSPSModel`.
        """
        mixture = getattr(model, "mixture", None)
        if not isinstance(mixture, StateMixture):
            raise RuntimeError(
                f"Cannot find the model's state mixture: {type(model).__name__} has no "
                "`mixture` attribute of type pypsps.torch.mixture.StateMixture."
            )
        return mixture

    # --- serialization ---

    def get_config(self) -> dict:
        """The two families as ``{"class_name", "config"}`` and ``n_states``; JSON-friendly."""
        return {
            "n_states": self.n_states,
            "outcome_family": _family_config(self._outcome_family),
            "treatment_family": _family_config(self._treatment_family),
        }

    @classmethod
    def from_config(cls, config: dict) -> "StateMixture":
        """Rebuilds the mixture and its families from `get_config`."""
        return cls(
            outcome_family=families_lib.from_config(
                config["outcome_family"]["class_name"], config["outcome_family"]["config"]
            ),
            treatment_family=families_lib.from_config(
                config["treatment_family"]["class_name"], config["treatment_family"]["config"]
            ),
            n_states=config["n_states"],
        )


def _family_config(family: families_lib.Family) -> dict:
    """``{"class_name": ..., "config": family.get_config()}``."""
    return {"class_name": type(family).__name__, "config": family.get_config()}
