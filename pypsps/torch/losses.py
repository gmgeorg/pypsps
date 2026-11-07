"""The PSPS causal loss on PyTorch: a plain callable ``nn.Module``.

    loss = outcome_loss_weight * outcome NLL + alpha * treatment NLL (+ state regularizer)

The same contract as `pypsps.keras.losses.CausalLoss`: ``alpha`` weights the treatment
log-likelihood (λ in Eq. 10 of https://proceedings.mlr.press/v177/kelly22a/kelly22a.pdf), the
per-unit loss is multiplied by ``sample_weight`` and reduced (``"sum_over_batch_size"``
divides by the number of units, not by the sum of the weights), and the regularizer on the
state weights is added once after the reduction, never scaled by ``sample_weight``.

Unlike in Keras, ``alpha`` is a plain float: change it between steps with ``loss.alpha = x``.

Note on alpha = 0: this zeroes only the explicit treatment log-likelihood. The outcome loss
still mixes with the posterior state weights under ``outcome_likelihood="observational"``,
which depend on the treatment head, so alpha = 0 is not an "outcome-only" ablation.
"""

import math
from typing import Callable, Optional, Union

import torch
from torch import nn

from ..core.layout import ColumnLayout
from ..core.likelihood import OutcomeLikelihood
from . import mixture as mixture_lib

Regularizer = Callable[[torch.Tensor], torch.Tensor]


def state_penalty(regularizer: Callable, weights: torch.Tensor) -> torch.Tensor:
    """The amount ``regularizer`` adds to the loss for a batch of state weights.

    A pypress regularizer has a ``penalty()`` method that returns the penalty independent of
    the batch size; any other callable is called on the weights.
    """
    penalty = getattr(regularizer, "penalty", None)
    return penalty(weights) if callable(penalty) else regularizer(weights)


class CausalLoss(nn.Module):
    """PSPS causal loss: outcome NLL plus ``alpha`` times treatment NLL.

    Args:
      mixture: the model's `pypsps.torch.mixture.StateMixture`.
      alpha: weight of the treatment log-likelihood. 1.0 gives the joint log-likelihood;
        values above 1 prioritize fitting the treatment (the propensity and the states).
      outcome_loss_weight: weight of the outcome log-likelihood, >= 0.
      outcome_likelihood: ``"interventional"`` (default) or ``"observational"``; see
        `pypsps.core.likelihood.OutcomeLikelihood`.
      predictive_states_regularizer: optional regularizer of the state weights, e.g. a
        `pypress.torch.regularizers` instance. It is a submodule, so its running averages
        follow ``train()`` / ``eval()`` and ``state_dict()``.
      reduction: ``"sum_over_batch_size"`` (default), ``"sum"`` or ``"none"``.
    """

    def __init__(
        self,
        mixture: mixture_lib.StateMixture,
        alpha: float = 1.0,
        outcome_loss_weight: float = 1.0,
        outcome_likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
        predictive_states_regularizer: Optional[Regularizer] = None,
        reduction: str = "sum_over_batch_size",
    ):
        """Validates the arguments."""
        super().__init__()
        if not isinstance(mixture, mixture_lib.StateMixture):
            raise TypeError(f"mixture must be a StateMixture, got {type(mixture).__name__}")
        if outcome_loss_weight < 0:
            raise ValueError(f"outcome_loss_weight must be >= 0, got {outcome_loss_weight!r}")
        self._mixture = mixture
        self.alpha = alpha
        self.outcome_loss_weight = float(outcome_loss_weight)
        self.outcome_likelihood = OutcomeLikelihood(outcome_likelihood)
        self.reduction = _checked_reduction(reduction)
        if isinstance(predictive_states_regularizer, nn.Module):
            self.predictive_states_regularizer = predictive_states_regularizer
        else:
            # a plain callable is not a module; keep it as an attribute
            self.__dict__["predictive_states_regularizer"] = predictive_states_regularizer

    @property
    def mixture(self) -> mixture_lib.StateMixture:
        """The state mixture."""
        return self._mixture

    @property
    def layout(self) -> ColumnLayout:
        """The column layout of ``y_pred`` and ``y_true``."""
        return self._mixture.layout

    @property
    def alpha(self) -> float:
        """Weight of the treatment log-likelihood; assign to change it."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"alpha must be finite, got {value!r}")
        self._alpha = value

    def per_unit(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """The causal loss per unit ``[N]``, without the regularizer."""
        outcome = self._mixture.outcome_nll(y_true, y_pred, self.outcome_likelihood)
        treatment = self._mixture.treatment_nll(y_true, y_pred)
        return self.outcome_loss_weight * outcome + self._alpha * treatment

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The reduced causal loss, plus the predictive-state regularizer.

        ``sample_weight`` (``[N]`` or ``[N, 1]``) multiplies each unit's loss before the
        reduction. The regularizer is a penalty on the batch's state weights, not a per-unit
        loss, so it is added once, after the reduction, and ``sample_weight`` does not scale it.
        Under ``reduction="none"`` it is added to every unit.
        """
        loss = self.per_unit(y_true, y_pred)
        if sample_weight is not None:
            loss = loss * torch.as_tensor(
                sample_weight, dtype=loss.dtype, device=loss.device
            ).reshape(-1)
        loss = mixture_lib.reduce(loss, self.reduction)
        regularizer = self.predictive_states_regularizer
        if regularizer is None:
            return loss
        weights = self.layout.decode_pred(y_pred).state_weights
        return loss + state_penalty(regularizer, weights).to(loss.dtype)

    def extra_repr(self) -> str:
        """Alpha, weight, likelihood and reduction, for ``print(loss)``."""
        return (
            f"alpha={self._alpha}, outcome_loss_weight={self.outcome_loss_weight}, "
            f"outcome_likelihood={self.outcome_likelihood.value!r}, reduction={self.reduction!r}"
        )


def _checked_reduction(reduction: str) -> str:
    """The normalized reduction key; ValueError for an unknown one."""
    from ..core import mixture as core_mixture

    return core_mixture.reduction_key(reduction)
