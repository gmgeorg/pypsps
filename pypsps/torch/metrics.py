"""Training metrics on PyTorch, with the names and meaning of `pypsps.keras.metrics`.

A metric accumulates over the batches of an epoch: ``update(y_true, y_pred, sample_weight)``,
then ``result()``, then ``reset()``. ``fit`` logs each result under the metric's ``name``, so a
callback can monitor ``"val_causal_loss_metric"`` as in Keras.

The means are weighted by ``sample_weight`` and divide by the sum of the weights (as Keras
metrics do; the loss divides by the number of units instead). The state-weight metrics
``predictive_state_df`` and ``uniform_entropy`` average the batch values, one per batch.
Differences from Keras: ``propensity_score_auroc`` / ``propensity_score_aupr`` are exact
(Keras bins the scores into 200 thresholds).
"""

from typing import List, Optional, Sequence, Union

import numpy as np
import pypress.torch.statistics as pr_stats
import torch

from ..core.layout import ColumnLayout
from ..core.likelihood import OutcomeLikelihood
from . import families as families_lib
from . import mixture as mixture_lib

Array = Union[np.ndarray, torch.Tensor]


class Metric:
    """A weighted-mean metric: subclasses define ``value(y_true, y_pred) -> [N]`` per unit."""

    name = "metric"

    def __init__(self, name: Optional[str] = None):
        """Creates the metric; ``name`` defaults to the class's ``name``."""
        if name is not None:
            self.name = name
        self.reset()

    def reset(self) -> None:
        """Clears the accumulated state."""
        self._total = 0.0
        self._weight = 0.0

    def values(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """The per-unit values ``[N]``, or a scalar for a per-batch value."""
        raise NotImplementedError

    def unit_weights(self, y_true: torch.Tensor, sample_weight: Optional[torch.Tensor]):
        """The weights of the units, ``sample_weight`` by default."""
        return sample_weight

    @torch.no_grad()
    def update(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Adds one batch."""
        values = self.values(y_true, y_pred).detach().to(torch.float64)
        if values.ndim == 0:  # a per-batch value counts once
            self._total += float(values)
            self._weight += 1.0
            return
        weight = self.unit_weights(y_true, sample_weight)
        if weight is None:
            weight = torch.ones_like(values)
        weight = torch.as_tensor(weight, dtype=torch.float64).reshape(-1)
        self._total += float((values * weight).sum())
        self._weight += float(weight.sum())

    def result(self) -> float:
        """The weighted mean so far; NaN before any weight was seen."""
        return self._total / self._weight if self._weight > 0 else float("nan")

    def __repr__(self) -> str:
        """The class and name."""
        return f"{type(self).__name__}(name={self.name!r})"


class CausalLossMetric(Metric):
    """The joint NLL ``outcome_loss_weight * outcome NLL + 1 * treatment NLL`` per unit.

    No regularizer and alpha fixed at 1, whatever the training loss uses, so it is comparable
    across epochs and alpha schedules (a sound early-stopping criterion).
    """

    name = "causal_loss_metric"

    def __init__(
        self,
        mixture: mixture_lib.StateMixture,
        outcome_loss_weight: float = 1.0,
        outcome_likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
        name: Optional[str] = None,
    ):
        """Stores the mixture and the training loss's outcome weight and likelihood."""
        self._mixture = mixture
        self._weight_outcome = float(outcome_loss_weight)
        self._likelihood = OutcomeLikelihood(outcome_likelihood)
        super().__init__(name)

    def values(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Per-unit joint NLL."""
        return self._mixture.causal_nll(y_true, y_pred, self._likelihood, self._weight_outcome)


class OutcomeNLLMetric(Metric):
    """The outcome NLL alone, without alpha or penalties."""

    name = "outcome_nll_metric"

    def __init__(
        self,
        mixture: mixture_lib.StateMixture,
        outcome_likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
        name: Optional[str] = None,
    ):
        """Stores the mixture and the outcome likelihood."""
        self._mixture = mixture
        self._likelihood = OutcomeLikelihood(outcome_likelihood)
        super().__init__(name)

    def values(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Per-unit outcome NLL."""
        return self._mixture.outcome_nll(y_true, y_pred, self._likelihood)


class PredictiveStateDF(Metric):
    """Degrees of freedom, trace(H), of the state weights (batch average)."""

    name = "predictive_state_df"

    def __init__(self, layout: ColumnLayout, name: Optional[str] = None):
        """Stores the layout."""
        self._layout = layout
        super().__init__(name)

    def values(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """The batch's degrees of freedom."""
        weights = self._layout.decode_pred(y_pred).state_weights
        return pr_stats.degrees_of_freedom(
            pr_stats.compute("trace_numerator", weights),
            pr_stats.compute("trace_denominator", weights),
        )


class UniformEntropy(Metric):
    """Mean Shannon entropy of the state weights (batch average); higher is more uniform."""

    name = "uniform_entropy"

    def __init__(self, layout: ColumnLayout, name: Optional[str] = None):
        """Stores the layout."""
        self._layout = layout
        super().__init__(name)

    def values(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """The batch's mean entropy."""
        weights = self._layout.decode_pred(y_pred).state_weights
        return (-pr_stats.safe_xlogx(weights).sum(dim=1)).mean()


class _PropensityMetric(Metric):
    """Scores the marginal propensity ``sum_k P(s_k | x) p_k`` against the treatment."""

    def __init__(self, mixture: mixture_lib.StateMixture, name: Optional[str] = None):
        """Stores the mixture; its treatment family must be binary."""
        if not mixture.treatment_family.is_binary:
            raise ValueError(
                f"{type(self).__name__} needs a binary treatment family, got "
                f"{mixture.treatment_family!r}"
            )
        self._mixture = mixture
        super().__init__(name)

    def _pair(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """(treatment ``[N]``, propensity ``[N]``)."""
        treatment = self._mixture.layout.decode_true(y_true).treatment.reshape(-1)
        return treatment, self._mixture.treatment_mean(y_pred).reshape(-1)


class PropensityScoreBinaryCrossentropy(_PropensityMetric):
    """Cross entropy of the marginal propensity score P(A = 1 | x)."""

    name = "propensity_score_binary_crossentropy"

    def values(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Per-unit cross entropy, with the propensity clipped to [1e-7, 1 - 1e-7]."""
        treatment, prob = self._pair(y_true, y_pred)
        prob = prob.clamp(1e-7, 1.0 - 1e-7)
        return -(treatment * prob.log() + (1.0 - treatment) * (1.0 - prob).log())


class PropensityScoreAUC(_PropensityMetric):
    """Exact area under the ROC curve (``curve="ROC"``, trapezoid) or the precision-recall curve
    (``"PR"``, average precision) of the marginal propensity score, weighted by
    ``sample_weight``."""

    name = "propensity_score_auc"

    def __init__(
        self,
        mixture: mixture_lib.StateMixture,
        curve: str = "ROC",
        name: Optional[str] = None,
    ):
        """``curve`` is ``"ROC"`` or ``"PR"``."""
        curve = str(curve).upper()
        if curve not in ("ROC", "PR"):
            raise ValueError(f"curve must be 'ROC' or 'PR', got {curve!r}")
        self._curve = curve
        super().__init__(mixture, name)

    def reset(self) -> None:
        """Clears the stored scores."""
        self._scores: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        self._weights: List[np.ndarray] = []

    @torch.no_grad()
    def update(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Stores the batch's scores, labels and weights."""
        treatment, prob = self._pair(y_true, y_pred)
        n = treatment.shape[0]
        weight = (
            np.ones(n)
            if sample_weight is None
            else torch.as_tensor(sample_weight, dtype=torch.float64).reshape(-1).numpy()
        )
        self._scores.append(prob.detach().double().numpy())
        self._labels.append((treatment.detach().numpy() > 0.5).astype(float))
        self._weights.append(weight)

    def result(self) -> float:
        """The area under the curve; NaN unless both classes have weight."""
        if not self._scores:
            return float("nan")
        score = np.concatenate(self._scores)
        label = np.concatenate(self._labels)
        weight = np.concatenate(self._weights)
        order = np.argsort(-score, kind="stable")
        score, label, weight = score[order], label[order], weight[order]
        tp = np.cumsum(weight * label)
        fp = np.cumsum(weight * (1.0 - label))
        last = np.r_[score[1:] != score[:-1], True]  # one point per distinct score
        tp, fp = tp[last], fp[last]
        positives, negatives = tp[-1], fp[-1]
        if positives <= 0 or negatives <= 0:
            return float("nan")
        recall = tp / positives
        if self._curve == "ROC":
            x, y = np.r_[0.0, fp / negatives], np.r_[0.0, recall]
            return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
        # PR: average precision, sum (R_i - R_{i-1}) * P_i; it is the prevalence for tied scores
        precision = tp / np.maximum(tp + fp, 1e-300)
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


class _TreatmentError(Metric):
    """Error of the mixed treatment mean against the observed treatment."""

    def __init__(self, mixture: mixture_lib.StateMixture, name: Optional[str] = None):
        """Stores the mixture."""
        self._mixture = mixture
        super().__init__(name)

    def _diff(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        treatment = self._mixture.layout.decode_true(y_true).treatment.reshape(-1)
        return self._mixture.treatment_mean(y_pred).reshape(-1) - treatment


class TreatmentMeanSquaredError(_TreatmentError):
    """MSE of the mixed treatment mean."""

    name = "treatment_mean_squared_error"

    def values(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Squared errors."""
        return self._diff(y_true, y_pred) ** 2


class TreatmentMeanAbsoluteError(_TreatmentError):
    """MAE of the mixed treatment mean."""

    name = "treatment_mean_absolute_error"

    def values(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Absolute errors."""
        return self._diff(y_true, y_pred).abs()


class _OutcomeError(Metric):
    """Error of the mixed outcome mean; for a censored family only units with an event count.

    The result is NaN while no event has been seen since the last reset.
    """

    def __init__(self, mixture: mixture_lib.StateMixture, name: Optional[str] = None):
        """Stores the mixture."""
        self._mixture = mixture
        super().__init__(name)

    def _diff(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        family = self._mixture.outcome_family
        outcome = self._mixture.layout.decode_true(y_true).outcome
        response = family.response(outcome).reshape(-1)
        return self._mixture.outcome_mean(y_pred).reshape(-1) - response

    def unit_weights(self, y_true: torch.Tensor, sample_weight: Optional[torch.Tensor]):
        """``sample_weight`` times the event indicator for a censored family."""
        family = self._mixture.outcome_family
        event = family.event_indicator(self._mixture.layout.decode_true(y_true).outcome)
        if event is None:
            return sample_weight
        event = event.reshape(-1).to(torch.float64)
        if sample_weight is None:
            return event
        return event * torch.as_tensor(sample_weight, dtype=torch.float64).reshape(-1)


class OutcomeMeanSquaredError(_OutcomeError):
    """MSE of the mixed outcome mean (events only for censored outcomes)."""

    name = "outcome_mean_squared_error"

    def values(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Squared errors."""
        return self._diff(y_true, y_pred) ** 2


class OutcomeMeanAbsoluteError(_OutcomeError):
    """MAE of the mixed outcome mean (events only for censored outcomes)."""

    name = "outcome_mean_absolute_error"

    def values(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Absolute errors."""
        return self._diff(y_true, y_pred).abs()


def default_metrics(
    mixture: mixture_lib.StateMixture,
    outcome_loss_weight: float = 1.0,
    outcome_likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
) -> Sequence[Metric]:
    """The metrics `pypsps.torch.training.fit` logs by default, chosen by the families.

    Always ``causal_loss_metric``, ``outcome_nll_metric``, ``predictive_state_df`` and
    ``uniform_entropy``. A binary treatment adds ``propensity_score_binary_crossentropy``,
    ``propensity_score_auroc`` and ``propensity_score_aupr``; any other treatment adds
    ``treatment_mean_squared_error``. Every outcome family but `LossFamily` adds
    ``outcome_mean_squared_error``.
    """
    chosen: List[Metric] = [
        CausalLossMetric(mixture, outcome_loss_weight, outcome_likelihood),
        OutcomeNLLMetric(mixture, outcome_likelihood),
        PredictiveStateDF(mixture.layout),
        UniformEntropy(mixture.layout),
    ]
    if mixture.treatment_family.is_binary:
        chosen += [
            PropensityScoreBinaryCrossentropy(mixture),
            PropensityScoreAUC(mixture, curve="ROC", name="propensity_score_auroc"),
            PropensityScoreAUC(mixture, curve="PR", name="propensity_score_aupr"),
        ]
    else:
        chosen.append(TreatmentMeanSquaredError(mixture))
    if not isinstance(mixture.outcome_family, families_lib.LossFamily):
        chosen.append(OutcomeMeanSquaredError(mixture))
    return chosen
