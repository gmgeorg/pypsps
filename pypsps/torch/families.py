"""Outcome and treatment families for PyTorch: one class per within-state distribution.

The same families as `pypsps.keras.families` (same names, parameters, bounds and config); the
parameter specs and validation are shared in `pypsps.core.families`, and the classes here add
the PyTorch ``nll`` and ``mean``. The numeric conventions match the Keras backend exactly: the
Bernoulli probability is clipped to ``[1e-7, 1 - 1e-7]`` as ``keras`` does, the Normal scale
and the survival log-time get a ``1e-6`` epsilon.

Every per-state method takes ``{param name: [N, K]}`` blocks (as `ColumnLayout.decode_pred`
returns them) and gives ``[N, K]``: one value per unit and state.

Built-in families: `Normal`, `Bernoulli`, `Exponential` and `Weibull`. `LossFamily` wraps any
elementwise callable on tensors. `get` looks a family up by name.
"""

import abc
import math
from typing import Any, Mapping, Optional

import torch
from torch import nn

from ..core import families as core
from ..core.families import (  # noqa: F401  (re-exported)
    Bounds,
    ParamSpec,
    check_bounds,
)
from . import layers

_EPS = 1e-6  # added to the Normal scale and the survival times, as in the Keras backend
_PROB_EPS = 1e-7  # Keras clips probabilities to [eps, 1 - eps] in binary cross-entropy


def _float(x: Any) -> torch.Tensor:
    """``x`` as a float32 tensor."""
    return torch.as_tensor(x, dtype=torch.float32)


# --- elementwise kernels; the same formulas as `pypsps.keras.neglogliks` ---


def negloglik_normal(y: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of ``y ~ Normal(loc, scale)``."""
    return (
        math.log(2.0 * math.pi) / 2.0
        + torch.log(scale + _EPS)
        + 0.5 * torch.square((y - loc) / (scale + _EPS))
    )


def negloglik_bernoulli(y: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with the probability clipped to ``[1e-7, 1 - 1e-7]`` (Keras)."""
    prob = torch.clamp(prob, _PROB_EPS, 1.0 - _PROB_EPS)
    return -(y * torch.log(prob) + (1.0 - y) * torch.log(1.0 - prob))


def negloglik_exponential(
    time: torch.Tensor, event: torch.Tensor, rate: torch.Tensor
) -> torch.Tensor:
    """Censored exponential NLL with ``rate``: ``rate * t - event * log(rate)``."""
    return rate * time - event * torch.log(rate + _EPS)


def negloglik_exponential_scale(
    time: torch.Tensor, event: torch.Tensor, log_scale: torch.Tensor
) -> torch.Tensor:
    """Censored exponential NLL with ``log_scale = log(mean time)``."""
    return time * torch.exp(-log_scale) + event * log_scale


def negloglik_weibull(
    time: torch.Tensor, event: torch.Tensor, log_scale: torch.Tensor, log_shape: torch.Tensor
) -> torch.Tensor:
    """Censored Weibull NLL in the ``(log_scale, log_shape)`` parameterization."""
    log_time = torch.log(time + _EPS)
    k = torch.exp(log_shape)
    z = torch.exp(k * (log_time - log_scale))
    log_pdf_term = log_shape - k * log_scale + (k - 1.0) * log_time
    return z - event * log_pdf_term


# --- families ---


def head_activation(param: ParamSpec) -> nn.Module:
    """Module that maps a raw head output onto ``param``'s support."""
    if param.bounds is not None:
        return layers.SoftClip(*param.bounds)
    return layers.activation(param.activation)


class Family(core.Family):
    """The distribution of a unit's outcome (or treatment) within one predictive state.

    Subclasses are frozen dataclasses whose fields are their config, so ``==`` compares
    configs. The same interface serves both roles; see the `OutcomeFamily` and
    `TreatmentFamily` aliases.
    """

    @abc.abstractmethod
    def nll(self, y_true: torch.Tensor, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """-log p(y | state k) for every unit and state.

        Args:
          y_true: ``[N, C]``.
          params: ``{name: [N, K]}``.

        Returns:
          ``[N, K]``, elementwise with no reduction.
        """

    @abc.abstractmethod
    def mean(self, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """E[response | state k]: ``{name: [N, K]}`` -> ``[N, K]``."""

    def head_activation(self, param: str) -> nn.Module:
        """Module that maps a raw head output onto parameter ``param``'s support."""
        return head_activation(self.param(param))


OutcomeFamily = Family
TreatmentFamily = Family


class SurvivalFamily(core.SurvivalFamily, Family):
    """A right-censored time-to-event family: ``y_true`` is ``[time, event]``."""

    def event_indicator(self, y_true: torch.Tensor) -> torch.Tensor:
        """``y_true[:, 1:2]`` as floats: 1 if the event was observed, 0 if censored."""
        return y_true[:, 1:2].to(torch.float32)

    def _time_and_event(self, y_true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``([N, 1] time, [N, 1] event)`` as float32."""
        y_true = _float(y_true)
        return y_true[:, 0:1], y_true[:, 1:2]


class Normal(core.Normal, Family):
    """``y ~ Normal(loc, scale)``; ``scale`` goes through a softplus."""

    def nll(self, y_true: torch.Tensor, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Gaussian NLL."""
        return negloglik_normal(_float(y_true)[:, :1], params["loc"], params["scale"])

    def mean(self, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``loc``."""
        return _float(params["loc"])


class Bernoulli(core.Bernoulli, Family):
    """``y ~ Bernoulli(prob)``; ``prob`` goes through a sigmoid."""

    def nll(self, y_true: torch.Tensor, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Binary cross-entropy, clipped as Keras clips it."""
        return negloglik_bernoulli(_float(y_true)[:, :1], _float(params["prob"]))

    def mean(self, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``prob``."""
        return _float(params["prob"])


class Exponential(core.Exponential, SurvivalFamily):
    """Right-censored exponential time to event.

    Attributes:
      parameterization: ``"log_scale"`` (log mean time; default), ``"log_rate"`` (log hazard)
        or ``"rate"`` (hazard, through a softplus).
      bounds: optional ``(min, max)`` of the ``log_scale`` or ``log_rate`` head.
    """

    def nll(self, y_true: torch.Tensor, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Censored exponential NLL."""
        time, event = self._time_and_event(y_true)
        if self.parameterization == "log_scale":
            return negloglik_exponential_scale(time, event, params["log_scale"])
        if self.parameterization == "log_rate":
            return negloglik_exponential(time, event, torch.exp(params["log_rate"]))
        return negloglik_exponential(time, event, params["rate"])

    def mean(self, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The mean time: ``exp(log_scale)``, ``exp(-log_rate)`` or ``1 / rate``."""
        if self.parameterization == "log_scale":
            return torch.exp(_float(params["log_scale"]))
        if self.parameterization == "log_rate":
            return torch.exp(-_float(params["log_rate"]))
        return 1.0 / _float(params["rate"])


class Weibull(core.Weibull, SurvivalFamily):
    """Right-censored Weibull time to event, parameterized by ``log_scale`` and ``log_shape``.

    Attributes:
      log_scale_bounds: optional ``(min, max)`` of the ``log_scale`` head.
      log_shape_bounds: optional ``(min, max)`` of the ``log_shape`` head.
    """

    def nll(self, y_true: torch.Tensor, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Censored Weibull NLL."""
        time, event = self._time_and_event(y_true)
        return negloglik_weibull(time, event, params["log_scale"], params["log_shape"])

    def mean(self, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """``exp(log_scale) * Gamma(1 + 1 / shape)``."""
        log_scale = _float(params["log_scale"])
        log_shape = _float(params["log_shape"])
        return torch.exp(log_scale) * torch.exp(torch.lgamma(1.0 + torch.exp(-log_shape)))


class LossFamily(core.LossFamily, Family):
    """Any elementwise loss on tensors as a family; its mean is unknown.

    Attributes:
      loss: a callable that maps ``y_true [N, C]`` and one state's ``[N, P]`` parameters to
        ``[N]``, with no reduction.
      n_params: P, the parameters per state, named ``param_0, ...``.
      n_true_cols: C.

    A model with a `LossFamily` cannot be saved with `pypsps.torch.io.save`: a callable
    cannot be serialized.
    """

    def nll(self, y_true: torch.Tensor, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Calls the loss once per state on that state's ``[N, P]`` parameters."""
        y_true = _float(y_true)
        if y_true.ndim == 1:
            y_true = y_true[:, None]
        blocks = [params[name] for name in self.param_names]
        n_states = blocks[0].shape[-1]
        per_state = []
        for j in range(n_states):
            state_params = torch.cat([b[:, j : j + 1] for b in blocks], dim=1)
            per_state.append(self.loss(y_true, state_params))
        return torch.stack(per_state, dim=1)

    def mean(self, params: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Raises: a generic loss does not say what its mean is."""
        raise NotImplementedError(
            f"the mean of a LossFamily ({type(self.loss).__name__}) is unknown; use one of the "
            "built-in families (Normal, Bernoulli, Exponential, Weibull) instead"
        )

    def get_config(self) -> dict:
        """Raises: a callable cannot be serialized."""
        raise NotImplementedError(
            "a LossFamily wraps a callable, which cannot be serialized; use one of the "
            "built-in families to save a model"
        )


_BY_NAME = {
    "normal": Normal,
    "bernoulli": Bernoulli,
    "exponential": Exponential,
    "weibull": Weibull,
}


def get(name: str, **kwargs: Any) -> Family:
    """The built-in family called ``name`` (case-insensitive), built with ``kwargs``.

    Example: ``families.get("weibull", log_scale_bounds=(0, 5), log_shape_bounds=(-1, 1))``.
    """
    return core.lookup(_BY_NAME, name, **kwargs)


def from_config(name: str, config: Optional[dict] = None) -> Family:
    """The family called ``name`` rebuilt from its `get_config`."""
    return get(name, **(config or {}))
