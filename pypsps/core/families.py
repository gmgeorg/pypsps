"""Family specs: what is distribution-specific about a unit's outcome (or treatment).

A family owns everything distribution-specific within one predictive state: its parameter
names, how a raw head output is mapped onto each parameter's support (`ParamSpec`), the
bounds, the defaults and the validation. The tensor math (the elementwise negative
log-likelihood and the mean) is not here: `pypsps.keras.families` and `pypsps.torch.families`
subclass these specs and add ``nll`` and ``mean`` for their framework. This module imports
neither TensorFlow nor PyTorch.

Built-in families: `Normal`, `Bernoulli`, `Exponential` and `Weibull`. `LossFamily` wraps any
elementwise loss of the backend. `lookup` finds a family by name.
"""

import abc
import dataclasses
from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

Bounds = Optional[tuple[float, float]]

ACTIVATIONS = ("linear", "softplus", "sigmoid")

# float32 exp() overflows near 88.7; keep the largest exponent of the survival NLL below this,
# with a wide margin.
_MAX_SURVIVAL_EXPONENT = 50.0


def check_bounds(name: str, bounds: Any) -> tuple[float, float]:
    """Validates a ``(min, max)`` pair of finite floats with ``min < max``."""
    if bounds is None or len(bounds) != 2:
        raise ValueError(f"{name} must be a (min, max) pair, got {bounds!r}")
    low, high = float(bounds[0]), float(bounds[1])
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        raise ValueError(f"{name} must be finite with min < max, got {bounds!r}")
    return low, high


def _optional_bounds(name: str, bounds: Any) -> Bounds:
    """`check_bounds`, but ``None`` (unbounded) passes through."""
    return None if bounds is None else check_bounds(name, bounds)


def _midpoint(bounds: Bounds) -> float | None:
    """Midpoint of ``bounds``, or None if unbounded."""
    return None if bounds is None else 0.5 * (bounds[0] + bounds[1])


def _check_exponent(exponent: float) -> None:
    """Raises if a survival NLL exponent can overflow float32."""
    if exponent > _MAX_SURVIVAL_EXPONENT:
        raise ValueError(
            f"the largest exponent of the loss is {exponent:.1f}, which exceeds "
            f"{_MAX_SURVIVAL_EXPONENT}: the loss would overflow float32. Tighten the bounds "
            "(raise the lower log-scale bound, lower the upper log-shape or log-rate bound) "
            "or rescale time."
        )


def _check_max_time_positive(max_time: float) -> float:
    """Returns ``max_time`` as a float; raises unless it is positive."""
    if not max_time > 0:
        raise ValueError(f"max_time must be positive, got {max_time!r}")
    return float(max_time)


@dataclasses.dataclass(frozen=True)
class ParamSpec:
    """One parameter of a family, per state.

    Attributes:
      name: the parameter name, e.g. ``"loc"``.
      activation: ``"linear"``, ``"softplus"`` or ``"sigmoid"``; maps a raw head output onto
        the support. Unused if ``bounds`` is set.
      bounds: if set, the head is a soft clip into ``(min, max)`` and the support is that
        interval.
      init: optional initial value on the support (after the activation). A training default,
        so it is excluded from equality and from the family config.
    """

    name: str
    activation: str = "linear"
    bounds: Bounds = None
    init: float | None = dataclasses.field(default=None, compare=False)

    def __post_init__(self):
        """Validates the activation and the bounds."""
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}, got {self.activation!r}")
        object.__setattr__(self, "bounds", _optional_bounds(f"{self.name} bounds", self.bounds))

    def raw_value(self, value: float) -> float:
        """Inverse of the head activation: the raw output that maps to ``value``.

        linear: ``v``; softplus: ``log(expm1(v))``, ``v > 0``; sigmoid: ``logit(v)``,
        ``0 < v < 1``; bounds: the inverse of the soft clip, ``min < v < max``.

        Raises:
          ValueError: if ``value`` lies outside the support.
        """
        value = float(value)
        if self.bounds is not None:
            low, high = self.bounds
            if not low < value < high:
                raise ValueError(f"{self.name} must lie in ({low}, {high}), got {value!r}")
            frac = (value - low) / (high - low)
            return float(np.log(frac / (1.0 - frac)))
        if self.activation == "softplus":
            if not value > 0:
                raise ValueError(f"{self.name} must be > 0 (softplus), got {value!r}")
            return float(np.log(np.expm1(value)))
        if self.activation == "sigmoid":
            if not 0 < value < 1:
                raise ValueError(f"{self.name} must lie in (0, 1) (sigmoid), got {value!r}")
            return float(np.log(value / (1.0 - value)))
        return value


class Family(abc.ABC):
    """The spec of the distribution of a unit's outcome (or treatment) within one state.

    Subclasses are frozen dataclasses whose fields are their config, so ``==`` compares
    configs (and the backend: a Keras `Normal` never equals a PyTorch one). The same interface
    serves both roles; see the `OutcomeFamily` and `TreatmentFamily` aliases.
    """

    n_true_cols: int = 1  # C: the number of y_true columns the family reads

    @property
    @abc.abstractmethod
    def params(self) -> tuple[ParamSpec, ...]:
        """The per-state parameters, in ``y_pred`` order."""

    @property
    def param_names(self) -> tuple[str, ...]:
        """``tuple(p.name for p in self.params)``."""
        return tuple(p.name for p in self.params)

    def param(self, name: str) -> ParamSpec:
        """The `ParamSpec` called ``name``."""
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"{type(self).__name__} has no parameter {name!r}; has {self.param_names}")

    def response(self, y_true: Any) -> Any:
        """The ``[N, 1]`` column of ``y_true`` that `mean` predicts."""
        return y_true[:, :1]

    def event_indicator(self, y_true: Any) -> Any | None:
        """``[N, 1]`` 0/1 mask of units whose response is fully observed; None if all are."""
        del y_true
        return None

    @property
    def is_binary(self) -> bool:
        """True if the response is 0/1, so classification metrics apply."""
        return False

    @property
    def is_censored(self) -> bool:
        """True if the family masks units by an event indicator."""
        return False

    def get_config(self) -> dict:
        """The dataclass fields, with tuples as lists."""
        return {
            f.name: list(v) if isinstance(v, tuple) else v
            for f in dataclasses.fields(self)
            for v in [getattr(self, f.name)]
        }

    @classmethod
    def from_config(cls, config: dict) -> "Family":
        """Rebuilds a family from `get_config`."""
        return cls(**config)


OutcomeFamily = Family
TreatmentFamily = Family


class SurvivalFamily(Family):
    """A right-censored time-to-event family: ``y_true`` is ``[time, event]``."""

    n_true_cols = 2

    @property
    def is_censored(self) -> bool:
        """True."""
        return True

    def response(self, y_true: Any) -> Any:
        """The event (or censoring) time, ``y_true[:, 0:1]``."""
        return y_true[:, 0:1]

    @abc.abstractmethod
    def check_max_time(self, max_time: float) -> None:
        """Raises ValueError if the NLL can overflow float32 for times up to ``max_time``.

        Also raises if a bound the check needs is None, because then no bound can be proven.
        """


@dataclasses.dataclass(frozen=True)
class Normal(Family):
    """``y ~ Normal(loc, scale)``; ``scale`` goes through a softplus."""

    @property
    def params(self) -> tuple[ParamSpec, ...]:
        """``loc`` (linear) and ``scale`` (softplus)."""
        return (ParamSpec("loc"), ParamSpec("scale", activation="softplus"))


@dataclasses.dataclass(frozen=True)
class Bernoulli(Family):
    """``y ~ Bernoulli(prob)``; ``prob`` goes through a sigmoid."""

    @property
    def params(self) -> tuple[ParamSpec, ...]:
        """``prob`` (sigmoid), initialized at 0.5."""
        return (ParamSpec("prob", activation="sigmoid", init=0.5),)

    @property
    def is_binary(self) -> bool:
        """True."""
        return True


EXPONENTIAL_PARAMETERIZATIONS = ("log_scale", "log_rate", "rate")


@dataclasses.dataclass(frozen=True)
class Exponential(SurvivalFamily):
    """Right-censored exponential time to event.

    Attributes:
      parameterization: ``"log_scale"`` (log mean time; default), ``"log_rate"`` (log hazard)
        or ``"rate"`` (hazard, through a softplus).
      bounds: optional ``(min, max)`` of the ``log_scale`` or ``log_rate`` head.
    """

    parameterization: str = "log_scale"
    bounds: Bounds = None

    def __post_init__(self):
        """Validates the fields."""
        if self.parameterization not in EXPONENTIAL_PARAMETERIZATIONS:
            raise ValueError(
                f"parameterization must be one of {EXPONENTIAL_PARAMETERIZATIONS}, "
                f"got {self.parameterization!r}"
            )
        object.__setattr__(self, "bounds", _optional_bounds("bounds", self.bounds))
        if self.parameterization == "rate" and self.bounds is not None:
            raise ValueError("bounds apply to parameterization 'log_scale' or 'log_rate' only")

    @property
    def params(self) -> tuple[ParamSpec, ...]:
        """The one parameter of the chosen parameterization."""
        if self.parameterization == "rate":
            return (ParamSpec("rate", activation="softplus"),)
        return (ParamSpec(self.parameterization, bounds=self.bounds, init=_midpoint(self.bounds)),)

    def check_max_time(self, max_time: float) -> None:
        """Raises if ``t / scale`` (or ``rate * t``) can overflow float32 up to ``max_time``.

        ``log(max_time) - log_scale_min`` is the exponent of ``t / scale``;
        ``log(max_time) + log_rate_max`` that of ``rate * t``. An unbounded rate admits no
        bound at all.
        """
        log_time = float(np.log(_check_max_time_positive(max_time)))
        if self.parameterization == "rate":
            raise ValueError(
                "check_max_time needs a bounded rate; use parameterization 'log_rate' with bounds"
            )
        if self.bounds is None:
            raise ValueError(f"check_max_time needs bounds on {self.parameterization}")
        if self.parameterization == "log_scale":
            _check_exponent(log_time - self.bounds[0])
        else:
            _check_exponent(log_time + self.bounds[1])


@dataclasses.dataclass(frozen=True)
class Weibull(SurvivalFamily):
    """Right-censored Weibull time to event, parameterized by ``log_scale`` and ``log_shape``.

    Attributes:
      log_scale_bounds: optional ``(min, max)`` of the ``log_scale`` head.
      log_shape_bounds: optional ``(min, max)`` of the ``log_shape`` head.
    """

    log_scale_bounds: Bounds = None
    log_shape_bounds: Bounds = None

    def __post_init__(self):
        """Validates the bounds."""
        object.__setattr__(
            self, "log_scale_bounds", _optional_bounds("log_scale_bounds", self.log_scale_bounds)
        )
        object.__setattr__(
            self, "log_shape_bounds", _optional_bounds("log_shape_bounds", self.log_shape_bounds)
        )

    @property
    def params(self) -> tuple[ParamSpec, ...]:
        """``log_scale`` and ``log_shape``, soft-clipped if bounded."""
        shape_bounds = self.log_shape_bounds
        if shape_bounds is not None and shape_bounds[0] < 0.0 < shape_bounds[1]:
            shape_init = 0.0  # shape 1: the exponential
        else:
            shape_init = _midpoint(shape_bounds)
        return (
            ParamSpec(
                "log_scale", bounds=self.log_scale_bounds, init=_midpoint(self.log_scale_bounds)
            ),
            ParamSpec("log_shape", bounds=shape_bounds, init=shape_init),
        )

    def check_max_time(self, max_time: float) -> None:
        """Raises if ``(t / scale) ** shape`` can overflow float32 up to ``max_time``.

        The exponent is at most ``shape_max * (log(max_time) - log_scale_min)``.
        """
        log_time = float(np.log(_check_max_time_positive(max_time)))
        if self.log_scale_bounds is None or self.log_shape_bounds is None:
            raise ValueError("check_max_time needs log_scale_bounds and log_shape_bounds")
        shape_max = float(np.exp(self.log_shape_bounds[1]))
        _check_exponent(shape_max * (log_time - self.log_scale_bounds[0]))


@dataclasses.dataclass(frozen=True, eq=False)
class LossFamily(Family):
    """Any elementwise loss as a family; its mean is unknown.

    Attributes:
      loss: a loss with ``reduction="none"`` (or a plain function) that maps ``y_true [N, C]``
        and one state's ``[N, P]`` parameters to ``[N]``. A Keras loss in `pypsps.keras`, a
        callable on torch tensors in `pypsps.torch`.
      n_params: P, the parameters per state, named ``param_0, ...``.
      n_true_cols: C.
    """

    loss: Any
    n_params: int
    n_true_cols: int = 1

    def __post_init__(self):
        """Validates the loss reduction and the counts."""
        reduction = getattr(self.loss, "reduction", None)
        if reduction is not None and reduction != "none":
            raise ValueError(f"loss must have reduction='none', got {reduction!r}")
        if int(self.n_params) < 1 or int(self.n_true_cols) < 1:
            raise ValueError(
                f"need n_params >= 1 and n_true_cols >= 1, got {self.n_params}, {self.n_true_cols}"
            )
        object.__setattr__(self, "n_params", int(self.n_params))
        object.__setattr__(self, "n_true_cols", int(self.n_true_cols))

    @property
    def params(self) -> tuple[ParamSpec, ...]:
        """``param_0, ...`` (linear)."""
        return tuple(ParamSpec(f"param_{i}") for i in range(self.n_params))

    def __eq__(self, other: object):
        """Same type, same loss object and counts."""
        return (
            type(other) is type(self)
            and other.loss is self.loss
            and (other.n_params, other.n_true_cols) == (self.n_params, self.n_true_cols)
        )

    def __hash__(self):
        """Hashes by loss identity and counts, consistent with `__eq__`."""
        return hash((id(self.loss), self.n_params, self.n_true_cols))


def lookup(table: Mapping[str, type], name: str, **kwargs: Any) -> Family:
    """The family called ``name`` (case-insensitive) in ``table``, built with ``kwargs``."""
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"unknown family {name!r}; known: {tuple(table)}")
    return table[key](**kwargs)
