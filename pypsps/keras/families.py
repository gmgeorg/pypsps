"""Outcome and treatment families: one module per within-state distribution.

A family owns everything distribution-specific about a unit's outcome (or treatment) within
one predictive state: its parameter names, how a raw head output is mapped onto each
parameter's support, the elementwise negative log-likelihood, the mean, and which ``y_true``
column that mean predicts. The column order of ``y_pred`` is not its concern; see
`pypsps.keras.layout.ColumnLayout`.

Every per-state method takes ``{param name: [N, K]}`` blocks (as `ColumnLayout.decode_pred`
returns them) and gives ``[N, K]``: one value per unit and state.

Built-in families: `Normal`, `Bernoulli`, `Exponential` and `Weibull`. `LossFamily` wraps any
elementwise Keras loss. `get` looks a family up by name.
"""

import abc
import dataclasses
from typing import Any, Mapping, Optional

import numpy as np
import tensorflow as tf

from . import layers, neglogliks

Bounds = Optional[tuple[float, float]]

_ACTIVATIONS = ("linear", "softplus", "sigmoid")

# float32 exp() overflows near 88.7; keep the largest exponent of the survival NLL below this,
# with a wide margin.
_MAX_SURVIVAL_EXPONENT = 50.0


def _check_bounds(name: str, bounds: Any) -> tuple[float, float]:
    """Validates a ``(min, max)`` pair of finite floats with ``min < max``."""
    if bounds is None or len(bounds) != 2:
        raise ValueError(f"{name} must be a (min, max) pair, got {bounds!r}")
    low, high = float(bounds[0]), float(bounds[1])
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        raise ValueError(f"{name} must be finite with min < max, got {bounds!r}")
    return low, high


def _optional_bounds(name: str, bounds: Any) -> Bounds:
    """`_check_bounds`, but ``None`` (unbounded) passes through."""
    return None if bounds is None else _check_bounds(name, bounds)


def _midpoint(bounds: Bounds) -> Optional[float]:
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


def _float(x: Any) -> tf.Tensor:
    """``x`` as a float32 tensor."""
    return tf.cast(tf.convert_to_tensor(x), tf.float32)


def _pin_states(nll: tf.Tensor, params: Mapping[str, Any]) -> tf.Tensor:
    """Pins the static shape of a per-state result to ``[None, K]``."""
    n_states = next(iter(params.values())).shape[-1]
    return tf.ensure_shape(nll, [None, n_states])


@dataclasses.dataclass(frozen=True)
class ParamSpec:
    """One parameter of a family, per state.

    Attributes:
      name: the parameter name, e.g. ``"loc"``.
      activation: ``"linear"``, ``"softplus"`` or ``"sigmoid"``; maps a raw head output onto
        the support. Unused if ``bounds`` is set.
      bounds: if set, the head is ``SoftClip(*bounds)`` and the support is ``(min, max)``.
      init: optional initial value on the support (after the activation). A training default,
        so it is excluded from equality and from the family config.
    """

    name: str
    activation: str = "linear"
    bounds: Bounds = None
    init: Optional[float] = dataclasses.field(default=None, compare=False)

    def __post_init__(self):
        """Validates the activation and the bounds."""
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}, got {self.activation!r}")
        object.__setattr__(self, "bounds", _optional_bounds(f"{self.name} bounds", self.bounds))

    def head_activation(self, name: Optional[str] = None) -> tf.keras.layers.Layer:
        """Keras layer that maps a raw head output onto this parameter's support."""
        if self.bounds is not None:
            return layers.SoftClip(*self.bounds, name=name)
        return tf.keras.layers.Activation(self.activation, name=name)

    def raw_value(self, value: float) -> float:
        """Inverse of the head activation: the raw output that maps to ``value``.

        linear: ``v``; softplus: ``log(expm1(v))``, ``v > 0``; sigmoid: ``logit(v)``,
        ``0 < v < 1``; bounds: ``SoftClip(*bounds).inverse(v)``, ``min < v < max``.

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
    """The distribution of a unit's outcome (or treatment) within one predictive state.

    Subclasses are frozen dataclasses whose fields are their config, so ``==`` compares
    configs. The same interface serves both roles; see the `OutcomeFamily` and
    `TreatmentFamily` aliases.
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

    @abc.abstractmethod
    def nll(self, y_true: tf.Tensor, params: Mapping[str, tf.Tensor]) -> tf.Tensor:
        """-log p(y | state k) for every unit and state.

        Args:
          y_true: ``[N, C]``.
          params: ``{name: [N, K]}``.

        Returns:
          ``[N, K]``, elementwise with no reduction.
        """

    @abc.abstractmethod
    def mean(self, params: Mapping[str, tf.Tensor]) -> tf.Tensor:
        """E[response | state k]: ``{name: [N, K]}`` -> ``[N, K]``."""

    def response(self, y_true: tf.Tensor) -> tf.Tensor:
        """The ``[N, 1]`` column of ``y_true`` that `mean` predicts."""
        return y_true[:, :1]

    def event_indicator(self, y_true: tf.Tensor) -> Optional[tf.Tensor]:
        """``[N, 1]`` 0/1 mask of units whose response is fully observed; None if all are."""
        del y_true
        return None

    @property
    def is_binary(self) -> bool:
        """True if the response is 0/1, so classification metrics apply."""
        return False

    def head_activation(self, param: str, name: Optional[str] = None) -> tf.keras.layers.Layer:
        """Keras layer that maps a raw head output onto parameter ``param``'s support."""
        return self.param(param).head_activation(name=name)

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

    def response(self, y_true: tf.Tensor) -> tf.Tensor:
        """The event (or censoring) time, ``y_true[:, 0:1]``."""
        return y_true[:, 0:1]

    def event_indicator(self, y_true: tf.Tensor) -> tf.Tensor:
        """``y_true[:, 1:2]`` as floats: 1 if the event was observed, 0 if censored."""
        return tf.cast(y_true[:, 1:2], tf.keras.backend.floatx())

    def _time_and_event(self, y_true: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """``([N, 1] time, [N, 1] event)`` as float32."""
        y_true = _float(y_true)
        return y_true[:, 0:1], y_true[:, 1:2]

    @abc.abstractmethod
    def check_max_time(self, max_time: float) -> None:
        """Raises ValueError if the NLL can overflow float32 for times up to ``max_time``.

        Also raises if a bound the check needs is None, because then no bound can be proven.
        """


@tf.keras.utils.register_keras_serializable(package="pypsps")
@dataclasses.dataclass(frozen=True)
class Normal(Family):
    """``y ~ Normal(loc, scale)``; ``scale`` goes through a softplus."""

    @property
    def params(self) -> tuple[ParamSpec, ...]:
        """``loc`` (linear) and ``scale`` (softplus)."""
        return (ParamSpec("loc"), ParamSpec("scale", activation="softplus"))

    def nll(self, y_true: tf.Tensor, params: Mapping[str, tf.Tensor]):
        """Gaussian NLL via `neglogliks._negloglik_normal`."""
        nll = neglogliks._negloglik_normal(
            y=_float(y_true)[:, :1], loc=params["loc"], scale=params["scale"]
        )
        return _pin_states(nll, params)

    def mean(self, params: Mapping[str, tf.Tensor]):
        """``loc``."""
        return _float(params["loc"])


@tf.keras.utils.register_keras_serializable(package="pypsps")
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

    def nll(self, y_true: tf.Tensor, params: Mapping[str, tf.Tensor]):
        """Binary cross-entropy, computed exactly as `tf.keras.losses.BinaryCrossentropy`."""
        prob = _float(params["prob"])
        y = tf.zeros_like(prob) + _float(y_true)[:, :1]  # [N, K], exact broadcast
        nll = tf.keras.losses.binary_crossentropy(y[..., None], prob[..., None])
        return _pin_states(nll, params)

    def mean(self, params: Mapping[str, tf.Tensor]):
        """``prob``."""
        return _float(params["prob"])


class _ExponentialLogScale:
    """Exponential with parameter ``log_scale = log(mean time)``."""

    def __init__(self, bounds: Bounds):
        """Stores the optional ``log_scale`` bounds."""
        self.params = (ParamSpec("log_scale", bounds=bounds, init=_midpoint(bounds)),)

    def nll(self, time: tf.Tensor, event: tf.Tensor, params: Mapping[str, tf.Tensor]):
        """Censored NLL via `neglogliks._negloglik_exponential_scale`."""
        return neglogliks._negloglik_exponential_scale(time, event, params["log_scale"])

    def mean(self, params: Mapping[str, tf.Tensor]):
        """``exp(log_scale)``."""
        return tf.exp(_float(params["log_scale"]))

    def max_exponent(self, log_time: float) -> float:
        """``log(max_time) - log_scale_min``: the exponent of ``t / scale``."""
        bounds = self.params[0].bounds
        if bounds is None:
            raise ValueError("check_max_time needs bounds on log_scale")
        return log_time - bounds[0]


class _ExponentialLogRate:
    """Exponential with parameter ``log_rate = log(hazard)``."""

    def __init__(self, bounds: Bounds):
        """Stores the optional ``log_rate`` bounds."""
        self.params = (ParamSpec("log_rate", bounds=bounds, init=_midpoint(bounds)),)

    def nll(self, time: tf.Tensor, event: tf.Tensor, params: Mapping[str, tf.Tensor]):
        """Censored NLL via `neglogliks._negloglik_exponential` with ``rate = exp(log_rate)``."""
        return neglogliks._negloglik_exponential(time, event, tf.exp(params["log_rate"]))

    def mean(self, params: Mapping[str, tf.Tensor]):
        """``exp(-log_rate)``."""
        return tf.exp(-_float(params["log_rate"]))

    def max_exponent(self, log_time: float) -> float:
        """``log(max_time) + log_rate_max``: the exponent of ``rate * t``."""
        bounds = self.params[0].bounds
        if bounds is None:
            raise ValueError("check_max_time needs bounds on log_rate")
        return log_time + bounds[1]


class _ExponentialRate:
    """Exponential with parameter ``rate`` (hazard), through a softplus."""

    def __init__(self, bounds: Bounds):
        """Rejects bounds: the rate is constrained by its softplus head."""
        if bounds is not None:
            raise ValueError("bounds apply to parameterization 'log_scale' or 'log_rate' only")
        self.params = (ParamSpec("rate", activation="softplus"),)

    def nll(self, time: tf.Tensor, event: tf.Tensor, params: Mapping[str, tf.Tensor]):
        """Censored NLL via `neglogliks._negloglik_exponential`."""
        return neglogliks._negloglik_exponential(time, event, params["rate"])

    def mean(self, params: Mapping[str, tf.Tensor]):
        """``1 / rate``."""
        return 1.0 / _float(params["rate"])

    def max_exponent(self, log_time: float) -> float:
        """Always raises: an unbounded rate admits no overflow bound."""
        raise ValueError(
            "check_max_time needs a bounded rate; use parameterization 'log_rate' with bounds"
        )


_EXPONENTIAL_IMPLS = {
    "log_scale": _ExponentialLogScale,
    "log_rate": _ExponentialLogRate,
    "rate": _ExponentialRate,
}


@tf.keras.utils.register_keras_serializable(package="pypsps")
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
        """Validates the fields and picks the private implementation."""
        if self.parameterization not in _EXPONENTIAL_IMPLS:
            raise ValueError(
                f"parameterization must be one of {tuple(_EXPONENTIAL_IMPLS)}, "
                f"got {self.parameterization!r}"
            )
        object.__setattr__(self, "bounds", _optional_bounds("bounds", self.bounds))
        object.__setattr__(self, "_impl", _EXPONENTIAL_IMPLS[self.parameterization](self.bounds))

    @property
    def params(self) -> tuple[ParamSpec, ...]:
        """The one parameter of the chosen parameterization."""
        return self._impl.params

    def nll(self, y_true: tf.Tensor, params: Mapping[str, tf.Tensor]):
        """Censored exponential NLL."""
        time, event = self._time_and_event(y_true)
        return _pin_states(self._impl.nll(time, event, params), params)

    def mean(self, params: Mapping[str, tf.Tensor]):
        """The mean time: ``exp(log_scale)``, ``exp(-log_rate)`` or ``1 / rate``."""
        return self._impl.mean(params)

    def check_max_time(self, max_time: float) -> None:
        """Raises if ``t / scale`` (or ``rate * t``) can overflow float32 up to ``max_time``."""
        log_time = float(np.log(_check_max_time_positive(max_time)))
        _check_exponent(self._impl.max_exponent(log_time))


@tf.keras.utils.register_keras_serializable(package="pypsps")
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

    def nll(self, y_true: tf.Tensor, params: Mapping[str, tf.Tensor]):
        """Censored Weibull NLL via `neglogliks._negloglik_weibull`."""
        time, event = self._time_and_event(y_true)
        nll = neglogliks._negloglik_weibull(
            time, event, log_scale=params["log_scale"], log_shape=params["log_shape"]
        )
        return _pin_states(nll, params)

    def mean(self, params: Mapping[str, tf.Tensor]):
        """``exp(log_scale) * Gamma(1 + 1 / shape)``."""
        log_scale = _float(params["log_scale"])
        log_shape = _float(params["log_shape"])
        return tf.exp(log_scale) * tf.exp(tf.math.lgamma(1.0 + tf.exp(-log_shape)))

    def check_max_time(self, max_time: float) -> None:
        """Raises if ``(t / scale) ** shape`` can overflow float32 up to ``max_time``.

        The exponent is at most ``shape_max * (log(max_time) - log_scale_min)``.
        """
        log_time = float(np.log(_check_max_time_positive(max_time)))
        if self.log_scale_bounds is None or self.log_shape_bounds is None:
            raise ValueError("check_max_time needs log_scale_bounds and log_shape_bounds")
        shape_max = float(np.exp(self.log_shape_bounds[1]))
        _check_exponent(shape_max * (log_time - self.log_scale_bounds[0]))


@tf.keras.utils.register_keras_serializable(package="pypsps")
@dataclasses.dataclass(frozen=True, eq=False)
class LossFamily(Family):
    """Any elementwise Keras loss as a family; its mean is unknown.

    Attributes:
      loss: a loss with ``reduction="none"`` (or a plain function) that maps ``y_true [N, C]``
        and one state's ``[N, P]`` parameters to ``[N]``.
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

    def nll(self, y_true: tf.Tensor, params: Mapping[str, tf.Tensor]):
        """Calls the loss once per state on that state's ``[N, P]`` parameters."""
        y_true = _float(y_true)
        if y_true.shape.rank == 1:
            y_true = y_true[:, None]
        blocks = [params[name] for name in self.param_names]
        n_states = blocks[0].shape[-1]
        per_state = []
        for j in range(n_states):
            state_params = tf.concat([b[:, j : j + 1] for b in blocks], axis=1)
            per_state.append(self.loss(y_true, state_params))
        return _pin_states(tf.stack(per_state, axis=1), params)

    def mean(self, params: Mapping[str, tf.Tensor]):
        """Raises: a generic loss does not say what its mean is."""
        raise NotImplementedError(
            f"the mean of a LossFamily ({type(self.loss).__name__}) is unknown; use one of the "
            "built-in families (Normal, Bernoulli, Exponential, Weibull) instead"
        )

    def __eq__(self, other: object):
        """Same loss object and counts."""
        return (
            type(other) is LossFamily
            and other.loss is self.loss
            and (other.n_params, other.n_true_cols) == (self.n_params, self.n_true_cols)
        )

    def __hash__(self):
        """Hashes by loss identity and counts, consistent with `__eq__`."""
        return hash((id(self.loss), self.n_params, self.n_true_cols))

    def get_config(self) -> dict:
        """Serializes the loss with Keras."""
        return {
            "loss": tf.keras.utils.serialize_keras_object(self.loss),
            "n_params": self.n_params,
            "n_true_cols": self.n_true_cols,
        }

    @classmethod
    def from_config(cls, config: dict) -> "LossFamily":
        """Rebuilds the family and its loss."""
        config = dict(config)
        config["loss"] = tf.keras.utils.deserialize_keras_object(config["loss"])
        return cls(**config)


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
    key = str(name).lower()
    if key not in _BY_NAME:
        raise ValueError(f"unknown family {name!r}; known: {tuple(_BY_NAME)}")
    return _BY_NAME[key](**kwargs)
