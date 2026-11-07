"""Outcome and treatment families for Keras: one class per within-state distribution.

A family owns everything distribution-specific about a unit's outcome (or treatment) within
one predictive state: its parameter names, how a raw head output is mapped onto each
parameter's support, the elementwise negative log-likelihood, the mean, and which ``y_true``
column that mean predicts. The column order of ``y_pred`` is not its concern; see
`pypsps.keras.layout.ColumnLayout`.

The parameter specs, bounds and validation are framework-free and live in
`pypsps.core.families`; the classes here add the TensorFlow ``nll`` and ``mean``.

Every per-state method takes ``{param name: [N, K]}`` blocks (as `ColumnLayout.decode_pred`
returns them) and gives ``[N, K]``: one value per unit and state.

Built-in families: `Normal`, `Bernoulli`, `Exponential` and `Weibull`. `LossFamily` wraps any
elementwise Keras loss. `get` looks a family up by name.
"""

import abc
from collections.abc import Mapping
from typing import Any

import tensorflow as tf

from ..core import families as core
from ..core.families import (  # noqa: F401  (re-exported)
    Bounds,
    ParamSpec,
    _check_exponent,
    _midpoint,
    _optional_bounds,
    check_bounds,
)
from . import layers, neglogliks

_check_bounds = check_bounds


def _float(x: Any) -> tf.Tensor:
    """``x`` as a float32 tensor."""
    return tf.cast(tf.convert_to_tensor(x), tf.float32)


def _pin_states(nll: tf.Tensor, params: Mapping[str, Any]) -> tf.Tensor:
    """Pins the static shape of a per-state result to ``[None, K]``."""
    n_states = next(iter(params.values())).shape[-1]
    return tf.ensure_shape(nll, [None, n_states])


def head_activation(param: ParamSpec, name: str | None = None) -> tf.keras.layers.Layer:
    """Keras layer that maps a raw head output onto ``param``'s support."""
    if param.bounds is not None:
        return layers.SoftClip(*param.bounds, name=name)
    return tf.keras.layers.Activation(param.activation, name=name)


class Family(core.Family):
    """The distribution of a unit's outcome (or treatment) within one predictive state.

    Subclasses are frozen dataclasses whose fields are their config, so ``==`` compares
    configs. The same interface serves both roles; see the `OutcomeFamily` and
    `TreatmentFamily` aliases.
    """

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

    def head_activation(self, param: str, name: str | None = None) -> tf.keras.layers.Layer:
        """Keras layer that maps a raw head output onto parameter ``param``'s support."""
        return head_activation(self.param(param), name=name)


OutcomeFamily = Family
TreatmentFamily = Family


class SurvivalFamily(core.SurvivalFamily, Family):
    """A right-censored time-to-event family: ``y_true`` is ``[time, event]``."""

    def event_indicator(self, y_true: tf.Tensor) -> tf.Tensor:
        """``y_true[:, 1:2]`` as floats: 1 if the event was observed, 0 if censored."""
        return tf.cast(y_true[:, 1:2], tf.keras.backend.floatx())

    def _time_and_event(self, y_true: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """``([N, 1] time, [N, 1] event)`` as float32."""
        y_true = _float(y_true)
        return y_true[:, 0:1], y_true[:, 1:2]


@tf.keras.utils.register_keras_serializable(package="pypsps")
class Normal(core.Normal, Family):
    """``y ~ Normal(loc, scale)``; ``scale`` goes through a softplus."""

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
class Bernoulli(core.Bernoulli, Family):
    """``y ~ Bernoulli(prob)``; ``prob`` goes through a sigmoid."""

    def nll(self, y_true: tf.Tensor, params: Mapping[str, tf.Tensor]):
        """Binary cross-entropy, computed exactly as `tf.keras.losses.BinaryCrossentropy`."""
        prob = _float(params["prob"])
        y = tf.zeros_like(prob) + _float(y_true)[:, :1]  # [N, K], exact broadcast
        nll = tf.keras.losses.binary_crossentropy(y[..., None], prob[..., None])
        return _pin_states(nll, params)

    def mean(self, params: Mapping[str, tf.Tensor]):
        """``prob``."""
        return _float(params["prob"])


@tf.keras.utils.register_keras_serializable(package="pypsps")
class Exponential(core.Exponential, SurvivalFamily):
    """Right-censored exponential time to event.

    Attributes:
      parameterization: ``"log_scale"`` (log mean time; default), ``"log_rate"`` (log hazard)
        or ``"rate"`` (hazard, through a softplus).
      bounds: optional ``(min, max)`` of the ``log_scale`` or ``log_rate`` head.
    """

    def nll(self, y_true: tf.Tensor, params: Mapping[str, tf.Tensor]):
        """Censored exponential NLL."""
        time, event = self._time_and_event(y_true)
        if self.parameterization == "log_scale":
            nll = neglogliks._negloglik_exponential_scale(time, event, params["log_scale"])
        elif self.parameterization == "log_rate":
            nll = neglogliks._negloglik_exponential(time, event, tf.exp(params["log_rate"]))
        else:
            nll = neglogliks._negloglik_exponential(time, event, params["rate"])
        return _pin_states(nll, params)

    def mean(self, params: Mapping[str, tf.Tensor]):
        """The mean time: ``exp(log_scale)``, ``exp(-log_rate)`` or ``1 / rate``."""
        if self.parameterization == "log_scale":
            return tf.exp(_float(params["log_scale"]))
        if self.parameterization == "log_rate":
            return tf.exp(-_float(params["log_rate"]))
        return 1.0 / _float(params["rate"])


@tf.keras.utils.register_keras_serializable(package="pypsps")
class Weibull(core.Weibull, SurvivalFamily):
    """Right-censored Weibull time to event, parameterized by ``log_scale`` and ``log_shape``.

    Attributes:
      log_scale_bounds: optional ``(min, max)`` of the ``log_scale`` head.
      log_shape_bounds: optional ``(min, max)`` of the ``log_shape`` head.
    """

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


@tf.keras.utils.register_keras_serializable(package="pypsps")
class LossFamily(core.LossFamily, Family):
    """Any elementwise Keras loss as a family; its mean is unknown.

    Attributes:
      loss: a loss with ``reduction="none"`` (or a plain function) that maps ``y_true [N, C]``
        and one state's ``[N, P]`` parameters to ``[N]``.
      n_params: P, the parameters per state, named ``param_0, ...``.
      n_true_cols: C.
    """

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
    return core.lookup(_BY_NAME, name, **kwargs)
