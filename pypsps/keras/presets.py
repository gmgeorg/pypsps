"""Named argument sets for `models.build_model`, one per model the `build_*` builders make.

Each preset takes the arguments of the builder it replaces (same names, order and defaults)
and returns a new dict of `build_model` keyword arguments, compile keywords included:

    model = models.build_model(**presets.continuous_weibull(3, 10, [(16, "relu")], ...))

A preset only builds families, head specs and regularizer objects; it creates no layer and no
TF state. So the dict can be printed, compared and changed before building. To replace an
entry, merge a dict: ``build_model(**(presets.toy(3, 10) | {"alpha": tf.Variable(2.0)}))``.
With ``compile=False`` a preset leaves out the compile keywords, which `build_model` rejects
without compiling.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pypress.keras.regularizers
import tensorflow as tf

from . import families, losses
from .models import HeadSpec

_SURVIVAL_DISTRIBUTIONS = ("exponential", "weibull")


def _with_compile(arguments: Dict[str, Any], compile: bool, **compile_kwargs: Any) -> dict:
    """``arguments`` plus the compile keywords, or ``compile=False`` without them."""
    if compile:
        return {**arguments, **compile_kwargs}
    return {**arguments, "compile": False}


def _df_regularizer(n_states: int, df_penalty_l2: float) -> tf.keras.regularizers.Regularizer:
    """The degrees-of-freedom penalty of the toy and binary-normal models.

    The target ``max(1, log(n_states / 2))`` is 1.0 for K <= 5 and 1.6 for K = 10. At
    initialization the degrees of freedom are exactly 1 for any K and grow as the states
    differentiate, so a target well below K limits that from the first step. It is a starting
    point: tune ``target`` and ``l2`` for the data by passing your own regularizer.
    """
    return pypress.keras.regularizers.DegreesOfFreedom(
        l2=df_penalty_l2, target=max(1, np.log(0.5 * n_states))
    )


def toy(
    n_states: int,
    n_features: int,
    compile: bool = True,
    alpha: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
) -> Dict[str, Any]:
    """The toy model: binary treatment, Normal outcome, small fixed layers.

    A demo model. Same as `binary_normal` with state layers ``[(10, "relu"), (10, "selu")]``,
    outcome layers ``[(10, "tanh")]``, a ``(5, "selu")`` location head and a constant scale.
    The degrees-of-freedom penalty is described in `binary_normal`.
    """
    return binary_normal(
        n_states,
        n_features,
        predictive_state_hidden_layers=[(10, "relu"), (10, "selu")],
        outcome_hidden_layers=[(10, "tanh")],
        loc_layer=(5, "selu"),
        scale_layer=None,
        compile=compile,
        alpha=alpha,
        df_penalty_l2=df_penalty_l2,
        learning_rate=learning_rate,
    )


def binary_normal(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    loc_layer: Optional[Tuple[int, str]] = None,
    scale_layer: Optional[Tuple[int, str]] = None,
    compile: bool = True,
    alpha: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
    dropout_rate: float = 0.2,
) -> Dict[str, Any]:
    """Binary treatment and Normal outcome, on a shared outcome trunk.

    Args:
      n_states: number of predictive states.
      n_features: number of numeric features.
      predictive_state_hidden_layers: ``(units, activation)`` per layer of the state network.
      outcome_hidden_layers: ``(units, activation)`` per layer of the outcome trunk.
      loc_layer: ``(units, activation)`` of each state's location head; None is linear.
      scale_layer: ``(units, activation)`` of each state's scale head; None makes the scale a
        constant per state.
      compile: if False, the dict has no compile keywords.
      alpha: weight of the treatment log-likelihood in the loss.
      df_penalty_l2: l2 of the degrees-of-freedom penalty on the states. Its target is
        ``max(1, log(n_states / 2))``: 1.0 for K <= 5, 1.6 for K = 10. The degrees of
        freedom start at exactly 1 and grow as the states differentiate, so a target well
        below K limits that from the first step. Tune it for the data by replacing
        ``predictive_states_regularizer``.
      learning_rate: Nadam learning rate (no gradient clipping).
      dropout_rate: dropout rate of the hidden layers.

    Returns:
      `models.build_model` keyword arguments.
    """
    scale = HeadSpec(constant=True) if scale_layer is None else HeadSpec(hidden_layer=scale_layer)
    return _with_compile(
        dict(
            n_states=n_states,
            n_features=n_features,
            outcome_family=families.Normal(),
            treatment_family=families.Bernoulli(),
            architecture="shared_trunk",
            predictive_state_hidden_layers=predictive_state_hidden_layers,
            outcome_hidden_layers=outcome_hidden_layers,
            outcome_heads={"loc": HeadSpec(hidden_layer=loc_layer), "scale": scale},
            treatment_embedding_units=None,
            dropout_rate=dropout_rate,
        ),
        compile,
        alpha=alpha,
        learning_rate=learning_rate,
        clipnorm=None,
        predictive_states_regularizer=_df_regularizer(n_states, df_penalty_l2),
    )


def binary_exponential(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    log_rate_layer: Optional[Tuple[int, str]] = None,
    compile: bool = True,
    alpha: float = 1.0,
    outcome_loss_weight: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
    dropout_rate: float = 0.2,
) -> Dict[str, Any]:
    """Binary treatment and an exponential (log-rate) time-to-event outcome.

    Args:
      n_states: number of predictive states.
      n_features: number of numeric features.
      predictive_state_hidden_layers: ``(units, activation)`` per layer of the state network.
      outcome_hidden_layers: ``(units, activation)`` per layer of the outcome trunk.
      log_rate_layer: ``(units, activation)`` of each state's log-rate head; None is linear.
      compile: if False, the dict has no compile keywords.
      alpha: weight of the treatment log-likelihood in the loss.
      outcome_loss_weight: weight of the outcome log-likelihood in the loss.
      df_penalty_l2: l2 of the `pypress.keras.regularizers.Uniform` penalty on the states.
      learning_rate: Nadam learning rate (no gradient clipping).
      dropout_rate: dropout rate of the hidden layers.

    Returns:
      `models.build_model` keyword arguments.
    """
    return _with_compile(
        dict(
            n_states=n_states,
            n_features=n_features,
            outcome_family=families.Exponential(parameterization="log_rate", bounds=None),
            treatment_family=families.Bernoulli(),
            architecture="shared_trunk",
            predictive_state_hidden_layers=predictive_state_hidden_layers,
            outcome_hidden_layers=outcome_hidden_layers,
            outcome_heads={"log_rate": HeadSpec(hidden_layer=log_rate_layer)},
            treatment_embedding_units=None,
            dropout_rate=dropout_rate,
        ),
        compile,
        alpha=alpha,
        outcome_loss_weight=outcome_loss_weight,
        learning_rate=learning_rate,
        clipnorm=None,
        predictive_states_regularizer=pypress.keras.regularizers.Uniform(l2=df_penalty_l2),
    )


def continuous_survival(
    distribution: str,
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    log_scale_bounds: Tuple[float, float],
    log_shape_bounds: Optional[Tuple[float, float]] = None,
    max_time: Optional[float] = None,
    state_hidden_layer: Tuple[int, str] = (16, "swish"),
    treatment_embedding_units: int = 8,
    separate_shape_per_state: bool = True,
    init_scale: Optional[float] = None,
    treatment_init_values: Optional[Sequence[Union[float, Sequence[float]]]] = None,
    compile: bool = True,
    alpha: Union[float, tf.Variable] = 1.0,
    outcome_loss_weight: float = 1.0,
    outcome_likelihood: Union[
        losses.OutcomeLikelihood, str
    ] = losses.OutcomeLikelihood.INTERVENTIONAL,
    predictive_states_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    learning_rate: float = 0.01,
    clipnorm: Optional[float] = 1.0,
    dropout_rate: float = 0.2,
    l1: float = 0.0,
) -> Dict[str, Any]:
    """Continuous (Normal) treatment and an exponential or Weibull time-to-event outcome.

    The ``"gated"`` architecture: each outcome parameter (``log_scale``, and ``log_shape`` for
    Weibull) has its own trunk, and each state's head sees the treatment embedding gated by
    that state's weight. The parameters are soft-clipped to their bounds. ``y_true`` is
    ``[time, event indicator, treatment]``.

    Args:
      distribution: "exponential" or "weibull".
      n_states: number of predictive states.
      n_features: number of numeric features.
      predictive_state_hidden_layers: ``(units, activation)`` per layer of the state network.
      outcome_hidden_layers: ``(units, activation)`` per layer of each parameter's trunk.
      log_scale_bounds: (min, max) of the log scale, in log time units. Required.
      log_shape_bounds: (min, max) of the Weibull log shape. Required for Weibull only.
      max_time: optional largest time in the data; raises a ValueError when the bounds let
        the loss overflow float32 at that time.
      state_hidden_layer: ``(units, activation)`` of each state's head.
      treatment_embedding_units: width of the treatment embedding.
      separate_shape_per_state: Weibull only; False shares one shape head across states.
      init_scale: optional initial scale (in time units, > 0) of every state; defaults to the
        midpoint of ``log_scale_bounds`` on the log scale.
      treatment_init_values: optional initial ``(loc, scale)`` of the treatment density: two
        floats for every state, or two rows of one value per state.
      compile: if False, the dict has no compile keywords.
      alpha: weight of the treatment log-likelihood; a float32 tf.Variable is shared.
      outcome_loss_weight: weight of the outcome log-likelihood.
      outcome_likelihood: see `losses.OutcomeLikelihood`.
      predictive_states_regularizer: optional regularizer on the state weights.
      learning_rate: Nadam learning rate.
      clipnorm: gradient-norm clipping; None disables it.
      dropout_rate: dropout rate of the hidden layers.
      l1: L1 penalty on the hidden-layer kernels.

    Returns:
      `models.build_model` keyword arguments.
    """
    if distribution not in _SURVIVAL_DISTRIBUTIONS:
        raise ValueError(
            f"distribution must be one of {_SURVIVAL_DISTRIBUTIONS}, got {distribution!r}"
        )
    # The families allow unbounded parameters; these models require bounds.
    log_scale_bounds = families._check_bounds("log_scale_bounds", log_scale_bounds)
    heads = {"log_scale": HeadSpec(hidden_layer=state_hidden_layer)}
    if distribution == "weibull":
        log_shape_bounds = families._check_bounds("log_shape_bounds", log_shape_bounds)
        outcome_family = families.Weibull(log_scale_bounds, log_shape_bounds)
        heads["log_shape"] = HeadSpec(
            hidden_layer=state_hidden_layer, per_state=separate_shape_per_state
        )
    elif log_shape_bounds is not None:
        raise ValueError("log_shape_bounds applies to distribution='weibull' only")
    else:
        outcome_family = families.Exponential("log_scale", bounds=log_scale_bounds)

    outcome_init = None
    if init_scale is not None:
        if not init_scale > 0:
            raise ValueError(f"init_scale must be > 0, got {init_scale!r}")
        outcome_init = {"log_scale": float(np.log(init_scale))}
    treatment_init = None
    if treatment_init_values is not None:
        if len(treatment_init_values) != 2:
            raise ValueError(
                f"treatment_init_values must be (loc, scale), got {treatment_init_values!r}"
            )
        treatment_init = dict(zip(("loc", "scale"), treatment_init_values))

    return _with_compile(
        dict(
            n_states=n_states,
            n_features=n_features,
            outcome_family=outcome_family,
            treatment_family=families.Normal(),
            architecture="gated",
            predictive_state_hidden_layers=predictive_state_hidden_layers,
            outcome_hidden_layers=outcome_hidden_layers,
            outcome_heads=heads,
            treatment_embedding_units=treatment_embedding_units,
            dropout_rate=dropout_rate,
            l1=l1,
            outcome_init=outcome_init,
            treatment_init=treatment_init,
            max_time=max_time,
        ),
        compile,
        alpha=alpha,
        outcome_loss_weight=outcome_loss_weight,
        outcome_likelihood=outcome_likelihood,
        predictive_states_regularizer=predictive_states_regularizer,
        learning_rate=learning_rate,
        clipnorm=clipnorm,
    )


def continuous_exponential(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    log_scale_bounds: Tuple[float, float],
    **kwargs: Any,
) -> Dict[str, Any]:
    """`continuous_survival` with an exponential outcome (``log_scale = log mean``)."""
    return continuous_survival(
        "exponential",
        n_states=n_states,
        n_features=n_features,
        predictive_state_hidden_layers=predictive_state_hidden_layers,
        outcome_hidden_layers=outcome_hidden_layers,
        log_scale_bounds=log_scale_bounds,
        **kwargs,
    )


def continuous_weibull(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    log_scale_bounds: Tuple[float, float],
    log_shape_bounds: Tuple[float, float],
    **kwargs: Any,
) -> Dict[str, Any]:
    """`continuous_survival` with a Weibull outcome (``log_scale``, ``log_shape``)."""
    return continuous_survival(
        "weibull",
        n_states=n_states,
        n_features=n_features,
        predictive_state_hidden_layers=predictive_state_hidden_layers,
        outcome_hidden_layers=outcome_hidden_layers,
        log_scale_bounds=log_scale_bounds,
        log_shape_bounds=log_shape_bounds,
        **kwargs,
    )
