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

from collections.abc import Sequence
from typing import Any

import pypress.keras.regularizers
import tensorflow as tf

from ..core import presets as core_presets
from . import families, losses

_SURVIVAL_DISTRIBUTIONS = core_presets.SURVIVAL_DISTRIBUTIONS

_BACKEND = core_presets.Backend(
    families,
    degrees_of_freedom=pypress.keras.regularizers.DegreesOfFreedom,
    uniform=pypress.keras.regularizers.Uniform,
)


def _with_compile(parts: core_presets.Parts, compile: bool) -> dict:
    """The ``build_model`` arguments plus the compile keywords, or ``compile=False`` without."""
    arguments, compile_kwargs = parts
    if compile:
        return {**arguments, **compile_kwargs}
    return {**arguments, "compile": False}


def _df_regularizer(n_states: int, df_penalty_l2: float) -> tf.keras.regularizers.Regularizer:
    """The degrees-of-freedom penalty of the toy and binary-normal models; see
    `pypsps.core.presets.df_regularizer`."""
    return core_presets.df_regularizer(_BACKEND, n_states, df_penalty_l2)


def toy(
    n_states: int,
    n_features: int,
    compile: bool = True,
    alpha: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
) -> dict[str, Any]:
    """The toy model: binary treatment, Normal outcome, small fixed layers.

    A demo model. Same as `binary_normal` with state layers ``[(10, "relu"), (10, "selu")]``,
    outcome layers ``[(10, "tanh")]``, a ``(5, "selu")`` location head and a constant scale.
    The degrees-of-freedom penalty is described in `binary_normal`.
    """
    return _with_compile(
        core_presets.toy(_BACKEND, n_states, n_features, alpha, df_penalty_l2, learning_rate),
        compile,
    )


def binary_normal(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: list[tuple[int, str]],
    outcome_hidden_layers: list[tuple[int, str]],
    loc_layer: tuple[int, str] | None = None,
    scale_layer: tuple[int, str] | None = None,
    compile: bool = True,
    alpha: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
    dropout_rate: float = 0.2,
) -> dict[str, Any]:
    """Binary treatment and Normal outcome, on a shared outcome trunk.

    Warning: a binary model's ``.weights.h5`` file saved before v0.4.0 does not load into this
    model. ``load_weights`` raises ``ValueError``, but only after it has overwritten part of the
    model; discard that model. Migrate with ``models.copy_legacy_weights`` of pypsps v0.4.x.

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
    return _with_compile(
        core_presets.binary_normal(
            _BACKEND,
            n_states,
            n_features,
            predictive_state_hidden_layers,
            outcome_hidden_layers,
            loc_layer=loc_layer,
            scale_layer=scale_layer,
            alpha=alpha,
            df_penalty_l2=df_penalty_l2,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
        ),
        compile,
    )


def binary_exponential(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: list[tuple[int, str]],
    outcome_hidden_layers: list[tuple[int, str]],
    log_rate_layer: tuple[int, str] | None = None,
    compile: bool = True,
    alpha: float = 1.0,
    outcome_loss_weight: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
    dropout_rate: float = 0.2,
) -> dict[str, Any]:
    """Binary treatment and an exponential (log-rate) time-to-event outcome.

    Warning: a binary model's ``.weights.h5`` file saved before v0.4.0 does not load into this
    model. ``load_weights`` raises ``ValueError``, but only after it has overwritten part of the
    model; discard that model. Migrate with ``models.copy_legacy_weights`` of pypsps v0.4.x.

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
        core_presets.binary_exponential(
            _BACKEND,
            n_states,
            n_features,
            predictive_state_hidden_layers,
            outcome_hidden_layers,
            log_rate_layer=log_rate_layer,
            alpha=alpha,
            outcome_loss_weight=outcome_loss_weight,
            df_penalty_l2=df_penalty_l2,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
        ),
        compile,
    )


def continuous_survival(
    distribution: str,
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: list[tuple[int, str]],
    outcome_hidden_layers: list[tuple[int, str]],
    log_scale_bounds: tuple[float, float],
    log_shape_bounds: tuple[float, float] | None = None,
    max_time: float | None = None,
    state_hidden_layer: tuple[int, str] = (16, "swish"),
    treatment_embedding_units: int = 8,
    separate_shape_per_state: bool = True,
    init_scale: float | None = None,
    treatment_init_values: Sequence[float | Sequence[float]] | None = None,
    compile: bool = True,
    alpha: float | tf.Variable = 1.0,
    outcome_loss_weight: float = 1.0,
    outcome_likelihood: losses.OutcomeLikelihood | str = losses.OutcomeLikelihood.INTERVENTIONAL,
    predictive_states_regularizer: tf.keras.regularizers.Regularizer | None = None,
    learning_rate: float = 0.01,
    clipnorm: float | None = 1.0,
    dropout_rate: float = 0.2,
    l1: float = 0.0,
) -> dict[str, Any]:
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
    return _with_compile(
        core_presets.continuous_survival(
            _BACKEND,
            distribution,
            n_states,
            n_features,
            predictive_state_hidden_layers,
            outcome_hidden_layers,
            log_scale_bounds,
            log_shape_bounds=log_shape_bounds,
            max_time=max_time,
            state_hidden_layer=state_hidden_layer,
            treatment_embedding_units=treatment_embedding_units,
            separate_shape_per_state=separate_shape_per_state,
            init_scale=init_scale,
            treatment_init_values=treatment_init_values,
            alpha=alpha,
            outcome_loss_weight=outcome_loss_weight,
            outcome_likelihood=outcome_likelihood,
            predictive_states_regularizer=predictive_states_regularizer,
            learning_rate=learning_rate,
            clipnorm=clipnorm,
            dropout_rate=dropout_rate,
            l1=l1,
        ),
        compile,
    )


def continuous_exponential(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: list[tuple[int, str]],
    outcome_hidden_layers: list[tuple[int, str]],
    log_scale_bounds: tuple[float, float],
    **kwargs: Any,
) -> dict[str, Any]:
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
    predictive_state_hidden_layers: list[tuple[int, str]],
    outcome_hidden_layers: list[tuple[int, str]],
    log_scale_bounds: tuple[float, float],
    log_shape_bounds: tuple[float, float],
    **kwargs: Any,
) -> dict[str, Any]:
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
