"""The argument logic of the named presets, shared by the backends.

Each function takes the backend (a `Backend`: its families and its regularizers) first and
returns a pair of dicts: the ``build_model`` arguments, and the training arguments (alpha,
learning rate, clipnorm, regularizer, ...). `pypsps.keras.presets` joins the two into one
``build_model`` call, since a Keras model is compiled by its builder; `pypsps.torch.presets`
keeps them apart, since a PyTorch model is trained by ``fit``. This module imports neither
TensorFlow nor PyTorch.
"""

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

import numpy as np

from . import families as core_families
from .spec import HeadSpec

SURVIVAL_DISTRIBUTIONS = ("exponential", "weibull")

Parts = tuple[dict[str, Any], dict[str, Any]]  # (build_model arguments, training arguments)


class Backend:
    """What a preset needs from a backend.

    Attributes:
      families: the backend's families module (``Normal``, ``Bernoulli``, ``Exponential``,
        ``Weibull``).
      degrees_of_freedom: ``(l2, target) -> regularizer``, the pypress degrees-of-freedom
        penalty of the backend.
      uniform: ``(l2) -> regularizer``, the pypress uniform-entropy penalty.
    """

    def __init__(
        self,
        families: ModuleType,
        degrees_of_freedom: Callable[..., Any],
        uniform: Callable[..., Any],
    ):
        """Stores the backend's families and regularizer constructors."""
        self.families = families
        self.degrees_of_freedom = degrees_of_freedom
        self.uniform = uniform


def df_regularizer(backend: Backend, n_states: int, df_penalty_l2: float) -> Any:
    """The degrees-of-freedom penalty of the toy and binary-normal models.

    The target ``max(1, log(n_states / 2))`` is 1.0 for K <= 5 and 1.6 for K = 10. At
    initialization the degrees of freedom are exactly 1 for any K and grow as the states
    differentiate, so a target well below K limits that from the first step. It is a starting
    point: tune ``target`` and ``l2`` for the data by passing your own regularizer.
    """
    return backend.degrees_of_freedom(l2=df_penalty_l2, target=max(1, np.log(0.5 * n_states)))


def toy(
    backend: Backend,
    n_states: int,
    n_features: int,
    alpha: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
) -> Parts:
    """The toy model: `binary_normal` with small fixed layers and a constant scale."""
    return binary_normal(
        backend,
        n_states,
        n_features,
        predictive_state_hidden_layers=[(10, "relu"), (10, "selu")],
        outcome_hidden_layers=[(10, "tanh")],
        loc_layer=(5, "selu"),
        scale_layer=None,
        alpha=alpha,
        df_penalty_l2=df_penalty_l2,
        learning_rate=learning_rate,
    )


def binary_normal(
    backend: Backend,
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: list[tuple[int, str]],
    outcome_hidden_layers: list[tuple[int, str]],
    loc_layer: tuple[int, str] | None = None,
    scale_layer: tuple[int, str] | None = None,
    alpha: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
    dropout_rate: float = 0.2,
) -> Parts:
    """Binary treatment and Normal outcome, on a shared outcome trunk."""
    fam = backend.families
    scale = HeadSpec(constant=True) if scale_layer is None else HeadSpec(hidden_layer=scale_layer)
    return (
        dict(
            n_states=n_states,
            n_features=n_features,
            outcome_family=fam.Normal(),
            treatment_family=fam.Bernoulli(),
            architecture="shared_trunk",
            predictive_state_hidden_layers=predictive_state_hidden_layers,
            outcome_hidden_layers=outcome_hidden_layers,
            outcome_heads={"loc": HeadSpec(hidden_layer=loc_layer), "scale": scale},
            treatment_embedding_units=None,
            dropout_rate=dropout_rate,
        ),
        dict(
            alpha=alpha,
            learning_rate=learning_rate,
            clipnorm=None,
            predictive_states_regularizer=df_regularizer(backend, n_states, df_penalty_l2),
        ),
    )


def binary_exponential(
    backend: Backend,
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: list[tuple[int, str]],
    outcome_hidden_layers: list[tuple[int, str]],
    log_rate_layer: tuple[int, str] | None = None,
    alpha: float = 1.0,
    outcome_loss_weight: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
    dropout_rate: float = 0.2,
) -> Parts:
    """Binary treatment and an exponential (log-rate) time-to-event outcome."""
    fam = backend.families
    return (
        dict(
            n_states=n_states,
            n_features=n_features,
            outcome_family=fam.Exponential(parameterization="log_rate", bounds=None),
            treatment_family=fam.Bernoulli(),
            architecture="shared_trunk",
            predictive_state_hidden_layers=predictive_state_hidden_layers,
            outcome_hidden_layers=outcome_hidden_layers,
            outcome_heads={"log_rate": HeadSpec(hidden_layer=log_rate_layer)},
            treatment_embedding_units=None,
            dropout_rate=dropout_rate,
        ),
        dict(
            alpha=alpha,
            outcome_loss_weight=outcome_loss_weight,
            learning_rate=learning_rate,
            clipnorm=None,
            predictive_states_regularizer=backend.uniform(l2=df_penalty_l2),
        ),
    )


def continuous_survival(
    backend: Backend,
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
    alpha: Any = 1.0,
    outcome_loss_weight: float = 1.0,
    outcome_likelihood: Any = "interventional",
    predictive_states_regularizer: Any = None,
    learning_rate: float = 0.01,
    clipnorm: float | None = 1.0,
    dropout_rate: float = 0.2,
    l1: float = 0.0,
) -> Parts:
    """Continuous (Normal) treatment and an exponential or Weibull time-to-event outcome."""
    fam = backend.families
    if distribution not in SURVIVAL_DISTRIBUTIONS:
        raise ValueError(
            f"distribution must be one of {SURVIVAL_DISTRIBUTIONS}, got {distribution!r}"
        )
    # The families allow unbounded parameters; these models require bounds.
    log_scale_bounds = core_families.check_bounds("log_scale_bounds", log_scale_bounds)
    heads = {"log_scale": HeadSpec(hidden_layer=state_hidden_layer)}
    if distribution == "weibull":
        log_shape_bounds = core_families.check_bounds("log_shape_bounds", log_shape_bounds)
        outcome_family = fam.Weibull(log_scale_bounds, log_shape_bounds)
        heads["log_shape"] = HeadSpec(
            hidden_layer=state_hidden_layer, per_state=separate_shape_per_state
        )
    elif log_shape_bounds is not None:
        raise ValueError("log_shape_bounds applies to distribution='weibull' only")
    else:
        outcome_family = fam.Exponential("log_scale", bounds=log_scale_bounds)

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

    return (
        dict(
            n_states=n_states,
            n_features=n_features,
            outcome_family=outcome_family,
            treatment_family=fam.Normal(),
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
        dict(
            alpha=alpha,
            outcome_loss_weight=outcome_loss_weight,
            outcome_likelihood=outcome_likelihood,
            predictive_states_regularizer=predictive_states_regularizer,
            learning_rate=learning_rate,
            clipnorm=clipnorm,
        ),
    )
