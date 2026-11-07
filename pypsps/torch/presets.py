"""Named argument sets for `models.build_model` on PyTorch.

The same presets as `pypsps.keras.presets` (same names, arguments, defaults and values; the
argument logic is shared in `pypsps.core.presets`), with the regularizers from
`pypress.torch`. A PyTorch model is built by ``build_model`` and trained by ``fit``, so a
preset has two parts: the ``build_model`` arguments (the dict itself) and the ``fit``
arguments (``preset.fit_kwargs``: alpha, learning rate, clipnorm, regularizer, ...):

    preset = presets.continuous_weibull(3, 10, [(16, "relu")], [(16, "relu")], ...)
    model = models.build_model(**preset)
    history = training.fit(model, features, treatment, outcome, epochs=50, **preset.fit_kwargs)

A preset only builds families, head specs and regularizer objects; it creates no layer. So the
dict can be printed, compared and changed before building.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pypress.torch.regularizers as pr_regularizers

from ..core import presets as core_presets
from ..core.likelihood import OutcomeLikelihood
from . import families

_BACKEND = core_presets.Backend(
    families,
    degrees_of_freedom=pr_regularizers.DegreesOfFreedom,
    uniform=pr_regularizers.Uniform,
)


class Preset(dict):
    """``build_model`` keyword arguments, with the ``fit`` keyword arguments alongside.

    Unpack the dict into `pypsps.torch.models.build_model`; unpack ``fit_kwargs`` into
    `pypsps.torch.training.fit`.
    """

    fit_kwargs: Dict[str, Any]

    def __init__(self, parts: core_presets.Parts):
        """Splits the ``(build_model arguments, fit arguments)`` pair."""
        arguments, fit_kwargs = parts
        super().__init__(arguments)
        self.fit_kwargs = dict(fit_kwargs)


def toy(
    n_states: int,
    n_features: int,
    alpha: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
) -> Preset:
    """The toy model: `binary_normal` with state layers ``[(10, "relu"), (10, "selu")]``,
    outcome layers ``[(10, "tanh")]``, a ``(5, "selu")`` location head and a constant scale."""
    return Preset(
        core_presets.toy(_BACKEND, n_states, n_features, alpha, df_penalty_l2, learning_rate)
    )


def binary_normal(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    loc_layer: Optional[Tuple[int, str]] = None,
    scale_layer: Optional[Tuple[int, str]] = None,
    alpha: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
    dropout_rate: float = 0.2,
) -> Preset:
    """Binary treatment and Normal outcome, on a shared outcome trunk.

    The degrees-of-freedom penalty has target ``max(1, log(n_states / 2))`` and l2
    ``df_penalty_l2``; the learning rate is that of a Nadam-style optimizer without gradient
    clipping. See `pypsps.keras.presets.binary_normal` for the arguments.
    """
    return Preset(
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
        )
    )


def binary_exponential(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    log_rate_layer: Optional[Tuple[int, str]] = None,
    alpha: float = 1.0,
    outcome_loss_weight: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
    dropout_rate: float = 0.2,
) -> Preset:
    """Binary treatment and an exponential (log-rate) time-to-event outcome.

    The states carry a `pypress.torch.regularizers.Uniform` penalty with l2 ``df_penalty_l2``.
    See `pypsps.keras.presets.binary_exponential` for the arguments.
    """
    return Preset(
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
        )
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
    alpha: float = 1.0,
    outcome_loss_weight: float = 1.0,
    outcome_likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
    predictive_states_regularizer: Any = None,
    learning_rate: float = 0.01,
    clipnorm: Optional[float] = 1.0,
    dropout_rate: float = 0.2,
    l1: float = 0.0,
) -> Preset:
    """Continuous (Normal) treatment and an exponential or Weibull time-to-event outcome.

    The ``"gated"`` architecture, with soft-clipped parameters. See
    `pypsps.keras.presets.continuous_survival` for the arguments; the torch version has no
    ``compile`` argument, and ``alpha`` is a float.
    """
    return Preset(
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
        )
    )


def continuous_exponential(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    log_scale_bounds: Tuple[float, float],
    **kwargs: Any,
) -> Preset:
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
) -> Preset:
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
