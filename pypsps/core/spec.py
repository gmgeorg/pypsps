"""The validated arguments of ``build_model``, shared by every backend.

`make_spec` checks every argument before any layer is created and returns a `Spec`; a
backend's ``build_model`` turns it into layers. This module imports neither TensorFlow nor
PyTorch.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from . import families
from .mixture import MixtureSpec

Architecture = Literal["shared_trunk", "gated"]
HiddenLayers = Sequence[tuple[int, str]]  # (units, activation) per Dense layer
ARCHITECTURES = ("shared_trunk", "gated")


def check_hidden_layer(entry: Any, what: str) -> tuple[int, str]:
    """Validates one ``(units, activation)`` entry and returns it as a tuple."""
    if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 2:
        raise ValueError(f"{what} must be a (units, activation) pair, got {entry!r}")
    units, activation = entry
    if isinstance(units, bool) or not isinstance(units, (int, np.integer)) or units < 1:
        raise ValueError(f"{what}: units must be a positive int, got {units!r}")
    if not isinstance(activation, str):
        raise ValueError(f"{what}: activation must be a string, got {activation!r}")
    return int(units), activation


def check_hidden_layers(entries: HiddenLayers, what: str) -> tuple[tuple[int, str], ...]:
    """Validates a list of ``(units, activation)`` entries; empty is allowed."""
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise ValueError(f"{what} must be a sequence of (units, activation), got {entries!r}")
    return tuple(check_hidden_layer(e, f"{what}[{i}]") for i, e in enumerate(entries))


@dataclasses.dataclass(frozen=True)
class HeadSpec:
    """How one outcome parameter's state-conditional outcome model is built.

    Attributes:
      hidden_layer: ``(units, activation)`` of one Dense layer before the raw ``Dense(1)``;
        None puts the raw ``Dense(1)`` directly on its input (a linear head).
      constant: True makes the parameter a feature-independent constant per state
        (`layers.BiasOnly`); ``hidden_layer`` is then ignored.
      per_state: False builds one head and uses it for every state.
    """

    hidden_layer: tuple[int, str] | None = (16, "swish")
    constant: bool = False
    per_state: bool = True

    def __post_init__(self):
        """Validates ``hidden_layer``."""
        if self.hidden_layer is not None:
            object.__setattr__(
                self, "hidden_layer", check_hidden_layer(self.hidden_layer, "HeadSpec.hidden_layer")
            )


@dataclasses.dataclass(frozen=True)
class Spec:
    """The validated arguments of `build_model`; the backend turns it into layers."""

    mixture: MixtureSpec
    n_features: int
    outcome_params: tuple[families.ParamSpec, ...]  # family params with init overrides
    treatment_init: np.ndarray | None  # (n_params, n_states) on the support, or None
    architecture: str
    predictive_state_hidden_layers: tuple[tuple[int, str], ...]
    outcome_hidden_layers: tuple[tuple[int, str], ...]
    heads: dict[str, HeadSpec]
    treatment_embedding_units: int | None
    dropout_rate: float
    l1: float

    @property
    def n_states(self) -> int:
        """K."""
        return self.mixture.n_states


def positive_int(value: Any, what: str) -> int:
    """Raises ValueError unless ``value`` is an int >= 1."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{what} must be an int >= 1, got {value!r}")
    return int(value)


def check_keys(given: Mapping[str, Any] | None, family: families.Family, what: str) -> dict:
    """Raises KeyError for a key that is not a parameter of ``family``."""
    given = dict(given or {})
    unknown = sorted(set(given) - set(family.param_names))
    if unknown:
        raise KeyError(
            f"{what} has unknown parameters {unknown}; {type(family).__name__} has "
            f"{family.param_names}"
        )
    return given


def treatment_init_array(
    family: families.Family, given: Mapping[str, Any] | None, n_states: int
) -> np.ndarray | None:
    """The ``(n_params, n_states)`` init of the treatment head, or None for Keras defaults.

    A value is one float for every state, or one float per state. Without ``given``, the
    family's defaults apply if every parameter has one (e.g. Bernoulli's 0.5).
    """
    params = family.params
    given = check_keys(given, family, "treatment_init")
    bounded = [p.name for p in params if p.bounds is not None]
    if not given:
        if bounded or any(p.init is None for p in params):
            return None
        given = {p.name: p.init for p in params}
    elif bounded:
        raise ValueError(
            f"treatment_init is not supported for bounded treatment parameters {bounded}"
        )
    missing = [p.name for p in params if p.name not in given]
    if missing:
        raise ValueError(f"treatment_init must give every treatment parameter; missing {missing}")
    rows = []
    for p in params:
        row = np.broadcast_to(np.asarray(given[p.name], dtype=float), (n_states,))
        for value in row:
            p.raw_value(value)  # raises outside the support
        rows.append(row)
    return np.stack(rows)


def make_spec(
    mixture_cls: type,
    n_states: int,
    n_features: int,
    outcome_family: families.Family,
    treatment_family: families.Family,
    architecture: str,
    predictive_state_hidden_layers: HiddenLayers,
    outcome_hidden_layers: HiddenLayers,
    outcome_heads: Mapping[str, HeadSpec] | None,
    treatment_embedding_units: int | None,
    dropout_rate: float,
    l1: float,
    outcome_init: Mapping[str, float] | None,
    treatment_init: Mapping[str, Any] | None,
    max_time: float | None,
) -> Spec:
    """Validates every `build_model` argument before any layer is created."""
    n_states = positive_int(n_states, "n_states")
    n_features = positive_int(n_features, "n_features")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")
    mixture = mixture_cls(outcome_family, treatment_family, n_states=n_states)

    heads = check_keys(outcome_heads, outcome_family, "outcome_heads")
    for name, head in heads.items():
        if not isinstance(head, HeadSpec):
            raise TypeError(f"outcome_heads[{name!r}] must be a HeadSpec, got {head!r}")
    heads = {p: heads.get(p, HeadSpec()) for p in outcome_family.param_names}

    inits = check_keys(outcome_init, outcome_family, "outcome_init")
    outcome_params = []
    for p in outcome_family.params:
        if p.name in inits:
            p = dataclasses.replace(p, init=float(inits[p.name]))
        if p.init is not None:
            p.raw_value(p.init)  # raises outside the support
        outcome_params.append(p)

    if max_time is not None:
        if not isinstance(outcome_family, families.SurvivalFamily):
            raise ValueError("max_time applies to survival outcome families")
        outcome_family.check_max_time(max_time)

    if treatment_embedding_units is not None:
        treatment_embedding_units = positive_int(
            treatment_embedding_units, "treatment_embedding_units"
        )
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(f"dropout_rate must lie in [0, 1), got {dropout_rate!r}")
    if l1 < 0:
        raise ValueError(f"l1 must be >= 0, got {l1!r}")

    return Spec(
        mixture=mixture,
        n_features=n_features,
        outcome_params=tuple(outcome_params),
        treatment_init=treatment_init_array(treatment_family, treatment_init, n_states),
        architecture=architecture,
        predictive_state_hidden_layers=check_hidden_layers(
            predictive_state_hidden_layers, "predictive_state_hidden_layers"
        ),
        outcome_hidden_layers=check_hidden_layers(outcome_hidden_layers, "outcome_hidden_layers"),
        heads=heads,
        treatment_embedding_units=treatment_embedding_units,
        dropout_rate=float(dropout_rate),
        l1=float(l1),
    )
