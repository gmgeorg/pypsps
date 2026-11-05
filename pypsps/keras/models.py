"""Model builders for pypsps: `build_model`, `compile_model`, and the older `build_*`."""

import dataclasses
import inspect
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pypress
import pypress.keras.layers
import pypress.keras.regularizers
import tensorflow as tf

from . import families, layers, losses, metrics
from . import mixture as mixture_lib

tfk = tf.keras


def build_toy_model(
    n_states: int,
    n_features: int,
    compile: bool = True,
    alpha: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
) -> tf.keras.Model:
    """Deprecated: use ``build_model(**presets.toy(...))``; removed in v0.5.0.

    The arguments are those of `presets.toy`, which documents them.
    """
    arguments = dict(locals())
    _warn_deprecated("build_toy_model", "toy")
    # Lazy: presets imports this module.
    from . import presets

    return build_model(**presets.toy(**arguments))


def build_model_binary_normal(
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
) -> tf.keras.Model:
    """Deprecated: use ``build_model(**presets.binary_normal(...))``; removed in v0.5.0.

    The arguments are those of `presets.binary_normal`, which documents them. Weight files
    saved before v0.4.0 don't load into the new model; see `copy_legacy_weights`.
    """
    arguments = dict(locals())
    _warn_deprecated("build_model_binary_normal", "binary_normal")
    # Lazy: presets imports this module.
    from . import presets

    return build_model(**presets.binary_normal(**arguments))


def build_model_binary_exponential(
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
) -> tf.keras.Model:
    """Deprecated: use ``build_model(**presets.binary_exponential(...))``; removed in v0.5.0.

    The arguments are those of `presets.binary_exponential`, which documents them. Weight
    files saved before v0.4.0 don't load into the new model; see `copy_legacy_weights`.
    """
    arguments = dict(locals())
    _warn_deprecated("build_model_binary_exponential", "binary_exponential")
    # Lazy: presets imports this module.
    from . import presets

    return build_model(**presets.binary_exponential(**arguments))


def _dense_stack(
    x: tf.Tensor,
    hidden_layers: List[Tuple[int, str]],
    dropout_rate: float,
    l1: float,
    name: str,
) -> tf.Tensor:
    """Applies Dense -> Dropout -> BatchNormalization for each ``(units, activation)``."""
    for i, (units, activation) in enumerate(hidden_layers):
        x = tfk.layers.Dense(
            units,
            activation,
            kernel_regularizer=tf.keras.regularizers.l1(l1) if l1 > 0 else None,
            name=f"{name}_dense_{i}",
        )(x)
        x = tfk.layers.Dropout(dropout_rate, name=f"{name}_dropout_{i}")(x)
        x = tfk.layers.BatchNormalization(name=f"{name}_bn_{i}")(x)
    return x


def _warn_deprecated(name: str, preset: str) -> None:
    """The `FutureWarning` of a `build_*` builder that is now a preset of `build_model`."""
    warnings.warn(
        f"{name} is deprecated and will be removed in v0.5.0; use "
        f"models.build_model(**presets.{preset}(...)) with the same arguments",
        FutureWarning,
        stacklevel=3,
    )


def build_model_continuous_survival(
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
) -> tf.keras.Model:
    """Deprecated: use ``build_model(**presets.continuous_survival(...))``; removed in v0.5.0.

    The arguments are those of `presets.continuous_survival`, which documents them.
    """
    arguments = dict(locals())
    _warn_deprecated("build_model_continuous_survival", "continuous_survival")
    # Lazy: presets imports this module.
    from . import presets

    return build_model(**presets.continuous_survival(**arguments))


def build_model_continuous_exponential(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    log_scale_bounds: Tuple[float, float],
    **kwargs: Any,
) -> tf.keras.Model:
    """Deprecated: use ``build_model(**presets.continuous_exponential(...))``; removed in v0.5.0."""
    _warn_deprecated("build_model_continuous_exponential", "continuous_exponential")
    # Lazy: presets imports this module.
    from . import presets

    return build_model(
        **presets.continuous_exponential(
            n_states,
            n_features,
            predictive_state_hidden_layers,
            outcome_hidden_layers,
            log_scale_bounds,
            **kwargs,
        )
    )


def build_model_continuous_weibull(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    log_scale_bounds: Tuple[float, float],
    log_shape_bounds: Tuple[float, float],
    **kwargs: Any,
) -> tf.keras.Model:
    """Deprecated: use ``build_model(**presets.continuous_weibull(...))``; removed in v0.5.0."""
    _warn_deprecated("build_model_continuous_weibull", "continuous_weibull")
    # Lazy: presets imports this module.
    from . import presets

    return build_model(
        **presets.continuous_weibull(
            n_states,
            n_features,
            predictive_state_hidden_layers,
            outcome_hidden_layers,
            log_scale_bounds,
            log_shape_bounds,
            **kwargs,
        )
    )


def _consumers(model: tf.keras.Model, tensor: Any) -> list:
    """The layers of ``model`` that take ``tensor`` as (one of) their inputs."""
    return [
        layer
        for layer in model.layers
        if not isinstance(layer, tfk.layers.InputLayer)
        and any(t is tensor for t in tf.nest.flatten(layer.input))
    ]


def _weighted_chain(model: tf.keras.Model, first: tf.keras.layers.Layer) -> list:
    """The weighted layers from ``first`` along its single-consumer chain, up to the first
    layer without weights that is not Dropout (the skip ``Concatenate``)."""
    chain, layer = [], first
    while True:
        if layer.weights:
            chain.append(layer)
        elif not isinstance(layer, tfk.layers.Dropout):
            return chain
        nxt = _consumers(model, layer.output)
        if len(nxt) != 1:
            return chain
        layer = nxt[0]


def _legacy_pairs(old: tf.keras.Model, new: tf.keras.Model) -> list:
    """``(old layer, new layer)`` for every weighted layer of a pre-v0.4.0 binary model,
    except the propensity logits; raises ValueError if the two models don't match."""
    mixture = mixture_lib.StateMixture.from_model(new)
    if mixture != mixture_lib.StateMixture.from_model(old):
        raise ValueError("old and new models have different families or n_states")
    n_states = mixture.n_states

    features_bn = _consumers(old, old.inputs[0])
    features_bn = [x for x in features_bn if isinstance(x, tfk.layers.BatchNormalization)]
    if len(features_bn) != 1:
        raise ValueError("cannot find the old model's feature BatchNormalization")
    state_dense = [
        x for x in _consumers(old, features_bn[0].output) if isinstance(x, tfk.layers.Dense)
    ]
    if len(state_dense) != 1:
        raise ValueError("cannot find the old model's predictive-state network")
    old_state = _weighted_chain(old, state_dense[0])
    old_trunk = _weighted_chain(old, old.get_layer("inputs_processing"))

    def chain(prefix: str) -> list:
        return [
            x
            for x in new.layers
            if x.name.startswith(prefix)
            and isinstance(x, (tfk.layers.Dense, tfk.layers.BatchNormalization))
        ]

    new_state = chain("predictive_state_")
    new_trunk = chain("outcome_trunk_")
    if len(old_state) != len(new_state) or len(old_trunk) != len(new_trunk):
        raise ValueError(
            "the hidden stacks differ (e.g. a toy model, whose state network has no "
            "BatchNormalization after its last layer); its weights don't map"
        )
    pairs = [(features_bn[0], new.get_layer("features_bn"))]
    pairs += list(zip(old_state, new_state)) + list(zip(old_trunk, new_trunk))
    for p in mixture.outcome_family.param_names:
        for k in range(n_states):
            names = {x.name for x in old.layers}
            if f"{p}_feat_eng_state_{k}" in names:
                pairs.append(
                    (
                        old.get_layer(f"{p}_feat_eng_state_{k}"),
                        new.get_layer(f"{p}_hidden_state_{k}"),
                    )
                )
            raw = f"{p}_pred_state_{k}" if f"{p}_pred_state_{k}" in names else f"{p}_logit_{k}"
            pairs.append((old.get_layer(raw), new.get_layer(f"{p}_raw_state_{k}")))
    simplex = [x for x in old.layers if isinstance(x, pypress.keras.layers.PredictiveStateSimplex)]
    pairs.append((simplex[0], new.get_layer("state_weights")))
    for a, b in pairs:
        shapes_a = [tuple(w.shape) for w in a.weights]
        shapes_b = [tuple(w.shape) for w in b.weights]
        if type(a) is not type(b) or shapes_a != shapes_b:
            raise ValueError(
                f"layer {a.name!r} ({type(a).__name__}, {shapes_a}) does not map onto "
                f"{b.name!r} ({type(b).__name__}, {shapes_b})"
            )
    return pairs


def copy_legacy_weights(old: tf.keras.Model, new: tf.keras.Model) -> tf.keras.Model:
    """Copies the weights of a model built before v0.4.0 into its preset model.

    Deprecated with the ``build_*`` builders; removed in v0.5.0.

    Since v0.4.0 binary-treatment models hold the propensity in one `PredictiveStateParams`
    layer (``treatment_params``) instead of one ``propensity_logit_state_<k>`` layer per state,
    so their ``.weights.h5`` files saved earlier don't load. Load the old model's ``.keras``
    file with ``compile=False``, build the new one with the same arguments, and copy:

        old = tf.keras.models.load_model("model.keras", compile=False)
        new = models.build_model(**presets.binary_normal(...same arguments...))
        models.copy_legacy_weights(old, new)

    **Warning:** ``new.load_weights`` on an old binary ``.weights.h5`` raises ``ValueError``,
    but only after overwriting part of the model; discard that model and use this function.

    The layers are matched by their role in the graph, so the copy is exact: the new model's
    outcome and state-weight columns equal the old ones, and the propensity agrees to float32
    rounding. For a continuous-treatment model, whose layers are unchanged, this is
    ``new.set_weights(old.get_weights())``.

    Args:
      old: a model from a ``build_*`` builder before v0.4.0 (or loaded from its file).
      new: the model from the matching preset, built with the same arguments.

    Returns:
      ``new``, with ``old``'s weights.

    Raises:
      ValueError: if the two models don't match, for example a toy model (whose architecture
        changed in v0.4.0) or different presets or arguments.
    """
    names = {x.name for x in old.layers}
    if "propensity_logit_state_0" not in names:
        old_shapes = [tuple(w.shape) for w in old.weights]
        if old_shapes != [tuple(w.shape) for w in new.weights]:
            raise ValueError("old and new models have different weights; they don't match")
        new.set_weights(old.get_weights())
        return new
    pairs = _legacy_pairs(old, new)
    n_states = mixture_lib.StateMixture.from_model(new).n_states
    logits = np.stack(
        [old.get_layer(f"propensity_logit_state_{k}").get_weights()[0] for k in range(n_states)]
    )
    for a, b in pairs:
        b.set_weights(a.get_weights())
    new.get_layer("treatment_params").set_weights([logits.reshape(n_states, 1)])
    return new


def get_propensity_state_conditional_means(model: tf.keras.Model) -> tf.Tensor:
    """Each predictive state's mean treatment, shape ``(n_states, 1)``.

    For a binary treatment this is the propensity score of each state; for a Normal treatment,
    each state's ``loc``. It works for every pypsps model, before and after training, and
    for models loaded with ``compile=False``: a useful check that the per-state treatment
    models are reasonable (e.g. not degenerate).

    The treatment block of ``y_pred`` does not depend on the features (each state has one
    treatment distribution), so one all-zero input row gives it.
    """
    try:
        mixture = mixture_lib.StateMixture.from_model(model)
    except RuntimeError:
        # Saved before v0.3.4 and loaded with compile=False: no mixture to read.
        return _legacy_propensity_means(model)
    probe = [tf.zeros((1,) + tuple(x.shape[1:])) for x in model.inputs]
    params = mixture.layout.decode_pred(model(probe, training=False)).treatment
    return tf.transpose(mixture.treatment_family.mean(params))


def _legacy_propensity_means(model: tf.keras.Model) -> tf.Tensor:
    """The per-state propensity of a pre-v0.3.4 binary model, from its layer names.

    Removed in v0.5.0, which stops loading those models.
    """
    n_states = model.get_layer("propensity_score").output.shape[-1]
    logits = tf.stack(
        [
            tf.convert_to_tensor(model.get_layer(f"propensity_logit_state_{k}").weights[0])
            for k in range(n_states)
        ]
    )
    return tf.sigmoid(logits)


# --- build_model and compile_model ---

Architecture = Literal["shared_trunk", "gated"]
HiddenLayers = Sequence[Tuple[int, str]]  # (units, activation) per Dense layer
_ARCHITECTURES = ("shared_trunk", "gated")


def _hidden_layer(entry: Any, what: str) -> Tuple[int, str]:
    """Validates one ``(units, activation)`` entry and returns it as a tuple."""
    if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 2:
        raise ValueError(f"{what} must be a (units, activation) pair, got {entry!r}")
    units, activation = entry
    if isinstance(units, bool) or not isinstance(units, (int, np.integer)) or units < 1:
        raise ValueError(f"{what}: units must be a positive int, got {units!r}")
    if not isinstance(activation, str):
        raise ValueError(f"{what}: activation must be a string, got {activation!r}")
    return int(units), activation


def _hidden_layers(entries: HiddenLayers, what: str) -> Tuple[Tuple[int, str], ...]:
    """Validates a list of ``(units, activation)`` entries; empty is allowed."""
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise ValueError(f"{what} must be a sequence of (units, activation), got {entries!r}")
    return tuple(_hidden_layer(e, f"{what}[{i}]") for i, e in enumerate(entries))


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

    hidden_layer: Optional[Tuple[int, str]] = (16, "swish")
    constant: bool = False
    per_state: bool = True

    def __post_init__(self):
        """Validates ``hidden_layer``."""
        if self.hidden_layer is not None:
            object.__setattr__(
                self, "hidden_layer", _hidden_layer(self.hidden_layer, "HeadSpec.hidden_layer")
            )


@dataclasses.dataclass(frozen=True)
class _Spec:
    """The validated arguments of `build_model`; `_assemble` turns it into layers."""

    mixture: mixture_lib.StateMixture
    n_features: int
    outcome_params: Tuple[families.ParamSpec, ...]  # family params with init overrides
    treatment_init: Optional[np.ndarray]  # (n_params, n_states) on the support, or None
    architecture: str
    predictive_state_hidden_layers: Tuple[Tuple[int, str], ...]
    outcome_hidden_layers: Tuple[Tuple[int, str], ...]
    heads: Dict[str, HeadSpec]
    treatment_embedding_units: Optional[int]
    dropout_rate: float
    l1: float

    @property
    def n_states(self) -> int:
        """K."""
        return self.mixture.n_states


def _positive_int(value: Any, what: str) -> int:
    """Raises ValueError unless ``value`` is an int >= 1."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{what} must be an int >= 1, got {value!r}")
    return int(value)


def _check_keys(given: Optional[Mapping[str, Any]], family: families.Family, what: str) -> dict:
    """Raises KeyError for a key that is not a parameter of ``family``."""
    given = dict(given or {})
    unknown = sorted(set(given) - set(family.param_names))
    if unknown:
        raise KeyError(
            f"{what} has unknown parameters {unknown}; {type(family).__name__} has "
            f"{family.param_names}"
        )
    return given


def _treatment_init(
    family: families.Family, given: Optional[Mapping[str, Any]], n_states: int
) -> Optional[np.ndarray]:
    """The ``(n_params, n_states)`` init of the treatment head, or None for Keras defaults.

    A value is one float for every state, or one float per state. Without ``given``, the
    family's defaults apply if every parameter has one (e.g. Bernoulli's 0.5).
    """
    params = family.params
    given = _check_keys(given, family, "treatment_init")
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


def _make_spec(
    n_states: int,
    n_features: int,
    outcome_family: families.Family,
    treatment_family: families.Family,
    architecture: str,
    predictive_state_hidden_layers: HiddenLayers,
    outcome_hidden_layers: HiddenLayers,
    outcome_heads: Optional[Mapping[str, HeadSpec]],
    treatment_embedding_units: Optional[int],
    dropout_rate: float,
    l1: float,
    outcome_init: Optional[Mapping[str, float]],
    treatment_init: Optional[Mapping[str, Any]],
    max_time: Optional[float],
) -> _Spec:
    """Validates every `build_model` argument before any layer is created."""
    n_states = _positive_int(n_states, "n_states")
    n_features = _positive_int(n_features, "n_features")
    if architecture not in _ARCHITECTURES:
        raise ValueError(f"architecture must be one of {_ARCHITECTURES}, got {architecture!r}")
    mixture = mixture_lib.StateMixture(outcome_family, treatment_family, n_states=n_states)

    heads = _check_keys(outcome_heads, outcome_family, "outcome_heads")
    for name, head in heads.items():
        if not isinstance(head, HeadSpec):
            raise TypeError(f"outcome_heads[{name!r}] must be a HeadSpec, got {head!r}")
    heads = {p: heads.get(p, HeadSpec()) for p in outcome_family.param_names}

    inits = _check_keys(outcome_init, outcome_family, "outcome_init")
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
        treatment_embedding_units = _positive_int(
            treatment_embedding_units, "treatment_embedding_units"
        )
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(f"dropout_rate must lie in [0, 1), got {dropout_rate!r}")
    if l1 < 0:
        raise ValueError(f"l1 must be >= 0, got {l1!r}")

    return _Spec(
        mixture=mixture,
        n_features=n_features,
        outcome_params=tuple(outcome_params),
        treatment_init=_treatment_init(treatment_family, treatment_init, n_states),
        architecture=architecture,
        predictive_state_hidden_layers=_hidden_layers(
            predictive_state_hidden_layers, "predictive_state_hidden_layers"
        ),
        outcome_hidden_layers=_hidden_layers(outcome_hidden_layers, "outcome_hidden_layers"),
        heads=heads,
        treatment_embedding_units=treatment_embedding_units,
        dropout_rate=float(dropout_rate),
        l1=float(l1),
    )


def _treatment_head(spec: _Spec, state_weights: tf.Tensor) -> tf.Tensor:
    """One feature-independent constant per state and treatment parameter, ``[N, P * K]``."""
    params = spec.mixture.treatment_family.params
    return pypress.keras.layers.PredictiveStateParams(
        n_params_per_state=len(params),
        activations=[
            p.activation if p.bounds is None else layers.SoftClip(*p.bounds) for p in params
        ],
        init_values=spec.treatment_init,
        flatten_output=True,  # param-major, as in ColumnLayout
        name="treatment_params",
    )(state_weights)


def _head(
    p: families.ParamSpec,
    head: HeadSpec,
    head_input: tf.Tensor,
    features_and_treatment: tf.Tensor,
    suffix: str,
) -> tf.Tensor:
    """One outcome parameter for one state (or shared), on the parameter's support."""
    init = "zeros" if p.init is None else tf.keras.initializers.Constant(p.raw_value(p.init))
    if head.constant:
        raw = layers.BiasOnly(initializer=init, name=f"{p.name}_raw_{suffix}")(
            features_and_treatment
        )
    else:
        x = head_input
        if head.hidden_layer is not None:
            x = tfk.layers.Dense(*head.hidden_layer, name=f"{p.name}_hidden_{suffix}")(x)
        raw = tfk.layers.Dense(1, bias_initializer=init, name=f"{p.name}_raw_{suffix}")(x)
    return p.head_activation(name=f"{p.name}_{suffix}")(raw)


def _shared_trunk_outcome(
    spec: _Spec, features_and_treatment: tf.Tensor
) -> Dict[str, List[tf.Tensor]]:
    """One outcome trunk with a skip connection; heads are built state by state."""
    trunk = _dense_stack(
        features_and_treatment, spec.outcome_hidden_layers, spec.dropout_rate, spec.l1,
        "outcome_trunk",
    )  # fmt: skip
    if spec.outcome_hidden_layers:
        trunk = tfk.layers.Concatenate(name="outcome_trunk_skip")([trunk, features_and_treatment])
    columns: Dict[str, List[tf.Tensor]] = {p.name: [] for p in spec.outcome_params}
    shared: Dict[str, tf.Tensor] = {}
    for k in range(spec.n_states):
        for p in spec.outcome_params:
            head = spec.heads[p.name]
            if head.per_state:
                columns[p.name].append(_head(p, head, trunk, features_and_treatment, f"state_{k}"))
                continue
            if p.name not in shared:
                shared[p.name] = _head(p, head, trunk, features_and_treatment, "shared")
            columns[p.name].append(shared[p.name])
    return columns


def _gated_outcome(
    spec: _Spec,
    features_and_treatment: tf.Tensor,
    treatment_repr: tf.Tensor,
    state_weights: tf.Tensor,
) -> Dict[str, List[tf.Tensor]]:
    """One trunk per outcome parameter; each state's head sees the treatment gated by its
    state weight. Heads are built parameter by parameter."""
    columns: Dict[str, List[tf.Tensor]] = {}
    for p in spec.outcome_params:
        head = spec.heads[p.name]
        if head.constant:
            per = range(spec.n_states) if head.per_state else [None]
            values = [
                _head(
                    p, head, None, features_and_treatment, "shared" if k is None else f"state_{k}"
                )
                for k in per
            ]
            columns[p.name] = values if head.per_state else values * spec.n_states
            continue
        trunk = _dense_stack(
            features_and_treatment, spec.outcome_hidden_layers, spec.dropout_rate, spec.l1,
            f"{p.name}_trunk",
        )  # fmt: skip
        if not head.per_state:
            columns[p.name] = [_head(p, head, trunk, features_and_treatment, "shared")] * (
                spec.n_states
            )
            continue
        values = []
        for k in range(spec.n_states):
            s = f"state_{k}"
            gate = layers.StateColumnGate(k, name=f"{p.name}_gate_{s}")(state_weights)
            gated = tfk.layers.Multiply(name=f"{p.name}_gated_treatment_{s}")(
                [treatment_repr, gate]
            )
            head_input = tfk.layers.Concatenate(name=f"{p.name}_input_{s}")([trunk, gated])
            values.append(_head(p, head, head_input, features_and_treatment, s))
        columns[p.name] = values
    return columns


def _assemble(spec: _Spec) -> tf.keras.Model:
    """Builds the network: inputs, state network, treatment head, outcome heads, output."""
    features = tfk.layers.Input(shape=(spec.n_features,), name="features")
    treatment = tfk.layers.Input(shape=(1,), name="treatment")
    features_bn = tfk.layers.BatchNormalization(name="features_bn")(features)
    if spec.treatment_embedding_units is None:
        treatment_repr = treatment
    else:
        treatment_bn = tfk.layers.BatchNormalization(name="treatment_bn")(treatment)
        treatment_repr = tfk.layers.Dense(
            spec.treatment_embedding_units, "swish", name="treatment_embedding"
        )(treatment_bn)
    features_and_treatment = tfk.layers.Concatenate(name="features_and_treatment")(
        [features_bn, treatment_repr]
    )

    state_hidden = _dense_stack(
        features_bn, spec.predictive_state_hidden_layers, spec.dropout_rate, spec.l1,
        "predictive_state",
    )  # fmt: skip
    if spec.architecture == "shared_trunk" and spec.predictive_state_hidden_layers:
        state_hidden = tfk.layers.Concatenate(name="predictive_state_skip")(
            [state_hidden, features_bn]
        )
    state_weights = pypress.keras.layers.PredictiveStateSimplex(
        n_states=spec.n_states, name="state_weights"
    )(state_hidden)
    treatment_params = _treatment_head(spec, state_weights)

    if spec.architecture == "shared_trunk":
        columns = _shared_trunk_outcome(spec, features_and_treatment)
    else:
        columns = _gated_outcome(spec, features_and_treatment, treatment_repr, state_weights)
    outcome = {
        name: tfk.layers.Concatenate(name=f"{name}_all_states")(values)
        for name, values in columns.items()
    }
    output = layers.CausalOutput(spec.mixture, name="output_tensor")(
        {"outcome": outcome, "state_weights": state_weights, "treatment": treatment_params}
    )
    return tfk.models.Model(inputs=[features, treatment], outputs=output)


def build_model(
    n_states: int,
    n_features: int,
    outcome_family: families.Family,
    treatment_family: families.Family,
    *,
    architecture: Architecture = "gated",
    predictive_state_hidden_layers: HiddenLayers = ((16, "relu"),),
    outcome_hidden_layers: HiddenLayers = ((16, "relu"),),
    outcome_heads: Optional[Mapping[str, HeadSpec]] = None,
    treatment_embedding_units: Optional[int] = 8,
    dropout_rate: float = 0.2,
    l1: float = 0.0,
    outcome_init: Optional[Mapping[str, float]] = None,
    treatment_init: Optional[Mapping[str, Union[float, Sequence[float]]]] = None,
    max_time: Optional[float] = None,
    compile: bool = True,
    **compile_kwargs: Any,
) -> tf.keras.Model:
    """Builds a pypsps model for any outcome family and treatment family.

    The network has four parts: a state network from the features to the state weights
    (layer ``state_weights``); a treatment head with one feature-independent constant per
    state and treatment parameter (``treatment_params``); one head per outcome parameter and
    state, built as ``outcome_heads`` says; and a `layers.CausalOutput` (``output_tensor``)
    that holds the model's `StateMixture`. The families choose the parameters, their
    activations or `layers.SoftClip` bounds, and their initial values.

    Args:
      n_states: number of predictive states, K >= 1.
      n_features: number of numeric features, >= 1.
      outcome_family: the within-state outcome distribution, e.g. ``families.Weibull(...)``;
        `families.get` maps names to families.
      treatment_family: the within-state treatment distribution, e.g.
        ``families.Bernoulli()`` or ``families.Normal()``.
      architecture: ``"gated"`` (one trunk per outcome parameter; each state's head sees the
        treatment gated by its state weight) or ``"shared_trunk"`` (one outcome trunk and a
        skip connection, shared by every head).
      predictive_state_hidden_layers: ``(units, activation)`` per layer of the state network;
        may be empty.
      outcome_hidden_layers: ``(units, activation)`` per layer of the outcome trunk; may be
        empty.
      outcome_heads: optional `HeadSpec` per outcome parameter; missing ones get `HeadSpec()`.
      treatment_embedding_units: width of a learned treatment embedding; None uses the raw
        treatment column.
      dropout_rate: dropout rate of the hidden layers.
      l1: L1 penalty on the hidden-layer kernels.
      outcome_init: optional initial value per outcome parameter, on its support. Overrides
        the family default (e.g. the midpoint of a bounded parameter).
      treatment_init: optional initial value per treatment parameter, on its support: one
        value for every state, or one per state. Give all parameters or none.
      max_time: optional largest time in the data, for survival outcome families: raises a
        ValueError when the family's bounds let the loss overflow float32 at that time.
      compile: if True, compiles the model with `compile_model`.
      **compile_kwargs: passed to `compile_model`: ``alpha``, ``outcome_loss_weight``,
        ``outcome_likelihood``, ``predictive_states_regularizer``, ``optimizer``,
        ``learning_rate``, ``clipnorm``, ``metrics``, ``extra_metrics``.

    Returns:
      A tf.keras Model with inputs ``[features, treatment]``, compiled if ``compile=True``.

    Raises:
      ValueError, KeyError, TypeError: for invalid arguments, before any layer is created.
    """
    if not compile and compile_kwargs:
        raise TypeError(f"compile keywords given with compile=False: {sorted(compile_kwargs)}")
    if compile:
        # An unknown compile keyword raises TypeError here, before any layer is created.
        inspect.signature(compile_model).bind(None, **compile_kwargs)
    spec = _make_spec(
        n_states,
        n_features,
        outcome_family,
        treatment_family,
        architecture,
        predictive_state_hidden_layers,
        outcome_hidden_layers,
        outcome_heads,
        treatment_embedding_units,
        dropout_rate,
        l1,
        outcome_init,
        treatment_init,
        max_time,
    )
    model = _assemble(spec)
    if compile:
        compile_model(model, **compile_kwargs)
    return model


_UNSET: Any = object()
_DEFAULT_LEARNING_RATE = 0.01
_DEFAULT_CLIPNORM = 1.0


def compile_model(
    model: tf.keras.Model,
    *,
    alpha: Union[float, tf.Variable] = 1.0,
    outcome_loss_weight: float = 1.0,
    outcome_likelihood: Union[
        losses.OutcomeLikelihood, str
    ] = losses.OutcomeLikelihood.INTERVENTIONAL,
    predictive_states_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
    learning_rate: float = _UNSET,
    clipnorm: Optional[float] = _UNSET,
    metrics: Optional[Sequence[tf.keras.metrics.Metric]] = None,
    extra_metrics: Sequence[tf.keras.metrics.Metric] = (),
) -> tf.keras.Model:
    """Compiles a pypsps model in place with the causal loss of its state mixture.

    Works on any model whose output layer is a `layers.CausalOutput`, including hand-built
    networks: the loss is ``CausalLoss.from_mixture(StateMixture.from_model(model), ...)``.

    Args:
      model: the model to compile.
      alpha: weight of the treatment log-likelihood; a float32 tf.Variable is shared, not
        copied, so a callback can change it.
      outcome_loss_weight: weight of the outcome log-likelihood, >= 0.
      outcome_likelihood: see `losses.OutcomeLikelihood`.
      predictive_states_regularizer: optional regularizer on the state weights, e.g. a
        `pypress.keras.regularizers` instance.
      optimizer: optional optimizer; by default Nadam with ``learning_rate`` (0.01) and
        ``clipnorm`` (1.0; None disables clipping). Pass either this or those two.
      learning_rate: learning rate of the default optimizer.
      clipnorm: gradient-norm clipping of the default optimizer.
      metrics: the metrics; None uses `default_metrics`.
      extra_metrics: metrics appended to ``metrics``.

    Returns:
      ``model``, compiled.

    Raises:
      RuntimeError: if the model has no `CausalOutput` layer (and no pypsps loss).
      TypeError: if ``optimizer`` is given with ``learning_rate`` or ``clipnorm``.
      ValueError: if ``outcome_loss_weight`` is negative.
    """
    if optimizer is not None and (learning_rate is not _UNSET or clipnorm is not _UNSET):
        raise TypeError("pass learning_rate/clipnorm or optimizer, not both")
    if outcome_loss_weight < 0:
        raise ValueError(f"outcome_loss_weight must be >= 0, got {outcome_loss_weight!r}")
    causal_loss = losses.CausalLoss.from_mixture(
        mixture_lib.StateMixture.from_model(model),
        alpha=alpha,
        outcome_likelihood=outcome_likelihood,
        outcome_loss_weight=outcome_loss_weight,
        predictive_states_regularizer=predictive_states_regularizer,
        reduction="sum_over_batch_size",
    )
    if optimizer is None:
        optimizer = tfk.optimizers.Nadam(
            learning_rate=_DEFAULT_LEARNING_RATE if learning_rate is _UNSET else learning_rate,
            clipnorm=_DEFAULT_CLIPNORM if clipnorm is _UNSET else clipnorm,
        )
    chosen = default_metrics(causal_loss) if metrics is None else list(metrics)
    model.compile(loss=causal_loss, optimizer=optimizer, metrics=chosen + list(extra_metrics))
    return model


def default_metrics(causal_loss: losses.CausalLoss) -> List[tf.keras.metrics.Metric]:
    """The metrics `compile_model` uses by default, chosen by the families.

    Always ``causal_loss_metric`` (penalty-free, alpha = 1), ``outcome_nll_metric``,
    ``predictive_state_df`` and ``uniform_entropy``. A binary treatment adds
    ``propensity_score_binary_crossentropy``, ``propensity_score_auroc`` and
    ``propensity_score_aupr``; any other treatment
    adds ``treatment_mean_squared_error``. Every outcome family but `families.LossFamily` adds
    ``outcome_mean_squared_error`` (over units with an event, for survival families).
    """
    mixture = causal_loss.mixture
    outcome_loss = causal_loss._outcome_loss
    chosen: List[tf.keras.metrics.Metric] = [
        metrics.CausalLossMetric(
            outcome_loss,
            causal_loss._treatment_loss,
            outcome_loss_weight=causal_loss._outcome_loss_weight,
        ),
        metrics.outcome_nll_metric_gen(outcome_loss),
        metrics.predictive_state_df_gen(mixture.layout),
        metrics.uniform_entropy_gen(mixture.layout),
    ]
    if mixture.treatment_family.is_binary:
        chosen += [
            metrics.PropensityScoreBinaryCrossentropy(mixture),
            metrics.PropensityScoreAUC(mixture, curve="ROC", name="propensity_score_auroc"),
            metrics.PropensityScoreAUC(mixture, curve="PR", name="propensity_score_aupr"),
        ]
    else:
        chosen.append(metrics.TreatmentMeanSquaredError(mixture))
    if not isinstance(mixture.outcome_family, families.LossFamily):
        chosen.append(metrics.OutcomeMeanSquaredError(mixture))
    return chosen
