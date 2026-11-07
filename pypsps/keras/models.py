"""Model builders for pypsps: `build_model` and `compile_model`."""

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import pypress
import pypress.keras.layers
import pypress.keras.regularizers
import tensorflow as tf

from ..core import spec as core_spec
from . import families, layers, losses, metrics
from . import mixture as mixture_lib

tfk = tf.keras


def _dense_stack(
    x: tf.Tensor,
    hidden_layers: list[tuple[int, str]],
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


def get_propensity_state_conditional_means(model: tf.keras.Model) -> tf.Tensor:
    """Each predictive state's mean treatment, shape ``(n_states, 1)``.

    For a binary treatment this is the propensity score of each state; for a Normal treatment,
    each state's ``loc``. It works for every pypsps model, before and after training, and
    for models loaded with ``compile=False``: a useful check that the per-state treatment
    models are reasonable (e.g. not degenerate).

    The treatment block of ``y_pred`` does not depend on the features (each state has one
    treatment distribution), so one all-zero input row gives it.
    """
    mixture = mixture_lib.StateMixture.from_model(model)
    probe = [tf.zeros((1,) + tuple(x.shape[1:])) for x in model.inputs]
    params = mixture.layout.decode_pred(model(probe, training=False)).treatment
    return tf.transpose(mixture.treatment_family.mean(params))


# --- build_model and compile_model ---

Architecture = core_spec.Architecture
HiddenLayers = core_spec.HiddenLayers
_ARCHITECTURES = core_spec.ARCHITECTURES
HeadSpec = core_spec.HeadSpec
_Spec = core_spec.Spec
_hidden_layer = core_spec.check_hidden_layer
_hidden_layers = core_spec.check_hidden_layers
_positive_int = core_spec.positive_int
_check_keys = core_spec.check_keys
_treatment_init = core_spec.treatment_init_array


def _make_spec(*args: Any, **kwargs: Any) -> _Spec:
    """`pypsps.core.spec.make_spec` with the Keras state mixture."""
    return core_spec.make_spec(mixture_lib.StateMixture, *args, **kwargs)


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
    return families.head_activation(p, name=f"{p.name}_{suffix}")(raw)


def _shared_trunk_outcome(
    spec: _Spec, features_and_treatment: tf.Tensor
) -> dict[str, list[tf.Tensor]]:
    """One outcome trunk with a skip connection; heads are built state by state."""
    trunk = _dense_stack(
        features_and_treatment, spec.outcome_hidden_layers, spec.dropout_rate, spec.l1,
        "outcome_trunk",
    )  # fmt: skip
    if spec.outcome_hidden_layers:
        trunk = tfk.layers.Concatenate(name="outcome_trunk_skip")([trunk, features_and_treatment])
    columns: dict[str, list[tf.Tensor]] = {p.name: [] for p in spec.outcome_params}
    shared: dict[str, tf.Tensor] = {}
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
) -> dict[str, list[tf.Tensor]]:
    """One trunk per outcome parameter; each state's head sees the treatment gated by its
    state weight. Heads are built parameter by parameter."""
    columns: dict[str, list[tf.Tensor]] = {}
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
    outcome_heads: Mapping[str, HeadSpec] | None = None,
    treatment_embedding_units: int | None = 8,
    dropout_rate: float = 0.2,
    l1: float = 0.0,
    outcome_init: Mapping[str, float] | None = None,
    treatment_init: Mapping[str, float | Sequence[float]] | None = None,
    max_time: float | None = None,
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
    alpha: float | tf.Variable = 1.0,
    outcome_loss_weight: float = 1.0,
    outcome_likelihood: losses.OutcomeLikelihood | str = losses.OutcomeLikelihood.INTERVENTIONAL,
    predictive_states_regularizer: tf.keras.regularizers.Regularizer | None = None,
    optimizer: tf.keras.optimizers.Optimizer | None = None,
    learning_rate: float = _UNSET,
    clipnorm: float | None = _UNSET,
    metrics: Sequence[tf.keras.metrics.Metric] | None = None,
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


def default_metrics(causal_loss: losses.CausalLoss) -> list[tf.keras.metrics.Metric]:
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
    chosen: list[tf.keras.metrics.Metric] = [
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
