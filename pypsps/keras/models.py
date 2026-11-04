"""Example model architectures for pypsps."""

from typing import List, Optional, Tuple, Union

import numpy as np
import pypress
import pypress.keras.layers
import pypress.keras.regularizers
import tensorflow as tf

from . import families, layers, losses, metrics
from . import mixture as mixture_lib

tfk = tf.keras

_EPS = 1e-3


def _build_binary_exponential_causal_loss(
    n_states: int,
    alpha: float,
    df_penalty_l2: float,
    outcome_loss_weight: float,
) -> losses.CausalLoss:
    """Causal loss for a binary treatment and an exponential (log-rate) survival outcome."""
    return losses.CausalLoss.from_mixture(
        mixture_lib.StateMixture(
            families.Exponential("log_rate"), families.Bernoulli(), n_states=n_states
        ),
        alpha=alpha,
        outcome_loss_weight=outcome_loss_weight,
        predictive_states_regularizer=pypress.keras.regularizers.Uniform(l2=df_penalty_l2),
        reduction="sum_over_batch_size",
    )


def _build_binary_normal_causal_loss(
    n_states: int,
    alpha: float,
    df_penalty_l2: float,
) -> losses.CausalLoss:
    """Causal loss for a binary treatment and a Normal outcome."""
    return losses.CausalLoss.from_mixture(
        mixture_lib.StateMixture(families.Normal(), families.Bernoulli(), n_states=n_states),
        alpha=alpha,
        outcome_loss_weight=1.0,
        predictive_states_regularizer=pypress.keras.regularizers.DegreesOfFreedom(
            l2=df_penalty_l2,
            target=max(1, np.log(0.5 * n_states)),
        ),
        reduction="sum_over_batch_size",
    )


def _binary_metrics(
    causal_loss: losses.CausalLoss,
) -> Tuple[List[tf.keras.metrics.Metric], metrics.CausalLossMetric]:
    """``(propensity metrics, causal loss metric)`` of the binary-treatment builders."""
    mixture = causal_loss.mixture
    return [
        metrics.PropensityScoreBinaryCrossentropy(mixture),
        metrics.PropensityScoreAUC(mixture, curve="ROC", name="propensity_score_auroc"),
        metrics.PropensityScoreAUC(mixture, curve="PR", name="propensity_score_aupr"),
    ], metrics.CausalLossMetric(
        causal_loss._outcome_loss,
        causal_loss._treatment_loss,
        outcome_loss_weight=causal_loss._outcome_loss_weight,
    )


def build_toy_model(
    n_states: int,
    n_features: int,
    compile: bool = True,
    alpha: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
) -> tf.keras.Model:
    """Builds a pypsps toy model for binary treatment & continuous outcome.

    All pypsps keras layers can be used to build more complex causal model architectures
    within a TensorFlow graph.  The specific model structure here is only used
    for proof-of-concept / demo purposes.

    Args:
      n_states: number of predictive states to use in the pypsps model.
      n_features: number of (numeric) features to use as input.
      compile: if True, compiles pypsps model with the appropriate pypsps causal loss functions.
      alpha: weight of the treatment (propensity) log-likelihood relative to the outcome
        log-likelihood in the causal loss (λ in Eq. 10 of the paper). The default 1.0 gives
        the joint log-likelihood. Values above 1 act as a penalty that prioritizes fitting
        treatment over fitting outcome: training then learns the propensity and the
        predictive states first, and refines the outcome models afterwards.
      df_penalty_l2: l2 parameter for the DF regularization
      learning_rate: learning rate of the optimizer.

    Returns:
      A tf.keras Model with the pypsps architecture (compiled model if `compile=True`).
    """

    assert n_states >= 1, f"Got n_states={n_states}"
    assert n_features >= 1, f"Got n_features={n_features}"

    features = tfk.layers.Input(shape=(n_features,), name="features")
    treat = tfk.layers.Input(shape=(1,), name="treatment")

    features_bn = tfk.layers.BatchNormalization(name="features_bn")(features)
    feat_treat_bn = tfk.layers.Concatenate(name="features_bn_and_treatment")([features_bn, treat])

    ps_hidden = tfk.layers.Dense(10, "relu")(features_bn)
    ps_hidden = tfk.layers.BatchNormalization()(ps_hidden)
    ps_hidden = tfk.layers.Dropout(0.2)(ps_hidden)
    ps_hidden = tfk.layers.Dense(10, "selu")(ps_hidden)

    ps_hidden = tf.keras.layers.Concatenate(name="ps_hidden_and_features_bn")(
        [ps_hidden, features_bn]
    )
    pss = pypress.keras.layers.PredictiveStateSimplex(n_states=n_states, input_dim=n_features)
    pred_states = pss(ps_hidden)
    # State-conditional propensity for binary treatment (--> "sigmoid" activation), one constant
    # per state (the "balancedness" assumption: treatment probability is homogeneous within a
    # predictive state). Kept state-conditional (not pre-mixed) so the causal loss can compute
    # the exact posterior P(state | X, treatment) via Bayes' rule.
    prop_score_preds = []
    for state_id in range(n_states):
        prop_score_preds.append(
            tf.keras.activations.sigmoid(
                layers.BiasOnly(name="propensity_logit_state_" + str(state_id))(features_bn)
            )
        )
    prop_score = tfk.layers.Concatenate(name="propensity_score")(prop_score_preds)

    outcome_hidden = tf.keras.layers.Dense(10, "tanh", name="inputs_processing")(feat_treat_bn)
    outcome_hidden = tf.keras.layers.Dropout(0.2)(outcome_hidden)
    outcome_hidden = tf.keras.layers.BatchNormalization()(outcome_hidden)

    outcome_hidden = tf.keras.layers.Concatenate(name="outcome_hidden_and_ft")(
        [outcome_hidden, feat_treat_bn]
    )

    loc_preds = []
    scale_preds = []
    # One outcome model per state.
    for state_id in range(n_states):
        loc_preds.append(
            tfk.layers.Dense(1, name="loc_pred_state_" + str(state_id))(
                tfk.layers.Dense(5, "selu", name="feat_eng_state_" + str(state_id))(outcome_hidden)
            )
        )

        # In this toy model use a constant scale estimate (BiasOnly); if needed
        # change this to a scale parameter that changes as a function of inputs / hidden layers.
        scale_preds.append(
            tf.keras.activations.softplus(
                layers.BiasOnly(name="scale_logit_" + str(state_id))(feat_treat_bn)
            )
        )

    loc_comb = tfk.layers.Concatenate(name="loc_all_states")(loc_preds)
    scale_comb = tfk.layers.Concatenate(name="scale_all_states")(scale_preds)

    outcome_pred = tfk.layers.Concatenate(name="loc_scale_all_states")([loc_comb, scale_comb])
    outputs_concat = tfk.layers.Concatenate(name="output_tensor")(
        [outcome_pred, pred_states, prop_score]
    )

    model = tfk.models.Model(inputs=[features, treat], outputs=outputs_concat)

    if compile:
        psps_causal_loss = _build_binary_normal_causal_loss(
            n_states=n_states,
            alpha=alpha,
            df_penalty_l2=df_penalty_l2,
        )
        propensity_metrics, causal_loss_metric = _binary_metrics(psps_causal_loss)
        model.compile(
            loss=psps_causal_loss,
            optimizer=tfk.optimizers.Nadam(learning_rate=learning_rate),
            metrics=[
                *propensity_metrics,
                metrics.OutcomeMeanSquaredError(psps_causal_loss.mixture),
                causal_loss_metric,
            ],
        )

    return model


def build_model_binary_normal(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    loc_layer: Tuple[int, str] = None,
    scale_layer: Tuple[int, str] = None,
    compile: bool = True,
    alpha: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
    dropout_rate: float = 0.2,
) -> tf.keras.Model:
    """Builds a pypsps toy model for binary treatment & continuous outcome.

    All pypsps keras layers can be used to build more complex causal model architectures
    within a TensorFlow graph.  The specific model structure here is only used
    for proof-of-concept / demo purposes.

    Args:
      n_states: number of predictive states to use in the pypsps model.
      n_features: number of (numeric) features to use as input.
      compile: if True, compiles pypsps model with the appropriate pypsps causal loss functions.
      alpha: weight of the treatment (propensity) log-likelihood relative to the outcome
        log-likelihood in the causal loss (λ in Eq. 10 of the paper). The default 1.0 gives
        the joint log-likelihood. Values above 1 act as a penalty that prioritizes fitting
        treatment over fitting outcome: training then learns the propensity and the
        predictive states first, and refines the outcome models afterwards.
      df_penalty_l2: l2 parameter for the DF regularization
      learning_rate: learning rate of the optimizer.

    Returns:
      A tf.keras Model with the pypsps architecture (compiled model if `compile=True`).
    """

    assert n_states >= 1, f"Got n_states={n_states}"
    assert n_features >= 1, f"Got n_features={n_features}"

    features = tfk.layers.Input(shape=(n_features,))
    treat = tfk.layers.Input(shape=(1,))

    features_bn = tfk.layers.BatchNormalization()(features)
    feat_treat_bn = tfk.layers.Concatenate(name="features_and_treatment")([features_bn, treat])

    ps_hidden = tf.keras.layers.Dense(
        predictive_state_hidden_layers[0][0],
        predictive_state_hidden_layers[0][1],
        name="features_processing_for_propensity_score",
    )(features_bn)
    ps_hidden = tf.keras.layers.Dropout(dropout_rate)(ps_hidden)
    ps_hidden = tf.keras.layers.BatchNormalization()(ps_hidden)

    for units, act in predictive_state_hidden_layers[1:]:
        ps_hidden = tf.keras.layers.Dense(units, act)(ps_hidden)
        ps_hidden = tf.keras.layers.Dropout(dropout_rate)(ps_hidden)
        ps_hidden = tf.keras.layers.BatchNormalization()(ps_hidden)

    ps_hidden = tf.keras.layers.Concatenate()([ps_hidden, features_bn])
    pss = pypress.keras.layers.PredictiveStateSimplex(n_states=n_states, input_dim=n_features)
    pred_states = pss(ps_hidden)

    # State-conditional propensity for binary treatment (--> "sigmoid" activation), one constant
    # per state; kept state-conditional (not pre-mixed) so the causal loss can compute the exact
    # posterior P(state | X, treatment) via Bayes' rule.
    prop_score_preds = []
    for state_id in range(n_states):
        prop_score_preds.append(
            tf.keras.activations.sigmoid(
                layers.BiasOnly(name="propensity_logit_state_" + str(state_id))(features_bn)
            )
        )
    prop_score = tfk.layers.Concatenate(name="propensity_score")(prop_score_preds)

    outcome_hidden = tf.keras.layers.Dense(
        outcome_hidden_layers[0][0], outcome_hidden_layers[0][1], name="inputs_processing"
    )(feat_treat_bn)
    outcome_hidden = tf.keras.layers.Dropout(dropout_rate)(outcome_hidden)
    outcome_hidden = tf.keras.layers.BatchNormalization()(outcome_hidden)

    for units, act in outcome_hidden_layers[1:]:
        outcome_hidden = tf.keras.layers.Dense(units, act)(outcome_hidden)
        outcome_hidden = tf.keras.layers.Dropout(dropout_rate)(outcome_hidden)
        outcome_hidden = tf.keras.layers.BatchNormalization()(outcome_hidden)

    outcome_hidden = tf.keras.layers.Concatenate(name="outcome_hidden_and_ft")(
        [outcome_hidden, feat_treat_bn]
    )

    loc_preds = []
    scale_preds = []
    # One outcome model per state.
    for state_id in range(n_states):
        loc_preds.append(
            tfk.layers.Dense(1, name="loc_pred_state_" + str(state_id))(
                tfk.layers.Dense(
                    loc_layer[0],
                    loc_layer[1],
                    name="loc_feat_eng_state_" + str(state_id),
                )(outcome_hidden)
            )
        )

        if scale_layer is None:
            # In this toy model use a constant scale estimate (BiasOnly); if needed
            # change this to a scale parameter that changes as a function of inputs / hidden layers.
            scale_preds.append(
                tf.keras.activations.softplus(
                    layers.BiasOnly(name="scale_logit_" + str(state_id))(feat_treat_bn)
                )
            )
        else:
            scale_preds.append(
                tfk.layers.Dense(
                    1, activation="softplus", name="scale_pred_state_" + str(state_id)
                )(
                    tfk.layers.Dense(
                        scale_layer[0],
                        scale_layer[1],
                        name="scale_feat_eng_state_" + str(state_id),
                    )(outcome_hidden)
                )
            )

    loc_comb = tfk.layers.Concatenate(name="loc_all_states")(loc_preds)
    scale_comb = tfk.layers.Concatenate(name="scale_all_states")(scale_preds)

    outputs_concat = tfk.layers.Concatenate(name="output_tensor")(
        [loc_comb, scale_comb, pred_states, prop_score]
    )

    model = tfk.models.Model(inputs=[features, treat], outputs=outputs_concat)

    if compile:
        psps_causal_loss = _build_binary_normal_causal_loss(
            n_states=n_states,
            alpha=alpha,
            df_penalty_l2=df_penalty_l2,
        )
        propensity_metrics, causal_loss_metric = _binary_metrics(psps_causal_loss)
        model.compile(
            loss=psps_causal_loss,
            optimizer=tfk.optimizers.Nadam(learning_rate=learning_rate),
            metrics=[
                *propensity_metrics,
                metrics.OutcomeMeanSquaredError(psps_causal_loss.mixture),
                causal_loss_metric,
            ],
        )

    return model


def build_model_binary_exponential(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    log_rate_layer: Tuple[int, str] = None,
    compile: bool = True,
    alpha: float = 1.0,
    outcome_loss_weight: float = 1.0,
    df_penalty_l2: float = 1.0,
    learning_rate: float = 0.01,
    dropout_rate: float = 0.2,
) -> tf.keras.Model:
    """Builds a pypsps toy model for binary treatment & continuous outcome.

    All pypsps keras layers can be used to build more complex causal model architectures
    within a TensorFlow graph.  The specific model structure here is only used
    for proof-of-concept / demo purposes.

    Args:
      n_states: number of predictive states to use in the pypsps model.
      n_features: number of (numeric) features to use as input.
      compile: if True, compiles pypsps model with the appropriate pypsps causal loss functions.
      alpha: weight of the treatment (propensity) log-likelihood relative to the outcome
        log-likelihood in the causal loss (λ in Eq. 10 of the paper). The default 1.0 gives
        the joint log-likelihood. Values above 1 act as a penalty that prioritizes fitting
        treatment over fitting outcome: training then learns the propensity and the
        predictive states first, and refines the outcome models afterwards.
      df_penalty_l2: l2 parameter for the DF regularization
      learning_rate: learning rate of the optimizer.

    Returns:
      A tf.keras Model with the pypsps architecture (compiled model if `compile=True`).
    """

    assert n_states >= 1, f"Got n_states={n_states}"
    assert n_features >= 1, f"Got n_features={n_features}"

    features = tfk.layers.Input(shape=(n_features,), name="features")
    treat = tfk.layers.Input(shape=(1,), name="treatment")

    features_bn = tfk.layers.BatchNormalization(name="features_bn")(features)
    feat_treat_bn = tfk.layers.Concatenate(name="features_and_treatment")([features_bn, treat])

    ps_hidden = tf.keras.layers.Dense(
        predictive_state_hidden_layers[0][0], predictive_state_hidden_layers[0][1]
    )(features_bn)
    ps_hidden = tf.keras.layers.Dropout(dropout_rate)(ps_hidden)
    ps_hidden = tf.keras.layers.BatchNormalization()(ps_hidden)

    for units, act in predictive_state_hidden_layers[1:]:
        ps_hidden = tf.keras.layers.Dense(units, act)(ps_hidden)
        ps_hidden = tf.keras.layers.Dropout(dropout_rate)(ps_hidden)
        ps_hidden = tf.keras.layers.BatchNormalization()(ps_hidden)

    ps_hidden = tf.keras.layers.Concatenate(name="concat_skip_connection_treatment")(
        [ps_hidden, features_bn]
    )
    pss = pypress.keras.layers.PredictiveStateSimplex(n_states=n_states, input_dim=n_features)
    pred_states = pss(ps_hidden)

    # State-conditional propensity for binary treatment (--> "sigmoid" activation), one constant
    # per state; kept state-conditional (not pre-mixed) so the causal loss can compute the exact
    # posterior P(state | X, treatment) via Bayes' rule.
    prop_score_preds = []
    for state_id in range(n_states):
        prop_score_preds.append(
            tf.keras.activations.sigmoid(
                layers.BiasOnly(name="propensity_logit_state_" + str(state_id))(features_bn)
            )
        )
    prop_score = tfk.layers.Concatenate(name="propensity_score")(prop_score_preds)

    outcome_hidden = tf.keras.layers.Dense(
        outcome_hidden_layers[0][0], outcome_hidden_layers[0][1], name="inputs_processing"
    )(feat_treat_bn)
    outcome_hidden = tf.keras.layers.Dropout(dropout_rate)(outcome_hidden)
    outcome_hidden = tf.keras.layers.BatchNormalization()(outcome_hidden)

    for units, act in outcome_hidden_layers[1:]:
        outcome_hidden = tf.keras.layers.Dense(units, act)(outcome_hidden)
        outcome_hidden = tf.keras.layers.Dropout(dropout_rate)(outcome_hidden)
        outcome_hidden = tf.keras.layers.BatchNormalization()(outcome_hidden)

    outcome_hidden = tf.keras.layers.Concatenate(name="concat_skip_connection_outcome")(
        [outcome_hidden, feat_treat_bn]
    )

    log_rate_preds = []
    # One outcome model per state.
    for state_id in range(n_states):
        log_rate_preds.append(
            tfk.layers.Dense(1, name="log_rate_pred_state_" + str(state_id))(
                tfk.layers.Dense(
                    log_rate_layer[0],
                    log_rate_layer[1],
                    name="log_rate_feat_eng_state_" + str(state_id),
                )(outcome_hidden)
            )
        )

    log_rate_comb = tfk.layers.Concatenate(name="log_rate_all_states")(log_rate_preds)

    outputs_concat = tfk.layers.Concatenate(name="output_tensor")(
        [log_rate_comb, pred_states, prop_score]
    )

    model = tfk.models.Model(inputs=[features, treat], outputs=outputs_concat)

    if compile:
        psps_causal_loss = _build_binary_exponential_causal_loss(
            n_states=n_states,
            alpha=alpha,
            df_penalty_l2=df_penalty_l2,
            outcome_loss_weight=outcome_loss_weight,
        )
        propensity_metrics, causal_loss_metric = _binary_metrics(psps_causal_loss)
        model.compile(
            loss=psps_causal_loss,
            optimizer=tfk.optimizers.Nadam(learning_rate=learning_rate),
            metrics=[
                *propensity_metrics,
                metrics.predictive_state_df_gen(psps_causal_loss.layout),
                causal_loss_metric,
            ],
        )

    return model


_SURVIVAL_DISTRIBUTIONS = ("exponential", "weibull")


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
    treatment_init_values: Optional[List[float]] = None,
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
    """Builds a pypsps model for a continuous treatment and an exponential or Weibull outcome.

    The output is ``[outcome params | predictive states | treatment (loc, scale)]``, the outcome
    params being ``log_scale`` (and ``log_shape`` for Weibull) per state, soft-clipped to the
    given bounds. Each state's outcome head sees the shared feature trunk and the treatment
    embedding gated by that state's probability. For Weibull, scale and shape have separate
    hidden layers that share no weights. y_true is ``[time, event indicator, treatment]``.

    TODO: explore other outcome-head architectures.

    Args:
      distribution: "exponential" or "weibull".
      n_states: number of predictive states.
      n_features: number of (numeric) features.
      predictive_state_hidden_layers: (units, activation) per layer of the state network.
      outcome_hidden_layers: (units, activation) per layer of the shared outcome trunk.
      log_scale_bounds: (min, max) of the predicted log scale, in log time units. Required.
      log_shape_bounds: (min, max) of the predicted Weibull log shape. Required for Weibull.
      max_time: optional largest time in the data; if set, raises a ValueError when the bounds
        let the loss overflow float32 at that time.
      state_hidden_layer: (units, activation) of each state's hidden layer per parameter.
      treatment_embedding_units: width of the treatment embedding.
      separate_shape_per_state: Weibull only; if False, all states share one shape head.
      init_scale: optional initial scale (in time units) of every state; defaults to the
        midpoint of ``log_scale_bounds`` on the log scale.
      treatment_init_values: optional initial (loc, scale) of each state's treatment density.
      compile: if True, compiles the model with the causal loss.
      alpha: weight of the treatment loss; a float32 tf.Variable is shared, not copied.
      outcome_loss_weight: weight of the outcome loss.
      outcome_likelihood: see `losses.OutcomeLikelihood`.
      predictive_states_regularizer: optional regularizer on the predictive states, e.g. a
        `pypress.keras.regularizers` instance.
      learning_rate: Nadam learning rate.
      clipnorm: gradient-norm clipping of the optimizer; None disables it.
      dropout_rate: dropout rate of the hidden layers.
      l1: L1 penalty on the hidden kernels.

    Returns:
      A tf.keras Model (compiled if ``compile=True``).
    """
    if distribution not in _SURVIVAL_DISTRIBUTIONS:
        raise ValueError(
            f"distribution must be one of {_SURVIVAL_DISTRIBUTIONS}, got {distribution!r}"
        )
    assert n_states >= 1, f"Got n_states={n_states}"
    assert n_features >= 1, f"Got n_features={n_features}"
    is_weibull = distribution == "weibull"
    # The builder requires bounds; the families alone allow unbounded parameters.
    log_scale_min, log_scale_max = families._check_bounds("log_scale_bounds", log_scale_bounds)
    if is_weibull:
        log_shape_min, log_shape_max = families._check_bounds("log_shape_bounds", log_shape_bounds)
        outcome_family = families.Weibull(log_scale_bounds, log_shape_bounds)
    elif log_shape_bounds is not None:
        raise ValueError("log_shape_bounds applies to distribution='weibull' only")
    else:
        outcome_family = families.Exponential("log_scale", bounds=log_scale_bounds)
    if max_time is not None:
        outcome_family.check_max_time(max_time)

    features = tfk.layers.Input(shape=(n_features,), name="features")
    treat = tfk.layers.Input(shape=(1,), name="treatment")
    features_bn = tfk.layers.BatchNormalization(name="features_bn")(features)
    treat_bn = tfk.layers.BatchNormalization(name="treatment_bn")(treat)
    treat_emb = tfk.layers.Dense(treatment_embedding_units, "swish", name="treatment_embedding")(
        treat_bn
    )
    feat_treat = tfk.layers.Concatenate(name="features_and_treatment")([features_bn, treat_emb])

    ps_hidden = _dense_stack(
        features_bn, predictive_state_hidden_layers, dropout_rate, l1, "predictive_state"
    )
    pred_states = pypress.keras.layers.PredictiveStateSimplex(n_states=n_states)(ps_hidden)
    treatment_params = pypress.keras.layers.PredictiveStateParams(
        n_params_per_state=2,
        activations=["linear", "softplus"],
        flatten_output=True,
        init_values=treatment_init_values,
        name="treatment_loc_scale",
    )(pred_states)

    def _param_heads(param: str, bounds: Tuple[float, float], init: float, per_state: bool):
        """One soft-clipped head per state (or one shared head) with its own trunk."""
        clip = layers.SoftClip(*bounds)
        bias = tf.keras.initializers.Constant(clip.inverse(init))
        trunk = _dense_stack(feat_treat, outcome_hidden_layers, dropout_rate, l1, f"{param}_trunk")
        heads = []
        for state_id in range(n_states if per_state else 1):
            suffix = f"state_{state_id}" if per_state else "shared"
            head_input = trunk
            if per_state:
                gate = layers.StateColumnGate(state_id, name=f"{param}_gate_{suffix}")(pred_states)
                gated_treat = tfk.layers.Multiply(name=f"{param}_gated_treatment_{suffix}")(
                    [treat_emb, gate]
                )
                head_input = tfk.layers.Concatenate(name=f"{param}_input_{suffix}")(
                    [trunk, gated_treat]
                )
            hidden = tfk.layers.Dense(
                state_hidden_layer[0], state_hidden_layer[1], name=f"{param}_hidden_{suffix}"
            )(head_input)
            raw = tfk.layers.Dense(1, bias_initializer=bias, name=f"{param}_raw_{suffix}")(hidden)
            heads.append(layers.SoftClip(*bounds, name=f"{param}_{suffix}")(raw))
        return heads if per_state else heads * n_states

    init_log_scale = (
        0.5 * (log_scale_min + log_scale_max) if init_scale is None else float(np.log(init_scale))
    )
    outcome_params = _param_heads(
        "log_scale", (log_scale_min, log_scale_max), init_log_scale, per_state=True
    )
    if is_weibull:
        outcome_params += _param_heads(
            "log_shape",
            (log_shape_min, log_shape_max),
            float(np.clip(0.0, log_shape_min, log_shape_max)),
            per_state=separate_shape_per_state,
        )
    outcome_pred = tfk.layers.Concatenate(name="outcome_params_all_states")(outcome_params)
    outputs = tfk.layers.Concatenate(name="output_tensor")(
        [outcome_pred, pred_states, treatment_params]
    )
    model = tfk.models.Model(inputs=[features, treat], outputs=outputs)

    if compile:
        causal_loss = losses.CausalLoss.from_mixture(
            mixture_lib.StateMixture(outcome_family, families.Normal(), n_states=n_states),
            alpha=alpha,
            outcome_likelihood=outcome_likelihood,
            outcome_loss_weight=outcome_loss_weight,
            predictive_states_regularizer=predictive_states_regularizer,
            reduction="sum_over_batch_size",
        )
        model.compile(
            loss=causal_loss,
            optimizer=tfk.optimizers.Nadam(learning_rate=learning_rate, clipnorm=clipnorm),
            metrics=[
                metrics.outcome_nll_metric_gen(causal_loss._outcome_loss),
                metrics.CausalLossMetric(
                    causal_loss._outcome_loss,
                    causal_loss._treatment_loss,
                    outcome_loss_weight=causal_loss._outcome_loss_weight,
                ),
                metrics.predictive_state_df_gen(causal_loss.layout),
                metrics.uniform_entropy_gen(causal_loss.layout),
            ],
        )
    return model


def build_model_continuous_exponential(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    log_scale_bounds: Tuple[float, float],
    **kwargs,
) -> tf.keras.Model:
    """`build_model_continuous_survival` with an exponential outcome (``log_scale = log mean``)."""
    return build_model_continuous_survival(
        "exponential",
        n_states=n_states,
        n_features=n_features,
        predictive_state_hidden_layers=predictive_state_hidden_layers,
        outcome_hidden_layers=outcome_hidden_layers,
        log_scale_bounds=log_scale_bounds,
        **kwargs,
    )


def build_model_continuous_weibull(
    n_states: int,
    n_features: int,
    predictive_state_hidden_layers: List[Tuple[int, str]],
    outcome_hidden_layers: List[Tuple[int, str]],
    log_scale_bounds: Tuple[float, float],
    log_shape_bounds: Tuple[float, float],
    **kwargs,
) -> tf.keras.Model:
    """`build_model_continuous_survival` with a Weibull outcome (``log_scale``, ``log_shape``)."""
    return build_model_continuous_survival(
        "weibull",
        n_states=n_states,
        n_features=n_features,
        predictive_state_hidden_layers=predictive_state_hidden_layers,
        outcome_hidden_layers=outcome_hidden_layers,
        log_scale_bounds=log_scale_bounds,
        log_shape_bounds=log_shape_bounds,
        **kwargs,
    )


def get_propensity_state_conditional_means(model: tf.keras.Model) -> tf.Tensor:
    """Returns each state's propensity constant (post-sigmoid), for models built by the
    `build_*` functions in this module.

    The propensity head is one `BiasOnly` logit per state (layers named
    "propensity_logit_state_<k>"), not a single mixed `pypress.keras.layers.PredictiveStateMeans`
    layer, so there's no single layer exposing a `.state_conditional_means` property anymore.
    This reconstructs the analogous `(n_states, 1)` tensor from each state's bias weight
    directly. Works both before and after training -- a useful sanity check that per-state
    propensity estimates are reasonable (e.g., not degenerate) at initialization and after
    fitting.
    """
    n_states = model.get_layer("propensity_score").output.shape[-1]
    logits = tf.stack(
        [
            tf.convert_to_tensor(model.get_layer(f"propensity_logit_state_{k}").weights[0])
            for k in range(n_states)
        ]
    )
    return tf.sigmoid(logits)
