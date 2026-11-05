"""Model builders for pypsps: `build_model`, `compile_model`, and the older `build_*`."""

import dataclasses
import inspect
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

_EPS = 1e-3


def _binary_exponential_mixture(n_states: int) -> mixture_lib.StateMixture:
    """Exponential (log-rate) outcome and Bernoulli treatment."""
    return mixture_lib.StateMixture(
        families.Exponential("log_rate"), families.Bernoulli(), n_states=n_states
    )


def _binary_normal_mixture(n_states: int) -> mixture_lib.StateMixture:
    """Normal outcome and Bernoulli treatment."""
    return mixture_lib.StateMixture(families.Normal(), families.Bernoulli(), n_states=n_states)


def _build_binary_exponential_causal_loss(
    mixture: mixture_lib.StateMixture,
    alpha: float,
    df_penalty_l2: float,
    outcome_loss_weight: float,
) -> losses.CausalLoss:
    """Causal loss for a binary treatment and an exponential (log-rate) survival outcome."""
    return losses.CausalLoss.from_mixture(
        mixture,
        alpha=alpha,
        outcome_loss_weight=outcome_loss_weight,
        predictive_states_regularizer=pypress.keras.regularizers.Uniform(l2=df_penalty_l2),
        reduction="sum_over_batch_size",
    )


def _build_binary_normal_causal_loss(
    mixture: mixture_lib.StateMixture,
    alpha: float,
    df_penalty_l2: float,
) -> losses.CausalLoss:
    """Causal loss for a binary treatment and a Normal outcome."""
    return losses.CausalLoss.from_mixture(
        mixture,
        alpha=alpha,
        outcome_loss_weight=1.0,
        predictive_states_regularizer=pypress.keras.regularizers.DegreesOfFreedom(
            l2=df_penalty_l2,
            target=max(1, np.log(0.5 * mixture.n_states)),
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

    # Kept so the layer order, hence the `.weights.h5` layout, matches earlier releases.
    outcome_pred = tfk.layers.Concatenate(name="loc_scale_all_states")([loc_comb, scale_comb])
    mixture = _binary_normal_mixture(n_states)
    outputs_concat = layers.CausalOutput(mixture, name="output_tensor")(
        {"outcome": outcome_pred, "state_weights": pred_states, "treatment": prop_score}
    )

    model = tfk.models.Model(inputs=[features, treat], outputs=outputs_concat)

    if compile:
        psps_causal_loss = _build_binary_normal_causal_loss(
            mixture,
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

    mixture = _binary_normal_mixture(n_states)
    outputs_concat = layers.CausalOutput(mixture, name="output_tensor")(
        {
            "outcome": {"loc": loc_comb, "scale": scale_comb},
            "state_weights": pred_states,
            "treatment": prop_score,
        }
    )

    model = tfk.models.Model(inputs=[features, treat], outputs=outputs_concat)

    if compile:
        psps_causal_loss = _build_binary_normal_causal_loss(
            mixture,
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

    mixture = _binary_exponential_mixture(n_states)
    outputs_concat = layers.CausalOutput(mixture, name="output_tensor")(
        {
            "outcome": {"log_rate": log_rate_comb},
            "state_weights": pred_states,
            "treatment": prop_score,
        }
    )

    model = tfk.models.Model(inputs=[features, treat], outputs=outputs_concat)

    if compile:
        psps_causal_loss = _build_binary_exponential_causal_loss(
            mixture,
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
    mixture = mixture_lib.StateMixture(outcome_family, families.Normal(), n_states=n_states)
    outputs = layers.CausalOutput(mixture, name="output_tensor")(
        {"outcome": outcome_pred, "state_weights": pred_states, "treatment": treatment_params}
    )
    model = tfk.models.Model(inputs=[features, treat], outputs=outputs)

    if compile:
        causal_loss = losses.CausalLoss.from_mixture(
            mixture,
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
    **kwargs: Any,
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
    **kwargs: Any,
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

    Experimental in v0.3.x: the interface may still change before v0.4.0.

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

    Experimental in v0.3.x: the interface may still change before v0.4.0.

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
    ``propensity_score_binary_crossentropy`` and ``propensity_score_aupr``; any other treatment
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
            metrics.PropensityScoreAUC(mixture, curve="PR", name="propensity_score_aupr"),
        ]
    else:
        chosen.append(metrics.TreatmentMeanSquaredError(mixture))
    if not isinstance(mixture.outcome_family, families.LossFamily):
        chosen.append(metrics.OutcomeMeanSquaredError(mixture))
    return chosen
