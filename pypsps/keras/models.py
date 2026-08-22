"""Example model architectures for pypsps."""

from typing import List, Tuple

import numpy as np
import pypress
import pypress.keras.layers
import pypress.keras.regularizers
import tensorflow as tf

from . import layers, losses, metrics, neglogliks

tfk = tf.keras

_EPS = 1e-3


def _build_binary_exponential_causal_loss(
    n_states: int,
    alpha: float,
    df_penalty_l1: float,
    outcome_loss_weight: float,
) -> losses.CausalLoss:
    """Builds an example of binary treatment & continuous outcome causal loss."""

    del n_states  # not used in this function, but kept for consistency with other loss builders

    treatment_nll = tf.keras.losses.BinaryCrossentropy(reduction="none")
    psps_outcome_loss = losses.OutcomeLoss(
        loss=neglogliks.NegloglikExponential(reduction="none", log_rate=True),
        treatment_loss=treatment_nll,
        n_outcome_true_cols=2,
        n_outcome_pred_cols=1,
        n_treatment_pred_cols=1,
        reduction="sum_over_batch_size",
    )
    psps_treat_loss = losses.TreatmentLoss(
        loss=treatment_nll,
        n_outcome_true_cols=2,
        n_outcome_pred_cols=1,
        n_treatment_pred_cols=1,
        reduction="sum_over_batch_size",
    )
    psps_causal_loss = losses.CausalLoss(
        outcome_loss=psps_outcome_loss,
        treatment_loss=psps_treat_loss,
        alpha=alpha,
        outcome_loss_weight=outcome_loss_weight,
        predictive_states_regularizer=pypress.keras.regularizers.Uniform(l2=df_penalty_l1),
        reduction="sum_over_batch_size",
    )
    return psps_causal_loss


def _build_binary_normal_causal_loss(
    n_states: int,
    alpha: float,
    df_penalty_l1: float,
) -> losses.CausalLoss:
    """Builds an example of binary treatment & Normal outcome causal loss."""
    treatment_nll = tf.keras.losses.BinaryCrossentropy(reduction="none")
    psps_outcome_loss = losses.OutcomeLoss(
        loss=neglogliks.NegloglikNormal(reduction="none"),
        treatment_loss=treatment_nll,
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction="sum_over_batch_size",
    )
    psps_treat_loss = losses.TreatmentLoss(
        loss=treatment_nll,
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction="sum_over_batch_size",
    )
    psps_causal_loss = losses.CausalLoss(
        outcome_loss=psps_outcome_loss,
        treatment_loss=psps_treat_loss,
        alpha=alpha,
        outcome_loss_weight=1.0,
        predictive_states_regularizer=pypress.keras.regularizers.DegreesOfFreedom(
            l1=df_penalty_l1,
            target=max(1, np.log(0.5 * n_states)),
        ),
        reduction="sum_over_batch_size",
    )
    return psps_causal_loss


def build_toy_model(
    n_states: int,
    n_features: int,
    compile: bool = True,
    alpha: float = 1.0,
    df_penalty_l1: float = 1.0,
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
      alpha: propensity score penalty (by default alpha = 1., which corresponds to equal weight)
      df_penalty_l1: l1 parameter for the DF regularization
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
            df_penalty_l1=df_penalty_l1,
        )
        model.compile(
            loss=psps_causal_loss,
            optimizer=tfk.optimizers.Nadam(learning_rate=learning_rate),
            metrics=[
                metrics.PropensityScoreBinaryCrossentropy(
                    n_outcome_pred_cols=2, n_treatment_pred_cols=1
                ),
                metrics.PropensityScoreAUC(
                    n_outcome_pred_cols=2, n_treatment_pred_cols=1, curve="PR"
                ),
                metrics.OutcomeMeanSquaredError(
                    n_outcome_pred_cols=2, n_treatment_pred_cols=1, n_outcome_true_cols=1
                ),
                metrics.causal_loss_metric_gen(
                    outcome_loss=psps_causal_loss._outcome_loss,
                    treatment_loss=psps_causal_loss._treatment_loss,
                ),
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
    df_penalty_l1: float = 1.0,
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
      alpha: propensity score penalty (by default alpha = 1., which corresponds to equal weight)
      df_penalty_l1: l1 parameter for the DF regularization
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
            df_penalty_l1=df_penalty_l1,
        )
        model.compile(
            loss=psps_causal_loss,
            optimizer=tfk.optimizers.Nadam(learning_rate=learning_rate),
            metrics=[
                metrics.PropensityScoreBinaryCrossentropy(
                    n_outcome_pred_cols=2, n_treatment_pred_cols=1
                ),
                metrics.PropensityScoreAUC(
                    n_outcome_pred_cols=2, n_treatment_pred_cols=1, curve="PR"
                ),
                metrics.OutcomeMeanSquaredError(
                    n_outcome_pred_cols=2, n_treatment_pred_cols=1, n_outcome_true_cols=1
                ),
                metrics.causal_loss_metric_gen(
                    outcome_loss=psps_causal_loss._outcome_loss,
                    treatment_loss=psps_causal_loss._treatment_loss,
                ),
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
    df_penalty_l1: float = 1.0,
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
      alpha: propensity score penalty (by default alpha = 1., which corresponds to equal weight)
      df_penalty_l1: l1 parameter for the DF regularization
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

    # Binary treatment & continuous exponential survival model architecture.
    _n_outcome_true_cols = 2  # (outcome, event indicator)
    _n_outcome_pred_cols = 1  # (log_rate)
    _n_treatment_pred_cols = 1  # (propensity score)

    if compile:
        psps_causal_loss = _build_binary_exponential_causal_loss(
            n_states=n_states,
            alpha=alpha,
            df_penalty_l1=df_penalty_l1,
            outcome_loss_weight=outcome_loss_weight,
        )
        model.compile(
            loss=psps_causal_loss,
            optimizer=tfk.optimizers.Nadam(learning_rate=learning_rate),
            metrics=[
                metrics.PropensityScoreBinaryCrossentropy(
                    n_treatment_pred_cols=_n_treatment_pred_cols,
                    n_outcome_pred_cols=_n_outcome_pred_cols,
                ),
                metrics.PropensityScoreAUC(
                    curve="PR",
                    n_treatment_pred_cols=_n_treatment_pred_cols,
                    n_outcome_pred_cols=_n_outcome_pred_cols,
                ),
                metrics.predictive_state_df_gen(
                    n_treatment_pred_cols=_n_treatment_pred_cols,
                    n_outcome_pred_cols=_n_outcome_pred_cols,
                ),
                metrics.causal_loss_metric_gen(
                    outcome_loss=psps_causal_loss._outcome_loss,
                    treatment_loss=psps_causal_loss._treatment_loss,
                ),
            ],
        )

    return model


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
