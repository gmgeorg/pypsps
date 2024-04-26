"""Example model architectures for pypsps."""

from typing import List, Tuple

import tensorflow as tf

import pypress
import pypress.keras.layers
import pypress.keras.regularizers

from . import losses, layers, metrics, neglogliks


tfk = tf.keras


def recommended_callbacks(monitor="val_loss") -> List[tf.keras.callbacks.Callback]:
    """Return a list of recommended callbacks.

    This list is subject to change w/o notice. Do not rely on this in production.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=20, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=10),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    return callbacks


def _build_binary_continuous_causal_loss(
    n_states: int,
    alpha: float,
    df_penalty_l1: float,
) -> losses.CausalLoss:
    """Builds an example of binary treatment & continuous outcome causal loss."""
    psps_outcome_loss = losses.OutcomeLoss(
        loss=neglogliks.NegloglikNormal(reduction="none"),
        reduction="sum_over_batch_size",
    )
    psps_treat_loss = losses.TreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        reduction="sum_over_batch_size",
    )
    psps_causal_loss = losses.CausalLoss(
        outcome_loss=psps_outcome_loss,
        treatment_loss=psps_treat_loss,
        alpha=alpha,
        outcome_loss_weight=1.0,
        predictive_states_regularizer=pypress.keras.regularizers.DegreesOfFreedom(
            l1=df_penalty_l1, df=n_states - 1
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
    """Builds a pypsps toy model for binary treatment & continous outcome.

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
    feat_treat = tfk.layers.Concatenate(name="features_and_treatment")(
        [features_bn, treat]
    )

    ps_hidden = tfk.layers.Dense(10, "relu")(features_bn)
    ps_hidden = tfk.layers.BatchNormalization()(ps_hidden)
    ps_hidden = tfk.layers.Dropout(0.2)(ps_hidden)
    ps_hidden = tfk.layers.Dense(10, "selu")(ps_hidden)

    ps_hidden = tf.keras.layers.Concatenate()([ps_hidden, features_bn])
    pss = pypress.keras.layers.PredictiveStateSimplex(
        n_states=n_states, input_dim=n_features
    )
    pred_states = pss(ps_hidden)
    # Propensity score for binary treatment (--> "sigmoid" activation).
    prop_score = pypress.keras.layers.PredictiveStateMeans(
        units=1, activation="sigmoid", name="propensity_score"
    )(pred_states)

    outcome_hidden = tf.keras.layers.Dense(10, "tanh")(feat_treat)
    outcome_hidden = tf.keras.layers.Dropout(0.2)(outcome_hidden)
    outcome_hidden = tf.keras.layers.BatchNormalization()(outcome_hidden)

    outcome_hidden = tf.keras.layers.Concatenate()([outcome_hidden, feat_treat])

    outcome_preds = []
    scale_preds = []
    # One outcome model per state.
    for state_id in range(n_states):
        outcome_preds.append(
            tfk.layers.Dense(1, name="outcome_pred_state_" + str(state_id))(
                tfk.layers.Dense(5, "selu", name="feat_eng_state_" + str(state_id))(
                    outcome_hidden
                )
            )
        )

        # In this toy model use a constant scale estimate (BiasOnly); if needed
        # change this to a scale parameter that changes as a function of inputs / hidden layers.
        scale_preds.append(
            tf.keras.activations.softplus(
                layers.BiasOnly(name="scale_logit_" + str(state_id))(feat_treat)
            )
        )

    outcome_comb = tfk.layers.Concatenate(name="outcome_pred_combined")(outcome_preds)
    scale_comb = tfk.layers.Concatenate(name="scale_pred_combined")(scale_preds)

    outputs_concat = tfk.layers.Concatenate(name="output_tensor")(
        [outcome_comb, scale_comb, pred_states, prop_score]
    )

    model = tfk.models.Model(inputs=[features, treat], outputs=outputs_concat)

    if compile:

        psps_causal_loss = _build_binary_continuous_causal_loss(
            n_states=n_states,
            alpha=alpha,
            df_penalty_l1=df_penalty_l1,
        )
        model.compile(
            loss=psps_causal_loss,
            optimizer=tfk.optimizers.Nadam(learning_rate=learning_rate),
            metrics=[
                metrics.PropensityScoreBinaryCrossentropy(),
                metrics.PropensityScoreAUC(curve="PR"),
                metrics.OutcomeMeanSquaredError(),
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
    """Builds a pypsps toy model for binary treatment & continous outcome.

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
    feat_treat = tfk.layers.Concatenate(name="features_and_treatment")(
        [features_bn, treat]
    )

    ps_hidden = tf.keras.layers.Dense(
        predictive_state_hidden_layers[0][0], predictive_state_hidden_layers[0][1]
    )(features_bn)
    ps_hidden = tf.keras.layers.Dropout(dropout_rate)(ps_hidden)
    ps_hidden = tf.keras.layers.BatchNormalization()(ps_hidden)

    for units, act in predictive_state_hidden_layers[1:]:
        ps_hidden = tf.keras.layers.Dense(units, act)(ps_hidden)
        ps_hidden = tf.keras.layers.Dropout(dropout_rate)(ps_hidden)
        ps_hidden = tf.keras.layers.BatchNormalization()(ps_hidden)

    ps_hidden = tf.keras.layers.Concatenate()([ps_hidden, features_bn])
    pss = pypress.keras.layers.PredictiveStateSimplex(
        n_states=n_states, input_dim=n_features
    )
    pred_states = pss(ps_hidden)

    # Propensity score for binary treatment (--> "sigmoid" activation).
    prop_score = pypress.keras.layers.PredictiveStateMeans(
        units=1, activation="sigmoid", name="propensity_score"
    )(pred_states)

    outcome_hidden = tf.keras.layers.Dense(
        outcome_hidden_layers[0][0], outcome_hidden_layers[0][1]
    )(feat_treat)
    outcome_hidden = tf.keras.layers.Dropout(dropout_rate)(outcome_hidden)
    outcome_hidden = tf.keras.layers.BatchNormalization()(outcome_hidden)

    for units, act in outcome_hidden_layers[1:]:
        outcome_hidden = tf.keras.layers.Dense(units, act)(outcome_hidden)
        outcome_hidden = tf.keras.layers.Dropout(dropout_rate)(outcome_hidden)
        outcome_hidden = tf.keras.layers.BatchNormalization()(outcome_hidden)

    outcome_hidden = tf.keras.layers.Concatenate()([outcome_hidden, feat_treat])

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
                    layers.BiasOnly(name="scale_logit_" + str(state_id))(feat_treat)
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

    loc_comb = tfk.layers.Concatenate(name="loc_pred_combined")(loc_preds)
    scale_comb = tfk.layers.Concatenate(name="scale_pred_combined")(scale_preds)

    outputs_concat = tfk.layers.Concatenate(name="output_tensor")(
        [loc_comb, scale_comb, pred_states, prop_score]
    )

    model = tfk.models.Model(inputs=[features, treat], outputs=outputs_concat)

    if compile:

        psps_causal_loss = _build_binary_continuous_causal_loss(
            n_states=n_states,
            alpha=alpha,
            df_penalty_l1=df_penalty_l1,
        )
        model.compile(
            loss=psps_causal_loss,
            optimizer=tfk.optimizers.Nadam(learning_rate=learning_rate),
            metrics=[
                metrics.PropensityScoreBinaryCrossentropy(),
                metrics.PropensityScoreAUC(curve="PR"),
                metrics.OutcomeMeanSquaredError(),
            ],
        )

    return model
