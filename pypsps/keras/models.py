"""Example model architectures for pypsps."""

from typing import List

import tensorflow as tf

import pypress
import pypress.keras.layers
import pypress.keras.regularizers
from . import losses
from . import layers
from . import metrics


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
    n_states: int, alpha: float = 1.0
) -> losses.CausalLoss:
    """Builds an example of binary treatment & continuous outcome causal loss."""
    psps_outcome_loss = losses.OutcomeLoss(
        loss=losses.NegloglikNormal(reduction="none"), reduction="auto"
    )
    psps_treat_loss = losses.TreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"), reduction="auto"
    )
    psps_causal_loss = losses.CausalLoss(
        outcome_loss=psps_outcome_loss,
        treatment_loss=psps_treat_loss,
        alpha=alpha,
        outcome_loss_weight=1.0,
        predictive_states_regularizer=pypress.keras.regularizers.DegreesOfFreedom(
            10.0, df=n_states - 1
        ),
        reduction="auto",
    )
    return psps_causal_loss


def build_toy_model(
    n_states: int,
    n_features: int,
    compile: bool = True,
    alpha: float = 1.0,
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
    constant_scale_preds = []
    # One outcome model per state.
    for state_id in range(n_states):
        outcome_preds.append(
            tfk.layers.Dense(1, name="outcome_pred_state_" + str(state_id))(
                tfk.layers.Dense(5, "selu", name="feat_eng_state_" + str(state_id))(
                    outcome_hidden
                )
            )
        )

        constant_scale_preds.append(
            tf.keras.activations.softplus(
                layers.BiasOnly(name="scale_logit_" + str(state_id))(feat_treat)
            )
        )

    outcome_comb = tfk.layers.Concatenate(name="outcome_pred_combined")(outcome_preds)
    constant_scale_comb = tfk.layers.Concatenate(name="constant_scale_combined")(
        constant_scale_preds
    )

    outputs_concat = tfk.layers.Concatenate(name="output_tensor")(
        [outcome_comb, constant_scale_comb, prop_score, pred_states]
    )

    model = tfk.models.Model(inputs=[features, treat], outputs=outputs_concat)

    if compile:

        psps_causal_loss = _build_binary_continuous_causal_loss(
            n_states=n_states, alpha=alpha
        )
        model.compile(
            loss=psps_causal_loss,
            optimizer=tfk.optimizers.Nadam(learning_rate=0.01),
            metrics=[metrics.propensity_score_crossentropy],
        )

    return model
