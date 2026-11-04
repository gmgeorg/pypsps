"""Example model architectures for pypsps."""

from typing import List, Optional, Tuple, Union

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
    df_penalty_l2: float,
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
        predictive_states_regularizer=pypress.keras.regularizers.Uniform(l2=df_penalty_l2),
        reduction="sum_over_batch_size",
    )
    return psps_causal_loss


def _build_binary_normal_causal_loss(
    n_states: int,
    alpha: float,
    df_penalty_l2: float,
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
            l2=df_penalty_l2,
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
      alpha: propensity score penalty (by default alpha = 1., which corresponds to equal weight)
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
      alpha: propensity score penalty (by default alpha = 1., which corresponds to equal weight)
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
      alpha: propensity score penalty (by default alpha = 1., which corresponds to equal weight)
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

    # Binary treatment & continuous exponential survival model architecture.
    _n_outcome_true_cols = 2  # (outcome, event indicator)
    _n_outcome_pred_cols = 1  # (log_rate)
    _n_treatment_pred_cols = 1  # (propensity score)

    if compile:
        psps_causal_loss = _build_binary_exponential_causal_loss(
            n_states=n_states,
            alpha=alpha,
            df_penalty_l2=df_penalty_l2,
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


_SURVIVAL_DISTRIBUTIONS = ("exponential", "weibull")

# float32 exp() overflows near 88.7; keep the largest exponent of the survival NLL below this,
# with a wide margin.
_MAX_SURVIVAL_EXPONENT = 50.0


def _check_bounds(name: str, bounds: Tuple[float, float]) -> Tuple[float, float]:
    """Validates a ``(min, max)`` pair of finite floats with ``min < max``."""
    if bounds is None or len(bounds) != 2:
        raise ValueError(f"{name} must be a (min, max) pair, got {bounds!r}")
    low, high = float(bounds[0]), float(bounds[1])
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        raise ValueError(f"{name} must be finite with min < max, got {bounds!r}")
    return low, high


def _check_survival_exponent(
    max_time: float, log_scale_min: float, log_shape_max: Optional[float]
) -> None:
    """Raises if the outcome NLL can overflow float32 for times up to ``max_time``.

    The exponent of ``(t / scale) ** shape`` is at most
    ``shape_max * (log(max_time) - log_scale_min)``, with ``shape_max = 1`` for exponential.
    """
    if not max_time > 0:
        raise ValueError(f"max_time must be positive, got {max_time!r}")
    shape_max = 1.0 if log_shape_max is None else float(np.exp(log_shape_max))
    exponent = shape_max * (np.log(max_time) - log_scale_min)
    if exponent > _MAX_SURVIVAL_EXPONENT:
        raise ValueError(
            f"shape_max * (log(max_time) - log_scale_min) = {exponent:.1f} exceeds "
            f"{_MAX_SURVIVAL_EXPONENT}: the loss would overflow float32. Raise the lower "
            "log-scale bound, lower the upper log-shape bound, or rescale time."
        )


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


def _build_continuous_survival_causal_loss(
    distribution: str,
    alpha: float,
    outcome_loss_weight: float,
    outcome_likelihood: Union[losses.OutcomeLikelihood, str],
    predictive_states_regularizer: Optional[tf.keras.regularizers.Regularizer],
) -> losses.CausalLoss:
    """Causal loss for a Normal continuous treatment and an exponential or Weibull outcome."""
    n_outcome_pred_cols = 2 if distribution == "weibull" else 1
    outcome_nll = (
        neglogliks.NegloglikWeibull(reduction="none")
        if distribution == "weibull"
        else neglogliks.NegloglikExponentialScale(reduction="none")
    )
    treatment_nll = neglogliks.NegloglikNormal(reduction="none")
    psps_outcome_loss = losses.OutcomeLoss(
        loss=outcome_nll,
        treatment_loss=treatment_nll,
        n_outcome_true_cols=2,
        n_outcome_pred_cols=n_outcome_pred_cols,
        n_treatment_pred_cols=2,
        outcome_likelihood=outcome_likelihood,
        reduction="sum_over_batch_size",
    )
    psps_treat_loss = losses.TreatmentLoss(
        loss=treatment_nll,
        n_outcome_true_cols=2,
        n_outcome_pred_cols=n_outcome_pred_cols,
        n_treatment_pred_cols=2,
        reduction="sum_over_batch_size",
    )
    return losses.CausalLoss(
        outcome_loss=psps_outcome_loss,
        treatment_loss=psps_treat_loss,
        alpha=alpha,
        outcome_loss_weight=outcome_loss_weight,
        predictive_states_regularizer=predictive_states_regularizer,
        reduction="sum_over_batch_size",
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
    log_scale_min, log_scale_max = _check_bounds("log_scale_bounds", log_scale_bounds)
    if is_weibull:
        log_shape_min, log_shape_max = _check_bounds("log_shape_bounds", log_shape_bounds)
    elif log_shape_bounds is not None:
        raise ValueError("log_shape_bounds applies to distribution='weibull' only")
    if max_time is not None:
        _check_survival_exponent(max_time, log_scale_min, log_shape_max if is_weibull else None)

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
        causal_loss = _build_continuous_survival_causal_loss(
            distribution=distribution,
            alpha=alpha,
            outcome_loss_weight=outcome_loss_weight,
            outcome_likelihood=outcome_likelihood,
            predictive_states_regularizer=predictive_states_regularizer,
        )
        n_outcome_pred_cols = 2 if is_weibull else 1
        model.compile(
            loss=causal_loss,
            optimizer=tfk.optimizers.Nadam(learning_rate=learning_rate, clipnorm=clipnorm),
            metrics=[
                metrics.outcome_nll_metric_gen(causal_loss._outcome_loss),
                metrics.causal_loss_metric_gen(
                    outcome_loss=causal_loss._outcome_loss,
                    treatment_loss=causal_loss._treatment_loss,
                    alpha=causal_loss._alpha,
                    outcome_loss_weight=outcome_loss_weight,
                ),
                metrics.predictive_state_df_gen(
                    n_outcome_pred_cols=n_outcome_pred_cols, n_treatment_pred_cols=2
                ),
                metrics.uniform_entropy_gen(
                    n_outcome_pred_cols=n_outcome_pred_cols, n_treatment_pred_cols=2
                ),
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
