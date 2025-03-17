"""Test module for loss functions."""

import random

import numpy as np
import pytest
import tensorflow as tf

from .. import datasets, inference, utils
from ..keras import losses, models, neglogliks

tfk = tf.keras


def test_psps_model_build_and_predict():
    """Test build & predict"""
    tf.random.set_seed(0)
    random.seed(0)
    np.random.seed(0)

    ks_data = datasets.KangSchafer(true_ate=10, seed=10).sample(n_samples=1000)

    inputs, outputs = ks_data.to_keras_inputs_outputs()
    assert outputs.shape == (1000, 2)

    tf.random.set_seed(10)
    model = models.build_toy_model(n_states=3, n_features=ks_data.n_features, compile=True)
    preds = model.predict(inputs)
    outcome_params_pred, weights, propensity_score = utils.split_y_pred(preds, 2, 1)

    assert outcome_params_pred.shape == (1000, 3 * 2)  # (obs, states * 2) for (loc, scale)
    assert propensity_score.shape[0] == 1000
    assert weights.shape == (1000, 3)


@pytest.mark.parametrize(
    "reduction,expected_shape",
    [("sum", ()), ("sum_over_batch_size", ()), ("none", (5,))],  # ("auto", 1),
)
def test_psps_causal_loss(reduction, expected_shape):
    """Test psps causal loss"""
    tf.random.set_seed(0)
    random.seed(0)
    np.random.seed(0)

    pypsps_outcome_loss = losses.OutcomeLoss(
        loss=neglogliks.NegloglikNormal(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction=reduction,
    )

    pypsps_treat_loss = losses.TreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction=reduction,
    )
    pypsps_causal_loss = losses.CausalLoss(
        outcome_loss=pypsps_outcome_loss,
        treatment_loss=pypsps_treat_loss,
        alpha=1.0,
        outcome_loss_weight=1.0,
        predictive_states_regularizer=tf.keras.regularizers.l2(0.1),
        reduction=reduction,
    )

    ks_data = datasets.KangSchafer(true_ate=10, seed=10).sample(n_samples=5)

    inputs, outputs = ks_data.to_keras_inputs_outputs()
    assert outputs.shape == (5, 2)

    tf.random.set_seed(10)
    model = models.build_toy_model(n_states=3, n_features=ks_data.n_features, compile=True)
    preds = model.predict(inputs)

    causal_loss = pypsps_causal_loss(outputs, preds)
    print(reduction, causal_loss)
    assert causal_loss.shape == expected_shape


def test_end_to_end_dataset_model_fit():
    """test fitting data & model"""
    ks_data = datasets.KangSchafer(true_ate=10, seed=13).sample(n_samples=1000)
    tf.random.set_seed(13)
    model = models.build_toy_model(n_states=5, n_features=ks_data.features.shape[1], compile=True)
    inputs, outputs = ks_data.to_keras_inputs_outputs()
    history = model.fit(
        inputs,
        outputs,
        epochs=20,
        batch_size=64,
        verbose=2,
        validation_split=0.2,
    )
    losses = history.history["loss"]
    assert losses[0] > losses[-1]
    preds = model.predict(inputs)

    assert preds.shape[0] == ks_data.n_samples

    ute = inference.predict_ute_binary(model, inputs[0])
    assert ute.shape[0] == ks_data.n_samples

    ate = inference.predict_ate_binary(model, inputs[0])
    print(ate)
    assert ate == pytest.approx(10.0, 2.0)
