"""Test module for loss functions."""

import numpy as np
import pytest
import tensorflow as tf
import random

from .. import datasets
from ..keras import losses, models
from ..keras import neglogliks
from .. import utils, inference


tfk = tf.keras


def test_psps_model_and_causal_loss():
    tf.random.set_seed(0)
    random.seed(0)
    np.random.seed(0)

    pypsps_outcome_loss = losses.OutcomeLoss(
        loss=neglogliks.NegloglikNormal(reduction="none"),
        reduction="sum_over_batch_size",
    )

    pypsps_treat_loss = losses.TreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        reduction="sum_over_batch_size",
    )
    pypsps_causal_loss = losses.CausalLoss(
        outcome_loss=pypsps_outcome_loss,
        treatment_loss=pypsps_treat_loss,
        alpha=1.0,
        outcome_loss_weight=0.0,
        predictive_states_regularizer=tf.keras.regularizers.l2(0.1),
        reduction="sum_over_batch_size",
    )

    np.random.seed(10)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=1000)

    inputs, outputs = ks_data.to_keras_inputs_outputs()
    assert outputs.shape == (1000, 2)

    tf.random.set_seed(10)
    model = models.build_toy_model(
        n_states=3, n_features=ks_data.n_features, compile=True
    )
    preds = model.predict(inputs)
    outcome_pred, const_scale, weights, propensity_score = utils.split_y_pred(preds)

    assert outcome_pred.shape == (1000, 3)  # (obs, states)
    assert const_scale.shape == (1000, 3)
    assert propensity_score.shape[0] == 1000
    assert weights.shape == (1000, 3)
    causal_loss = pypsps_causal_loss(outputs, preds)
    assert causal_loss.numpy() == pytest.approx(25.88, 0.01)


def test_end_to_end_dataset_model_fit():
    np.random.seed(13)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=1000)
    tf.random.set_seed(13)
    model = models.build_toy_model(
        n_states=5, n_features=ks_data.features.shape[1], compile=True
    )
    inputs, outputs = ks_data.to_keras_inputs_outputs()
    history = model.fit(
        inputs,
        outputs,
        epochs=2,
        batch_size=64,
        verbose=2,
        validation_split=0.2,
    )
    losses = history.history["loss"]
    assert losses[0] > losses[-1]
    preds = model.predict(inputs)

    assert preds.shape[0] == ks_data.n_samples

    ate = inference.predict_ate(model, inputs[0])
    assert ate > 0
