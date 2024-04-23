"""Test module for loss functions."""

from typing import Tuple

import numpy as np
import pytest
import pandas as pd
import tensorflow as tf

from .. import datasets
from ..keras import losses, models
from ..keras import layers as pypsps_layers
from .. import utils, inference
from ..keras import metrics

from pypress.keras import layers as press_layers
from pypress.keras import regularizers


tfk = tf.keras


def _test_data() -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([[0.0, 1.0], [-1, 0.1], [0.1, 0.5]])
    return y_true, y_pred


@pytest.mark.parametrize(
    "reduction,expected_len",
    [("auto", 1), ("sum", 1), ("sum_over_batch_size", 1), ("none", 3)],
)
def test_negloglik_normal_loss(reduction, expected_len):
    y_true, y_pred = _test_data()
    loss = losses.NegloglikNormal(reduction=reduction)(
        y_true=y_true.astype("float32"), y_pred=y_pred.astype("float32")
    )
    if expected_len == 1:
        assert not len(loss.numpy().shape)
    else:
        assert loss.shape[0] == expected_len


def test_psps_model_and_causal_loss():
    pypsps_outcome_loss = losses.OutcomeLoss(
        loss=losses.NegloglikNormal(reduction="none"), reduction="auto"
    )

    pypsps_treat_loss = losses.TreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"), reduction="auto"
    )
    pypsps_causal_loss = losses.CausalLoss(
        outcome_loss=pypsps_outcome_loss,
        treatment_loss=pypsps_treat_loss,
        alpha=1.0,
        outcome_loss_weight=0.0,
        predictive_states_regularizer=tf.keras.regularizers.l2(0.1),
        reduction="auto",
    )

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
    assert causal_loss.numpy() == pytest.approx(25.44, 0.1)


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
    l = history.history["loss"]
    assert l[0] > l[-1]
    preds = model.predict(inputs)

    assert preds.shape[0] == ks_data.n_samples

    outcome_pred, scale_pred, weights, prop_score = utils.split_y_pred(preds)

    preds_comb = np.hstack([outcome_pred, scale_pred, weights, prop_score])
    np.testing.assert_allclose(preds, preds_comb)
    ate = inference.predict_ate(model, inputs[0])
    assert ate > 0
