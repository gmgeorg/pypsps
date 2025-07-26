"""Test module for inference functions."""

import pytest
import tensorflow as tf

from .. import datasets, inference
from ..keras import models

tfk = tf.keras


def test_end_to_end_dataset_model_fit_and_inference():
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
