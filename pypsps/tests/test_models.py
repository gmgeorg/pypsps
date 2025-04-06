"""Test module for model functions."""

import numpy as np
import tensorflow as tf

from .. import datasets
from ..keras import models

tfk = tf.keras


def test_build_toy_model():
    """test toy model"""
    np.random.seed(10)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=1000)

    inputs, outputs = ks_data.to_keras_inputs_outputs()
    tf.random.set_seed(10)
    model = models.build_toy_model(n_states=3, n_features=ks_data.n_features, compile=True)
    preds = model.predict(inputs)
    assert preds.shape[0] == ks_data.n_samples
    assert not np.isnan(preds.sum().sum())


def test_build_model_binary_normal():
    """test build models"""
    np.random.seed(10)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=1000)

    inputs, _ = ks_data.to_keras_inputs_outputs()
    tf.random.set_seed(10)
    model = models.build_model_binary_normal(
        n_states=3,
        n_features=ks_data.n_features,
        compile=True,
        predictive_state_hidden_layers=[(10, "selu"), (20, "relu")],
        outcome_hidden_layers=[(30, "tanh"), (20, "selu")],
        loc_layer=(20, "selu"),
        scale_layer=(10, "tanh"),
    )
    preds = model.predict(inputs)
    assert preds.shape[0] == ks_data.n_samples
    assert not np.isnan(preds.sum().sum())


def test_build_model_binary_exponential():
    """test build models"""
    np.random.seed(10)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=1000)

    inputs, _ = ks_data.to_keras_inputs_outputs()
    tf.random.set_seed(10)
    model = models.build_model_binary_exponential(
        n_states=3,
        n_features=ks_data.n_features,
        compile=True,
        predictive_state_hidden_layers=[(10, "selu"), (20, "relu")],
        outcome_hidden_layers=[(30, "tanh"), (20, "selu")],
        log_rate_layer=(20, "selu"),
    )
    preds = model.predict(inputs)
    assert preds.shape[0] == ks_data.n_samples
    assert not np.isnan(preds.sum().sum())
