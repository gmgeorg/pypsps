"""Test module for inference functions."""

import types

import numpy as np
import pytest
import tensorflow as tf

from .. import datasets, inference, utils
from ..keras import models, neglogliks

tfk = tf.keras


def _make_stub_model(outcome_nll):
    """Minimal stand-in for a compiled model, exposing only what `_state_conditional_outcome_mean`
    reads (`model.loss._outcome_loss._loss`)."""
    outcome_loss = types.SimpleNamespace(_loss=outcome_nll)
    loss = types.SimpleNamespace(_outcome_loss=outcome_loss)
    return types.SimpleNamespace(loss=loss)


def test_state_conditional_outcome_mean_normal():
    """Normal outcome: the mean is the loc param block, unchanged."""
    model = _make_stub_model(neglogliks.NegloglikNormal(reduction="none"))
    layout = utils.ColumnLayout(
        n_outcome_pred_cols=2, n_treatment_pred_cols=1, n_outcome_true_cols=1
    )
    # 2 states, params interleaved as [loc_s0, loc_s1, scale_s0, scale_s1].
    outcome_params = np.array([[1.0, 2.0, 0.5, 0.5]], dtype="float32")
    mean = inference._state_conditional_outcome_mean(model, outcome_params, layout)
    np.testing.assert_allclose(mean, [[1.0, 2.0]])


def test_state_conditional_outcome_mean_exponential_log_rate():
    """Exponential outcome (log_rate=True): mean is exp(-log_rate), not log_rate itself."""
    model = _make_stub_model(neglogliks.NegloglikExponential(reduction="none", log_rate=True))
    layout = utils.ColumnLayout(
        n_outcome_pred_cols=1, n_treatment_pred_cols=1, n_outcome_true_cols=2
    )
    log_rate = np.array([[0.0, np.log(2.0)]], dtype="float32")  # rates 1, 2 -> means 1, 0.5
    mean = inference._state_conditional_outcome_mean(model, log_rate, layout)
    np.testing.assert_allclose(mean, [[1.0, 0.5]], rtol=1e-5)


def test_state_conditional_outcome_mean_unsupported_loss_raises():
    """An outcome loss this helper has no case for raises, instead of a silently wrong number."""
    model = _make_stub_model(tf.keras.losses.MeanSquaredError(reduction="none"))
    layout = utils.ColumnLayout(
        n_outcome_pred_cols=1, n_treatment_pred_cols=1, n_outcome_true_cols=1
    )
    with pytest.raises(NotImplementedError):
        inference._state_conditional_outcome_mean(model, np.zeros((1, 1), dtype="float32"), layout)


def test_ute_binary_works_for_exponential_survival_model():
    """predict_ute_binary/predict_ate_binary run end-to-end on an Exponential outcome model.

    Previously hardcoded n_outcome_pred_cols=2 (Normal-only), which silently mis-sliced columns
    for this model's n_outcome_pred_cols=1 (log_rate) and returned a meaningless number with no
    error -- see get_column_layout.
    """
    np.random.seed(10)
    simulator = datasets.CancerSurvivalSimulator(seed=42)
    surv_data = simulator.sample(n_samples=500)
    inputs, outputs = surv_data.to_keras_inputs_outputs()

    tf.random.set_seed(10)
    model = models.build_model_binary_exponential(
        n_states=4,
        n_features=surv_data.n_features,
        compile=True,
        predictive_state_hidden_layers=[(10, "selu")],
        outcome_hidden_layers=[(10, "selu")],
        log_rate_layer=(10, "selu"),
    )
    model.fit(inputs, outputs, epochs=1, batch_size=32, verbose=0)

    ute = inference.predict_ute_binary(model, inputs[0])
    assert ute.shape[0] == surv_data.n_samples
    assert np.all(np.isfinite(ute))

    ate = inference.predict_ate_binary(model, inputs[0])
    assert np.isfinite(ate)


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
