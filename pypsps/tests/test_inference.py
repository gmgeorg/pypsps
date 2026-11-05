"""Test module for inference functions."""

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from .. import datasets, inference
from ..keras import models, presets

tfk = tf.keras


def test_ute_binary_works_for_exponential_survival_model():
    """predict_ute_binary/predict_ate_binary run end-to-end on an Exponential outcome model.

    Previously hardcoded n_outcome_pred_cols=2 (Normal-only), which silently mis-sliced columns
    for this model's n_outcome_pred_cols=1 (log_rate) and returned a meaningless number with no
    error. The outcome family now comes from the model's StateMixture.
    """
    np.random.seed(10)
    simulator = datasets.CancerSurvivalSimulator(seed=42)
    surv_data = simulator.sample(n_samples=500)
    inputs, outputs = surv_data.to_keras_inputs_outputs()

    tf.random.set_seed(10)
    model = models.build_model(
        **presets.binary_exponential(
            n_states=4,
            n_features=surv_data.n_features,
            compile=True,
            predictive_state_hidden_layers=[(10, "selu")],
            outcome_hidden_layers=[(10, "selu")],
            log_rate_layer=(10, "selu"),
        )
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
    # Seeds Python, NumPy and TF: Keras draws default initializer seeds from Python's
    # `random`, so seeding TF alone makes the fit depend on which tests ran before.
    tf.keras.utils.set_random_seed(13)
    model = models.build_model(
        **presets.toy(n_states=5, n_features=ks_data.features.shape[1], compile=True)
    )
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
    # The mixed outcome mean tracks the observed outcome (moved from the deleted
    # test_utils.py::test_agg_outcome_preds_works, which pinned the raw-parameter mix).
    outcome_mean = np.asarray(model.loss.mixture.outcome_mean(preds))[:, 0]
    # After 20 epochs this correlation varies a lot by seed (0.15 to 0.95 over seeds 10-19
    # for the toy preset, 0.23 to 0.85 for the pre-v0.4.0 toy model); seed 13 gives 0.50.
    assert np.corrcoef(outcome_mean, outputs[:, 0])[0, 1] > 0.45

    ute = inference.predict_ute_binary(model, inputs[0])
    assert ute.shape[0] == ks_data.n_samples

    ate = inference.predict_ate_binary(model, inputs[0])
    print(ate)
    assert ate == pytest.approx(10.0, 2.0)


def test_predict_ate_continuous_weibull_end_to_end():
    """A continuous Weibull model's ATE curve has the true signs and returns an ndarray for
    array features and a Series named "ate" for a DataFrame."""
    from ._continuous import continuous_model, synthetic_continuous_survival

    tf.random.set_seed(0)
    inputs, y_true, true_ate = synthetic_continuous_survival(n_samples=1024)
    model = continuous_model("weibull")
    model.fit(inputs, y_true, epochs=20, batch_size=64, verbose=0)

    grid = [-1.0, 0.0, 1.0]
    ate = inference.predict_ate_continuous(model, inputs[0], treatment_grid=grid)
    assert isinstance(ate, np.ndarray) and ate.shape == (3,)
    want = np.array([true_ate(a, 0.0) for a in grid])
    assert ate[1] == 0.0
    assert np.all(np.sign(ate[[0, 2]]) == np.sign(want[[0, 2]]))
    np.testing.assert_allclose(ate[[0, 2]], want[[0, 2]], rtol=0.5)

    frame = pd.DataFrame(inputs[0], columns=["x0", "x1", "x2"])
    ate_series = inference.predict_ate_continuous(model, frame, treatment_grid=grid)
    assert isinstance(ate_series, pd.Series) and ate_series.name == "ate"
    np.testing.assert_allclose(ate_series.values, ate, rtol=1e-5)
