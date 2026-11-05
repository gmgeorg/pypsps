"""Test module for model functions."""

import numpy as np
import pytest
import tensorflow as tf

from .. import datasets
from ..keras import callbacks, models

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


def test_get_propensity_state_conditional_means():
    """state-conditional propensity means are 0.5 at zero-init and track weight updates"""
    model = models.build_toy_model(n_states=4, n_features=3, compile=False)
    means = models.get_propensity_state_conditional_means(model)
    assert means.shape == (4, 1)
    np.testing.assert_allclose(means.numpy().ravel(), 0.5)

    model.get_layer("propensity_logit_state_2").set_weights([np.array([2.0])])
    means = models.get_propensity_state_conditional_means(model)
    updated = means.numpy().ravel()
    np.testing.assert_allclose(updated[[0, 1, 3]], 0.5)
    np.testing.assert_allclose(updated[2], tf.sigmoid(2.0).numpy(), rtol=1e-5)


def test_build_toy_model_compiles_causal_loss_metric():
    """`causal_loss_metric` must be a compiled metric, so get_default_callbacks' recommended
    monitor ("val_causal_loss_metric") actually exists.

    Previously only build_model_binary_exponential compiled this metric; build_toy_model and
    build_model_binary_normal didn't, which silently disabled EarlyStopping/ReduceLROnPlateau
    for anyone following the documented recommended pattern (Keras only warns, it doesn't
    raise, when a monitored metric isn't found).
    """
    np.random.seed(10)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=50)
    inputs, outputs = ks_data.to_keras_inputs_outputs()
    tf.random.set_seed(10)
    model = models.build_toy_model(n_states=3, n_features=ks_data.n_features, compile=True)
    result = model.evaluate(inputs, outputs, verbose=0, return_dict=True)
    assert "causal_loss_metric" in result


def test_build_model_binary_normal():
    """test build models"""
    np.random.seed(10)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=1000)

    inputs, outputs = ks_data.to_keras_inputs_outputs()
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

    # get_default_callbacks' recommended monitor ("val_causal_loss_metric") must exist.
    result = model.evaluate(inputs, outputs, verbose=0, return_dict=True)
    assert "causal_loss_metric" in result


def test_build_model_binary_exponential():
    """test build models"""
    np.random.seed(10)

    simulator = datasets.CancerSurvivalSimulator(seed=42)
    surv_data = simulator.sample(n_samples=1000)
    inputs, outputs = surv_data.to_keras_inputs_outputs()

    tf.random.set_seed(10)
    model = models.build_model_binary_exponential(
        n_states=3,
        n_features=surv_data.n_features,
        compile=True,
        predictive_state_hidden_layers=[(10, "selu"), (20, "relu")],
        outcome_hidden_layers=[(30, "tanh"), (20, "selu")],
        log_rate_layer=(20, "selu"),
    )

    preds = model.predict(inputs)
    assert preds.shape[0] == surv_data.n_samples
    assert not np.isnan(preds.sum().sum())
    history = model.fit(
        inputs,
        outputs,
        validation_split=0.2,
        verbose=1,
        epochs=1,
        batch_size=32,
        callbacks=callbacks.get_default_callbacks(monitor="val_causal_loss_metric", patience=5),
    )
    assert not np.isnan(history.history["loss"][-1])


@pytest.mark.parametrize(
    "builder",
    [
        models.build_toy_model,
        models.build_model_binary_normal,
        models.build_model_binary_exponential,
    ],
)
def test_binary_builders_log_propensity_score_auroc_and_aupr(builder):
    """The binary builders log the propensity ROC AUC and PR AUC under explicit names."""
    if builder is models.build_model_binary_exponential:
        data = datasets.CancerSurvivalSimulator(seed=42).sample(n_samples=64)
        extra = {"log_rate_layer": (5, "selu")}
    else:
        np.random.seed(10)
        data = datasets.KangSchafer(true_ate=10).sample(n_samples=64)
        extra = (
            {"loc_layer": (5, "selu"), "scale_layer": (5, "tanh")}
            if builder is models.build_model_binary_normal
            else {}
        )
    if builder is not models.build_toy_model:
        extra.update(
            predictive_state_hidden_layers=[(5, "selu")], outcome_hidden_layers=[(5, "selu")]
        )
    inputs, outputs = data.to_keras_inputs_outputs()
    tf.random.set_seed(10)
    model = builder(n_states=2, n_features=data.n_features, compile=True, **extra)

    result = model.evaluate(inputs, outputs, verbose=0, return_dict=True)
    assert "propensity_score_auc" not in result

    compiled = {m.name: m for m in model.metrics}
    # Keras 3 nests compiled metrics in a CompileMetrics container.
    for container in model.metrics:
        compiled.update({m.name: m for m in getattr(container, "metrics", [])})
    for name, curve in [("propensity_score_auroc", "ROC"), ("propensity_score_aupr", "PR")]:
        assert name in result
        assert compiled[name].curve.name == curve
