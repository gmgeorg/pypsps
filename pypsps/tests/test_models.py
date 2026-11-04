"""Test module for model functions."""

import numpy as np
import pytest
import tensorflow as tf

from .. import datasets, utils
from ..keras import callbacks, layers, losses, models, neglogliks

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


_LOG_SCALE_BOUNDS = (-3.0, 6.0)
_LOG_SHAPE_BOUNDS = (-2.0, 2.0)


def _synthetic_continuous_survival(n_samples=512, seed=0):
    """Continuous treatment confounded by x0; Weibull times with scale exp(1 + 0.3 a + 0.2 x1).

    Returns (inputs, y_true, true ATE function); y_true is [time, event, treatment].
    """
    rng = np.random.RandomState(seed)
    features = rng.normal(size=(n_samples, 3)).astype("float32")
    treatment = (0.5 * features[:, 0] + rng.normal(size=n_samples)).astype("float32")
    scale = np.exp(1.0 + 0.3 * treatment + 0.2 * features[:, 1])
    event_time = scale * rng.weibull(1.5, size=n_samples)
    censor_time = rng.exponential(3.0 * scale.mean(), size=n_samples)
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype("float32")
    y_true = np.column_stack([time, event, treatment]).astype("float32")

    # E[T | a] = E[exp(0.2 x1)] * exp(1 + 0.3 a) * Gamma(1 + 1 / 1.5).
    mean_at = lambda a: np.exp(0.02 + 1.0 + 0.3 * a) * 0.9027452929509336  # noqa: E731

    return [features, treatment[:, None]], y_true, lambda a, base: mean_at(a) - mean_at(base)


def _continuous_model(distribution="weibull", **kwargs):
    """Small continuous survival model with the test bounds."""
    if distribution == "weibull":
        kwargs.setdefault("log_shape_bounds", _LOG_SHAPE_BOUNDS)
    return models.build_model_continuous_survival(
        distribution,
        n_states=3,
        n_features=3,
        predictive_state_hidden_layers=[(16, "relu")],
        outcome_hidden_layers=[(16, "relu")],
        log_scale_bounds=_LOG_SCALE_BOUNDS,
        **kwargs,
    )


@pytest.mark.parametrize("distribution, n_outcome_pred_cols", [("exponential", 1), ("weibull", 2)])
def test_continuous_survival_builder_layout_and_bounds(distribution, n_outcome_pred_cols):
    """Output is [outcome params | states | treatment (loc, scale)], params inside the bounds."""
    tf.random.set_seed(0)
    model = _continuous_model(distribution)
    layout = utils.get_column_layout(model)
    assert (layout.n_outcome_pred_cols, layout.n_treatment_pred_cols) == (n_outcome_pred_cols, 2)
    assert model.output_shape == (None, 3 * (n_outcome_pred_cols + 1 + 2))

    rng = np.random.RandomState(0)
    probe = [100.0 * rng.normal(size=(64, 3)), 100.0 * rng.normal(size=(64, 1))]
    preds = model.predict(probe, verbose=0)
    outcome_params, weights, _ = utils.split_y_pred(preds, n_outcome_pred_cols, 2)
    bounds = [_LOG_SCALE_BOUNDS, _LOG_SHAPE_BOUNDS][:n_outcome_pred_cols]
    for block, (low, high) in zip(
        utils.split_outcome_pred(outcome_params, n_outcome_pred_cols), bounds
    ):
        assert np.all(block >= low) and np.all(block <= high)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=1e-5)


def test_continuous_survival_builder_validates_its_arguments():
    """Bounds are required and checked; shape bounds belong to Weibull only."""
    with pytest.raises(ValueError, match="distribution"):
        _continuous_model("normal")
    with pytest.raises(ValueError, match="log_shape_bounds"):
        _continuous_model("weibull", log_shape_bounds=None)
    with pytest.raises(ValueError, match="weibull"):
        _continuous_model("exponential", log_shape_bounds=_LOG_SHAPE_BOUNDS)
    with pytest.raises(ValueError, match="log_shape_bounds"):
        _continuous_model("weibull", log_shape_bounds=(1.0, -1.0))
    with pytest.raises(ValueError, match="lie in"):
        _continuous_model("weibull", init_scale=np.exp(7.0))


def test_continuous_survival_max_time_check_rejects_overflow_with_a_margin():
    """max_time rejects bounds whose worst corner overflows the Weibull NLL, and also bounds
    within the margin (exponent above 50 but still finite), and accepts bounds well inside."""
    unsafe = dict(log_scale_bounds=(-10.0, 5.0), log_shape_bounds=(-1.0, 3.0))  # exponent 293
    margin = dict(log_scale_bounds=(-8.0, 5.0), log_shape_bounds=(-1.0, 1.5))  # exponent 56
    safe = dict(log_scale_bounds=(-3.0, 5.0), log_shape_bounds=(-1.0, 1.0))  # exponent 21
    max_time = 100.0
    nll = neglogliks.NegloglikWeibull(reduction="none")
    for bounds, overflows in ((unsafe, True), (margin, False), (safe, False)):
        worst = [[bounds["log_scale_bounds"][0], bounds["log_shape_bounds"][1]]]
        value = nll(tf.constant([[max_time, 1.0]]), tf.constant(worst)).numpy()
        assert np.isfinite(value).all() != overflows
    kwargs = dict(
        n_states=2,
        n_features=3,
        predictive_state_hidden_layers=[(4, "relu")],
        outcome_hidden_layers=[(4, "relu")],
        max_time=max_time,
        compile=False,
    )
    for rejected in (unsafe, margin):
        with pytest.raises(ValueError, match="overflow"):
            models.build_model_continuous_weibull(**rejected, **kwargs)
    models.build_model_continuous_weibull(**safe, **kwargs)


def test_continuous_weibull_scale_and_shape_share_no_weights():
    """No weight receives gradient from both the log-scale and the log-shape outputs except the
    shared input layers (feature and treatment normalization, treatment embedding, states)."""
    tf.random.set_seed(0)
    model = _continuous_model("weibull", compile=False)
    inputs, _, _ = _synthetic_continuous_survival(n_samples=32)
    with tf.GradientTape(persistent=True) as tape:
        preds = model(inputs, training=False)
        log_scale_sum = tf.reduce_sum(preds[:, :3])
        log_shape_sum = tf.reduce_sum(preds[:, 3:6])
    variables = model.trainable_variables
    grads_scale = tape.gradient(log_scale_sum, variables)
    grads_shape = tape.gradient(log_shape_sum, variables)
    used_by_both = {
        v.path.split("/")[0]
        for v, g_scale, g_shape in zip(variables, grads_scale, grads_shape)
        if g_scale is not None
        and g_shape is not None
        and np.any(g_scale.numpy() != 0)
        and np.any(g_shape.numpy() != 0)
    }
    assert used_by_both
    assert not any(name.startswith(("log_scale", "log_shape")) for name in used_by_both)
    scale_layers = {layer.name for layer in model.layers if layer.name.startswith("log_scale")}
    shape_layers = {layer.name for layer in model.layers if layer.name.startswith("log_shape")}
    assert "log_scale_trunk_dense_0" in scale_layers and "log_shape_trunk_dense_0" in shape_layers


@pytest.mark.parametrize("per_state", [True, False])
def test_continuous_weibull_separate_shape_per_state(per_state):
    """With separate_shape_per_state=False every state predicts the same log shape."""
    tf.random.set_seed(0)
    model = _continuous_model("weibull", separate_shape_per_state=per_state, compile=False)
    inputs, _, _ = _synthetic_continuous_survival(n_samples=32)
    log_shape = model.predict(inputs, verbose=0)[:, 3:6]
    all_equal = np.allclose(log_shape, log_shape[:, :1])
    assert all_equal != per_state


def test_continuous_survival_init_scale_sets_the_head_bias():
    """init_scale sets every state's raw log-scale bias to the soft-clip inverse of its log."""
    model = _continuous_model("exponential", init_scale=5.0, compile=False)
    want = layers.SoftClip(*_LOG_SCALE_BOUNDS).inverse(np.log(5.0))
    for state_id in range(3):
        bias = model.get_layer(f"log_scale_raw_state_{state_id}").bias.numpy()
        np.testing.assert_allclose(bias, [want], rtol=1e-6)


def test_continuous_survival_compiles_outcome_likelihood_and_shared_alpha():
    """The loss uses the requested outcome likelihood and the metrics share its alpha."""
    model = _continuous_model("weibull")
    assert model.loss._outcome_loss._outcome_likelihood is losses.OutcomeLikelihood.INTERVENTIONAL
    observational = _continuous_model("weibull", outcome_likelihood="observational")
    assert observational.loss._outcome_loss._outcome_likelihood is (
        losses.OutcomeLikelihood.OBSERVATIONAL
    )
    inputs, y_true, _ = _synthetic_continuous_survival(n_samples=64)
    before = model.evaluate(inputs, y_true, verbose=0, return_dict=True)
    model.loss._alpha.assign(5.0)
    after = model.evaluate(inputs, y_true, verbose=0, return_dict=True)
    assert after["causal_loss_metric"] > before["causal_loss_metric"]
    np.testing.assert_allclose(after["outcome_nll_metric"], before["outcome_nll_metric"], rtol=1e-6)


@pytest.mark.parametrize("distribution", ["exponential", "weibull"])
def test_continuous_survival_trains_stably_and_survives_save_load(distribution, tmp_path):
    """Training keeps every loss finite and lowers it; the model reloads without custom code."""
    tf.random.set_seed(0)
    inputs, y_true, _ = _synthetic_continuous_survival()
    model = _continuous_model(distribution, max_time=float(y_true[:, 0].max()))
    history = model.fit(inputs, y_true, epochs=10, batch_size=64, verbose=0)
    losses_seen = np.array(history.history["loss"])
    assert np.all(np.isfinite(losses_seen))
    assert losses_seen[-1] < losses_seen[0]

    path = str(tmp_path / "model.keras")
    model.save(path)
    reloaded = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(
        reloaded.predict(inputs, verbose=0), model.predict(inputs, verbose=0), rtol=1e-5
    )
