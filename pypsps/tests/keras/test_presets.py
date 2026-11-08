"""The presets: what each returns, how it validates, and that each model trains."""

import numpy as np
import pypress.keras.regularizers
import pytest
import tensorflow as tf

from ...keras import families, layers, models, neglogliks, presets
from ...keras.layout import ColumnLayout

N, K, F = 16, 3, 4


def _data(seed=0):
    """Inputs and a ``y_true`` whose width fits both binary and continuous models."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, F)).astype("float32")
    t = rng.integers(0, 2, size=(N, 1)).astype("float32")
    time = rng.uniform(0.5, 5.0, size=(N, 1))
    event = rng.integers(0, 2, size=(N, 1))
    return x, t, np.hstack([time, event, t]).astype("float32")


def test_binary_exponential_preset_fits_with_default_callbacks():
    """The binary exponential preset trains on simulated survival data with the default
    callbacks, which monitor its `causal_loss_metric`."""
    from ... import datasets
    from ...keras import callbacks

    data = datasets.CancerSurvivalSimulator(seed=42).sample(n_samples=300)
    inputs, outputs = data.to_keras_inputs_outputs()
    tf.keras.utils.set_random_seed(10)
    model = models.build_model(
        **presets.binary_exponential(
            n_states=3,
            n_features=data.n_features,
            predictive_state_hidden_layers=[(10, "selu"), (20, "relu")],
            outcome_hidden_layers=[(30, "tanh"), (20, "selu")],
            log_rate_layer=(20, "selu"),
        )
    )
    history = model.fit(
        inputs,
        outputs,
        validation_split=0.2,
        epochs=2,
        batch_size=32,
        verbose=0,
        callbacks=callbacks.get_default_callbacks(monitor="val_causal_loss_metric", patience=5),
    )
    assert np.isfinite(history.history["loss"]).all()


_HIDDEN = dict(predictive_state_hidden_layers=[(4, "relu")], outcome_hidden_layers=[(4, "relu")])


def _continuous(distribution="weibull", **kwargs):
    """`presets.continuous_survival` with small layers and the test bounds."""
    if distribution == "weibull":
        kwargs.setdefault("log_shape_bounds", (-2.0, 2.0))
    return presets.continuous_survival(
        distribution, n_states=K, n_features=F, log_scale_bounds=(-3.0, 6.0), **_HIDDEN, **kwargs
    )


def test_weibull_overflow_bounds_match_the_keras_nll():
    """The bounds the preset rejects (``tests/core/test_presets.py``) are exactly those whose
    worst corner overflows the Keras Weibull NLL, or lies within the 50-exponent margin."""
    unsafe = dict(log_scale_bounds=(-10.0, 5.0), log_shape_bounds=(-1.0, 3.0))  # exponent 293
    margin = dict(log_scale_bounds=(-8.0, 5.0), log_shape_bounds=(-1.0, 1.5))  # exponent 56
    safe = dict(log_scale_bounds=(-3.0, 5.0), log_shape_bounds=(-1.0, 1.0))  # exponent 21
    max_time = 100.0
    nll = neglogliks.NegloglikWeibull(reduction="none")
    for bounds, overflows in ((unsafe, True), (margin, False), (safe, False)):
        worst = [[bounds["log_scale_bounds"][0], bounds["log_shape_bounds"][1]]]
        value = nll(tf.constant([[max_time, 1.0]]), tf.constant(worst)).numpy()
        assert np.isfinite(value).all() != overflows
    kwargs = dict(n_states=2, n_features=F, max_time=max_time, compile=False, **_HIDDEN)
    for rejected in (unsafe, margin):
        with pytest.raises(ValueError, match="overflow"):
            models.build_model(**presets.continuous_weibull(**rejected, **kwargs))
    models.build_model(**presets.continuous_weibull(**safe, **kwargs))


def test_presets_create_no_layers():
    """A preset is data (``tests/core/test_presets.py``): no Keras layer is created."""
    before = tf.keras.backend.get_uid("dense")
    _continuous()
    assert tf.keras.backend.get_uid("dense") == before + 1


def test_compile_false_leaves_out_the_compile_keywords():
    """With compile=False the dict has no compile keywords; merging compile=False raises."""
    compiled = _continuous(alpha=2.0)
    assert compiled["alpha"] == 2.0 and "compile" not in compiled
    uncompiled = _continuous(compile=False, alpha=2.0)
    assert uncompiled["compile"] is False and "alpha" not in uncompiled
    assert getattr(models.build_model(**uncompiled), "optimizer", None) is None
    with pytest.raises(TypeError, match="compile=False"):
        models.build_model(**(compiled | {"compile": False}))


def test_continuous_preset_uses_the_keras_families():
    """The preset builds its families from the Keras backend (the argument mapping itself is
    tested in ``tests/core/test_presets.py``)."""
    arguments = _continuous()
    assert isinstance(arguments["outcome_family"], families.Weibull)
    assert isinstance(arguments["treatment_family"], families.Normal)


def test_flat_treatment_init_values_build():
    """A flat ``(loc, scale)`` crashed the old builder (pypress wants (2, K)); now it builds."""
    model = models.build_model(**_continuous(treatment_init_values=[0.5, 2.0], compile=False))
    x, t, _ = _data()
    block = ColumnLayout.from_model(model).decode_pred(model([x, t], training=False)).treatment
    np.testing.assert_allclose(block["loc"], 0.5, rtol=1e-6)
    np.testing.assert_allclose(block["scale"], 2.0, rtol=1e-5)


def test_log_shape_bounds_excluding_zero_build():
    """Shape bounds that exclude 0 crashed the old builder; the shape now starts mid-bounds."""
    model = models.build_model(**_continuous(log_shape_bounds=(0.2, 1.0), compile=False))
    want = layers.SoftClip(0.2, 1.0).inverse(0.6)
    np.testing.assert_allclose(model.get_layer("log_shape_raw_state_0").bias.numpy(), [want])


@pytest.mark.parametrize(
    "preset,kwargs",
    [
        ("toy", {}),
        ("binary_normal", dict(_HIDDEN)),  # loc_layer=None crashed the old builder
        ("binary_normal", dict(_HIDDEN, loc_layer=(3, "selu"), scale_layer=(3, "tanh"))),
        ("binary_exponential", dict(_HIDDEN, outcome_loss_weight=0.5)),
        ("binary_exponential", dict(predictive_state_hidden_layers=[], outcome_hidden_layers=[])),
    ],
)
def test_binary_presets_build_and_train(preset, kwargs):
    """The binary presets build with every head option and train to a finite loss."""
    arguments = getattr(presets, preset)(K, F, **kwargs)
    assert arguments["architecture"] == "shared_trunk"
    assert arguments["treatment_embedding_units"] is None
    assert arguments["clipnorm"] is None
    model = models.build_model(**arguments)
    x, t, y_true = _data()
    if preset != "binary_exponential":
        y_true = np.hstack([x[:, :1], t])
    history = model.fit([x, t], y_true, epochs=1, verbose=0)
    assert np.isfinite(history.history["loss"]).all()


def test_scheduled_regularizer_values_advance_during_fit():
    """A pypress `ScheduledValue` in the loss's state regularizer follows the training clock
    that `RegularizerScheduler` advances each epoch; after `fit` the clock stops."""
    from pypress.keras import callbacks as press_callbacks
    from pypress.keras import schedules

    l2 = schedules.ScheduledValue(start=0.0, end=1.0, duration_epochs=3)
    reg = pypress.keras.regularizers.DegreesOfFreedom(l2=l2, target=2.0)
    model = models.build_model(**(presets.toy(K, F) | {"predictive_states_regularizer": reg}))
    x, t, _ = _data()
    y = np.hstack([x[:, :1], t])  # [outcome, treatment]
    seen = []
    record = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: seen.append(float(l2.value()))
    )
    model.fit(
        [x, t],
        y,
        epochs=5,
        batch_size=32,
        verbose=0,
        callbacks=[press_callbacks.RegularizerScheduler(), record],
    )
    np.testing.assert_allclose(seen, [0.0, 1 / 3, 2 / 3, 1.0, 1.0], rtol=1e-6)
    assert schedules.current_epoch() is None


def test_binary_preset_regularizers():
    """Toy and binary normal penalize the degrees of freedom; binary exponential is uniform."""
    df = presets.binary_normal(10, F, [], [], df_penalty_l2=0.3)["predictive_states_regularizer"]
    assert isinstance(df, pypress.keras.regularizers.DegreesOfFreedom)
    uniform = presets.binary_exponential(K, F, [], [])["predictive_states_regularizer"]
    assert isinstance(uniform, pypress.keras.regularizers.Uniform)
    toy = presets.toy(K, F, compile=False)
    assert "predictive_states_regularizer" not in toy
    assert toy["outcome_heads"]["scale"].constant
