"""Presets, and their equivalence with the frozen v0.3.5 reference builders.

The new side is always `build_model(**presets.<name>(...))`. Continuous models take the old
weights with `set_weights` and match bit for bit. Binary models take them through
`models.copy_legacy_weights`; their treatment block differs from the old one by float32
rounding, so the loss and gradients get a tolerance.
"""

import warnings

import numpy as np
import pypress.keras.regularizers
import pytest
import tensorflow as tf

from ..keras import families, layers, models, neglogliks, presets
from ..keras.layout import ColumnLayout
from . import _reference_builders as reference

N, K, F = 16, 3, 4

_BINARY = dict(n_states=K, n_features=F, predictive_state_hidden_layers=[(6, "relu")])
_CONTINUOUS = dict(
    n_states=K,
    n_features=F,
    predictive_state_hidden_layers=[(6, "relu")],
    outcome_hidden_layers=[(6, "relu")],
    log_scale_bounds=(-2.0, 4.0),
)

CASES = {
    "binary_normal-constant_scale-depth1": (
        "build_model_binary_normal",
        dict(_BINARY, outcome_hidden_layers=[(6, "tanh")], loc_layer=(5, "selu")),
    ),
    "binary_normal-scale_layer-depth2": (
        "build_model_binary_normal",
        dict(
            _BINARY,
            predictive_state_hidden_layers=[(6, "relu"), (5, "selu")],
            outcome_hidden_layers=[(6, "tanh"), (4, "relu")],
            loc_layer=(5, "selu"),
            scale_layer=(10, "tanh"),
        ),
    ),
    "binary_exponential": (
        "build_model_binary_exponential",
        dict(_BINARY, outcome_hidden_layers=[(6, "tanh")], log_rate_layer=(5, "selu")),
    ),
    "continuous_exponential": ("build_model_continuous_exponential", dict(_CONTINUOUS)),
    "continuous_weibull-shared_shape-init": (
        "build_model_continuous_weibull",
        dict(
            _CONTINUOUS,
            log_shape_bounds=(-1.0, 1.0),
            separate_shape_per_state=False,
            init_scale=5.0,
            treatment_init_values=[[0.0] * K, [1.0] * K],  # pypress: (n_params, n_states)
        ),
    ),
    "continuous_weibull-per_state_shape": (
        "build_model_continuous_weibull",
        dict(_CONTINUOUS, log_shape_bounds=(-1.0, 1.0)),
    ),
}


# Every deprecated builder and its preset.
_PRESETS = {
    "build_toy_model": "toy",
    "build_model_binary_normal": "binary_normal",
    "build_model_binary_exponential": "binary_exponential",
    "build_model_continuous_exponential": "continuous_exponential",
    "build_model_continuous_weibull": "continuous_weibull",
}


def _new(name, kwargs):
    """The new side: the builder's preset through `build_model`."""
    return models.build_model(**getattr(presets, _PRESETS[name])(**kwargs))


def _is_binary(case):
    """Whether a case is a binary-treatment builder."""
    return "binary" in CASES[case][0]


def _data(seed=0):
    """Inputs and a ``y_true`` whose width fits both binary and continuous models."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, F)).astype("float32")
    t = rng.integers(0, 2, size=(N, 1)).astype("float32")
    time = rng.uniform(0.5, 5.0, size=(N, 1))
    event = rng.integers(0, 2, size=(N, 1))
    return x, t, np.hstack([time, event, t]).astype("float32")


def _randomize(model, seed=1):
    """Random weights, so a swap of two same-shape layers changes the predictions."""
    rng = np.random.default_rng(seed)
    values = []
    for w in model.weights:
        value = 0.3 * rng.normal(size=w.shape)
        if w.path.endswith("moving_variance"):
            value = np.abs(value) + 0.5  # a negative variance makes every output NaN
        values.append(value.astype("float32"))
    model.set_weights(values)


def _loss_and_grads(model, x, t, y_true):
    """The compiled loss at ``training=False`` and its gradients."""
    with tf.GradientTape() as tape:
        loss = model.loss(_y_true_for(model, y_true), model([x, t], training=False))
    return loss, tape.gradient(loss, model.trainable_weights)


def _y_true_for(model, y_true):
    """``[outcome, treatment]`` for binary models, ``[time, event, treatment]`` otherwise."""
    if model.loss.layout.n_outcome_true_cols == 1:
        return y_true[:, 1:]  # event as a 0/1 "outcome" is fine for the equivalence check
    return y_true


@pytest.mark.parametrize("case", sorted(c for c in CASES if not _is_binary(c)))
def test_continuous_equivalence(case, tmp_path):
    """Same predictions, loss and gradients under a weight transfer; old weight files load."""
    name, kwargs = CASES[case]
    old = getattr(reference, name)(**kwargs)
    new = _new(name, kwargs)
    _randomize(old)
    new.set_weights(old.get_weights())
    x, t, y_true = _data()

    expected = old([x, t], training=False)
    assert np.isfinite(expected).all()
    np.testing.assert_array_equal(new([x, t], training=False), expected)
    old_loss, old_grads = _loss_and_grads(old, x, t, y_true)
    new_loss, new_grads = _loss_and_grads(new, x, t, y_true)
    np.testing.assert_array_equal(new_loss, old_loss)
    for g_new, g_old in zip(new_grads, old_grads):
        np.testing.assert_array_equal(g_new, g_old)

    path = str(tmp_path / "old.weights.h5")
    old.save_weights(path)
    loaded = _new(name, kwargs)
    loaded.load_weights(path)
    np.testing.assert_array_equal(loaded([x, t], training=False), old([x, t], training=False))


def _grads_by_layer(model, x, t, y_true):
    """The loss, and its gradient per layer name (one list per layer)."""
    loss, grads = _loss_and_grads(model, x, t, y_true)
    by_layer = {}
    for v, g in zip(model.trainable_weights, grads):
        by_layer.setdefault(v.path.split("/")[0], []).append(g.numpy())
    return loss, by_layer


@pytest.mark.parametrize("case", sorted(c for c in CASES if _is_binary(c)))
def test_binary_equivalence_through_copy_legacy_weights(case, tmp_path):
    """After `copy_legacy_weights` the outcome and state columns are bit-identical, the
    propensity agrees to float32 rounding, and so do the loss and every gradient. The old
    weight file does not load into the new model."""
    name, kwargs = CASES[case]
    old = getattr(reference, name)(**kwargs)
    new = _new(name, kwargs)
    _randomize(old)
    models.copy_legacy_weights(old, new)
    x, t, y_true = _data()

    layout = ColumnLayout.from_model(new)
    want = layout.decode_pred(old([x, t], training=False))
    got = layout.decode_pred(new([x, t], training=False))
    for p in want.outcome:
        np.testing.assert_array_equal(got.outcome[p], want.outcome[p])
    np.testing.assert_array_equal(got.state_weights, want.state_weights)
    np.testing.assert_allclose(got.treatment["prob"], want.treatment["prob"], rtol=1e-6, atol=0)

    old_loss, old_grads = _grads_by_layer(old, x, t, y_true)
    new_loss, new_grads = _grads_by_layer(new, x, t, y_true)
    np.testing.assert_allclose(new_loss, old_loss, rtol=1e-6)
    for a, b in models._legacy_pairs(old, new):
        for g_old, g_new in zip(old_grads[a.name], new_grads[b.name]):
            np.testing.assert_allclose(g_new, g_old, rtol=1e-5, atol=1e-7, err_msg=b.name)
    propensity = np.ravel([old_grads[f"propensity_logit_state_{k}"] for k in range(K)])
    np.testing.assert_allclose(
        np.ravel(new_grads["treatment_params"]), propensity, rtol=1e-5, atol=1e-7
    )

    path = str(tmp_path / "old.weights.h5")
    old.save_weights(path)
    with pytest.raises(ValueError):
        _new(name, kwargs).load_weights(path)


def test_copy_legacy_weights_rejects_what_does_not_map():
    """A toy model (its state network changed) and mismatched models raise ValueError."""
    with pytest.raises(ValueError, match="toy"):
        models.copy_legacy_weights(
            reference.build_toy_model(K, F),
            _new("build_toy_model", dict(n_states=K, n_features=F)),
        )
    name, kwargs = CASES["binary_normal-constant_scale-depth1"]
    other = dict(kwargs, outcome_hidden_layers=[(7, "tanh")])
    with pytest.raises(ValueError, match="does not map"):
        models.copy_legacy_weights(getattr(reference, name)(**kwargs), _new(name, other))
    exp_name, exp_kwargs = CASES["binary_exponential"]
    with pytest.raises(ValueError, match="families"):
        models.copy_legacy_weights(getattr(reference, name)(**kwargs), _new(exp_name, exp_kwargs))
    name, kwargs = CASES["continuous_exponential"]
    with pytest.raises(ValueError, match="different weights"):
        models.copy_legacy_weights(
            getattr(reference, name)(**kwargs), _new(name, dict(kwargs, n_states=2))
        )


# --- deprecated builders ---

_DEPRECATED_CASES = {"toy": ("build_toy_model", dict(n_states=K, n_features=F)), **CASES}


@pytest.mark.parametrize("case", sorted(_DEPRECATED_CASES))
def test_deprecated_builders_warn_once_and_match_their_preset(case):
    """One FutureWarning, pointing at the caller and naming the preset; the same model."""
    name, kwargs = _DEPRECATED_CASES[case]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        old = getattr(models, name)(**kwargs)
    future = [w for w in record if issubclass(w.category, FutureWarning)]
    assert len(future) == 1
    assert f"presets.{_PRESETS[name]}(...)" in str(future[0].message)
    assert "v0.5.0" in str(future[0].message)
    assert future[0].filename == __file__
    new = _new(name, kwargs)
    _randomize(old)
    new.set_weights(old.get_weights())
    x, t, _ = _data()
    np.testing.assert_array_equal(new([x, t], training=False), old([x, t], training=False))


def test_deprecated_continuous_survival_builder_warns():
    """`build_model_continuous_survival` forwards every argument to its preset."""
    with pytest.warns(FutureWarning, match=r"presets\.continuous_survival"):
        model = models.build_model_continuous_survival(
            "exponential",
            2,
            F,
            [(4, "relu")],
            [(4, "relu")],
            (-2.0, 4.0),
            state_hidden_layer=(3, "tanh"),
            compile=False,
        )
    assert model.get_layer("log_scale_hidden_state_0").units == 3
    assert getattr(model, "optimizer", None) is None


def test_binary_exponential_preset_fits_with_default_callbacks():
    """The binary exponential preset trains on simulated survival data with the default
    callbacks, which monitor its `causal_loss_metric`."""
    from .. import datasets
    from ..keras import callbacks

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


# --- the presets ---

_HIDDEN = dict(predictive_state_hidden_layers=[(4, "relu")], outcome_hidden_layers=[(4, "relu")])


def _continuous(distribution="weibull", **kwargs):
    """`presets.continuous_survival` with small layers and the test bounds."""
    if distribution == "weibull":
        kwargs.setdefault("log_shape_bounds", (-2.0, 2.0))
    return presets.continuous_survival(
        distribution, n_states=K, n_features=F, log_scale_bounds=(-3.0, 6.0), **_HIDDEN, **kwargs
    )


def test_continuous_preset_validates_its_arguments():
    """Bounds are required and checked; shape bounds belong to Weibull only."""
    with pytest.raises(ValueError, match="distribution"):
        _continuous("normal")
    with pytest.raises(ValueError, match="log_shape_bounds"):
        _continuous("weibull", log_shape_bounds=None)
    with pytest.raises(ValueError, match="weibull"):
        _continuous("exponential", log_shape_bounds=(-2.0, 2.0))
    with pytest.raises(ValueError, match="log_shape_bounds"):
        _continuous("weibull", log_shape_bounds=(1.0, -1.0))
    with pytest.raises(ValueError, match="log_scale_bounds"):
        presets.continuous_exponential(K, F, log_scale_bounds=None, **_HIDDEN)
    with pytest.raises(ValueError, match="init_scale"):
        _continuous("weibull", init_scale=0.0)
    with pytest.raises(ValueError, match="treatment_init_values"):
        _continuous("weibull", treatment_init_values=[0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="lie in"):
        models.build_model(**_continuous("weibull", init_scale=np.exp(7.0)))


def test_continuous_preset_max_time_rejects_overflow_with_a_margin():
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
    kwargs = dict(n_states=2, n_features=F, max_time=max_time, compile=False, **_HIDDEN)
    for rejected in (unsafe, margin):
        with pytest.raises(ValueError, match="overflow"):
            models.build_model(**presets.continuous_weibull(**rejected, **kwargs))
    models.build_model(**presets.continuous_weibull(**safe, **kwargs))


def test_presets_return_fresh_dicts_and_create_no_layers():
    """A preset is data: a new dict per call, and no Keras layer is created."""
    before = tf.keras.backend.get_uid("dense")
    first, second = _continuous(), _continuous()
    assert first is not second and first.keys() == second.keys()
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


def test_continuous_preset_maps_its_arguments():
    """init_scale, treatment_init_values and separate_shape_per_state reach build_model."""
    arguments = _continuous(
        init_scale=5.0, treatment_init_values=[0.5, 2.0], separate_shape_per_state=False
    )
    assert arguments["outcome_init"] == {"log_scale": pytest.approx(np.log(5.0))}
    assert arguments["treatment_init"] == {"loc": 0.5, "scale": 2.0}
    assert arguments["outcome_heads"]["log_shape"].per_state is False
    assert arguments["architecture"] == "gated"
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


def test_binary_preset_regularizers():
    """Toy and binary normal penalize the degrees of freedom; binary exponential is uniform."""
    df = presets.binary_normal(10, F, [], [], df_penalty_l2=0.3)["predictive_states_regularizer"]
    assert isinstance(df, pypress.keras.regularizers.DegreesOfFreedom)
    uniform = presets.binary_exponential(K, F, [], [])["predictive_states_regularizer"]
    assert isinstance(uniform, pypress.keras.regularizers.Uniform)
    toy = presets.toy(K, F, compile=False)
    assert "predictive_states_regularizer" not in toy
    assert toy["outcome_heads"]["scale"].constant
