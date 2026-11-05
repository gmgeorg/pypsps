"""Tests for `models.build_model`, `models.compile_model` and `models.default_metrics`."""

import functools

import numpy as np
import pypress.keras.regularizers
import pytest
import tensorflow as tf

from .. import inference
from ..keras import callbacks, families, layers, losses, mixture, models, presets
from ..keras.layout import ColumnLayout

N, F = 128, 4
_LS_BOUNDS, _SHAPE_BOUNDS = (-2.0, 4.0), (-1.0, 1.0)

TREATMENTS = {"bernoulli": families.Bernoulli, "normal": families.Normal}
OUTCOMES = {
    "normal": families.Normal,
    "exp_log_rate": lambda: families.Exponential("log_rate"),
    "exp_log_scale": lambda: families.Exponential("log_scale", bounds=_LS_BOUNDS),
    "weibull": lambda: families.Weibull(_LS_BOUNDS, _SHAPE_BOUNDS),
}
ARCHITECTURES = ("shared_trunk", "gated")
COMBOS = [(t, o, a, 3) for t in TREATMENTS for o in OUTCOMES for a in ARCHITECTURES] + [
    ("bernoulli", "normal", a, 1) for a in ARCHITECTURES
]
_IDS = [f"{t}-{o}-{a}-K{k}" for t, o, a, k in COMBOS]
_SMALL = dict(predictive_state_hidden_layers=[(8, "relu")], outcome_hidden_layers=[(8, "relu")])


def _data(treatment: str, outcome: str, n: int = N, seed: int = 0):
    """``([x, t], y_true)``: Bernoulli treatment logistic in x0, Normal treatment 0.5 x0 + noise;
    a Normal outcome, or censored survival times."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, F))
    if treatment == "bernoulli":
        t = (rng.random(n) < 1 / (1 + np.exp(-x[:, 0]))).astype(float)
    else:
        t = 0.5 * x[:, 0] + rng.normal(scale=0.5, size=n)
    t = t[:, None]
    if outcome == "normal":
        y = x[:, 1:2] + t + rng.normal(scale=0.5, size=(n, 1))
        y_true = np.hstack([y, t])
    else:
        time = rng.exponential(np.exp(0.5 * x[:, 1:2] + 0.3 * t))
        censor = rng.exponential(2.0, size=(n, 1))
        y_true = np.hstack([np.minimum(time, censor), (time <= censor).astype(float), t])
    return [x.astype("float32"), t.astype("float32")], y_true.astype("float32")


def _build(treatment, outcome, architecture="gated", n_states=3, **kwargs):
    """A model from the family names, with small hidden layers."""
    return models.build_model(
        n_states,
        F,
        OUTCOMES[outcome](),
        TREATMENTS[treatment](),
        architecture=architecture,
        **{**_SMALL, **kwargs},
    )


@functools.lru_cache(maxsize=None)
def _trained(combo):
    """A model per combination, fit for 5 epochs once and shared by the grid tests."""
    treatment, outcome, architecture, n_states = combo
    tf.keras.utils.set_random_seed(0)
    model = _build(treatment, outcome, architecture, n_states)
    inputs, y_true = _data(treatment, outcome)
    history = model.fit(inputs, y_true, epochs=5, batch_size=32, verbose=0)
    return model, history.history, inputs, y_true


def _treatment_block(model, inputs):
    """The decoded treatment parameters, ``{name: [N, K]}``."""
    layout = ColumnLayout.from_model(model)
    return layout.decode_pred(model(inputs, training=False)).treatment


# --- the family x architecture grid ---


@pytest.mark.parametrize("combo", COMBOS, ids=_IDS)
def test_output_matches_layout_and_bounds(combo):
    """The output width matches the layout; weights sum to 1; params stay in range."""
    model, _, inputs, _ = _trained(combo)
    mix = mixture.StateMixture.from_model(model)
    assert model.output_shape[-1] == mix.layout.y_pred_width()
    parts = mix.layout.decode_pred(model([inputs[0] * 100, inputs[1] * 100], training=False))
    np.testing.assert_allclose(np.sum(parts.state_weights, axis=1), 1.0, rtol=1e-5)
    for role, params in (("outcome", parts.outcome), ("treatment", parts.treatment)):
        family = mix.outcome_family if role == "outcome" else mix.treatment_family
        for p in family.params:
            values = np.asarray(params[p.name])
            assert np.isfinite(values).all(), (role, p.name)
            if p.bounds is not None:
                assert (values >= p.bounds[0]).all() and (values <= p.bounds[1]).all()
            elif p.activation == "softplus":
                assert (values >= 0).all()  # float32 softplus underflows to 0 on x100 inputs
            elif p.activation == "sigmoid":
                assert ((values >= 0) & (values <= 1)).all()


@pytest.mark.parametrize("combo", COMBOS, ids=_IDS)
def test_treatment_block_is_feature_independent(combo):
    """Treatment params are the same for every unit and any input."""
    model, _, inputs, _ = _trained(combo)
    other = [np.flip(inputs[0], axis=0) * 3.0, np.flip(inputs[1], axis=0)]
    first, second = _treatment_block(model, inputs), _treatment_block(model, other)
    for name in first:
        block = np.asarray(first[name])
        np.testing.assert_array_equal(block, np.asarray(second[name]))
        np.testing.assert_array_equal(block, np.broadcast_to(block[:1], block.shape))


@pytest.mark.parametrize("combo", COMBOS, ids=_IDS)
def test_trains_with_finite_decreasing_loss(combo):
    """Five epochs give a finite loss that ends below where it started."""
    _, history, _, _ = _trained(combo)
    assert np.isfinite(history["loss"]).all()
    assert history["loss"][-1] < history["loss"][0]


@pytest.mark.parametrize("combo", COMBOS, ids=_IDS)
def test_default_metric_keys_by_family(combo):
    """The default metrics depend on the families as documented."""
    model, _, inputs, y_true = _trained(combo)
    keys = set(model.evaluate(inputs, y_true, verbose=0, return_dict=True))
    expected = {"loss", "causal_loss_metric", "outcome_nll_metric", "predictive_state_df"}
    expected |= {"uniform_entropy", "outcome_mean_squared_error"}
    if combo[0] == "bernoulli":
        expected |= {"propensity_score_binary_crossentropy", "propensity_score_auroc"}
        expected |= {"propensity_score_aupr"}
    else:
        expected |= {"treatment_mean_squared_error"}
    assert keys == expected


@pytest.mark.parametrize("combo", COMBOS, ids=_IDS)
def test_inference_runs(combo):
    """UTE (binary) or ATE (continuous) inference runs and is finite."""
    model, _, inputs, _ = _trained(combo)
    x = inputs[0][:20]
    if combo[0] == "bernoulli":
        ute = np.asarray(inference.predict_ute_binary(model, x))
        assert ute.shape[0] == 20
    else:
        ate = np.asarray(inference.predict_ate_continuous(model, x, [-1.0, 0.0, 1.0]))
        assert ate.shape == (3,)
        ute = ate
    assert np.isfinite(ute).all()


@pytest.mark.parametrize(
    "combo",
    [
        ("bernoulli", "normal", "shared_trunk", 3),
        ("bernoulli", "exp_log_rate", "gated", 3),
        ("normal", "weibull", "gated", 3),
        ("normal", "exp_log_scale", "shared_trunk", 3),
    ],
    ids=lambda c: "-".join(map(str, c)),
)
def test_compile_true_round_trip(combo, tmp_path):
    """A compiled model saves and loads with the same metrics and UTEs."""
    model, _, inputs, y_true = _trained(combo)
    path = str(tmp_path / "model.keras")
    model.save(path)
    reloaded = tf.keras.models.load_model(path)
    before = model.evaluate(inputs, y_true, verbose=0, return_dict=True)
    after = reloaded.evaluate(inputs, y_true, verbose=0, return_dict=True)
    for key, value in before.items():
        np.testing.assert_allclose(after[key], value, rtol=1e-5, err_msg=key)
    if combo[0] == "bernoulli":
        np.testing.assert_allclose(
            inference.predict_ute_binary(reloaded, inputs[0][:10]),
            inference.predict_ute_binary(model, inputs[0][:10]),
            rtol=1e-5,
        )


# --- layer names, heads and initial values ---


def _contract_names(architecture, params, n_states, embedding, per_state=None):
    """The §3.8 layer names for one hidden layer in each stack."""
    per_state = per_state or {}
    names = {"features", "treatment", "features_bn", "features_and_treatment"}
    names |= {"state_weights", "treatment_params", "output_tensor"}
    names |= {f"predictive_state_{kind}_0" for kind in ("dense", "dropout", "bn")}
    if embedding:
        names |= {"treatment_bn", "treatment_embedding"}
    if architecture == "shared_trunk":
        names |= {"predictive_state_skip", "outcome_trunk_skip"}
        names |= {f"outcome_trunk_{kind}_0" for kind in ("dense", "dropout", "bn")}
    for p in params:
        names.add(f"{p}_all_states")
        if architecture == "gated":
            names |= {f"{p}_trunk_{kind}_0" for kind in ("dense", "dropout", "bn")}
        suffixes = [f"state_{k}" for k in range(n_states)]
        if not per_state.get(p, True):
            suffixes = ["shared"]
        for s in suffixes:
            names |= {f"{p}_hidden_{s}", f"{p}_raw_{s}", f"{p}_{s}"}
            if architecture == "gated" and s != "shared":
                names |= {f"{p}_gate_{s}", f"{p}_gated_treatment_{s}", f"{p}_input_{s}"}
    return names


@pytest.mark.parametrize(
    "architecture,outcome,embedding",
    [("gated", "weibull", 8), ("shared_trunk", "normal", None), ("shared_trunk", "weibull", 8)],
)
def test_layer_names_follow_the_contract(architecture, outcome, embedding):
    """Layer names are exactly the documented set."""
    model = _build(
        "bernoulli", outcome, architecture, 2, treatment_embedding_units=embedding, compile=False
    )
    params = OUTCOMES[outcome]().param_names
    assert {layer.name for layer in model.layers} == _contract_names(
        architecture, params, 2, embedding
    )
    assert isinstance(model.get_layer("output_tensor"), layers.CausalOutput)
    assert not any(layer.name.startswith("propensity") for layer in model.layers)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_head_spec_per_state_false_shares_one_head(architecture):
    """``per_state=False`` builds one head and repeats it across states."""
    model = _build(
        "normal",
        "weibull",
        architecture,
        outcome_heads={"log_shape": models.HeadSpec(per_state=False)},
        compile=False,
    )
    names = {layer.name for layer in model.layers}
    assert "log_shape_raw_shared" in names and "log_shape_raw_state_0" not in names
    assert "log_scale_raw_state_2" in names
    inputs, _ = _data("normal", "weibull", n=8)
    parts = ColumnLayout.from_model(model).decode_pred(model(inputs, training=False))
    shape = np.asarray(parts.outcome["log_shape"])
    np.testing.assert_array_equal(shape, np.repeat(shape[:, :1], 3, axis=1))


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_head_spec_constant_is_feature_independent(architecture):
    """``constant=True`` gives a BiasOnly head that ignores the inputs."""
    model = _build(
        "bernoulli",
        "normal",
        architecture,
        outcome_heads={"scale": models.HeadSpec(constant=True)},
        compile=False,
    )
    assert isinstance(model.get_layer("scale_raw_state_0"), layers.BiasOnly)
    inputs, _ = _data("bernoulli", "normal", n=8)
    parts = ColumnLayout.from_model(model).decode_pred(model(inputs, training=False))
    scale = np.asarray(parts.outcome["scale"])
    np.testing.assert_array_equal(scale, np.broadcast_to(scale[:1], scale.shape))
    np.testing.assert_allclose(scale, np.log(2.0), rtol=1e-6)  # softplus(0)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_head_spec_linear_head_builds(architecture):
    """``hidden_layer=None`` drops the hidden layer of that head only."""
    model = _build(
        "bernoulli",
        "normal",
        architecture,
        outcome_heads={"loc": models.HeadSpec(hidden_layer=None)},
        compile=False,
    )
    names = {layer.name for layer in model.layers}
    assert "loc_hidden_state_0" not in names and "loc_raw_state_0" in names
    assert "scale_hidden_state_0" in names


@pytest.mark.parametrize("outcome", ["normal", "weibull"])
def test_gated_parameters_share_no_weights_beyond_inputs_and_states(outcome):
    """No weight gets gradient from two outcome parameters, except the shared input layers
    (feature and treatment normalization, treatment embedding, state network)."""
    model = _build("normal", outcome, "gated", compile=False)
    inputs, _ = _data("normal", outcome, n=32)
    first, second = OUTCOMES[outcome]().param_names
    layout = ColumnLayout.from_model(model)
    with tf.GradientTape(persistent=True) as tape:
        parts = layout.decode_pred(model(inputs, training=False))
        sums = [tf.reduce_sum(parts.outcome[first]), tf.reduce_sum(parts.outcome[second])]
    variables = model.trainable_variables
    grads = [tape.gradient(s, variables) for s in sums]
    used_by_both = {
        v.path.split("/")[0]
        for v, g1, g2 in zip(variables, *grads)
        if g1 is not None and g2 is not None and np.any(g1 != 0) and np.any(g2 != 0)
    }
    assert used_by_both
    assert not any(name.startswith((first, second)) for name in used_by_both)


def _raw_bias(model, layer_name):
    """The bias (last weight) of a raw head layer."""
    layer = model.get_layer(layer_name)
    weights = layer.get_weights()
    return float(weights[-1][0])


def test_param_init_sets_head_bias():
    """Family default inits, and an outcome_init override, set the raw head's bias."""
    family = families.Weibull(_LS_BOUNDS, (0.5, 1.5))  # 0 outside: the midpoint, 1.0
    model = models.build_model(3, F, family, families.Normal(), compile=False)
    for p in family.params:
        for k in range(3):
            assert _raw_bias(model, f"{p.name}_raw_state_{k}") == pytest.approx(
                p.raw_value(p.init), abs=1e-6
            )
    assert family.param("log_shape").init == pytest.approx(1.0)

    override = models.build_model(
        3, F, family, families.Normal(), outcome_init={"log_scale": np.log(5.0)}, compile=False
    )
    expected = family.param("log_scale").raw_value(np.log(5.0))
    assert _raw_bias(override, "log_scale_raw_state_1") == pytest.approx(expected, abs=1e-6)

    constant = models.build_model(
        2,
        F,
        families.Normal(),
        families.Bernoulli(),
        outcome_heads={"scale": models.HeadSpec(constant=True)},
        outcome_init={"scale": 2.0},
        compile=False,
    )
    assert _raw_bias(constant, "scale_raw_state_0") == pytest.approx(np.log(np.expm1(2.0)))


@pytest.mark.parametrize(
    "init", [{"loc": 0.5, "scale": 2.0}, {"loc": [0.5, 0.5, 0.5], "scale": [2.0, 2.0, 2.0]}]
)
def test_treatment_init_sets_state_params(init):
    """A scalar or per-state treatment init sets the treatment params."""
    model = _build("normal", "normal", treatment_init=init, compile=False)
    inputs, _ = _data("normal", "normal", n=4)
    block = _treatment_block(model, inputs)
    np.testing.assert_allclose(block["loc"], 0.5, rtol=1e-5)
    np.testing.assert_allclose(block["scale"], 2.0, rtol=1e-5)


def test_bernoulli_treatment_starts_at_one_half():
    """The Bernoulli default init is prob 0.5."""
    model = _build("bernoulli", "normal", compile=False)
    inputs, _ = _data("bernoulli", "normal", n=4)
    np.testing.assert_allclose(_treatment_block(model, inputs)["prob"], 0.5, rtol=1e-6)


def test_partial_treatment_init_raises():
    """A treatment init must cover every parameter."""
    with pytest.raises(ValueError, match="missing"):
        _build("normal", "normal", treatment_init={"loc": 0.0}, compile=False)


# --- argument validation ---

_NORMAL, _BERNOULLI = families.Normal(), families.Bernoulli()


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        (dict(n_states=0), ValueError, "n_states"),
        (dict(n_features=0), ValueError, "n_features"),
        (dict(n_states=2.5), ValueError, "n_states"),
        (dict(architecture="deep"), ValueError, "shared_trunk"),
        (dict(outcome_heads={"rate": models.HeadSpec()}), KeyError, "loc"),
        (dict(outcome_heads={"loc": (16, "relu")}), TypeError, "HeadSpec"),
        (dict(outcome_init={"mean": 0.0}), KeyError, "scale"),
        (dict(treatment_init={"loc": 0.0}), KeyError, "prob"),
        (dict(outcome_init={"scale": -1.0}), ValueError, "scale"),
        (dict(treatment_init={"prob": 1.5}), ValueError, "prob"),
        (dict(max_time=10.0), ValueError, "survival"),
        (dict(treatment_family=families.Weibull()), ValueError, ""),
        (dict(predictive_state_hidden_layers=[(0, "relu")]), ValueError, "units"),
        (dict(outcome_hidden_layers=[16]), ValueError, "pair"),
        (dict(outcome_hidden_layers=[(16, None)]), ValueError, "activation"),
        (dict(dropout_rate=1.0), ValueError, "dropout"),
        (dict(l1=-0.1), ValueError, "l1"),
        (dict(treatment_embedding_units=0), ValueError, "treatment_embedding_units"),
        (dict(compile=False, alpha=2.0), TypeError, "compile=False"),
        (dict(learning_rte=0.1), TypeError, "learning_rte"),
    ],
)
def test_build_model_validates_arguments(kwargs, error, match):
    """Bad arguments raise before any layer is created."""
    args = dict(n_states=2, n_features=F, outcome_family=_NORMAL, treatment_family=_BERNOULLI)
    args.update(kwargs)
    before = tf.keras.backend.get_uid("dense")
    with pytest.raises(error, match=match):
        models.build_model(**args)
    assert tf.keras.backend.get_uid("dense") == before + 1  # no layer was created


def test_head_spec_validates_its_hidden_layer():
    """HeadSpec rejects an invalid hidden layer."""
    with pytest.raises(ValueError, match="units"):
        models.HeadSpec(hidden_layer=(-1, "relu"))


def test_max_time_check_passes_through():
    """The family's float32 overflow check raises through build_model."""
    kwargs = dict(n_states=2, n_features=F, treatment_family=_NORMAL, max_time=100.0)
    with pytest.raises(ValueError, match="overflow"):
        models.build_model(outcome_family=families.Weibull((-10.0, 5.0), (-1.0, 3.0)), **kwargs)
    models.build_model(outcome_family=families.Weibull((-3.0, 5.0), (-1.0, 1.0)), **kwargs)


# --- compile_model ---


def _nlls(model, inputs, y_true):
    """(mean outcome NLL, mean treatment NLL) of the model's mixture at the current weights."""
    mix = mixture.StateMixture.from_model(model)
    y_pred = model(inputs, training=False)
    return (
        float(tf.reduce_mean(mix.outcome_nll(y_true, y_pred))),
        float(tf.reduce_mean(mix.treatment_nll(y_true, y_pred))),
    )


def _evaluate(model, inputs, y_true):
    """``model.evaluate`` on one full batch, as a dict."""
    return model.evaluate(inputs, y_true, batch_size=len(y_true), verbose=0, return_dict=True)


@pytest.mark.parametrize("treatment", list(TREATMENTS))
def test_alpha_variable_reaches_the_loss(treatment):
    """Assigning the alpha variable changes the loss by the treatment NLL term."""
    alpha = tf.Variable(1.0, trainable=False, dtype=tf.float32)
    model = _build(treatment, "normal", alpha=alpha)
    assert model.loss.alpha is alpha
    inputs, y_true = _data(treatment, "normal", n=64)
    before = _evaluate(model, inputs, y_true)["loss"]
    alpha.assign(5.0)
    after = _evaluate(model, inputs, y_true)["loss"]
    _, treatment_nll = _nlls(model, inputs, y_true)
    np.testing.assert_allclose(after - before, 4.0 * treatment_nll, rtol=1e-3)


def test_causal_loss_metric_ignores_model_alpha():
    """The metric is ``w * outcome NLL + treatment NLL``, whatever alpha is."""
    alpha = tf.Variable(3.0, trainable=False, dtype=tf.float32)
    model = _build(
        "bernoulli",
        "normal",
        alpha=alpha,
        outcome_loss_weight=2.0,
        predictive_states_regularizer=pypress.keras.regularizers.DegreesOfFreedom(l2=10.0),
    )
    inputs, y_true = _data("bernoulli", "normal", n=64)
    outcome_nll, treatment_nll = _nlls(model, inputs, y_true)
    result = _evaluate(model, inputs, y_true)
    expected = 2.0 * outcome_nll + treatment_nll
    np.testing.assert_allclose(result["causal_loss_metric"], expected, rtol=1e-4)
    assert not np.isclose(result["loss"], result["causal_loss_metric"], rtol=1e-3)
    alpha.assign(5.0)
    np.testing.assert_allclose(
        _evaluate(model, inputs, y_true)["causal_loss_metric"], expected, rtol=1e-4
    )


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("via", ["build_model", "compile_model"])
def test_causal_loss_metric_uses_outcome_loss_weight(weight, via):
    """The metric weights the outcome NLL by ``outcome_loss_weight``."""
    if via == "build_model":
        model = _build("bernoulli", "normal", outcome_loss_weight=weight)
    else:
        model = models.compile_model(
            _build("bernoulli", "normal", compile=False), outcome_loss_weight=weight
        )
    inputs, y_true = _data("bernoulli", "normal", n=64)
    outcome_nll, treatment_nll = _nlls(model, inputs, y_true)
    np.testing.assert_allclose(
        _evaluate(model, inputs, y_true)["causal_loss_metric"],
        weight * outcome_nll + treatment_nll,
        rtol=1e-4,
    )
    if weight == 0.0:
        history = model.fit(inputs, y_true, epochs=1, verbose=0)
        assert np.isfinite(history.history["loss"]).all()


def test_negative_outcome_loss_weight_raises():
    """A negative outcome loss weight raises."""
    with pytest.raises(ValueError, match="outcome_loss_weight"):
        _build("bernoulli", "normal", outcome_loss_weight=-1.0)


def test_causal_loss_metric_ignores_alpha_schedule():
    """Under frozen weights the metric is constant across epochs while the loss moves."""
    model = _build("bernoulli", "normal", compile=False)
    model.trainable = False
    models.compile_model(model, alpha=2.0)
    inputs, y_true = _data("bernoulli", "normal", n=64)
    schedule = callbacks.AlphaScheduleCallback(
        alpha_start=3.0, alpha_end=1.0, half_life_epoch=1, steepness=2.0, verbose=False
    )
    history = model.fit(
        inputs,
        y_true,
        validation_data=(inputs, y_true),
        epochs=4,
        batch_size=64,
        callbacks=[schedule],
        verbose=0,
    ).history
    np.testing.assert_allclose(
        history["val_causal_loss_metric"], history["val_causal_loss_metric"][0], rtol=1e-6
    )
    assert np.ptp(history["val_loss"]) > 1e-3


@pytest.mark.parametrize("outcome", list(OUTCOMES))
@pytest.mark.parametrize("likelihood", ["interventional", "observational"])
def test_outcome_likelihood_reaches_the_loss(outcome, likelihood):
    """``outcome_likelihood`` is passed to the outcome loss."""
    model = _build("normal", outcome, outcome_likelihood=likelihood)
    assert model.loss._outcome_loss.outcome_likelihood is losses.OutcomeLikelihood(likelihood)


def test_compile_model_on_a_hand_built_network():
    """Any network ending in a CausalOutput can be compiled: the second adapter."""
    import pypress.keras.layers

    mix = mixture.StateMixture(families.Normal(), families.Bernoulli(), n_states=2)
    features = tf.keras.Input((F,), name="features")
    treatment = tf.keras.Input((1,), name="treatment")
    hidden = tf.keras.layers.Dense(8, "relu")(features)
    weights = pypress.keras.layers.PredictiveStateSimplex(n_states=2)(hidden)
    joint = tf.keras.layers.Concatenate()([hidden, treatment])
    loc = tf.keras.layers.Dense(2)(joint)
    scale = tf.keras.layers.Dense(2, activation="softplus")(joint)
    prob = pypress.keras.layers.PredictiveStateParams(1, activations="sigmoid")(weights)
    out = layers.CausalOutput(mix)(
        {"outcome": {"loc": loc, "scale": scale}, "state_weights": weights, "treatment": prob}
    )
    model = models.compile_model(tf.keras.Model([features, treatment], out))
    inputs, y_true = _data("bernoulli", "normal", n=64)
    history = model.fit(inputs, y_true, epochs=2, verbose=0)
    assert np.isfinite(history.history["loss"]).all()
    assert model.loss.mixture == mix


def test_compile_model_rejects_a_plain_keras_model():
    """A model without CausalOutput cannot be compiled."""
    plain = tf.keras.Sequential([tf.keras.Input((F,)), tf.keras.layers.Dense(1)])
    with pytest.raises(RuntimeError, match="CausalOutput"):
        models.compile_model(plain)


def test_compile_model_rejects_optimizer_with_learning_rate():
    """An optimizer excludes learning_rate and clipnorm."""
    model = _build("bernoulli", "normal", compile=False)
    with pytest.raises(TypeError, match="not both"):
        models.compile_model(model, optimizer=tf.keras.optimizers.Adam(), learning_rate=0.1)
    with pytest.raises(TypeError, match="not both"):
        models.compile_model(model, optimizer=tf.keras.optimizers.Adam(), clipnorm=None)
    models.compile_model(model, optimizer=tf.keras.optimizers.Adam(0.05))
    assert float(model.optimizer.learning_rate) == pytest.approx(0.05)


def test_compile_model_default_optimizer():
    """The default optimizer is Nadam with the given learning rate and clipnorm."""
    model = _build("bernoulli", "normal", learning_rate=0.02, clipnorm=None)
    assert isinstance(model.optimizer, tf.keras.optimizers.Nadam)
    assert float(model.optimizer.learning_rate) == pytest.approx(0.02)
    assert model.optimizer.clipnorm is None
    assert _build("bernoulli", "normal").optimizer.clipnorm == 1.0


def test_extra_metrics_are_appended_and_metrics_replace_the_defaults():
    """``extra_metrics`` add to the defaults; ``metrics`` replaces them."""
    extra = tf.keras.metrics.MeanMetricWrapper(lambda y, p: tf.reduce_mean(p), name="mean_pred")
    model = _build("bernoulli", "normal", extra_metrics=[extra])
    inputs, y_true = _data("bernoulli", "normal", n=32)
    keys = set(_evaluate(model, inputs, y_true))
    assert {"mean_pred", "causal_loss_metric"} <= keys

    only = tf.keras.metrics.MeanMetricWrapper(lambda y, p: tf.reduce_mean(p), name="only")
    model = _build("bernoulli", "normal", metrics=[only])
    assert set(_evaluate(model, inputs, y_true)) == {"loss", "only"}


def test_build_is_reproducible_under_set_random_seed():
    """The same seed gives the same initial weights."""
    weights = []
    for _ in range(2):
        tf.keras.utils.set_random_seed(7)
        weights.append(_build("normal", "weibull", compile=False).get_weights())
    for a, b in zip(*weights):
        np.testing.assert_array_equal(a, b)


# --- get_propensity_state_conditional_means ---


def test_propensity_means_for_every_model(tmp_path):
    """Each state's mean treatment, read through the mixture: toy preset and build_model
    models (0.5 at init, tracking the weights), a Normal treatment (the per-state loc), and a
    compile=False reload."""
    get = models.get_propensity_state_conditional_means

    toy = models.build_model(**presets.toy(n_states=4, n_features=3, compile=False))
    np.testing.assert_allclose(get(toy).numpy(), np.full((4, 1), 0.5), rtol=1e-6)
    toy.get_layer("treatment_params").set_weights([np.array([[0.0], [0.0], [2.0], [0.0]])])
    np.testing.assert_allclose(
        get(toy).numpy().ravel(), [0.5, 0.5, tf.sigmoid(2.0).numpy(), 0.5], rtol=1e-5
    )

    gated = _build("bernoulli", "weibull", "gated", compile=False)
    np.testing.assert_allclose(get(gated).numpy(), np.full((3, 1), 0.5), rtol=1e-6)
    logits = np.array([[-1.0], [0.0], [2.0]], dtype="float32")
    gated.get_layer("treatment_params").set_weights([logits])
    np.testing.assert_allclose(get(gated).numpy(), tf.sigmoid(logits).numpy(), rtol=1e-5)

    continuous = _build(
        "normal", "weibull", treatment_init={"loc": [-1.0, 0.5, 2.0], "scale": 1.0}, compile=False
    )
    np.testing.assert_allclose(get(continuous).numpy(), [[-1.0], [0.5], [2.0]], rtol=1e-5)

    path = str(tmp_path / "model.keras")
    continuous.save(path)
    reloaded = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(get(reloaded).numpy(), get(continuous).numpy(), rtol=1e-6)


def test_propensity_means_of_a_pre_v0_3_4_model():
    """A model without CausalOutput or a pypsps loss falls back to its propensity layer names."""
    features = tf.keras.Input((F,), name="features")
    logits = [
        layers.BiasOnly(
            initializer=tf.keras.initializers.Constant(v), name=f"propensity_logit_state_{k}"
        )(features)
        for k, v in enumerate([0.0, 1.0])
    ]
    propensity = tf.keras.layers.Concatenate(name="propensity_score")(
        [tf.keras.layers.Activation("sigmoid")(x) for x in logits]
    )
    model = tf.keras.Model(features, propensity)
    np.testing.assert_allclose(
        models.get_propensity_state_conditional_means(model).numpy(),
        [[0.5], [tf.sigmoid(1.0).numpy()]],
        rtol=1e-6,
    )
