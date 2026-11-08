"""Tests for the outcome and treatment families."""

import math

import numpy as np
import pytest
import tensorflow as tf

from ...keras import families, neglogliks
from ...keras.layout import ColumnLayout
from . import _reference_v0_2 as ref

_N, _K = 6, 3


def _params(family: families.Family, seed: int = 0) -> dict:
    """Valid per-state parameters ``{name: [N, K]}`` for ``family``."""
    rng = np.random.RandomState(seed)
    out = {}
    for p in family.params:
        if p.bounds is not None:
            low, high = p.bounds
            value = rng.uniform(low + 0.1 * (high - low), high - 0.1 * (high - low), (_N, _K))
        elif p.activation == "softplus":
            value = rng.uniform(0.5, 2.0, (_N, _K))
        elif p.activation == "sigmoid":
            value = rng.uniform(0.1, 0.9, (_N, _K))
        else:
            value = rng.uniform(-1.0, 1.0, (_N, _K))
        out[p.name] = value.astype("float32")
    return out


def _y_true(family: families.Family, seed: int = 1) -> np.ndarray:
    """A ``y_true`` that fits ``family``: [time, event], 0/1, or real."""
    rng = np.random.RandomState(seed)
    if isinstance(family, families.SurvivalFamily):
        return np.stack([rng.uniform(0.5, 5.0, _N), rng.randint(0, 2, _N)], axis=1).astype(
            "float32"
        )
    if family.is_binary:
        return rng.randint(0, 2, (_N, 1)).astype("float32")
    return rng.normal(size=(_N, 1)).astype("float32")


_FAMILIES = {
    "normal": families.Normal(),
    "bernoulli": families.Bernoulli(),
    "exponential_log_scale": families.Exponential("log_scale", bounds=(-1.0, 3.0)),
    "exponential_log_rate": families.Exponential("log_rate", bounds=(-3.0, 1.0)),
    "exponential_rate": families.Exponential("rate"),
    "weibull": families.Weibull(log_scale_bounds=(-1.0, 3.0), log_shape_bounds=(-1.0, 1.0)),
}
_IDS = list(_FAMILIES)
_ALL = [_FAMILIES[i] for i in _IDS]


def _closed_form_nll(family, y, p):
    """NumPy reference densities, in float64."""
    y, p = y.astype("float64"), {k: v.astype("float64") for k, v in p.items()}
    if isinstance(family, families.Normal):
        z = (y[:, :1] - p["loc"]) / p["scale"]
        return 0.5 * np.log(2 * np.pi) + np.log(p["scale"]) + 0.5 * z**2
    if isinstance(family, families.Bernoulli):
        return -(y * np.log(p["prob"]) + (1 - y) * np.log(1 - p["prob"]))
    t, d = y[:, :1], y[:, 1:2]
    if isinstance(family, families.Weibull):
        lam, k = np.exp(p["log_scale"]), np.exp(p["log_shape"])
        log_pdf = np.log(k) - k * np.log(lam) + (k - 1) * np.log(t)
        return (t / lam) ** k - d * log_pdf
    rate = {
        "log_scale": lambda: np.exp(-p.get("log_scale", 0.0)),
        "log_rate": lambda: np.exp(p.get("log_rate", 0.0)),
        "rate": lambda: p.get("rate", 0.0),
    }[family.parameterization]()
    return rate * t - d * np.log(rate)


def _closed_form_mean(family, p):
    """NumPy reference means, in float64."""
    p = {k: v.astype("float64") for k, v in p.items()}
    if isinstance(family, families.Normal):
        return p["loc"]
    if isinstance(family, families.Bernoulli):
        return p["prob"]
    if isinstance(family, families.Weibull):
        return np.exp(p["log_scale"]) * np.vectorize(math.gamma)(1 + 1 / np.exp(p["log_shape"]))
    return {
        "log_scale": lambda: np.exp(p.get("log_scale", 0.0)),
        "log_rate": lambda: np.exp(-p.get("log_rate", 0.0)),
        "rate": lambda: 1 / p.get("rate", 1.0),
    }[family.parameterization]()


@pytest.mark.parametrize("family", _ALL, ids=_IDS)
def test_nll_matches_closed_form(family):
    """Each family's NLL is its textbook (censored) negative log density."""
    y, p = _y_true(family), _params(family)
    nll = family.nll(tf.constant(y), {k: tf.constant(v) for k, v in p.items()})
    assert nll.shape == (_N, _K)
    np.testing.assert_allclose(nll.numpy(), _closed_form_nll(family, y, p), rtol=1e-4, atol=1e-4)


_LEGACY_LOSSES = {
    "normal": neglogliks.NegloglikNormal(reduction="none"),
    "bernoulli": tf.keras.losses.BinaryCrossentropy(reduction="none"),
    "exponential_log_scale": neglogliks.NegloglikExponentialScale(reduction="none"),
    "exponential_log_rate": neglogliks.NegloglikExponential(reduction="none", log_rate=True),
    "exponential_rate": neglogliks.NegloglikExponential(reduction="none"),
    "weibull": neglogliks.NegloglikWeibull(reduction="none"),
}


@pytest.mark.parametrize("name", _IDS)
def test_nll_matches_the_per_state_loop(name):
    """Family.nll equals the frozen per-state loop over the public losses to float32 rounding.

    Not bit for bit: TensorFlow's vectorized float32 log/exp can round an element differently
    depending on its position in the tensor, and the loop sees [N] columns where the family
    sees one [N, K] block. Differences are about one ulp.
    """
    family = _FAMILIES[name]
    y, p = _y_true(family), _params(family)
    layout = ColumnLayout(family.param_names, ("prob",), family.n_true_cols, n_states=_K)
    y_pred = layout.encode(p, np.full((_N, _K), 1.0 / _K, "float32"), np.zeros((_N, _K), "f4"))
    outcome_block = y_pred[:, : _K * len(family.params)]
    legacy = ref._negloglik_per_state(
        _LEGACY_LOSSES[name], tf.constant(y), tf.constant(outcome_block), n_states=_K
    )
    new = family.nll(tf.constant(y), layout.decode_pred(tf.constant(y_pred)).outcome)
    np.testing.assert_allclose(new.numpy(), legacy.numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("family", _ALL, ids=_IDS)
def test_mean_matches_closed_form(family):
    """mean is E[response | state]; works on NumPy input too."""
    p = _params(family)
    np.testing.assert_allclose(np.asarray(family.mean(p)), _closed_form_mean(family, p), rtol=1e-5)


def test_event_indicator_masks_survival_families():
    """Survival families mask on the event column as a float32 tensor; the others do not mask.

    The column order (``response``) is tested in ``tests/core/test_families.py``.
    """
    y2 = tf.constant([[3.0, 1.0, 7.0], [4.0, 0.0, 7.0]])
    for family in _ALL:
        if isinstance(family, families.SurvivalFamily):
            mask = family.event_indicator(y2)
            assert mask.dtype == tf.float32
            np.testing.assert_array_equal(mask, [[1.0], [0.0]])
        else:
            assert family.event_indicator(y2) is None
    assert families.LossFamily(lambda y, p: p[:, 0], 1).event_indicator(y2) is None


@pytest.mark.parametrize("family", _ALL, ids=_IDS)
def test_param_specs_and_head_activation_ranges(family):
    """Every head maps any raw value onto the parameter's support."""
    raw = tf.constant(np.linspace(-30.0, 30.0, 61, dtype="float32")[:, None])
    for p in family.params:
        out = family.head_activation(p.name, name=f"{p.name}_head")(raw).numpy()
        assert np.isfinite(out).all()
        if p.bounds is not None:
            assert isinstance(family.head_activation(p.name), tf.keras.layers.Layer)
            assert (out >= p.bounds[0]).all() and (out <= p.bounds[1]).all()
        elif p.activation == "softplus":
            assert (out >= 0).all()
        elif p.activation == "sigmoid":
            assert (out >= 0).all() and (out <= 1).all()
    with pytest.raises(KeyError, match="no parameter"):
        family.head_activation("nope")


@pytest.mark.parametrize("family", _ALL, ids=_IDS)
def test_raw_value_inverts_head_activation(family):
    """head_activation(raw_value(v)) == v inside the support (outside it raises, see core)."""
    for p in family.params:
        if p.bounds is not None:
            low, high = p.bounds
            grid = np.linspace(low, high, 9)[1:-1]
        elif p.activation == "softplus":
            grid = [0.01, 0.5, 1.0, 5.0]
        elif p.activation == "sigmoid":
            grid = [0.01, 0.3, 0.5, 0.99]
        else:
            grid = [-5.0, 0.0, 3.0]
        raw = tf.constant([[p.raw_value(v)] for v in grid], tf.float32)
        np.testing.assert_allclose(
            family.head_activation(p.name)(raw).numpy()[:, 0], grid, rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("family", _ALL, ids=_IDS)
def test_config_round_trip_and_equality(family):
    """Families serialize through Keras to configs and rebuild equal (the core config round
    trip is tested in ``tests/core/test_families.py``)."""
    config = tf.keras.utils.serialize_keras_object(family)
    rebuilt = tf.keras.utils.deserialize_keras_object(config)
    assert type(rebuilt) is type(family) and rebuilt == family
    assert rebuilt.param_names == family.param_names


def test_get_by_name():
    """get maps the builders' distribution strings onto classes."""
    assert families.get("Normal") == families.Normal()
    assert families.get("weibull", log_scale_bounds=(0, 1)) == families.Weibull((0.0, 1.0))
    assert families.get("exponential", parameterization="rate") == families.Exponential("rate")
    with pytest.raises(ValueError, match="known"):
        families.get("gamma")


@pytest.mark.parametrize(
    "parameterization, private_nll",
    [
        ("log_scale", lambda t, d, v: neglogliks._negloglik_exponential_scale(t, d, v)),
        ("log_rate", lambda t, d, v: neglogliks._negloglik_exponential(t, d, tf.exp(v))),
        ("rate", lambda t, d, v: neglogliks._negloglik_exponential(t, d, v)),
    ],
)
def test_exponential_routes_to_private_impl(parameterization, private_nll):
    """One public class; nll and mean come from the matching private implementation."""
    family = families.Exponential(parameterization)
    (name,) = family.param_names
    assert name == parameterization
    y, p = _y_true(family), _params(family)
    expected = private_nll(y[:, :1], y[:, 1:2], p[name])
    np.testing.assert_array_equal(family.nll(y, p).numpy(), expected.numpy())
    assert set(family.get_config()) == {"parameterization", "bounds"}
    with pytest.raises(ValueError, match="log_scale.*log_rate.*rate"):
        families.Exponential("scale")


def test_loss_family_wraps_custom_loss_and_mean_raises():
    """LossFamily loops a custom elementwise loss over states; its mean is unknown."""
    mse = tf.keras.losses.MeanSquaredError(reduction="none")
    family = families.LossFamily(mse, n_params=1, n_true_cols=1)
    assert family.param_names == ("param_0",)
    y = np.arange(_N, dtype="float32")[:, None]
    p = {"param_0": np.ones((_N, _K), "float32")}
    np.testing.assert_allclose(family.nll(y, p).numpy(), np.tile((y - 1) ** 2, (1, _K)))
    with pytest.raises(NotImplementedError, match="mean"):
        family.mean(p)
    with pytest.raises(ValueError, match="reduction='none'"):
        families.LossFamily(tf.keras.losses.MeanSquaredError(), 1)
    rebuilt = tf.keras.utils.deserialize_keras_object(tf.keras.utils.serialize_keras_object(family))
    np.testing.assert_allclose(rebuilt.nll(y, p).numpy(), family.nll(y, p).numpy())


@pytest.mark.parametrize("family", _ALL, ids=_IDS)
def test_nll_shape_survives_partial_batch_retrace(family):
    """The 3e4c9b9 regression through Family.nll: graph-mode fit with a partial last batch.

    n_samples=200 with batch_size=32 leaves an 8-row final batch, which retraces train_step
    with a different batch shape.
    """
    n_features = 5
    rng = np.random.RandomState(0)
    x = rng.randn(200, n_features).astype("float32")
    if isinstance(family, families.SurvivalFamily):
        y = np.stack([rng.uniform(1.0, 10.0, 200), rng.randint(0, 2, 200)], 1).astype("float32")
    else:
        y = (rng.uniform(size=(200, 1)) > 0.5).astype("float32")
    layout = ColumnLayout(family.param_names, ("prob",), family.n_true_cols, n_states=_K)
    raw = tf.keras.layers.Dense(_K * len(family.params))(inp := tf.keras.Input((n_features,)))

    def loss(y_true, y_pred):
        """Mean NLL over states with each parameter mapped onto its support."""
        blocks = layout.decode_pred(
            tf.concat([y_pred, tf.zeros_like(y_pred[:, :_K]), tf.zeros_like(y_pred[:, :_K])], 1)
        ).outcome
        params = {n: family.head_activation(n)(b) for n, b in blocks.items()}
        return tf.reduce_mean(family.nll(y_true, params), axis=1)

    model = tf.keras.Model(inp, raw)
    model.compile(optimizer="adam", loss=loss)
    history = model.fit(x, y, batch_size=32, epochs=2, verbose=0)
    assert np.isfinite(history.history["loss"][-1])
