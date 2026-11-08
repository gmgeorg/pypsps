"""A Keras model copied into PyTorch gives the same ``y_pred`` in eval mode."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("torch")

import torch  # noqa: E402

from ...core.spec import NonPortableActivationWarning  # noqa: E402
from ...keras import families as kf  # noqa: E402
from ...keras import models as kmodels  # noqa: E402
from ...keras import presets as kpresets  # noqa: E402
from ...torch import convert  # noqa: E402
from ...torch import models as tmodels  # noqa: E402
from ...torch import presets as tpresets  # noqa: E402
from ...torch.models import HeadSpec  # noqa: E402

_N = 64


def _data(n_features, binary_outcome=False, survival=False, seed=0):
    """Features, treatment and an ``[N, C + 1]`` ``y_true`` for the fitted outcome family."""
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(_N, n_features)).astype("f4")
    a = rng.randint(0, 2, _N).astype("f4")
    y = rng.uniform(0.5, 3.0, _N).astype("f4")
    cols = [y, rng.randint(0, 2, _N).astype("f4")] if survival else [y]
    return x, a, np.stack(cols + [a], axis=1)


def _assert_same(keras_model, x, a, y_true, fit_epochs=2):
    """Trains briefly (so BN statistics move), converts, and compares eval predictions."""
    keras_model.fit([x, a], y_true, epochs=fit_epochs, batch_size=16, verbose=0)
    expected = keras_model.predict([x, a], verbose=0)
    model = tmodels.PSPSModel.from_keras(keras_model)
    assert not model.training
    with torch.no_grad():
        actual = model(torch.tensor(x), torch.tensor(a)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("architecture", ["gated", "shared_trunk"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"treatment_embedding_units": None},
        {"predictive_state_hidden_layers": (), "outcome_hidden_layers": ()},
        {"predictive_state_hidden_layers": ((8, "tanh"), (4, "selu")), "l1": 0.01},
        {
            "outcome_heads": {
                "loc": HeadSpec(hidden_layer=(6, "relu")),
                "scale": HeadSpec(constant=True),
            }
        },
        {"outcome_heads": {"loc": HeadSpec(per_state=False), "scale": HeadSpec(constant=True)}},
        {
            "outcome_heads": {"loc": HeadSpec(constant=True, per_state=False)},
            "outcome_init": {"scale": 1.5},
        },
    ],
    ids=["default", "no-embedding", "no-hidden", "deep-state-l1", "heads", "shared-heads", "const"],
)
def test_normal_bernoulli(architecture, kwargs):
    """Every head, trunk and embedding variant converts exactly."""
    args = {"architecture": architecture, **kwargs}
    # the same arguments through the Keras builder (HeadSpec is shared core code)
    keras_model = kmodels.build_model(3, 5, kf.Normal(), kf.Bernoulli(), **args)
    _assert_same(keras_model, *_data(5))


@pytest.mark.parametrize("n_states", [1, 4])
def test_normal_treatment(n_states):
    """A Normal treatment (two treatment parameters) converts too."""
    keras_model = kmodels.build_model(n_states, 4, kf.Normal(), kf.Normal())
    x, a, y_true = _data(4)
    _assert_same(keras_model, x, np.random.RandomState(1).normal(size=_N).astype("f4"), y_true)


@pytest.mark.parametrize(
    "outcome",
    [
        kf.Exponential("log_scale"),
        kf.Exponential("log_rate", bounds=(-2.0, 2.0)),
        kf.Exponential("rate"),
        kf.Weibull(log_scale_bounds=(-1.0, 3.0), log_shape_bounds=(-1.0, 1.0)),
    ],
    ids=["exp-log-scale", "exp-log-rate-bounds", "exp-rate", "weibull-bounds"],
)
def test_survival_families(outcome):
    """Survival outcomes, with and without a SoftClip bound, convert exactly."""
    keras_model = kmodels.build_model(3, 5, outcome, kf.Bernoulli(), outcome_init={})
    _assert_same(keras_model, *_data(5, survival=True))


_PRESETS = {
    "toy": lambda: kpresets.toy(3, 5),
    "binary_normal": lambda: kpresets.binary_normal(
        3, 5, [(8, "relu")], [(8, "relu")], loc_layer=(4, "selu")
    ),
    "binary_exponential": lambda: kpresets.binary_exponential(3, 5, [(8, "relu")], [(8, "relu")]),
    "continuous_exponential": lambda: kpresets.continuous_exponential(
        3, 5, [(8, "relu")], [(8, "relu")], log_scale_bounds=(-2.0, 4.0)
    ),
    "continuous_weibull": lambda: kpresets.continuous_weibull(
        3, 5, [(8, "relu")], [(8, "relu")], log_scale_bounds=(-2.0, 4.0), log_shape_bounds=(-1, 1)
    ),
}


@pytest.mark.parametrize("name", list(_PRESETS))
def test_every_preset_converts(name):
    """A model built from each preset predicts the same after conversion."""
    keras_model = kmodels.build_model(**_PRESETS[name]())
    _assert_same(keras_model, *_data(5, survival=name != "toy" and name != "binary_normal"))


def test_loss_family_has_no_torch_counterpart():
    """A LossFamily wraps a callable, so converting it raises NotImplementedError."""
    import tensorflow as tf

    family = kf.LossFamily(tf.keras.losses.MeanSquaredError(reduction="none"), n_params=1)
    keras_model = kmodels.build_model(2, 3, family, kf.Bernoulli(), compile=False)
    with pytest.raises(NotImplementedError):
        tmodels.PSPSModel.from_keras(keras_model)


@pytest.mark.parametrize(
    "kwargs,field",
    [
        ({"predictive_state_hidden_layers": [(4, "relu"), (4, "leaky_relu")]}, "predictive_state"),
        ({"outcome_hidden_layers": [(4, "leaky_relu")]}, "outcome_hidden_layers"),
        ({"outcome_heads": {"loc": HeadSpec(hidden_layer=(4, "leaky_relu"))}}, "loc"),
    ],
)
def test_from_keras_rejects_a_non_portable_activation_up_front(monkeypatch, kwargs, field):
    """A Keras-only activation fails before any torch model is built, naming the field."""
    with pytest.warns(NonPortableActivationWarning):
        keras_model = kmodels.build_model(2, 5, kf.Normal(), kf.Bernoulli(), **kwargs)

    def fail(*args, **kwargs):
        raise AssertionError("torch build_model was called")

    monkeypatch.setattr(tmodels, "build_model", fail)
    monkeypatch.setattr(tmodels.PSPSModel, "__init__", fail)
    with pytest.raises(ValueError, match="leaky_relu") as caught:
        tmodels.PSPSModel.from_keras(keras_model)
    assert field in str(caught.value)
    assert "relu" in str(caught.value)


@pytest.mark.parametrize("name", list(_PRESETS))
def test_from_keras_reads_the_spec_not_the_layer_names(monkeypatch, name):
    """With a carried Spec the layer-name scraper is not used, and the copy is still exact."""

    def fail(*args, **kwargs):
        raise AssertionError("the layer-name scraper was used")

    monkeypatch.setattr(convert, "infer_build_args", fail)
    keras_model = kmodels.build_model(**_PRESETS[name]())
    _assert_same(keras_model, *_data(5, survival=name != "toy" and name != "binary_normal"))


def _silu(layers):
    """``(units, activation)`` pairs with Keras's canonical name for swish."""
    return [[u, "silu" if a == "swish" else a] for u, a in layers]


def _structure(config):
    """The part of a Spec config that the layer names determine.

    Keras writes "swish" as "silu", a constant head ignores its hidden layer, and a model
    without hidden layers has no dropout or L1 layer to read.
    """
    heads = {}
    for name, head in config["outcome_heads"].items():
        hidden = None if head["constant"] or head["hidden_layer"] is None else head["hidden_layer"]
        heads[name] = {**head, "hidden_layer": hidden and _silu([hidden])[0]}
    structure = {
        "n_features": config["n_features"],
        "architecture": config["architecture"],
        "predictive_state_hidden_layers": _silu(config["predictive_state_hidden_layers"]),
        "outcome_hidden_layers": _silu(config["outcome_hidden_layers"]),
        "outcome_heads": heads,
        "treatment_embedding_units": config["treatment_embedding_units"],
    }
    if config["predictive_state_hidden_layers"] or config["outcome_hidden_layers"]:
        structure.update(dropout_rate=config["dropout_rate"], l1=config["l1"])
    return structure


@pytest.mark.parametrize("architecture", ["gated", "shared_trunk"])
@pytest.mark.parametrize("name", list(_PRESETS))
def test_scraper_agrees_with_the_spec(name, architecture):
    """The layer-name fallback (for models saved before the Spec was carried) reads the same
    architecture the Spec records, so it can't rot unnoticed."""
    keras_model = kmodels.build_model(**{**_PRESETS[name](), "architecture": architecture})
    carried = kmodels.spec_of(keras_model).get_config()
    scraped = tmodels.build_model(**convert.infer_build_args(keras_model)).spec.get_config()
    assert _structure(scraped) == _structure(carried)


@pytest.mark.parametrize("architecture", ["gated", "shared_trunk"])
def test_from_keras_falls_back_to_the_layer_names_without_a_spec(architecture):
    """A Keras model without the carried Spec (saved before v0.6.3) still converts."""
    keras_model = kmodels.build_model(
        2, 5, kf.Normal(), kf.Bernoulli(), architecture=architecture, l1=0.01
    )
    config = keras_model.get_config()
    for layer in config["layers"]:
        if layer["class_name"].endswith("CausalOutput"):
            layer["config"].pop("spec")
    old = type(keras_model).from_config(config)
    old.set_weights(keras_model.get_weights())
    assert kmodels.spec_of(old) is None
    x, a, y_true = _data(5)
    expected = keras_model.predict([x, a], verbose=0)
    model = tmodels.PSPSModel.from_keras(old)
    with torch.no_grad():
        np.testing.assert_allclose(
            model(torch.tensor(x), torch.tensor(a)).numpy(), expected, rtol=1e-4, atol=1e-5
        )
    assert model.spec.l1 == pytest.approx(0.01, rel=1e-6)


def test_from_keras_keeps_max_time():
    """``max_time`` survives the conversion."""
    weibull = kf.Weibull(log_scale_bounds=(-2.0, 4.0), log_shape_bounds=(-1.0, 1.0))
    keras_model = kmodels.build_model(2, 5, weibull, kf.Bernoulli(), max_time=50.0)
    assert tmodels.PSPSModel.from_keras(keras_model).spec.max_time == 50.0


@pytest.mark.parametrize("architecture", ["gated", "shared_trunk"])
def test_native_models_from_one_spec_match(architecture):
    """Both backends build the same network from the same arguments: with the weights copied
    by name into a natively built torch model, the forward passes agree."""
    args = {**_PRESETS["continuous_weibull"](), "architecture": architecture}
    keras_model = kmodels.build_model(**args)
    x, a, y_true = _data(5, survival=True)
    keras_model.fit([x, a], y_true, epochs=2, batch_size=16, verbose=0)
    torch_args = tpresets.continuous_weibull(
        3, 5, [(8, "relu")], [(8, "relu")], log_scale_bounds=(-2.0, 4.0), log_shape_bounds=(-1, 1)
    )
    torch_model = tmodels.build_model(**{**torch_args, "architecture": architecture})
    convert._copy_weights(keras_model, torch_model)
    with torch.no_grad():
        actual = torch_model.eval()(torch.tensor(x), torch.tensor(a)).numpy()
    np.testing.assert_allclose(actual, keras_model.predict([x, a], verbose=0), rtol=1e-4, atol=1e-5)
