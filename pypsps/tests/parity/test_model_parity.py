"""A Keras model copied into PyTorch gives the same ``y_pred`` in eval mode."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("torch")

import torch  # noqa: E402

from ...keras import families as kf  # noqa: E402
from ...keras import models as kmodels  # noqa: E402
from ...keras import presets as kpresets  # noqa: E402
from ...torch import models as tmodels  # noqa: E402
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
