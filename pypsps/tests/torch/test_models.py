"""Tests for `pypsps.torch.models` and `pypsps.torch.layers` (no TensorFlow needed)."""

import numpy as np
import pytest
import torch

from ...core.spec import HeadSpec
from ...torch import families, layers, losses, models, presets


def _inputs(n=16, n_features=5, seed=0):
    rng = np.random.RandomState(seed)
    x = torch.tensor(rng.normal(size=(n, n_features)), dtype=torch.float32)
    a = torch.tensor(rng.randint(0, 2, n), dtype=torch.float32)
    return x, a


@pytest.mark.parametrize("architecture", ["gated", "shared_trunk"])
@pytest.mark.parametrize("n_states", [1, 3])
def test_forward_shape_and_state_weights_sum_to_one(architecture, n_states):
    """y_pred has the layout's width; its state weights are a simplex."""
    model = models.build_model(
        n_states, 5, families.Normal(), families.Bernoulli(), architecture=architecture
    )
    x, a = _inputs()
    y_pred = model(x, a)
    assert y_pred.shape == (16, model.layout.y_pred_width())
    weights = model.layout.decode_pred(y_pred).state_weights
    np.testing.assert_allclose(weights.sum(dim=1).detach().numpy(), 1.0, rtol=1e-5)


def test_treatment_column_shape_does_not_matter():
    """``treatment`` may be ``[N]`` or ``[N, 1]``."""
    model = models.build_model(2, 5, families.Normal(), families.Bernoulli()).eval()
    x, a = _inputs()
    np.testing.assert_array_equal(model(x, a).detach(), model(x, a[:, None]).detach())


def test_keras_conventions():
    """Batch norm and dense layers follow Keras: eps 1e-3, momentum 0.01, Glorot, zero bias."""
    model = models.build_model(2, 5, families.Normal(), families.Bernoulli())
    assert model.features_bn.eps == 1e-3
    assert model.features_bn.momentum == pytest.approx(0.01)
    dense = model.predictive_state_dense_0
    assert torch.all(dense.bias == 0)
    limit = np.sqrt(6.0 / (dense.in_features + dense.out_features))
    assert dense.weight.abs().max() <= limit + 1e-6
    # the state head starts uniform, the treatment head at the family's defaults
    weights = model.layout.decode_pred(model.eval()(*_inputs())).state_weights
    np.testing.assert_allclose(weights.detach().numpy(), 0.5, atol=1e-6)


def test_dropout_only_in_train_mode():
    """Eval mode is deterministic; train mode with dropout is not."""
    model = models.build_model(2, 5, families.Normal(), families.Bernoulli(), dropout_rate=0.5)
    x, a = _inputs(64)
    model.eval()
    np.testing.assert_array_equal(model(x, a).detach(), model(x, a).detach())
    model.train()
    assert not torch.equal(model(x, a), model(x, a))


def test_head_specs_shape_the_network():
    """Constant, shared and hidden-layer heads create the expected modules."""
    model = models.build_model(
        3,
        5,
        families.Normal(),
        families.Bernoulli(),
        outcome_heads={"loc": HeadSpec(hidden_layer=(6, "relu")), "scale": HeadSpec(constant=True)},
    )
    assert isinstance(model.loc_hidden_state_2, layers.Dense)
    assert isinstance(model.scale_raw_state_1, layers.BiasOnly)
    shared = models.build_model(
        3,
        5,
        families.Normal(),
        families.Bernoulli(),
        outcome_heads={"loc": HeadSpec(per_state=False)},
    )
    assert hasattr(shared, "loc_raw_shared") and not hasattr(shared, "loc_raw_state_0")


def test_outcome_init_sets_the_starting_value():
    """``outcome_init`` starts the (constant) scale head at the given value."""
    model = models.build_model(
        2,
        5,
        families.Normal(),
        families.Bernoulli(),
        outcome_heads={"scale": HeadSpec(constant=True, per_state=False)},
        outcome_init={"scale": 2.0},
        dropout_rate=0.0,
    ).eval()
    scale = model.layout.decode_pred(model(*_inputs())).outcome["scale"]
    np.testing.assert_allclose(scale.detach().numpy(), 2.0, rtol=1e-4)


def test_invalid_arguments_raise_before_any_layer_is_built(monkeypatch):
    """The shared checks (``tests/core/test_spec.py``) run before the first layer is created."""

    def fail(*args, **kwargs):
        raise AssertionError("a layer was created")

    monkeypatch.setattr(layers.Dense, "__init__", fail)
    for kwargs in ({"n_states": 0}, {"architecture": "nope"}, {"dropout_rate": 1.5}):
        args = {"n_states": 2, "n_features": 5, **kwargs}
        with pytest.raises(ValueError):
            models.build_model(
                args.pop("n_states"),
                args.pop("n_features"),
                families.Normal(),
                families.Bernoulli(),
                **args,
            )


def test_unknown_activation_in_a_hidden_layer_raises():
    """An unknown activation name is rejected (while the layer is built, for now)."""
    with pytest.raises(ValueError, match="nope"):
        models.build_model(
            2,
            5,
            families.Normal(),
            families.Bernoulli(),
            predictive_state_hidden_layers=[(8, "nope")],
        )


def test_unknown_activation_names_the_known_ones():
    with pytest.raises(ValueError, match="relu"):
        layers.activation("nope")


def test_softclip_bounds_and_inverse():
    clip = layers.SoftClip(-1.0, 3.0)
    out = clip(torch.tensor([-50.0, 0.0, 50.0]))
    assert out[0] >= -1.0 and out[2] <= 3.0
    assert clip(torch.tensor(clip.inverse(1.5))).item() == pytest.approx(1.5, abs=1e-5)
    with pytest.raises(ValueError):
        layers.SoftClip(2.0, 1.0)


def test_bounded_survival_head_stays_inside_the_bounds():
    """A bounded Weibull head stays in its bounds whatever the inputs."""
    model = models.build_model(
        2,
        5,
        families.Weibull(log_scale_bounds=(-1.0, 2.0), log_shape_bounds=(-0.5, 0.5)),
        families.Bernoulli(),
    ).eval()
    x, a = _inputs()
    outcome = model.layout.decode_pred(model(x * 100, a)).outcome
    assert outcome["log_scale"].min() >= -1.0 and outcome["log_scale"].max() <= 2.0
    assert outcome["log_shape"].min() >= -0.5 and outcome["log_shape"].max() <= 0.5


def test_kernel_l1_penalty():
    """The L1 penalty is zero without l1 and l1 * sum|kernel| over the hidden layers with it."""
    plain = models.build_model(2, 5, families.Normal(), families.Bernoulli())
    assert plain.kernel_l1_penalty().item() == 0.0
    model = models.build_model(2, 5, families.Normal(), families.Bernoulli(), l1=0.1)
    expected = 0.1 * sum(m.weight.abs().sum().item() for m in model._l1_modules)
    assert model.kernel_l1_penalty().item() == pytest.approx(expected, rel=1e-5)


def test_propensity_state_conditional_means_with_init():
    """A treatment init gives each state's mean treatment back."""
    model = models.build_model(
        2,
        5,
        families.Normal(),
        families.Bernoulli(),
        treatment_init={"prob": [0.2, 0.7]},
    )
    np.testing.assert_allclose(
        model.propensity_state_conditional_means().detach().numpy().ravel(),
        [0.2, 0.7],
        rtol=1e-4,
    )


def test_config_round_trip():
    """``from_config(get_config())`` rebuilds the same architecture."""
    model = models.build_model(
        3,
        5,
        families.Weibull(log_scale_bounds=(0.0, 4.0), log_shape_bounds=(-1.0, 1.0)),
        families.Bernoulli(),
        architecture="shared_trunk",
        outcome_heads={"log_scale": HeadSpec(hidden_layer=(4, "tanh"))},
        l1=0.01,
        treatment_init={"prob": [0.3, 0.5, 0.7]},
    )
    rebuilt = models.PSPSModel.from_config(model.get_config())
    assert rebuilt.get_config() == model.get_config()
    assert [k for k in rebuilt.state_dict()] == [k for k in model.state_dict()]


@pytest.mark.parametrize(
    "name, make",
    [
        ("toy", lambda: presets.toy(3, 5)),
        ("binary_normal", lambda: presets.binary_normal(3, 5, [(8, "relu")], [(8, "relu")])),
        (
            "binary_exponential",
            lambda: presets.binary_exponential(3, 5, [(8, "relu")], [(8, "relu")]),
        ),
        (
            "continuous_weibull",
            lambda: presets.continuous_weibull(
                3, 5, [(8, "relu")], [(8, "relu")], (-2.0, 4.0), (-1.0, 1.0)
            ),
        ),
    ],
)
def test_presets_build_and_carry_fit_arguments(name, make):
    """A preset is the build_model arguments plus ``fit_kwargs`` (alpha, regularizer, ...)."""
    preset = make()
    model = models.build_model(**preset)
    assert model.layout.y_pred_width() > 0
    assert {"alpha", "predictive_states_regularizer"} <= set(preset.fit_kwargs)
    loss = losses.CausalLoss(
        model.mixture,
        alpha=preset.fit_kwargs["alpha"],
        predictive_states_regularizer=preset.fit_kwargs["predictive_states_regularizer"],
    )
    assert loss.alpha == preset.fit_kwargs["alpha"]
