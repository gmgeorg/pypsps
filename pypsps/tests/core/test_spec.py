"""Tests for `core.spec.make_spec`: every build argument is checked before any layer exists."""

import json

import numpy as np
import pytest

from ...core import families
from ...core.mixture import MixtureSpec
from ...core.spec import (
    HIDDEN_ACTIVATIONS,
    HeadSpec,
    NonPortableActivationWarning,
    Spec,
    make_spec,
    require_portable,
)

F = 4
_NORMAL, _BERNOULLI = families.Normal(), families.Bernoulli()


def _spec(**kwargs):
    """`make_spec` with valid defaults for a Normal outcome and a Bernoulli treatment."""
    args = dict(
        n_states=2,
        n_features=F,
        outcome_family=_NORMAL,
        treatment_family=_BERNOULLI,
        architecture="shared_trunk",
        predictive_state_hidden_layers=[],
        outcome_hidden_layers=[],
        outcome_heads=None,
        treatment_embedding_units=None,
        dropout_rate=0.0,
        l1=0.0,
        outcome_init=None,
        treatment_init=None,
        max_time=None,
    )
    args.update(kwargs)
    return make_spec(MixtureSpec, **args)


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        (dict(n_states=0), ValueError, "n_states"),
        (dict(n_features=0), ValueError, "n_features"),
        (dict(n_states=2.5), ValueError, "n_states"),
        (dict(architecture="deep"), ValueError, "shared_trunk"),
        (dict(outcome_heads={"rate": HeadSpec()}), KeyError, "loc"),
        (dict(outcome_heads={"loc": (16, "relu")}), TypeError, "HeadSpec"),
        (dict(outcome_init={"mean": 0.0}), KeyError, "scale"),
        (dict(treatment_init={"loc": 0.0}), KeyError, "prob"),
        (dict(outcome_init={"scale": -1.0}), ValueError, "scale"),
        (dict(treatment_init={"prob": 1.5}), ValueError, "prob"),
        (dict(max_time=10.0), ValueError, "survival"),
        (dict(treatment_family=families.Weibull()), ValueError, "one y_true column"),
        (dict(predictive_state_hidden_layers=[(0, "relu")]), ValueError, "units"),
        (dict(outcome_hidden_layers=[16]), ValueError, "pair"),
        (dict(outcome_hidden_layers=[(16, None)]), ValueError, "activation"),
        (dict(dropout_rate=1.0), ValueError, "dropout"),
        (dict(dropout_rate=1.5), ValueError, "dropout"),
        (dict(l1=-0.1), ValueError, "l1"),
        (dict(treatment_embedding_units=0), ValueError, "treatment_embedding_units"),
    ],
)
def test_make_spec_validates_arguments(kwargs, error, match):
    """Bad arguments raise the documented error; the defaults of `_spec` are valid."""
    _spec()
    with pytest.raises(error, match=match):
        _spec(**kwargs)


def test_make_spec_fills_in_heads_inits_and_the_treatment_init():
    """Unlisted outcome parameters get a default HeadSpec; inits override the family's."""
    spec = _spec(
        outcome_heads={"scale": HeadSpec(constant=True)},
        outcome_init={"scale": 2.0},
    )
    assert set(spec.heads) == {"loc", "scale"}
    assert spec.heads["scale"].constant and not spec.heads["loc"].constant
    assert {p.name: p.init for p in spec.outcome_params}["scale"] == 2.0
    # Bernoulli's default init is 0.5 for every state.
    np.testing.assert_allclose(spec.treatment_init, [[0.5, 0.5]])
    assert spec.n_states == 2


def test_make_spec_accepts_a_survival_max_time_only_within_the_float32_margin():
    """The family's overflow check runs through make_spec."""
    kwargs = dict(treatment_family=families.Normal(), max_time=100.0)
    with pytest.raises(ValueError, match="overflow"):
        _spec(outcome_family=families.Weibull((-10.0, 5.0), (-1.0, 3.0)), **kwargs)
    _spec(outcome_family=families.Weibull((-3.0, 5.0), (-1.0, 1.0)), **kwargs)


@pytest.mark.parametrize(
    "kwargs,field",
    [
        (
            dict(predictive_state_hidden_layers=[(4, "relu"), (4, "leaky_relu")]),
            "predictive_state_hidden_layers[1]",
        ),
        (dict(outcome_hidden_layers=[(4, "leaky_relu")]), "outcome_hidden_layers[0]"),
        (
            dict(outcome_heads={"scale": HeadSpec(hidden_layer=(4, "leaky_relu"))}),
            "outcome_heads['scale']",
        ),
    ],
)
def test_make_spec_rejects_a_non_portable_activation(kwargs, field):
    """An activation outside `HIDDEN_ACTIVATIONS` is rejected, naming the field and the list."""
    with pytest.raises(ValueError, match="leaky_relu") as caught:
        _spec(**kwargs)
    message = str(caught.value)
    assert field in message
    for name in HIDDEN_ACTIVATIONS:
        assert repr(name) in message or name in message


def test_every_portable_activation_is_accepted():
    """Each name of `HIDDEN_ACTIVATIONS` passes the default check."""
    for name in HIDDEN_ACTIVATIONS:
        _spec(
            predictive_state_hidden_layers=[(2, name)],
            outcome_hidden_layers=[(2, name)],
            outcome_heads={"loc": HeadSpec(hidden_layer=(2, name))},
        )


def test_a_constant_head_ignores_its_hidden_layer():
    """A constant head has no hidden layer, so its activation is not checked."""
    _spec(outcome_heads={"scale": HeadSpec(hidden_layer=(4, "leaky_relu"), constant=True)})


def test_make_spec_uses_the_given_activation_check():
    """A custom check sees every hidden activation exactly once, and replaces the default."""
    seen = []
    _spec(
        predictive_state_hidden_layers=[(4, "relu"), (4, "leaky_relu")],
        outcome_hidden_layers=[(4, "mish")],
        outcome_heads={"loc": HeadSpec(hidden_layer=(4, "tanh")), "scale": HeadSpec(None)},
        check_activation=lambda name, what: seen.append((name, what)),
    )
    names = sorted(name for name, _ in seen)
    assert names == ["leaky_relu", "mish", "relu", "tanh"]
    assert all(isinstance(what, str) and what for _, what in seen)


def test_require_portable_names_what_and_the_portable_names():
    """`require_portable` is a no-op for a portable name and a ValueError otherwise."""
    require_portable("relu", "x")
    with pytest.raises(ValueError, match=r"x\[2\].*'foo'"):
        require_portable("foo", "x[2]")


def test_the_non_portable_warning_is_a_user_warning():
    """Users can filter `NonPortableActivationWarning` by class."""
    assert issubclass(NonPortableActivationWarning, UserWarning)


_WEIBULL = families.Weibull(log_scale_bounds=(-2.0, 4.0), log_shape_bounds=(-1.0, 1.0))
_HEADS = {
    "normal": [None, {"loc": HeadSpec(None), "scale": HeadSpec(constant=True, per_state=False)}],
    "weibull": [
        None,
        {"log_scale": HeadSpec((3, "tanh"), per_state=False), "log_shape": HeadSpec(None)},
    ],
}


@pytest.mark.parametrize("architecture", ["shared_trunk", "gated"])
@pytest.mark.parametrize("hidden", [[], [(4, "relu"), (3, "swish")]], ids=["flat", "deep"])
@pytest.mark.parametrize("outcome", ["normal", "weibull"])
@pytest.mark.parametrize("heads", [0, 1], ids=["default-heads", "custom-heads"])
@pytest.mark.parametrize("treatment_init", [None, {"prob": [0.3, 0.7]}], ids=["no-init", "init"])
def test_spec_config_round_trips(architecture, hidden, outcome, heads, treatment_init):
    """``get_config`` is JSON, leaves out the mixture, and ``from_config`` inverts it."""
    spec = _spec(
        outcome_family=_WEIBULL if outcome == "weibull" else _NORMAL,
        architecture=architecture,
        predictive_state_hidden_layers=hidden,
        outcome_hidden_layers=hidden,
        outcome_heads=_HEADS[outcome][heads],
        treatment_embedding_units=None if heads else 8,
        dropout_rate=0.1,
        l1=0.01,
        treatment_init=treatment_init,
        max_time=50.0 if outcome == "weibull" else None,
    )
    config = spec.get_config()
    assert "mixture" not in config
    assert json.loads(json.dumps(config)) == config
    rebuilt = Spec.from_config(json.loads(json.dumps(config)), spec.mixture)
    assert rebuilt.get_config() == config
    assert rebuilt.mixture is spec.mixture
    assert rebuilt.max_time == spec.max_time
    assert rebuilt.heads == spec.heads
    assert rebuilt.architecture == architecture
    np.testing.assert_array_equal(rebuilt.treatment_init, spec.treatment_init)


def test_spec_from_config_validates_like_arguments():
    """A config is checked as ``make_spec`` checks arguments, so an edited one can't skip it."""
    spec = _spec(outcome_hidden_layers=[(4, "relu")])
    config = spec.get_config()
    for edit, match in (
        ({"dropout_rate": 1.0}, "dropout"),
        ({"architecture": "deep"}, "shared_trunk"),
        ({"outcome_hidden_layers": [[4, "leaky_relu"]]}, "leaky_relu"),
        ({"predictive_state_hidden_layers": [[0, "relu"]]}, "units"),
    ):
        with pytest.raises(ValueError, match=match):
            Spec.from_config({**config, **edit}, spec.mixture)


def test_spec_from_config_uses_the_given_activation_check():
    """The check of the backend that loads the config replaces the default."""
    spec = _spec(outcome_hidden_layers=[(4, "relu")])
    config = {**spec.get_config(), "outcome_hidden_layers": [[4, "leaky_relu"]]}
    seen = []
    rebuilt = Spec.from_config(
        config, spec.mixture, check_activation=lambda name, what: seen.append(name)
    )
    assert seen.count("leaky_relu") == 1
    assert rebuilt.outcome_hidden_layers == ((4, "leaky_relu"),)


def test_spec_keeps_max_time():
    """``max_time`` is validated and kept; a config without the key reads as None."""
    spec = _spec(outcome_family=_WEIBULL, max_time=50.0)
    assert spec.max_time == 50.0
    assert spec.get_config()["max_time"] == 50.0
    assert _spec().max_time is None
    config = spec.get_config()
    del config["max_time"]
    assert Spec.from_config(config, spec.mixture).max_time is None
    with pytest.raises(ValueError, match="overflow|max_time"):
        Spec.from_config({**spec.get_config(), "max_time": 1e30}, spec.mixture)
