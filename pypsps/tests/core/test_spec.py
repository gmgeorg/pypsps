"""Tests for `core.spec.make_spec`: every build argument is checked before any layer exists."""

import numpy as np
import pytest

from ...core import families
from ...core.mixture import MixtureSpec
from ...core.spec import (
    HIDDEN_ACTIVATIONS,
    HeadSpec,
    NonPortableActivationWarning,
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
