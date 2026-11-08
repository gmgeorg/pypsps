"""Tests for the argument logic of the presets, run against core families only."""

import numpy as np
import pytest

from ...core import families, presets
from ...core.mixture import MixtureSpec
from ...core.spec import make_spec

K, F = 3, 4
_HIDDEN = dict(predictive_state_hidden_layers=[(4, "relu")], outcome_hidden_layers=[(4, "relu")])


class _Recorder:
    """A stand-in regularizer constructor that records its arguments."""

    def __init__(self, name):
        self.name = name

    def __call__(self, **kwargs):
        return (self.name, kwargs)


def _backend():
    """A backend of core families and recording regularizers: no framework."""
    return presets.Backend(families, _Recorder("df"), _Recorder("uniform"))


def _continuous(distribution="weibull", **kwargs):
    """The build arguments of `continuous_survival` with small layers and the test bounds."""
    if distribution == "weibull":
        kwargs.setdefault("log_shape_bounds", (-2.0, 2.0))
    build, _ = presets.continuous_survival(
        _backend(),
        distribution,
        n_states=K,
        n_features=F,
        log_scale_bounds=(-3.0, 6.0),
        **_HIDDEN,
        **kwargs,
    )
    return build


def _make_spec(build):
    """`make_spec` on a preset's build arguments."""
    return make_spec(MixtureSpec, **build)


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
        presets.continuous_survival(
            _backend(), "exponential", K, F, log_scale_bounds=None, **_HIDDEN
        )
    with pytest.raises(ValueError, match="init_scale"):
        _continuous("weibull", init_scale=0.0)
    with pytest.raises(ValueError, match="treatment_init_values"):
        _continuous("weibull", treatment_init_values=[0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="lie in"):
        _make_spec(_continuous("weibull", init_scale=np.exp(7.0)))


def test_continuous_preset_max_time_rejects_overflow_with_a_margin():
    """max_time rejects bounds whose worst corner overflows the Weibull NLL, and also bounds
    within the margin (exponent above 50 but still finite), and accepts bounds well inside.

    The numeric cross-check against the Keras NLL is in ``tests/keras/test_presets.py``.
    """
    unsafe = dict(log_scale_bounds=(-10.0, 5.0), log_shape_bounds=(-1.0, 3.0))  # exponent 293
    margin = dict(log_scale_bounds=(-8.0, 5.0), log_shape_bounds=(-1.0, 1.5))  # exponent 56
    safe = dict(log_scale_bounds=(-3.0, 5.0), log_shape_bounds=(-1.0, 1.0))  # exponent 21
    kwargs = dict(n_states=2, n_features=F, max_time=100.0, **_HIDDEN)

    def build(**bounds):
        arguments, _ = presets.continuous_survival(_backend(), "weibull", **bounds, **kwargs)
        return arguments

    for rejected in (unsafe, margin):
        with pytest.raises(ValueError, match="overflow"):
            _make_spec(build(**rejected))
    _make_spec(build(**safe))


def test_presets_return_fresh_dicts():
    """A preset is data: a new dict per call, with equal keys."""
    first, second = _continuous(), _continuous()
    assert first is not second and first.keys() == second.keys()
    assert first["outcome_heads"] is not second["outcome_heads"]


def test_continuous_preset_maps_its_arguments():
    """init_scale, treatment_init_values and separate_shape_per_state reach the arguments."""
    arguments = _continuous(
        init_scale=5.0, treatment_init_values=[0.5, 2.0], separate_shape_per_state=False
    )
    assert arguments["outcome_init"] == {"log_scale": pytest.approx(np.log(5.0))}
    assert arguments["treatment_init"] == {"loc": 0.5, "scale": 2.0}
    assert arguments["outcome_heads"]["log_shape"].per_state is False
    assert arguments["architecture"] == "gated"
    assert isinstance(arguments["outcome_family"], families.Weibull)
    assert isinstance(arguments["treatment_family"], families.Normal)
