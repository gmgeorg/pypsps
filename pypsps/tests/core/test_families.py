"""Tests for the framework-free family specs: parameters, defaults, bounds and configs."""

import dataclasses
import json

import numpy as np
import pytest

from ...core import families

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


def _loss_family() -> families.LossFamily:
    """A `LossFamily` around a plain function."""
    return families.LossFamily(lambda y, p: p[:, 0], 1)


def test_response_column():
    """`response` is the first y_true column for every family, on NumPy input."""
    y2 = np.array([[3.0, 1.0, 7.0], [4.0, 0.0, 7.0]])
    for family in [*_ALL, _loss_family()]:
        np.testing.assert_array_equal(family.response(y2), [[3.0], [4.0]])


def test_event_indicator_is_none_without_a_backend_for_every_family():
    """The core specs have no tensor math: only a backend's survival family builds the mask."""
    y2 = np.array([[3.0, 1.0], [4.0, 0.0]])
    for family in [_FAMILIES["normal"], _FAMILIES["bernoulli"], _loss_family()]:
        assert family.event_indicator(y2) is None


def test_survival_family_and_is_binary():
    """SurvivalFamily is exactly Exponential and Weibull; only Bernoulli is binary."""
    for name, family in _FAMILIES.items():
        assert isinstance(family, families.SurvivalFamily) == name.startswith(
            ("exponential", "weibull")
        )
        assert family.n_true_cols == (2 if isinstance(family, families.SurvivalFamily) else 1)
        assert family.is_binary == (name == "bernoulli")
        assert family.is_censored == isinstance(family, families.SurvivalFamily)
    assert not _loss_family().is_binary


@pytest.mark.parametrize("family", _ALL, ids=_IDS)
def test_param_lookup(family):
    """`param` finds a parameter by name; an unknown name raises."""
    for p in family.params:
        assert family.param(p.name) is p or family.param(p.name) == p
    with pytest.raises(KeyError, match="no parameter"):
        family.param("nope")


@pytest.mark.parametrize("family", _ALL, ids=_IDS)
def test_raw_value_rejects_values_outside_the_support(family):
    """raw_value raises for a value on or outside the support's edge, and names the parameter."""
    for p in family.params:
        if p.bounds is not None:
            low, high = p.bounds
            outside = [low, high, high + 1.0]
        elif p.activation == "softplus":
            outside = [0.0, -1.0]
        elif p.activation == "sigmoid":
            outside = [0.0, 1.0]
        else:
            outside = []
        for v in outside:
            with pytest.raises(ValueError, match=p.name):
                p.raw_value(v)


def test_raw_value_is_the_inverse_of_each_activation():
    """linear is the identity, softplus is log(expm1(v)), sigmoid is the logit, and the
    midpoint of a bounded support is raw 0."""
    assert families.ParamSpec("x").raw_value(-2.5) == -2.5
    assert families.ParamSpec("x", activation="softplus").raw_value(2.0) == pytest.approx(
        np.log(np.expm1(2.0))
    )
    assert families.ParamSpec("x", activation="sigmoid").raw_value(0.25) == pytest.approx(
        np.log(0.25 / 0.75)
    )
    assert families.ParamSpec("x", bounds=(-1.0, 3.0)).raw_value(1.0) == pytest.approx(0.0)


def test_default_inits():
    """The Group C init table: midpoints for bounded survival params, 0.5 for prob."""
    inits = lambda f: {p.name: p.init for p in f.params}  # noqa: E731
    assert inits(families.Normal()) == {"loc": None, "scale": None}
    assert inits(families.Bernoulli()) == {"prob": 0.5}
    assert inits(families.Exponential(bounds=(0.0, 4.0))) == {"log_scale": 2.0}
    assert inits(families.Exponential()) == {"log_scale": None}
    assert inits(families.Exponential("rate")) == {"rate": None}
    assert inits(families.Weibull((0.0, 4.0), (-1.0, 2.0))) == {"log_scale": 2.0, "log_shape": 0.0}
    assert inits(families.Weibull((0.0, 4.0), (0.5, 1.5))) == {"log_scale": 2.0, "log_shape": 1.0}
    assert inits(families.Weibull((0.0, 4.0), (0.0, 1.0)))["log_shape"] == 0.5  # 0 not inside
    assert inits(families.Weibull()) == {"log_scale": None, "log_shape": None}


def test_init_not_in_config_or_equality():
    """An init is a training default: replace(p, init=...) changes neither eq nor config."""
    spec = families.Bernoulli().params[0]
    assert dataclasses.replace(spec, init=0.9) == spec
    assert "init" not in families.Weibull((0.0, 4.0), (-1.0, 1.0)).get_config()


def test_bounds_validation():
    """Bounds are a finite (min, max) pair with min < max; rate takes none."""
    with pytest.raises(ValueError, match=r"must be a \(min, max\) pair"):
        families.Weibull(log_scale_bounds=(1.0,))
    with pytest.raises(ValueError, match="must be finite with min < max"):
        families.Weibull(log_shape_bounds=(1.0, -1.0))
    with pytest.raises(ValueError, match="must be finite with min < max"):
        families.Exponential(bounds=(0.0, np.inf))
    with pytest.raises(ValueError, match="log_scale' or 'log_rate"):
        families.Exponential("rate", bounds=(0.0, 1.0))
    with pytest.raises(ValueError, match="activation must be one of"):
        families.ParamSpec("x", activation="relu")
    assert families.Exponential(bounds=[0, 1]).bounds == (0.0, 1.0)


def test_survival_family_check_max_time():
    """check_max_time raises on overflow and when a bound it needs is None."""
    weibull = families.Weibull(log_scale_bounds=(-3.0, 5.0), log_shape_bounds=(-1.0, 1.0))
    weibull.check_max_time(100.0)  # exponent e * (log 100 + 3) = 20.7
    with pytest.raises(ValueError, match="overflow"):
        families.Weibull((-8.0, 5.0), (-1.0, 1.5)).check_max_time(100.0)  # exponent 56
    with pytest.raises(ValueError, match="must be positive"):
        weibull.check_max_time(0.0)
    with pytest.raises(ValueError, match="needs"):
        families.Weibull(log_scale_bounds=(-3.0, 5.0)).check_max_time(100.0)

    families.Exponential(bounds=(-40.0, 5.0)).check_max_time(100.0)  # 4.6 + 40
    with pytest.raises(ValueError, match="overflow"):
        families.Exponential(bounds=(-50.0, 5.0)).check_max_time(100.0)
    families.Exponential("log_rate", bounds=(-5.0, 40.0)).check_max_time(100.0)
    with pytest.raises(ValueError, match="overflow"):
        families.Exponential("log_rate", bounds=(-5.0, 50.0)).check_max_time(100.0)
    for unbounded in (families.Exponential(), families.Exponential("rate")):
        with pytest.raises(ValueError, match="needs"):
            unbounded.check_max_time(100.0)
    assert not hasattr(families.Normal(), "check_max_time")


@pytest.mark.parametrize("family", _ALL, ids=_IDS)
def test_config_round_trip_and_equality(family):
    """Families serialize to JSON-able configs and rebuild equal."""
    config = family.get_config()
    rebuilt = type(family).from_config(json.loads(json.dumps(config)))
    assert type(rebuilt) is type(family) and rebuilt == family
    assert rebuilt.param_names == family.param_names


def test_exponential_parameterization_is_validated_and_names_the_parameter():
    """One public class; the parameterization is the one parameter's name."""
    for parameterization in families.EXPONENTIAL_PARAMETERIZATIONS:
        family = families.Exponential(parameterization)
        assert family.param_names == (parameterization,)
        assert set(family.get_config()) == {"parameterization", "bounds"}
    with pytest.raises(ValueError, match="log_scale.*log_rate.*rate"):
        families.Exponential("scale")


def test_lookup_by_name():
    """lookup maps a case-insensitive name onto a class and forwards the keywords."""
    table = {"normal": families.Normal, "weibull": families.Weibull}
    assert families.lookup(table, "Normal") == families.Normal()
    assert families.lookup(table, "weibull", log_scale_bounds=(0, 1)) == families.Weibull(
        (0.0, 1.0)
    )
    with pytest.raises(ValueError, match="known"):
        families.lookup(table, "gamma")
