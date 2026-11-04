"""Test module for the half-Gaussian alpha schedule."""

import math

import numpy as np
import pytest

from ..keras import alpha_schedule


def test_formula_is_exact_zero_based_and_starts_at_alpha_start():
    """alpha(e) = end + (start - end) * exp(-c e^2), with alpha(0) == start exactly."""
    s = alpha_schedule.AlphaSchedule(alpha_start=10.0, alpha_end=1.0, decay_const=0.25)
    got = np.array([s.alpha(e) for e in range(10)])
    want = 1.0 + 9.0 * np.exp(-0.25 * np.arange(10) ** 2)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)
    assert got[0] == 10.0
    assert np.all(np.diff(got) < 0)


def test_zero_slope_at_epoch_zero():
    """The continuous curve has zero slope at e = 0, unlike exp(-c e)."""
    c, h = 0.25, 1e-6
    f = lambda e: alpha_schedule.gaussian_alpha(e, 10.0, 1.0, c)  # noqa: E731
    assert (f(h) - f(0)) / h == pytest.approx(0.0, abs=1e-4)


def test_increasing_schedule_is_allowed():
    """alpha_start < alpha_end ramps up along the same curve."""
    s = alpha_schedule.AlphaSchedule(alpha_start=1.0, alpha_end=5.0, decay_const=0.5)
    alphas = [s.alpha(e) for e in range(10)]
    assert alphas[0] == 1.0
    assert np.all(np.diff(alphas) > 0)
    assert alphas[-1] == pytest.approx(5.0, abs=1e-9)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"alpha_start": 0.0}, "alpha_start must be > 0"),
        ({"alpha_end": -1.0}, "alpha_end must be > 0"),
        ({"decay_const": 0.0}, "decay_const must be > 0"),
        ({"decay_const": float("nan")}, "finite"),
        ({"alpha_start": float("inf")}, "finite"),
        ({"alpha_start": "ten"}, "finite number"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"epsilon": 1.0}, "epsilon"),
    ],
)
def test_invalid_parameters_are_refused(kwargs, match):
    """Non-finite or non-positive parameters raise ValueError."""
    with pytest.raises(ValueError, match=match):
        alpha_schedule.AlphaSchedule(**kwargs)


def test_constant_schedule_has_no_warmup():
    """alpha_start == alpha_end is constant and settled from epoch 0."""
    s = alpha_schedule.AlphaSchedule(alpha_start=2.0, alpha_end=2.0)
    assert [s.alpha(e) for e in range(5)] == [2.0] * 5
    assert s.warmup_epochs == 0


@pytest.mark.parametrize("c, w", [(0.15, 7), (0.25, 6), (0.5, 4)])
def test_warmup_is_the_first_settled_epoch(c, w):
    """The residual is <= epsilon of the range at w and > epsilon at w - 1."""
    assert alpha_schedule.AlphaSchedule(decay_const=c).warmup_epochs == w
    assert math.exp(-c * w * w) <= 1e-3 < math.exp(-c * (w - 1) ** 2)


def test_invalid_epoch_is_refused():
    """Negative, fractional and boolean epochs raise ValueError."""
    s = alpha_schedule.AlphaSchedule()
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="epoch"):
            s.alpha(bad)


def test_epoch_cap_must_reach_the_settled_epoch():
    """An epoch cap that ends before alpha settles is refused."""
    s = alpha_schedule.AlphaSchedule(decay_const=0.25)
    with pytest.raises(ValueError, match="before alpha settles"):
        s.check_epoch_cap(6)
    s.check_epoch_cap(7)


def test_schedule_from_mapping_and_object():
    """Mappings and attribute objects both work; missing fields take the defaults."""
    from_dict = alpha_schedule.schedule_from_params({"alpha_start": 4.0, "decay_const": 0.5})
    assert from_dict == alpha_schedule.AlphaSchedule(alpha_start=4.0, decay_const=0.5)

    class Params:
        alpha_start = 3.0
        alpha_end = 1.5

    from_obj = alpha_schedule.schedule_from_params(Params())
    assert from_obj == alpha_schedule.AlphaSchedule(alpha_start=3.0, alpha_end=1.5)
