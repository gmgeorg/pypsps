"""Test module for the half-Gaussian alpha schedule."""

import math
import warnings

import numpy as np
import pytest

from ...core import alpha_schedule


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


# Values of the sigmoid schedule, frozen from the Keras callback before it moved to core.
_SIGMOID_EPOCHS = (0, 3, 8, 10, 12, 20, 35, 60)
_SIGMOID_CASES = {
    "decay": (
        dict(alpha_start=10.0, alpha_end=1.0, half_life_epoch=10, steepness=0.5),
        [9.939764341681437, 9.736189923237793, 7.579527207670044, 5.5, 3.420472792329956]
        + [1.0, 1.0, 1.0],
    ),
    "decay_without_clamp": (
        dict(
            alpha_start=10.0, alpha_end=1.0, half_life_epoch=10, steepness=0.5, clamp_threshold=0.0
        ),
        [9.939764341681437, 9.736189923237793, 7.579527207670044, 5.5, 3.420472792329956]
        + [1.0602356583185637, 1.0000335397535576, 1.0000000001249916],
    ),
    "increase": (
        dict(alpha_start=1.0, alpha_end=5.0, half_life_epoch=5, steepness=0.2),
        [2.0757656854799804, 2.605249359550192, 3.5826252249031816, 3.9242343145200196]
        + [4.208735554234327, 5.0, 5.0, 5.0],
    ),
    "constant": (dict(alpha_start=3.0, alpha_end=3.0), [3.0] * 8),
}


@pytest.mark.parametrize("case", list(_SIGMOID_CASES))
def test_sigmoid_schedule_matches_the_closed_form(case):
    """Decay, increase, constant and the clamp give the values the Keras callback gave."""
    kwargs, want = _SIGMOID_CASES[case]
    s = alpha_schedule.SigmoidAlphaSchedule(**kwargs)
    got = [s.alpha(e) for e in _SIGMOID_EPOCHS]
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    assert all(type(a) is float for a in got)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"steepness": 0.0}, "steepness must be positive"),
        ({"steepness": -1.0}, "steepness must be positive"),
        ({"half_life_epoch": -1}, "half_life_epoch must be non-negative"),
        ({"clamp_threshold": -0.1}, "clamp_threshold must be non-negative"),
    ],
)
def test_sigmoid_schedule_validates_its_arguments(kwargs, match):
    """The three argument checks keep their messages."""
    with pytest.raises(ValueError, match=match):
        alpha_schedule.SigmoidAlphaSchedule(**kwargs)


def test_sigmoid_clamp_threshold_defaults_to_a_tenth_of_the_range():
    """None resolves to 10% of the range, and to 0 for a constant schedule."""
    assert alpha_schedule.SigmoidAlphaSchedule(10.0, 1.0).clamp_threshold == pytest.approx(0.9)
    assert alpha_schedule.SigmoidAlphaSchedule(2.0, 2.0).clamp_threshold == 0.0
    assert (
        alpha_schedule.SigmoidAlphaSchedule(10.0, 1.0, clamp_threshold=0.5).clamp_threshold == 0.5
    )


@pytest.mark.parametrize("steepness, epoch", [(10.0, 100), (1.0, 5000), (0.1, 10**6)])
def test_sigmoid_schedule_does_not_overflow(steepness, epoch):
    """A steep or late schedule is exactly alpha_end, with no warning and no exception."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for start, end in ((10.0, 1.0), (1.0, 10.0)):
            s = alpha_schedule.SigmoidAlphaSchedule(
                alpha_start=start, alpha_end=end, half_life_epoch=10, steepness=steepness
            )
            assert s.alpha(epoch) == end
            assert s.alpha(-epoch) == pytest.approx(start)


def test_direction_properties_on_both_schedules():
    """is_decay / is_increase / is_constant follow alpha_start against alpha_end."""
    for cls in (alpha_schedule.SigmoidAlphaSchedule, alpha_schedule.AlphaSchedule):
        decay = cls(alpha_start=10.0, alpha_end=1.0)
        increase = cls(alpha_start=1.0, alpha_end=10.0)
        constant = cls(alpha_start=5.0, alpha_end=5.0)
        assert (decay.is_decay, decay.is_increase, decay.is_constant) == (True, False, False)
        assert (increase.is_decay, increase.is_increase, increase.is_constant) == (
            False,
            True,
            False,
        )
        assert (constant.is_decay, constant.is_increase, constant.is_constant) == (
            False,
            False,
            True,
        )


@pytest.mark.parametrize("c, want", [(0.25, 2), (0.5, 1), (0.01, 8), (0.15, 2)])
def test_gaussian_half_life_label(c, want):
    """The plot label is the epoch where exp(-c e^2) crosses 1/2, rounded."""
    s = alpha_schedule.AlphaSchedule(decay_const=c)
    assert s.half_life_epoch == round(math.sqrt(math.log(2.0) / c)) == want


def test_schedule_run_refuses_a_mid_fit_resume_and_resets_history():
    """With the guard, a first epoch other than 0 raises, and begin() clears the history."""
    schedule = alpha_schedule.AlphaSchedule()
    run = alpha_schedule.ScheduleRun(schedule, require_start_at_zero=True)
    run.begin()
    with pytest.raises(alpha_schedule.MidFitResumeError, match="epoch 3"):
        run.alpha_for(3)
    run.begin()
    assert [run.alpha_for(e) for e in range(3)] == [schedule.alpha(e) for e in range(3)]
    assert run.history == [schedule.alpha(e) for e in range(3)]
    run.begin()
    assert run.history == []
    assert run.alpha_for(0) == schedule.alpha(0)


def test_schedule_run_accumulates_history_without_the_guard():
    """Without the guard any first epoch is fine, and history accumulates across fits."""
    schedule = alpha_schedule.SigmoidAlphaSchedule(steepness=0.5)
    run = alpha_schedule.ScheduleRun(schedule)
    run.begin()
    assert run.alpha_for(4) == schedule.alpha(4)
    run.begin()
    run.alpha_for(0)
    assert run.history == [schedule.alpha(4), schedule.alpha(0)]


def test_mid_fit_resume_error_is_a_runtime_error():
    """It is one class, defined next to the schedules."""
    assert issubclass(alpha_schedule.MidFitResumeError, RuntimeError)


def test_plot_schedule_returns_a_figure():
    """Both schedules plot, labelled by direction."""
    plt = pytest.importorskip("matplotlib.pyplot")
    for schedule, title in (
        (alpha_schedule.SigmoidAlphaSchedule(10.0, 1.0), "Decay"),
        (alpha_schedule.SigmoidAlphaSchedule(1.0, 10.0), "Increase"),
        (alpha_schedule.AlphaSchedule(3.0, 3.0), "Constant"),
    ):
        fig = alpha_schedule.plot_schedule(schedule, max_epochs=30, figsize=(4, 3))
        assert title in fig.axes[0].get_title()
        assert len(fig.axes[0].lines[0].get_ydata()) == 30
        plt.close(fig)
