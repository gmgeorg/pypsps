"""Tests for the framework-free checkpoint selector and the default callback description."""

import math

import pytest

from ...core import checkpoint

NAN = float("nan")
INF = float("inf")


def _drive(series, *, start=6, patience=50, min_delta=0.0, alphas=None):
    """Feed ``series`` to a selector as epoch logs until it says stop; return it and the steps."""
    sel = checkpoint.CheckpointSelector(
        "val_m", patience=patience, start_from_epoch=start, min_delta=min_delta
    )
    sel.begin()
    steps = []
    for e, v in enumerate(series):
        alpha = None if alphas is None else alphas[e]
        steps.append(sel.observe(e, {"val_m": v}, alpha=alpha))
        if steps[-1].stop:
            break
    return sel, steps


def test_ignores_epochs_before_start():
    """Better values before start_from_epoch are never selected."""
    sel, steps = _drive([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 5.0, 4.0, 3.5, 3.9])
    assert sel.finish() == 8
    assert (sel.selected_epoch, sel.selected_value) == (8, 3.5)
    assert [t["eligible"] for t in sel.trace] == [False] * 6 + [True] * 4
    assert [s.improved for s in steps] == [False] * 6 + [True, True, True, False]


def test_skips_nan_and_inf_and_ties_go_to_earliest():
    """A NaN at the first eligible epoch is not selected; ties keep the earlier epoch."""
    sel, _ = _drive([1, 1, 1, 1, 1, 1, NAN, 3.0, 2.0, 2.0, INF])
    assert sel.finish() == 8


def test_min_delta_requires_a_real_improvement():
    """An improvement smaller than min_delta does not move the selection."""
    sel, _ = _drive([9] * 6 + [3.0, 2.95, 2.5], min_delta=0.1)
    assert sel.finish() == 8
    sel, _ = _drive([9] * 6 + [3.0, 2.95, 2.5], min_delta=-0.1)
    assert sel.min_delta == 0.1


def test_patience_counts_from_the_first_eligible_epoch():
    """Patience runs over eligible epochs only, and the stop step is the one that runs out."""
    sel, steps = _drive([9] * 6 + [3.0, 4.0, 4.0, 4.0, 1.0], patience=3)
    assert (sel.stopped_epoch, len(steps), steps[-1].stop) == (9, 10, True)
    assert not any(s.stop for s in steps[:-1])
    assert sel.finish() == 6


def test_a_non_finite_epoch_counts_against_patience():
    """A NaN after the best epoch is a wasted epoch, like any other non-improvement."""
    sel, steps = _drive([9] * 2 + [1.0, NAN, NAN, 0.5], start=0, patience=2)
    assert (sel.stopped_epoch, len(steps)) == (4, 5)
    assert sel.finish() == 2


def test_raises_when_no_epoch_is_eligible():
    """No finite value at or after start_from_epoch raises, naming the monitored values."""
    sel, _ = _drive([1.0] * 6 + [NAN] * 3)
    with pytest.raises(checkpoint.NoEligibleCheckpointError, match="nan"):
        sel.finish()
    sel, _ = _drive([NAN, NAN], start=0)
    with pytest.raises(checkpoint.NoEligibleCheckpointError):
        sel.finish()
    assert sel.selected_epoch is None


def test_raises_on_a_missing_monitor():
    """A monitor that is not in the logs raises KeyError listing the available keys."""
    sel = checkpoint.CheckpointSelector("val_missing", patience=2)
    sel.begin()
    with pytest.raises(KeyError, match="val_missing.*loss"):
        sel.observe(0, {"loss": 1.0})
    with pytest.raises(KeyError, match="val_missing"):
        sel.observe(0, None)


def test_validates_arguments():
    """patience < 1 and a negative start_from_epoch raise ValueError."""
    with pytest.raises(ValueError, match="patience"):
        checkpoint.CheckpointSelector("val_m", patience=0)
    with pytest.raises(ValueError, match="start_from_epoch"):
        checkpoint.CheckpointSelector("val_m", patience=1, start_from_epoch=-1)


def test_trace_records_epoch_value_eligibility_and_alpha():
    """Each observed epoch leaves one trace row; begin() clears the state of a previous fit."""
    sel, _ = _drive([4.0, 3.0, NAN, 2.0], start=1, alphas=[None, 1.5, 1.0, 1.0])
    assert sel.trace[0] == {"epoch": 0, "monitored": 4.0, "eligible": False, "alpha": None}
    assert sel.trace[1] == {"epoch": 1, "monitored": 3.0, "eligible": True, "alpha": 1.5}
    assert sel.trace[2]["eligible"] is False and math.isnan(sel.trace[2]["monitored"])
    assert sel.trace[3]["alpha"] == 1.0
    assert (sel.best, sel.best_epoch, sel.wait) == (2.0, 3, 0)
    sel.begin()
    assert (sel.trace, sel.best, sel.best_epoch, sel.wait, sel.stopped_epoch) == (
        [],
        None,
        None,
        0,
        0,
    )
    assert (sel.selected_epoch, sel.selected_value) == (None, None)


def test_default_callback_args():
    """One description of the three default callbacks, for both backends to build from."""
    args = checkpoint.default_callback_args("val_m", 10)
    assert args == {
        "early_stopping": {"monitor": "val_m", "patience": 10, "verbose": 0},
        "reduce_lr": {
            "monitor": "val_m",
            "factor": 0.2,
            "patience": 5,
            "min_lr": 1e-6,
            "verbose": 0,
        },
        "alpha_schedule": {
            "alpha_start": 2.0,
            "alpha_end": 1.0,
            "half_life_epoch": 10,
            "steepness": 0.1,
            "verbose": False,
        },
    }
    custom = checkpoint.default_callback_args(
        "loss",
        7,
        verbose=True,
        alpha_start=8.0,
        alpha_end=2.0,
        half_life_epoch=4,
        steepness=0.2,
    )
    assert custom["early_stopping"] == {"monitor": "loss", "patience": 7, "verbose": 1}
    assert custom["reduce_lr"]["patience"] == 3 and custom["reduce_lr"]["verbose"] == 1
    assert custom["alpha_schedule"] == {
        "alpha_start": 8.0,
        "alpha_end": 2.0,
        "half_life_epoch": 4,
        "steepness": 0.2,
        "verbose": True,
    }
