"""Tests for the y_pred / y_true column layout (NumPy only; no framework)."""

import itertools

import numpy as np
import pandas as pd
import pytest

from ...core import families
from ...core.layout import ColumnLayout

_OUTCOME_PARAMS = [("loc", "scale"), ("log_scale",), ("log_scale", "log_shape")]
_TREATMENT_PARAMS = [("prob",), ("loc", "scale")]


def _y_pred(layout: ColumnLayout, n_rows: int = 4) -> np.ndarray:
    """A y_pred with distinct values, so any misplaced column shows up."""
    width = layout.y_pred_width()
    return np.arange(n_rows * width, dtype="float32").reshape(n_rows, width)


@pytest.mark.parametrize(
    "n_states, outcome_params, treatment_params, n_true",
    list(itertools.product([1, 2, 5], _OUTCOME_PARAMS, _TREATMENT_PARAMS, [1, 2])),
)
def test_encode_decode_round_trip(n_states, outcome_params, treatment_params, n_true):
    """decode_pred(encode(parts)) == parts and encode(decode_pred(y)) == y."""
    layout = ColumnLayout(outcome_params, treatment_params, n_true, n_states)
    y = _y_pred(layout)
    parts = layout.decode_pred(y)
    assert tuple(parts.outcome) == outcome_params
    assert tuple(parts.treatment) == treatment_params
    for block in [*parts.outcome.values(), parts.state_weights, *parts.treatment.values()]:
        assert block.shape == (4, n_states)

    encoded = layout.encode(parts.outcome, parts.state_weights, parts.treatment)
    assert isinstance(encoded, np.ndarray)
    np.testing.assert_array_equal(encoded, y)

    again = layout.decode_pred(encoded)
    for name in outcome_params:
        np.testing.assert_array_equal(again.outcome[name], parts.outcome[name])
    np.testing.assert_array_equal(again.state_weights, parts.state_weights)


def test_decode_follows_the_interleaved_per_state_order():
    """Each param block holds one column per state; weights sit between the blocks."""
    layout = ColumnLayout(("loc", "scale"), ("prob",), 1, n_states=2)
    y = np.array([[10.0, 11.0, 20.0, 21.0, 0.3, 0.7, 0.4, 0.6]])
    parts = layout.decode_pred(y)
    np.testing.assert_array_equal(parts.outcome["loc"], [[10.0, 11.0]])
    np.testing.assert_array_equal(parts.outcome["scale"], [[20.0, 21.0]])
    np.testing.assert_array_equal(parts.state_weights, [[0.3, 0.7]])
    np.testing.assert_array_equal(parts.treatment["prob"], [[0.4, 0.6]])


def test_decode_true_reads_exactly_one_treatment_column_and_ignores_trailing():
    """The treatment is the one column right after the outcome columns (1a80698)."""
    y_true = np.array([[5.0, 1.0, 0.0, 99.0, 98.0]])
    parts = ColumnLayout(("log_scale",), ("prob",), n_outcome_true_cols=2, n_states=1).decode_true(
        y_true
    )
    np.testing.assert_array_equal(parts.outcome, [[5.0, 1.0]])
    np.testing.assert_array_equal(parts.treatment, [[0.0]])

    with pytest.raises(ValueError, match="at least 3"):
        ColumnLayout(("log_scale",), ("prob",), 2, 1).decode_true(y_true[:, :2])


def test_width_mismatch_raises():
    """A known K pins the width; the error names the layout."""
    layout = ColumnLayout(("loc", "scale"), ("prob",), 1, n_states=3)
    with pytest.raises(ValueError, match="y_pred has 8 columns.*expects 12"):
        layout.decode_pred(np.zeros((2, 8)))


def test_n_states_is_required():
    """K is part of the layout: None or < 1 raises (no inference from the width since v0.5.0)."""
    for n_states in (None, 0):
        with pytest.raises(ValueError, match="n_states must be an int >= 1"):
            ColumnLayout(("loc", "scale"), ("prob",), 1, n_states)
    layout = ColumnLayout(("loc", "scale"), ("prob",), 1, n_states=3)
    assert layout.resolve_n_states(np.zeros((2, 12))) == 3


def test_encode_rejects_missing_or_misshaped_param():
    """encode checks the keys and every block's shape."""
    layout = ColumnLayout(("loc", "scale"), ("prob",), 1, n_states=2)
    block = np.zeros((3, 2))
    with pytest.raises(KeyError, match="outcome keys"):
        layout.encode({"loc": block}, block, {"prob": block})
    with pytest.raises(ValueError, match=r"outcome\['scale'\] must be \[N, 2\]"):
        layout.encode({"loc": block, "scale": np.zeros((3, 1))}, block, {"prob": block})
    with pytest.raises(ValueError, match="K=2"):
        layout.encode({"loc": block, "scale": block}, np.zeros((3, 3)), {"prob": block})
    with pytest.raises(ValueError, match="treatment block must have 2 columns"):
        layout.encode({"loc": block, "scale": block}, block, np.zeros((3, 4)))
    # The whole treatment block also works.
    out = layout.encode({"loc": block, "scale": block}, block, block)
    assert isinstance(out, np.ndarray) and out.shape == (3, 8)


def test_names_are_validated():
    """Parameter names must be non-empty and unique."""
    with pytest.raises(ValueError, match="unique"):
        ColumnLayout(("loc", "loc"), ("prob",), 1, 2)
    with pytest.raises(ValueError, match="empty"):
        ColumnLayout((), ("prob",), 1, 2)


def test_layout_from_families():
    """from_families takes the parameter names and y_true width from the families."""
    layout = ColumnLayout.from_families(3, families.Weibull(), families.Normal())
    assert layout.outcome_params == ("log_scale", "log_shape")
    assert layout.treatment_params == ("loc", "scale")
    assert (layout.n_outcome_true_cols, layout.n_states) == (2, 3)


@pytest.mark.parametrize("n_states", [1, 4])
def test_config_round_trip(n_states):
    """The layout serializes to plain lists and rebuilds equal."""
    layout = ColumnLayout(("log_scale", "log_shape"), ("loc", "scale"), 2, n_states)
    config = layout.get_config()
    assert config["outcome_params"] == ["log_scale", "log_shape"]
    assert ColumnLayout.from_config(config) == layout


# --- encode_true ---


@pytest.mark.parametrize("n_true", [1, 2])
@pytest.mark.parametrize("treatment_shape", ["flat", "column"])
def test_encode_true_stacks_outcome_then_treatment(n_true, treatment_shape):
    """y_true is the outcome columns, then the one treatment column, in that order."""
    layout = ColumnLayout(("loc", "scale"), ("prob",), n_true, n_states=2)
    outcome = np.arange(5 * n_true, dtype="float64").reshape(5, n_true) + 10
    if n_true == 1:
        outcome = outcome[:, 0]  # [N] is accepted when C == 1
    treatment = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    if treatment_shape == "column":
        treatment = treatment[:, None]

    y = layout.encode_true(outcome, treatment)

    assert isinstance(y, np.ndarray) and y.shape == (5, n_true + 1)
    np.testing.assert_array_equal(y[:, :n_true], np.reshape(outcome, (5, n_true)))
    np.testing.assert_array_equal(y[:, n_true], [0.0, 1.0, 1.0, 0.0, 1.0])


@pytest.mark.parametrize("n_true", [1, 2])
def test_encode_true_round_trips_decode_true(n_true):
    """decode_true(encode_true(o, t)) gives back (o, t)."""
    layout = ColumnLayout(("loc", "scale"), ("prob",), n_true, n_states=3)
    rng = np.random.RandomState(0)
    outcome, treatment = rng.normal(size=(6, n_true)), rng.randint(0, 2, (6, 1)).astype(float)

    parts = layout.decode_true(layout.encode_true(outcome, treatment))

    np.testing.assert_array_equal(parts.outcome, outcome)
    np.testing.assert_array_equal(parts.treatment, treatment)


def test_encode_true_rejects_the_wrong_outcome_width():
    """The family fixes the outcome width: too few and too many columns both raise.

    Two columns for a Normal outcome is survival data ``[time, event]`` read as ``[outcome,
    treatment]``; ``decode_true`` alone would train on it silently.
    """
    weibull = ColumnLayout.from_families(2, families.Weibull(), families.Bernoulli())
    with pytest.raises(ValueError, match=r"outcome has 1 column.*reads 2"):
        weibull.encode_true(np.zeros((4, 1)), np.zeros(4))
    with pytest.raises(ValueError, match="outcome has 1 column"):
        weibull.encode_true(np.zeros(4), np.zeros(4))

    normal = ColumnLayout.from_families(2, families.Normal(), families.Bernoulli())
    with pytest.raises(ValueError, match=r"outcome has 2 column.*reads 1"):
        normal.encode_true(np.zeros((4, 2)), np.zeros(4))
    with pytest.raises(ValueError, match="outcome must be"):
        normal.encode_true(np.zeros((4, 1, 1)), np.zeros(4))


def test_encode_true_rejects_a_wide_treatment_or_mismatched_rows():
    """The treatment is one column and has one row per outcome row."""
    layout = ColumnLayout.from_families(2, families.Normal(), families.Bernoulli())
    with pytest.raises(ValueError, match="treatment has 2 column"):
        layout.encode_true(np.zeros(4), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="treatment must be"):
        layout.encode_true(np.zeros(4), np.zeros((4, 1, 1)))
    with pytest.raises(ValueError, match=r"outcome has 4 rows but treatment has 3"):
        layout.encode_true(np.zeros(4), np.zeros(3))


def test_encode_true_accepts_pandas():
    """A DataFrame outcome and a Series treatment give a NumPy y_true."""
    layout = ColumnLayout.from_families(2, families.Weibull(), families.Bernoulli())
    outcome = pd.DataFrame({"time": [1.0, 2.0, 3.0], "event": [1.0, 0.0, 1.0]}, index=[7, 8, 9])
    treatment = pd.Series([0, 1, 1], index=[7, 8, 9], name="a")

    y = layout.encode_true(outcome, treatment)

    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y, [[1.0, 1.0, 0.0], [2.0, 0.0, 1.0], [3.0, 1.0, 1.0]])
    one_column = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    normal = ColumnLayout.from_families(2, families.Normal(), families.Bernoulli())
    assert normal.encode_true(one_column, treatment.to_frame()).shape == (3, 2)
