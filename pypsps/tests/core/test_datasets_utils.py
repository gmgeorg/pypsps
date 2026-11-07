"""Test module for dataset utilities."""

import numpy as np
import pandas as pd
import pytest

from ...datasets.utils import identify_binary_features


def _frame():
    """Synthetic columns of 100 rows covering each case, with exact value counts."""
    n = 100
    return pd.DataFrame(
        {
            "b01": [0, 1] * 50,
            "const": np.ones(n),
            "with_nan": [0.0, 1.0, np.nan, 1.0] * 25,
            "quasi": [0.0] * 50 + [1.0] * 48 + [2.0, 3.0],  # top two values: 98% of rows
            "continuous": np.linspace(0.0, 1.0, n),
            "all_nan": np.full(n, np.nan),
            "labels": ["a"] * 60 + ["b"] * 39 + ["c"],  # strings are never quasi-binary
            "cat": pd.Categorical(["x"] * 50 + ["y"] * 49 + ["z"]),  # 99%
        }
    )


def test_strict_and_quasi_binary_columns():
    """Each column lands in the expected list; nulls are ignored when counting values."""
    strict, quasi = identify_binary_features(_frame())
    assert strict == ["b01", "const", "with_nan"]
    assert quasi == ["quasi", "cat"]


def test_threshold_controls_quasi_binary():
    """A stricter threshold drops the 98% column; 1.0 keeps only exact two-value columns."""
    _, quasi = identify_binary_features(_frame()[["quasi"]], threshold=0.99)
    assert quasi == []
    _, quasi = identify_binary_features(_frame()[["quasi"]], threshold=0.97)
    assert quasi == ["quasi"]


def test_invalid_threshold_is_refused():
    """threshold outside (0, 1] raises ValueError."""
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="threshold"):
            identify_binary_features(_frame(), threshold=bad)


def test_threshold_is_inclusive():
    """Top two values covering exactly ``threshold`` of the rows count as quasi-binary."""
    X = pd.DataFrame({"x": [0, 0, 1, 2]})  # top two cover exactly 3/4
    assert identify_binary_features(X, threshold=0.75) == ([], ["x"])
    assert identify_binary_features(X, threshold=0.76) == ([], [])
