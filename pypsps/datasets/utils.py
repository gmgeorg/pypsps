"""Utilities for datasets."""

from typing import List, Tuple

import pandas as pd

__all__ = ["identify_binary_features"]


def identify_binary_features(
    X: pd.DataFrame, threshold: float = 0.95
) -> Tuple[List[str], List[str]]:
    """Finds binary and quasi-binary columns of ``X``.

    A column is strictly binary if it has 1 or 2 distinct non-null values. A numeric or
    categorical column with more is quasi-binary if its two most frequent values cover at
    least ``threshold`` of its non-null rows. All-null columns are in neither list.

    Returns:
        ``(strictly_binary, quasi_binary)``, lists of column names as strings.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    strictly_binary: List[str] = []
    quasi_binary: List[str] = []
    nunique = X.nunique(dropna=True)
    for col in X.columns:
        count = nunique[col]
        if count == 0:
            continue
        if count <= 2:
            strictly_binary.append(str(col))
            continue
        if pd.api.types.is_numeric_dtype(X[col]) or isinstance(X[col].dtype, pd.CategoricalDtype):
            top_two = float(X[col].value_counts(normalize=True).iloc[:2].sum())
            if top_two >= threshold:
                quasi_binary.append(str(col))
    return strictly_binary, quasi_binary
