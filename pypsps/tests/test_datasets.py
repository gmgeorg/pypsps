"""Module for testing datasets modules."""

import pytest
import numpy as np
import pandas as pd

from .. import datasets


def test_kang_schafer():
    np.random.seed(123)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=1000)
    df = ks_data.to_data_frame()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1000
    assert ks_data.n_samples == 1000
    assert ks_data.naive_ate() == pytest.approx(-9.6, 0.1)


@pytest.mark.parametrize(
    "association,expected_ate", [("none", 2.04), ("moderate", 1.76), ("strong", 1.65)]
)
def test_lunceford_davidian(association, expected_ate):
    np.random.seed(123)
    ld_data = datasets.LuncefordDavidian(association=association).sample(n_samples=100)
    df = ld_data.to_data_frame()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 100
    assert ld_data.n_samples == 100
    assert ld_data.n_features == 6
    assert ld_data.n_treatments == 1
    assert ld_data.n_outcomes == 1

    assert ld_data.naive_ate() == pytest.approx(expected_ate, 0.01)
    pd.testing.assert_index_equal(ld_data.naive_ute().index, df.index)
