"""Module for testing datasets modules."""

import numpy as np
import pandas as pd
import pytest

from ... import datasets
from ...core import families
from ...core.layout import ColumnLayout


def test_kang_schafer():
    ks_data = datasets.KangSchafer(true_ate=10, seed=123).sample(n_samples=1000)
    df = ks_data.to_data_frame()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1000
    assert ks_data.n_samples == 1000
    assert ks_data.naive_ate() == pytest.approx(-9.41, 0.1)


@pytest.mark.skip(reason="seed issue on github CI/CD")
@pytest.mark.parametrize(
    "association,expected_ate", [("none", 1.95), ("moderate", 1.64), ("strong", 1.44)]
)
def test_lunceford_davidian(association, expected_ate):
    ld_data = datasets.LuncefordDavidian(association=association, seed=123).sample(n_samples=100)
    df = ld_data.to_data_frame()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 100
    assert ld_data.n_samples == 100
    assert ld_data.n_features == 6
    assert ld_data.n_treatments == 1
    assert ld_data.n_outcomes == 1

    assert ld_data.naive_ate() == pytest.approx(expected_ate, 0.01)
    pd.testing.assert_index_equal(ld_data.naive_ute().index, df.index)


def _layout(outcome_family, n_states=2):
    """The layout of a model with this outcome family and a Bernoulli treatment."""
    return ColumnLayout.from_families(n_states, outcome_family, families.Bernoulli())


def test_to_keras_inputs_outputs_without_a_layout_is_unchanged():
    """The targets are the outcome columns then the treatment, with no width check."""
    data = datasets.KangSchafer(true_ate=10, seed=123).sample(n_samples=20)
    inputs, outputs = data.to_keras_inputs_outputs()
    np.testing.assert_array_equal(inputs[0], data.features.values)
    np.testing.assert_array_equal(inputs[1], data.treatments.values)
    np.testing.assert_array_equal(
        outputs, np.hstack([data.outcomes.values, data.treatments.values])
    )


def test_to_keras_inputs_outputs_checks_width_with_a_layout():
    """With a layout the targets go through ``encode_true``: same array, plus the width check."""
    data = datasets.KangSchafer(true_ate=10, seed=123).sample(n_samples=20)
    _, plain = data.to_keras_inputs_outputs()

    inputs, outputs = data.to_keras_inputs_outputs(layout=_layout(families.Normal()))
    np.testing.assert_array_equal(outputs, plain)
    np.testing.assert_array_equal(inputs[0], data.features.values)

    with pytest.raises(ValueError, match=r"outcome has 1 column.*reads 2"):
        data.to_keras_inputs_outputs(layout=_layout(families.Weibull()))
