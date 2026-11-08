"""Tests for the framework-free helpers in `pypsps.utils`."""

import numpy as np
import pandas as pd
import pytest

from ... import utils
from ...core import families
from ...core.layout import ColumnLayout


def _data(n=6, survival=False):
    """Features, a binary treatment and an outcome (one column, or ``[time, event]``)."""
    rng = np.random.RandomState(0)
    features = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    treatments = pd.DataFrame({"t": rng.randint(0, 2, n)})
    columns = {"time": rng.uniform(1, 5, n), "event": rng.randint(0, 2, n)}
    outcomes = pd.DataFrame(columns if survival else {"y": rng.normal(size=n)})
    return features, treatments, outcomes


def test_prepare_keras_inputs_outputs_without_a_layout_is_unchanged():
    """Inputs are float32 features and the treatment; outputs are outcome then treatment."""
    features, treatments, outcomes = _data()
    (x, t), y = utils.prepare_keras_inputs_outputs(features, treatments, outcomes)
    assert x.dtype == np.float32
    np.testing.assert_array_equal(t, treatments.values)
    np.testing.assert_array_equal(
        y, np.hstack([outcomes.values.astype("float32"), treatments.values])
    )
    # Without a layout there is no width check: a survival outcome is accepted as given.
    _, y = utils.prepare_keras_inputs_outputs(*_data(survival=True))
    assert y.shape == (6, 3)
    (_, _), no_outcome = utils.prepare_keras_inputs_outputs(features, treatments, None)
    assert no_outcome is None


def test_prepare_keras_inputs_outputs_with_a_layout_gives_the_same_array():
    """``layout=`` builds the targets through ``encode_true``; the array is identical."""
    layout = ColumnLayout.from_families(2, families.Weibull(), families.Bernoulli())
    data = _data(survival=True)
    _, plain = utils.prepare_keras_inputs_outputs(*data)
    _, checked = utils.prepare_keras_inputs_outputs(*data, layout=layout)
    np.testing.assert_array_equal(checked, plain)
    assert checked.dtype == plain.dtype


def test_prepare_keras_inputs_outputs_with_a_layout_checks_the_outcome_width():
    """A survival outcome for a Normal layout (or the reverse) raises."""
    normal = ColumnLayout.from_families(2, families.Normal(), families.Bernoulli())
    with pytest.raises(ValueError, match=r"outcome has 2 column.*reads 1"):
        utils.prepare_keras_inputs_outputs(*_data(survival=True), layout=normal)
    weibull = ColumnLayout.from_families(2, families.Weibull(), families.Bernoulli())
    with pytest.raises(ValueError, match=r"outcome has 1 column.*reads 2"):
        utils.prepare_keras_inputs_outputs(*_data(), layout=weibull)
    # No outcome (prediction): nothing to check.
    _, none = utils.prepare_keras_inputs_outputs(*_data()[:2], None, layout=weibull)
    assert none is None
