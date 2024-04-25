"""Test module for loss functions."""

from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
import random

from .. import datasets
from ..keras import losses, models
from ..keras import neglogliks
from .. import utils, inference


tfk = tf.keras


def _test_data() -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([[0.0, 1.0], [-1, 0.1], [0.1, 0.5]])
    return y_true, y_pred


@pytest.mark.parametrize(
    "reduction,expected_len",
    [("sum", 1), ("sum_over_batch_size", 1), ("none", 3)],  # ("auto", 1),
)
def test_negloglik_normal_loss(reduction, expected_len):
    y_true, y_pred = _test_data()
    loss = neglogliks.NegloglikNormal(reduction=reduction)(
        y_true=y_true.astype("float32"), y_pred=y_pred.astype("float32")
    )
    if expected_len == 1:
        assert not len(loss.numpy().shape)
    else:
        assert loss.shape[0] == expected_len
