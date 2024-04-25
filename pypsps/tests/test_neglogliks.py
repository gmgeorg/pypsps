"""Test module for loss functions."""

from typing import Tuple

import numpy as np
import pytest
import tensorflow_probability as tfp

from ..keras import neglogliks


tfd = tfp.distributions


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


@pytest.mark.parametrize(
    "reduction",
    [("sum"), ("sum_over_batch_size"), ("none")],  # ("auto", 1),
)
def test_negloglik_loss_class_works(reduction):
    y_true, y_pred = _test_data()
    loss_normal = neglogliks.NegloglikNormal(reduction=reduction)(
        y_true=y_true.astype("float32"), y_pred=y_pred.astype("float32")
    )
    loss_class_normal = neglogliks.NegloglikLoss(
        reduction=reduction, distribution_constructor=tfd.Normal
    )(y_true=y_true.astype("float32"), y_pred=y_pred.astype("float32"))
    print(loss_normal)
    print(loss_class_normal)
    assert loss_normal.numpy() == pytest.approx(loss_class_normal.numpy(), 0.0001)
