"""Test module for loss functions."""

from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from ..keras import neglogliks

tfd = tfp.distributions


def _create_sample_data_exponential():
    """
    Creates a simple test case with two observations:
      - First observation: event_time=10, event_indicator=1, log_hazard=log(0.1)
      - Second observation: event_time=10, event_indicator=0, log_hazard=log(0.2)
    """
    # y_true has shape (n, 2): columns are event_time and event_indicator.
    y_true = tf.constant([[10.0, 1.0], [10.0, 0.0]])
    # y_pred has shape (n, 1): log_hazard predictions.
    y_pred = tf.constant([[0.1], [0.2]])
    return y_true, y_pred


def _test_data() -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([[0.0, 1.0], [-1, 0.1], [0.1, 0.5]])
    return y_true, y_pred


@pytest.mark.parametrize(
    "reduction,expected_shape",
    [("sum", ()), ("sum_over_batch_size", ()), ("none", (3,))],  # ("auto", 1),
)
def test_negloglik_normal_loss(reduction, expected_shape):
    y_true, y_pred = _test_data()
    loss = neglogliks.NegloglikNormal(reduction=reduction)(
        y_true=y_true.astype("float32"), y_pred=y_pred.astype("float32")
    )
    print(loss.shape)

    assert loss.shape == expected_shape


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


# --------------------------------------------------------------------
# Tests for _negloglik_exponential function.
# --------------------------------------------------------------------


def test_negloglik_exponential_event():
    """
    Test when an event occurs (event_indicator == 1).

    For an observation with:
      event_time = 10,
      event_indicator = 1,
      log_hazard = log(0.1) (so rate = 0.1),
    the loss should be: rate*event_time - log_hazard = 0.1*10 - log(0.1).
    """
    event_time = tf.constant([10.0])
    event_indicator = tf.constant([1.0])
    rate = tf.constant([0.1])

    loss = neglogliks._negloglik_exponential(event_time, event_indicator, rate)
    expected = 0.1 * 10 - np.log(0.1)
    np.testing.assert_allclose(loss.numpy(), [expected], atol=1e-5)


def test_negloglik_exponential_censored():
    """
    Test when an observation is censored (event_indicator == 0).

    For an observation with:
      event_time = 10,
      event_indicator = 0,
      log_hazard = log(0.1) (so rate = 0.1),
    the loss should be: rate*event_time = 0.1*10.
    """
    event_time = tf.constant([10.0])
    event_indicator = tf.constant([0.0])
    rate = tf.constant([0.1])

    loss = neglogliks._negloglik_exponential(event_time, event_indicator, rate)
    expected = 0.1 * 10
    np.testing.assert_allclose(loss.numpy(), [expected], atol=1e-5)


def test_NegloglikExponential_none():
    """
    Test NegloglikExponential with reduction NONE.

    Expected losses:
      Observation 1: 0.1*10 - log(0.1)
      Observation 2: 0.2*10
    """
    loss_obj = neglogliks.NegloglikExponential(reduction=tf.keras.losses.Reduction.NONE)
    y_true, y_pred = _create_sample_data_exponential()
    losses = loss_obj(y_true, y_pred)

    expected1 = 0.1 * 10 - np.log(0.1)
    expected2 = 0.2 * 10
    expected = np.array([expected1, expected2])
    np.testing.assert_allclose(losses.numpy(), expected, atol=1e-5)


def test_NegloglikExponential_sum():
    """
    Test NegloglikExponential with reduction SUM.

    Expected loss: sum over observations.
    """
    loss_obj = neglogliks.NegloglikExponential(reduction=tf.keras.losses.Reduction.SUM)
    y_true, y_pred = _create_sample_data_exponential()
    loss_value = loss_obj(y_true, y_pred)

    expected1 = 0.1 * 10 - np.log(0.1)
    expected2 = 0.2 * 10
    expected = expected1 + expected2
    np.testing.assert_allclose(loss_value.numpy(), expected, atol=1e-5)


def test_NegloglikExponential_sum_over_batch_size():
    """
    Test NegloglikExponential with reduction SUM_OVER_BATCH_SIZE.

    Expected loss: average loss over observations.
    """
    loss_obj = neglogliks.NegloglikExponential(
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )
    y_true, y_pred = _create_sample_data_exponential()
    loss_value = loss_obj(y_true, y_pred)

    expected1 = 0.1 * 10 - np.log(0.1)
    expected2 = 0.2 * 10
    expected = (expected1 + expected2) / 2.0
    np.testing.assert_allclose(loss_value.numpy(), expected, atol=1e-5)
