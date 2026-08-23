"""Test module for loss functions."""

from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from ..keras import losses, neglogliks

tfd = tfp.distributions


@pytest.mark.parametrize(
    "loss_obj",
    [
        neglogliks.NegloglikExponential(
            reduction="sum_over_batch_size", log_rate=True, name="my_exp"
        ),
        neglogliks.NegloglikExponentialScale(reduction="sum_over_batch_size", name="my_exp_scale"),
        neglogliks.NegloglikWeibull(reduction="sum_over_batch_size", name="my_weibull"),
    ],
)
def test_negloglik_loss_serialization_roundtrip(loss_obj):
    """These losses are registered as Keras-serializable; get_config/from_config must round-trip
    all constructor args (e.g. NegloglikExponential.log_rate), not just reduction/name."""
    config = loss_obj.get_config()
    restored = type(loss_obj).from_config(config)
    assert restored.get_config() == config

    y_true = tf.constant([[10.0, 1.0], [10.0, 0.0]])
    if isinstance(loss_obj, neglogliks.NegloglikWeibull):
        y_pred = tf.constant([[0.1, 0.2], [0.2, 0.1]])
    else:
        y_pred = tf.constant([[0.1], [0.2]])

    np.testing.assert_allclose(loss_obj(y_true, y_pred).numpy(), restored(y_true, y_pred).numpy())


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
    """Small fixed (y_true, y_pred) pair shared by the Normal-loss tests below."""
    y_true = np.array([0.0, 1.0, 2.0])
    y_pred = np.array([[0.0, 1.0], [-1, 0.1], [0.1, 0.5]])
    return y_true, y_pred


@pytest.mark.parametrize(
    "reduction,expected_shape",
    [("sum", ()), ("sum_over_batch_size", ()), ("none", (3,))],  # ("auto", 1),
)
def test_negloglik_normal_loss(reduction, expected_shape):
    """NegloglikNormal returns the expected output shape for each reduction mode."""
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
    """NegloglikNormal agrees with the generic NegloglikLoss(tfd.Normal) for the same inputs."""
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


# --------------------------------------------------------------------
# Tests for _negloglik_exponential_scale function / NegloglikExponentialScale.
# --------------------------------------------------------------------


def _create_sample_data_exponential_scale():
    """
    Creates a simple test case with two observations:
      - First observation: event_time=10, event_indicator=1, log_scale=log(5)
      - Second observation: event_time=10, event_indicator=0, log_scale=log(2)
    """
    y_true = tf.constant([[10.0, 1.0], [10.0, 0.0]])
    y_pred = tf.constant([[np.log(5.0)], [np.log(2.0)]])
    return y_true, y_pred


def test_negloglik_exponential_scale_event():
    """
    For event_time=10, event_indicator=1, log_scale=log(5) (scale=5, rate=0.2),
    the loss should be: t*exp(-log_scale) + log_scale = 10/5 + log(5).
    """
    event_time = tf.constant([10.0])
    event_indicator = tf.constant([1.0])
    log_scale = tf.constant([np.log(5.0)]).numpy().astype("float32")

    loss = neglogliks._negloglik_exponential_scale(event_time, event_indicator, log_scale)
    expected = 10.0 / 5.0 + np.log(5.0)
    np.testing.assert_allclose(loss.numpy(), [expected], atol=1e-5)


def test_negloglik_exponential_scale_censored():
    """
    For event_time=10, event_indicator=0, log_scale=log(5),
    the loss should be: t*exp(-log_scale) = 10/5.
    """
    event_time = tf.constant([10.0])
    event_indicator = tf.constant([0.0])
    log_scale = tf.constant([np.log(5.0)]).numpy().astype("float32")

    loss = neglogliks._negloglik_exponential_scale(event_time, event_indicator, log_scale)
    expected = 10.0 / 5.0
    np.testing.assert_allclose(loss.numpy(), [expected], atol=1e-5)


def test_negloglik_exponential_scale_matches_exponential_rate():
    """log_scale = -log(rate) parameterization should agree with the rate parameterization."""
    event_time = tf.constant([3.0, 7.5, 1.2])
    event_indicator = tf.constant([1.0, 0.0, 1.0])
    rate = tf.constant([0.1, 0.2, 0.5])
    log_scale = -tf.math.log(rate)

    loss_rate = neglogliks._negloglik_exponential(event_time, event_indicator, rate)
    loss_scale = neglogliks._negloglik_exponential_scale(event_time, event_indicator, log_scale)
    np.testing.assert_allclose(loss_rate.numpy(), loss_scale.numpy(), atol=1e-4)


def test_NegloglikExponentialScale_none():
    """NegloglikExponentialScale with reduction NONE returns per-observation losses."""
    loss_obj = neglogliks.NegloglikExponentialScale(reduction=tf.keras.losses.Reduction.NONE)
    y_true, y_pred = _create_sample_data_exponential_scale()
    losses = loss_obj(y_true, y_pred)

    expected1 = 10.0 / 5.0 + np.log(5.0)
    expected2 = 10.0 / 2.0
    expected = np.array([expected1, expected2])
    np.testing.assert_allclose(losses.numpy(), expected, atol=1e-5)


def test_NegloglikExponentialScale_sum():
    """NegloglikExponentialScale with reduction SUM sums losses over observations."""
    loss_obj = neglogliks.NegloglikExponentialScale(reduction=tf.keras.losses.Reduction.SUM)
    y_true, y_pred = _create_sample_data_exponential_scale()
    loss_value = loss_obj(y_true, y_pred)

    expected1 = 10.0 / 5.0 + np.log(5.0)
    expected2 = 10.0 / 2.0
    np.testing.assert_allclose(loss_value.numpy(), expected1 + expected2, atol=1e-5)


def test_NegloglikExponentialScale_sum_over_batch_size():
    """NegloglikExponentialScale with reduction SUM_OVER_BATCH_SIZE averages over observations."""
    loss_obj = neglogliks.NegloglikExponentialScale(
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )
    y_true, y_pred = _create_sample_data_exponential_scale()
    loss_value = loss_obj(y_true, y_pred)

    expected1 = 10.0 / 5.0 + np.log(5.0)
    expected2 = 10.0 / 2.0
    np.testing.assert_allclose(loss_value.numpy(), (expected1 + expected2) / 2.0, atol=1e-5)


# --------------------------------------------------------------------
# Tests for _negloglik_weibull function / NegloglikWeibull.
# --------------------------------------------------------------------


def _weibull_nll_reference(t, delta, log_scale, log_shape):
    """Reference NLL computed directly against tfp.distributions.Weibull."""
    k = np.exp(log_shape)
    scale = np.exp(log_scale)
    dist = tfd.Weibull(concentration=k, scale=scale)
    event_nll = -dist.log_prob(t).numpy()
    censored_nll = -dist.log_survival_function(t).numpy()
    return np.where(delta == 1.0, event_nll, censored_nll)


def _create_sample_data_weibull():
    """
    Creates a simple test case with two observations:
      - First observation: event_time=2, event_indicator=1, log_scale=log(3), log_shape=log(1.5)
      - Second observation: event_time=5, event_indicator=0, log_scale=log(4), log_shape=log(0.8)
    """
    y_true = tf.constant([[2.0, 1.0], [5.0, 0.0]])
    y_pred = tf.constant([[np.log(3.0), np.log(1.5)], [np.log(4.0), np.log(0.8)]], dtype=tf.float32)
    return y_true, y_pred


def test_negloglik_weibull_event():
    """_negloglik_weibull's event-term matches tfp.distributions.Weibull.log_prob."""
    event_time = np.array([2.0])
    event_indicator = np.array([1.0])
    log_scale = np.array([np.log(3.0)], dtype="float32")
    log_shape = np.array([np.log(1.5)], dtype="float32")

    loss = neglogliks._negloglik_weibull(event_time, event_indicator, log_scale, log_shape)
    expected = _weibull_nll_reference(event_time, event_indicator, log_scale, log_shape)
    np.testing.assert_allclose(loss.numpy(), expected, atol=1e-4)


def test_negloglik_weibull_censored():
    """_negloglik_weibull's censored-term matches tfp.distributions.Weibull.log_survival_function."""
    event_time = np.array([5.0])
    event_indicator = np.array([0.0])
    log_scale = np.array([np.log(4.0)], dtype="float32")
    log_shape = np.array([np.log(0.8)], dtype="float32")

    loss = neglogliks._negloglik_weibull(event_time, event_indicator, log_scale, log_shape)
    expected = _weibull_nll_reference(event_time, event_indicator, log_scale, log_shape)
    np.testing.assert_allclose(loss.numpy(), expected, atol=1e-4)


def test_negloglik_weibull_reduces_to_exponential_when_shape_one():
    """Weibull with shape k=1 (log_shape=0) is the Exponential distribution."""
    event_time = tf.constant([3.0, 7.5, 1.2])
    event_indicator = tf.constant([1.0, 0.0, 1.0])
    log_scale = tf.constant([0.3, -0.2, 1.0])
    log_shape = tf.zeros_like(log_scale)

    loss_weibull = neglogliks._negloglik_weibull(event_time, event_indicator, log_scale, log_shape)
    loss_exp_scale = neglogliks._negloglik_exponential_scale(event_time, event_indicator, log_scale)
    np.testing.assert_allclose(loss_weibull.numpy(), loss_exp_scale.numpy(), atol=1e-4)


def test_NegloglikWeibull_none():
    """NegloglikWeibull with reduction NONE returns per-observation losses."""
    loss_obj = neglogliks.NegloglikWeibull(reduction=tf.keras.losses.Reduction.NONE)
    y_true, y_pred = _create_sample_data_weibull()
    losses = loss_obj(y_true, y_pred)

    expected = _weibull_nll_reference(
        np.array([2.0, 5.0]),
        np.array([1.0, 0.0]),
        np.array([np.log(3.0), np.log(4.0)], dtype="float32"),
        np.array([np.log(1.5), np.log(0.8)], dtype="float32"),
    )
    np.testing.assert_allclose(losses.numpy(), expected, atol=1e-4)


def test_NegloglikWeibull_sum():
    """NegloglikWeibull with reduction SUM sums losses over observations."""
    loss_obj = neglogliks.NegloglikWeibull(reduction=tf.keras.losses.Reduction.SUM)
    y_true, y_pred = _create_sample_data_weibull()
    loss_value = loss_obj(y_true, y_pred)

    expected = _weibull_nll_reference(
        np.array([2.0, 5.0]),
        np.array([1.0, 0.0]),
        np.array([np.log(3.0), np.log(4.0)], dtype="float32"),
        np.array([np.log(1.5), np.log(0.8)], dtype="float32"),
    )
    np.testing.assert_allclose(loss_value.numpy(), expected.sum(), atol=1e-4)


def test_NegloglikWeibull_sum_over_batch_size():
    """NegloglikWeibull with reduction SUM_OVER_BATCH_SIZE averages over observations."""
    loss_obj = neglogliks.NegloglikWeibull(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    y_true, y_pred = _create_sample_data_weibull()
    loss_value = loss_obj(y_true, y_pred)

    expected = _weibull_nll_reference(
        np.array([2.0, 5.0]),
        np.array([1.0, 0.0]),
        np.array([np.log(3.0), np.log(4.0)], dtype="float32"),
        np.array([np.log(1.5), np.log(0.8)], dtype="float32"),
    )
    np.testing.assert_allclose(loss_value.numpy(), expected.mean(), atol=1e-4)


@pytest.mark.parametrize(
    "name,treatment_nll_loss,n_params,y_dim",
    [
        ("exponential", neglogliks.NegloglikExponential(reduction="none"), 1, 2),
        ("exponential_scale", neglogliks.NegloglikExponentialScale(reduction="none"), 1, 2),
        ("weibull", neglogliks.NegloglikWeibull(reduction="none"), 2, 2),
        ("normal", neglogliks.NegloglikNormal(reduction="none"), 2, 1),
    ],
)
def test_prob_state_given_treatment_features_survives_partial_batch_retrace(
    name, treatment_nll_loss, n_params, y_dim
):
    """Regression test for the bare-squeeze shape-inference bug: model.fit() crashed with
    `TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'` inside
    posterior_from_negloglik_per_state whenever Keras retraced train_step for a batch shaped
    differently from the one used to first trace the graph -- e.g. the partial/remainder final
    batch you get from n_samples=200, batch_size=32 (last batch has 8 rows). Reproduces without
    any downstream model builder: a minimal Dense model routed through
    prob_state_given_treatment_features is enough to trigger it in graph-mode model.fit().
    """
    n_states = 3
    n_features = 5
    n_samples = 200  # batch_size=32 -> last batch has 8 rows (partial/remainder)
    batch_size = 32

    np.random.seed(0)
    x = np.random.randn(n_samples, n_features).astype("float32")
    event_time = np.random.uniform(1.0, 10.0, size=(n_samples,)).astype("float32")
    event_indicator = (np.random.uniform(size=(n_samples,)) > 0.5).astype("float32")
    y = np.stack([event_time, event_indicator], axis=1) if y_dim == 2 else event_time[:, None]

    inp = tf.keras.Input(shape=(n_features,))
    raw = tf.keras.layers.Dense(units=n_states + n_states * n_params)(inp)

    def custom_loss(y_true, y_pred):
        weights = tf.nn.softmax(y_pred[:, :n_states], axis=1)
        # softplus keeps rate/scale params positive -- avoids NaNs from an untrained
        # random Dense layer, unrelated to the shape bug under test.
        treatment_pred = tf.nn.softplus(y_pred[:, n_states:]) + 1e-3
        gamma = losses.prob_state_given_treatment_features(
            weights=weights,
            treatment_true=y_true,
            treatment_pred=treatment_pred,
            treatment_nll_loss=treatment_nll_loss,
        )
        return -tf.reduce_sum(tf.math.log(gamma + 1e-6), axis=1)

    model = tf.keras.Model(inputs=inp, outputs=raw)
    model.compile(optimizer="adam", loss=custom_loss)

    # Prior to the fix, this raises TypeError: unsupported operand type(s) for -:
    # 'NoneType' and 'int' when Keras retraces train_step for the 8-row final batch.
    history = model.fit(x=x, y=y, batch_size=batch_size, epochs=2, verbose=0)
    assert np.isfinite(history.history["loss"][-1])
