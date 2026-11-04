import numpy as np
import pytest
import tensorflow as tf

from pypsps.keras import losses, metrics, neglogliks


def _make_y_pred(outcome_blocks, weights, treatment_blocks):
    """Builds an interleaved y_pred tensor: [outcome params..., weights, treatment params...].

    outcome_blocks / treatment_blocks: lists of per-param arrays, each of shape
    (n_rows, n_states), eg for a Normal outcome with 2 states: [loc_by_state, scale_by_state].
    weights: (n_rows, n_states) prior state weights (rows should sum to 1).
    """
    blocks = [np.asarray(b, dtype=np.float32) for b in outcome_blocks]
    blocks.append(np.asarray(weights, dtype=np.float32))
    blocks.extend(np.asarray(b, dtype=np.float32) for b in treatment_blocks)
    return tf.constant(np.concatenate(blocks, axis=1))


def test_propensity_score_binary_crossentropy():
    """PropensityScoreBinaryCrossentropy must compare treatment_true against the
    weighted (prior) average of the state-conditional propensity predictions."""
    weights = [[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]]
    treatment_by_state = [[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]]  # p(a=1 | s_k, x)

    y_pred = _make_y_pred(
        outcome_blocks=[np.zeros((3, 2))],  # n_outcome_pred_cols=1, unused by this metric
        weights=weights,
        treatment_blocks=[treatment_by_state],  # n_treatment_pred_cols=1
    )
    y_true = tf.constant([[0.0, 1.0], [0.0, 0.0], [0.0, 1.0]], dtype=tf.float32)

    metric = metrics.PropensityScoreBinaryCrossentropy(
        n_outcome_pred_cols=1, n_treatment_pred_cols=1
    )
    metric.update_state(y_true, y_pred)

    marginal_propensity = (np.asarray(weights) * np.asarray(treatment_by_state)).sum(axis=1)
    expected = tf.keras.metrics.BinaryCrossentropy()
    expected.update_state(y_true=y_true[:, -1:], y_pred=marginal_propensity[:, None])

    np.testing.assert_allclose(metric.result().numpy(), expected.result().numpy(), rtol=1e-5)


def test_propensity_score_auc():
    """Test propensity score AUC uses the weighted (prior) marginal propensity."""
    weights = [[0.5, 0.5], [0.3, 0.7], [0.9, 0.1], [0.1, 0.9]]
    treatment_by_state = [[0.9, 0.1], [0.5, 0.5], [0.2, 0.8], [0.7, 0.3]]

    y_pred = _make_y_pred(
        outcome_blocks=[np.zeros((4, 2))],
        weights=weights,
        treatment_blocks=[treatment_by_state],
    )
    y_true = tf.constant([[0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=tf.float32)

    metric = metrics.PropensityScoreAUC(n_outcome_pred_cols=1, n_treatment_pred_cols=1)
    metric.update_state(y_true, y_pred)

    marginal_propensity = (np.asarray(weights) * np.asarray(treatment_by_state)).sum(axis=1)
    expected = tf.keras.metrics.AUC()
    expected.update_state(y_true=y_true[:, -1:], y_pred=marginal_propensity[:, None])

    np.testing.assert_allclose(metric.result().numpy(), expected.result().numpy(), rtol=1e-5)


def test_propensity_score_binary_crossentropy_treatment_is_last_column():
    """Regression test: propensity metrics must read treatment from the LAST column of
    y_true, not from y_true[:, 1:]. Outcome data can have more than one column (e.g. a
    survival outcome of [event_time, event_indicator]), in which case y_true[:, 1:]
    would silently leak non-treatment columns into the propensity metric. Treatment is
    always appended as the final column by utils.prepare_keras_inputs_outputs, so
    y_true[:, -1:] must be used regardless of how many outcome columns precede it.
    """
    weights = [[1.0], [1.0], [1.0]]  # n_states=1: no mixing, marginal == state-0 prediction
    treatment_by_state = [[0.9], [0.1], [0.8]]

    y_pred = _make_y_pred(
        outcome_blocks=[np.zeros((3, 1)), np.zeros((3, 1))],  # n_outcome_pred_cols=2
        weights=weights,
        treatment_blocks=[treatment_by_state],
    )
    # 2 outcome-like columns followed by the treatment column (last).
    y_true = tf.constant([[5.0, 0.0, 1.0], [3.0, 1.0, 0.0], [8.0, 1.0, 1.0]], dtype=tf.float32)

    metric = metrics.PropensityScoreBinaryCrossentropy(
        n_outcome_pred_cols=2, n_treatment_pred_cols=1
    )
    metric.update_state(y_true, y_pred)

    expected = tf.keras.metrics.BinaryCrossentropy()
    expected.update_state(y_true=y_true[:, -1:], y_pred=np.asarray(treatment_by_state))

    np.testing.assert_allclose(metric.result().numpy(), expected.result().numpy(), rtol=1e-5)


def test_propensity_score_auc_treatment_is_last_column():
    """Regression test: see test_propensity_score_binary_crossentropy_treatment_is_last_column."""
    weights = [[1.0], [1.0], [1.0]]
    treatment_by_state = [[0.9], [0.1], [0.8]]

    y_pred = _make_y_pred(
        outcome_blocks=[np.zeros((3, 1)), np.zeros((3, 1))],
        weights=weights,
        treatment_blocks=[treatment_by_state],
    )
    y_true = tf.constant([[5.0, 0.0, 1.0], [3.0, 1.0, 0.0], [8.0, 1.0, 1.0]], dtype=tf.float32)

    metric = metrics.PropensityScoreAUC(n_outcome_pred_cols=2, n_treatment_pred_cols=1)
    metric.update_state(y_true, y_pred)

    expected = tf.keras.metrics.AUC()
    expected.update_state(y_true=y_true[:, -1:], y_pred=np.asarray(treatment_by_state))

    np.testing.assert_allclose(metric.result().numpy(), expected.result().numpy(), rtol=1e-5)


def test_treatment_mean_squared_error():
    """test for treatment MSE"""
    weights = [[1.0], [1.0], [1.0]]  # n_states=1: agg_treatment_pred passes values through
    treat_pred = [[2.1], [3.9], [6.2]]
    y_pred = _make_y_pred(
        outcome_blocks=[np.array([[9.0], [19.0], [31.0]])],
        weights=weights,
        treatment_blocks=[treat_pred],
    )
    y_true = tf.constant([[10, 2], [20, 4], [30, 6]], dtype=tf.float32)
    metric = metrics.TreatmentMeanSquaredError(
        n_outcome_pred_cols=1, n_treatment_pred_cols=1, n_outcome_true_cols=1
    )
    metric.update_state(y_true, y_pred)
    mse = metric.result().numpy()
    # Since predictions are close, MSE should be low.
    assert mse < 1.0


def test_treatment_mean_absolute_error():
    """test for MAE treatment"""
    weights = [[1.0], [1.0], [1.0]]
    treat_pred = [[2.0], [4.0], [6.0]]
    y_pred = _make_y_pred(
        outcome_blocks=[np.array([[9.0], [20.0], [30.0]])],
        weights=weights,
        treatment_blocks=[treat_pred],
    )
    y_true = tf.constant([[10, 2], [20, 4], [30, 6]], dtype=tf.float32)
    metric = metrics.TreatmentMeanAbsoluteError(
        n_outcome_pred_cols=1, n_treatment_pred_cols=1, n_outcome_true_cols=1
    )
    metric.update_state(y_true, y_pred)
    mae = metric.result().numpy()
    # Expect near zero error
    np.testing.assert_allclose(mae, 0.0, atol=1e-6)


def test_outcome_mean_squared_error():
    """Tests for MSE"""
    y_true = tf.constant(
        [
            [5, 100],  # outcome_true=5, treatment_true=100 (ignored)
            [10, 200],
            [15, 300],
        ],
        dtype=tf.float32,
    )
    # n_states=1: [outcome, weight=1.0, treatment]. Weight of 1.0 means agg_outcome_pred
    # passes the outcome prediction through unweighted.
    y_pred = tf.constant([[5, 1.0, 0.0], [10, 1.0, 0.0], [15, 1.0, 0.0]], dtype=tf.float32)
    metric = metrics.OutcomeMeanSquaredError(
        n_outcome_pred_cols=1, n_treatment_pred_cols=1, n_outcome_true_cols=1
    )
    metric.update_state(y_true, y_pred)
    mse = metric.result().numpy()
    np.testing.assert_allclose(mse, 0.0, atol=1e-6)


def test_outcome_mean_absolute_error():
    """tests for outcome MAE."""
    y_true = tf.constant([[5, 100], [10, 200], [15, 300]], dtype=tf.float32)
    y_pred = tf.constant([[5, 1.0, 0.0], [10, 1.0, 0.0], [15, 1.0, 0.0]], dtype=tf.float32)
    metric = metrics.OutcomeMeanAbsoluteError(
        n_outcome_pred_cols=1, n_treatment_pred_cols=1, n_outcome_true_cols=1
    )
    metric.update_state(y_true, y_pred)
    mae = metric.result().numpy()
    np.testing.assert_allclose(mae, 0.0, atol=1e-6)


def test_predictive_state_df_gen():
    """Test predictive state gen"""
    n_outcome_pred_cols = 1
    n_treatment_pred_cols = 1
    weights = [[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]]
    y_pred = _make_y_pred(
        outcome_blocks=[np.zeros((3, 2))],
        weights=weights,
        treatment_blocks=[np.zeros((3, 2))],
    )
    func = metrics.predictive_state_df_gen(n_outcome_pred_cols, n_treatment_pred_cols)
    result = func(None, y_pred)
    # Check that result is a scalar tensor.
    assert result.shape.ndims == 0 or (result.shape.ndims == 1 and result.shape[0] == 1)


def _make_treatment_loss(**overrides):
    """Builds a plain (no-penalty) `TreatmentLoss` with sensible test defaults."""
    kwargs = dict(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction="sum_over_batch_size",
    )
    kwargs.update(overrides)
    return losses.TreatmentLoss(**kwargs)


def test_treatment_loss_penalty_free_is_identity_when_no_penalty_declared():
    """Base `TreatmentLoss.penalty_free()` has no penalty args to drop, so it must return an
    equivalent (but distinct) instance -- not the same object."""
    original = _make_treatment_loss()
    clean = original.penalty_free()

    assert clean is not original
    assert clean._n_outcome_true_cols == original._n_outcome_true_cols
    assert clean._n_outcome_pred_cols == original._n_outcome_pred_cols
    assert clean._n_treatment_pred_cols == original._n_treatment_pred_cols
    assert clean.get_config() == original.get_config()


class _BalancePenalizedTreatmentLoss(losses.TreatmentLoss):
    """Test double simulating a hypothetical within-state balance penalty folded directly
    into `TreatmentLoss.call` (`self._lambda_balance * balance_penalty`), the pattern
    `causal_loss_metric_gen` must be robust to -- without relying on the `lambda_balance`
    name, since `penalty_free()` is reconstructed via explicit constructor arguments, not
    attribute-name sniffing."""

    def __init__(self, *args, lambda_balance: float = 0.0, **kwargs):
        """Stores `lambda_balance`, the (test-only) penalty weight."""
        super().__init__(*args, **kwargs)
        self._lambda_balance = lambda_balance

    def call(self, y_true, y_pred):
        """Adds a constant `lambda_balance` penalty on top of the real treatment NLL."""
        return super().call(y_true, y_pred) + self._lambda_balance


def test_treatment_loss_penalty_free_zeros_declared_penalty_without_mutating_original():
    """A subclass that inherits `penalty_free()` without overriding it drops any constructor
    argument not explicitly forwarded by the base implementation -- here, `lambda_balance`
    falls back to its own "off" default (0.0) -- and the original, live instance must be
    left untouched."""
    original = _BalancePenalizedTreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        lambda_balance=5.0,
        reduction="sum_over_batch_size",
    )
    clean = original.penalty_free()

    assert clean is not original
    assert clean._lambda_balance == 0.0
    assert original._lambda_balance == 5.0


def test_treatment_loss_penalty_free_fails_loudly_for_undeclared_required_penalty_arg():
    """If a subclass adds a *required* penalty argument (no safe "off" default) and doesn't
    override `penalty_free()` to account for it, reconstruction must raise rather than
    silently guess a value -- forcing whoever adds the penalty to explicitly decide how
    `penalty_free()` should handle it."""

    class _RequiredPenaltyTreatmentLoss(losses.TreatmentLoss):
        """Test double whose penalty weight has no safe "off" default."""

        def __init__(self, *args, lambda_balance: float, **kwargs):
            """Stores the required `lambda_balance` penalty weight."""
            super().__init__(*args, **kwargs)
            self._lambda_balance = lambda_balance

    original = _RequiredPenaltyTreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        lambda_balance=5.0,
        reduction="sum_over_batch_size",
    )
    try:
        original.penalty_free()
        assert False, "expected TypeError: lambda_balance is required and not forwarded"
    except TypeError:
        pass


def test_causal_loss_metric_gen_strips_embedded_treatment_loss_penalty():
    """causal_loss_metric_gen must report the exact joint likelihood even when the passed-in
    treatment_loss instance carries an embedded penalty term -- the metric used for
    EarlyStopping/checkpoint selection/Optuna must never be a penalized proxy."""
    n_outcome_pred_cols, n_treatment_pred_cols, n_outcome_true_cols = 2, 1, 1
    weights = [[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]]
    y_pred = _make_y_pred(
        outcome_blocks=[np.zeros((3, 2)), np.ones((3, 2))],
        weights=weights,
        treatment_blocks=[[[0.9, 0.7], [0.5, 0.5], [0.2, 0.6]]],
    )
    y_true = tf.constant([[5.0, 1.0], [3.0, 0.0], [8.0, 1.0]], dtype=tf.float32)

    def _build(lambda_balance):
        """Builds a matching (outcome_loss, treatment_loss) pair for a given penalty weight."""
        outcome_loss = losses.OutcomeLoss(
            loss=neglogliks.NegloglikNormal(reduction="none"),
            treatment_loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
            n_outcome_true_cols=n_outcome_true_cols,
            n_outcome_pred_cols=n_outcome_pred_cols,
            n_treatment_pred_cols=n_treatment_pred_cols,
            reduction="sum_over_batch_size",
        )
        treatment_loss = _BalancePenalizedTreatmentLoss(
            loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
            n_outcome_true_cols=n_outcome_true_cols,
            n_outcome_pred_cols=n_outcome_pred_cols,
            n_treatment_pred_cols=n_treatment_pred_cols,
            lambda_balance=lambda_balance,
            reduction="sum_over_batch_size",
        )
        return outcome_loss, treatment_loss

    penalized_outcome_loss, penalized_treatment_loss = _build(lambda_balance=1000.0)
    clean_outcome_loss, clean_treatment_loss = _build(lambda_balance=0.0)

    metric = metrics.causal_loss_metric_gen(
        outcome_loss=penalized_outcome_loss, treatment_loss=penalized_treatment_loss
    )
    assert isinstance(metric, metrics.CausalLossMetric)
    metric.update_state(y_true, y_pred)
    metric_value = metric.result().numpy()

    expected_clean = (
        clean_outcome_loss(y_true, y_pred) + clean_treatment_loss(y_true, y_pred)
    ).numpy()
    expected_penalized = (
        penalized_outcome_loss(y_true, y_pred) + penalized_treatment_loss(y_true, y_pred)
    ).numpy()

    np.testing.assert_allclose(metric_value, expected_clean, rtol=1e-5)
    assert not np.isclose(metric_value, expected_penalized, rtol=1e-5)
    # The original, live training-time loss must be unmodified.
    assert penalized_treatment_loss._lambda_balance == 1000.0


def _normal_outcome_loss(cls=losses.OutcomeLoss, **extra):
    """An OutcomeLoss for a normal outcome (2 params per state) and binary treatment."""
    return cls(
        loss=neglogliks.NegloglikNormal(reduction="none"),
        treatment_loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction="sum_over_batch_size",
        **extra,
    )


def _normal_fixture():
    """y_true = [outcome | treatment] and y_pred for 2 states, normal outcome."""
    weights = [[0.5, 0.5], [0.3, 0.7], [0.9, 0.1]]
    y_pred = _make_y_pred(
        outcome_blocks=[[[0.0, 1.0], [2.0, -1.0], [0.5, 0.5]], [[1.0, 2.0]] * 3],
        weights=weights,
        treatment_blocks=[[[0.8, 0.3], [0.4, 0.6], [0.5, 0.2]]],
    )
    y_true = tf.constant([[0.2, 1.0], [1.5, 0.0], [-0.3, 1.0]], dtype=tf.float32)
    return y_true, y_pred


class _PenalizedOutcomeLoss(losses.OutcomeLoss):
    """Test double: an OutcomeLoss whose call() adds a constant penalty."""

    def __init__(self, *args, lambda_penalty: float = 0.0, **kwargs):
        """Stores the penalty weight."""
        super().__init__(*args, **kwargs)
        self._lambda_penalty = lambda_penalty

    def call(self, y_true, y_pred):
        """Adds the penalty to the outcome negative log-likelihood."""
        return super().call(y_true, y_pred) + self._lambda_penalty


def test_outcome_nll_metric_equals_the_outcome_loss():
    """The metric is the outcome loss itself, logged as outcome_nll_metric."""
    outcome = _normal_outcome_loss()
    y_true, y_pred = _normal_fixture()
    metric = metrics.outcome_nll_metric_gen(outcome)
    assert metric.name == "outcome_nll_metric"
    np.testing.assert_allclose(
        float(metric(y_true, y_pred)), float(outcome(y_true, y_pred)), rtol=1e-7
    )


def test_outcome_nll_metric_is_penalty_free_and_leaves_the_loss_untouched():
    """A penalty in the training loss does not reach the metric, nor is it removed from the loss."""
    y_true, y_pred = _normal_fixture()
    penalized = _normal_outcome_loss(_PenalizedOutcomeLoss, lambda_penalty=5.0)
    clean = float(metrics.outcome_nll_metric_gen(_normal_outcome_loss())(y_true, y_pred))
    np.testing.assert_allclose(
        float(metrics.outcome_nll_metric_gen(penalized)(y_true, y_pred)), clean, rtol=1e-7
    )
    assert penalized._lambda_penalty == 5.0
    np.testing.assert_allclose(float(penalized(y_true, y_pred)), clean + 5.0, rtol=1e-6)


def test_outcome_nll_metric_does_not_move_with_alpha():
    """Changing the training loss's alpha leaves the metric unchanged."""
    from pypsps import datasets
    from pypsps.keras import models

    np.random.seed(3)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=128)
    inputs, outputs = ks_data.to_keras_inputs_outputs()
    tf.random.set_seed(3)
    model = models.build_toy_model(n_states=2, n_features=ks_data.n_features, alpha=1.0)
    metric = metrics.outcome_nll_metric_gen(model.loss._outcome_loss)
    model.compile(loss=model.loss, optimizer="adam", metrics=[metric])

    def evaluate():
        """Return (loss, outcome_nll_metric) on the sample."""
        res = model.evaluate(inputs, outputs, verbose=0, return_dict=True)
        return res["loss"], res["outcome_nll_metric"]

    loss_1, nll_1 = evaluate()
    model.loss.alpha = 10.0
    loss_10, nll_10 = evaluate()
    assert loss_10 > loss_1
    np.testing.assert_allclose(nll_10, nll_1, rtol=1e-6)


def _metric_value(metric, y_true, y_pred) -> float:
    """One-shot value of a stateful metric: reset, update once, read."""
    metric.reset_state()
    metric.update_state(y_true, y_pred)
    return float(metric.result())


def test_causal_loss_metric_ignores_the_training_alpha():
    """The metric is the alpha-1 joint NLL, whatever alpha the training loss holds."""
    outcome, treatment = _normal_outcome_loss(), _make_treatment_loss()
    y_true, y_pred = _normal_fixture()
    training = losses.CausalLoss(outcome, treatment, alpha=1.0)
    metric = metrics.CausalLossMetric(outcome, treatment)

    at_one = _metric_value(metric, y_true, y_pred)
    np.testing.assert_allclose(at_one, float(training(y_true, y_pred)), rtol=1e-6)
    training.alpha = 5.0
    assert float(training(y_true, y_pred)) > at_one
    np.testing.assert_allclose(_metric_value(metric, y_true, y_pred), at_one, rtol=1e-7)


def test_causal_loss_metric_weights_the_outcome_nll():
    """value = outcome_loss_weight * outcome NLL + treatment NLL."""
    outcome, treatment = _normal_outcome_loss(), _make_treatment_loss()
    y_true, y_pred = _normal_fixture()
    nll_y, nll_a = float(outcome(y_true, y_pred)), float(treatment(y_true, y_pred))
    for weight in (1.0, 2.5):
        metric = metrics.CausalLossMetric(outcome, treatment, outcome_loss_weight=weight)
        np.testing.assert_allclose(
            _metric_value(metric, y_true, y_pred), weight * nll_y + nll_a, rtol=1e-6
        )


@pytest.mark.parametrize("weight", [1.0, 2.5, 0.0])
def test_causal_loss_metric_config_round_trips(weight):
    """The config holds both losses and the weight, and no alpha; the rebuilt metric agrees."""
    import json

    y_true, y_pred = _normal_fixture()
    metric = metrics.CausalLossMetric(
        _normal_outcome_loss(), _make_treatment_loss(), outcome_loss_weight=weight
    )
    config = metric.get_config()
    json.dumps(config)
    assert "alpha" not in config
    assert config["outcome_loss_weight"] == weight
    assert {"outcome_loss", "treatment_loss"} <= set(config)

    rebuilt = metrics.CausalLossMetric.from_config(config)
    assert rebuilt._outcome_loss_weight == weight
    np.testing.assert_allclose(
        _metric_value(rebuilt, y_true, y_pred), _metric_value(metric, y_true, y_pred), rtol=1e-6
    )


def test_causal_loss_metric_allows_zero_outcome_loss_weight():
    """outcome_loss_weight=0 is legal: the metric is the treatment NLL alone."""
    treatment = _make_treatment_loss()
    y_true, y_pred = _normal_fixture()
    metric = metrics.CausalLossMetric(_normal_outcome_loss(), treatment, outcome_loss_weight=0)
    np.testing.assert_allclose(
        _metric_value(metric, y_true, y_pred), float(treatment(y_true, y_pred)), rtol=1e-6
    )


def test_causal_loss_metric_rejects_alpha_and_wrong_loss_types():
    """There is no alpha parameter; the two losses must be pypsps outcome/treatment losses."""
    outcome, treatment = _normal_outcome_loss(), _make_treatment_loss()
    with pytest.raises(TypeError):
        metrics.CausalLossMetric(outcome, treatment, alpha=1.0)
    with pytest.raises(TypeError, match="OutcomeLoss"):
        metrics.CausalLossMetric(treatment, treatment)
    with pytest.raises(TypeError, match="TreatmentLoss"):
        metrics.CausalLossMetric(outcome, outcome)


def test_causal_loss_metric_gen_alpha_is_deprecated_and_ignored():
    """A passed alpha warns (FutureWarning, naming v0.5.0) and doesn't change the value."""
    import warnings

    outcome, treatment = _normal_outcome_loss(), _make_treatment_loss()
    y_true, y_pred = _normal_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain = metrics.causal_loss_metric_gen(outcome, treatment)
        weighted = metrics.causal_loss_metric_gen(outcome, treatment, outcome_loss_weight=2.0)
    with pytest.warns(FutureWarning, match="v0.5.0"):
        with_alpha = metrics.causal_loss_metric_gen(outcome, treatment, alpha=5.0)

    value = _metric_value(plain, y_true, y_pred)
    np.testing.assert_allclose(_metric_value(with_alpha, y_true, y_pred), value, rtol=1e-7)
    assert _metric_value(weighted, y_true, y_pred) != pytest.approx(value)


def test_metric_generators_return_serializable_metrics():
    """The three MeanMetricWrapper generators round-trip through Keras serialization."""
    y_true, y_pred = _normal_fixture()
    for metric in (
        metrics.outcome_nll_metric_gen(_normal_outcome_loss()),
        metrics.predictive_state_df_gen(n_outcome_pred_cols=2, n_treatment_pred_cols=1),
        metrics.uniform_entropy_gen(n_outcome_pred_cols=2, n_treatment_pred_cols=1),
    ):
        rebuilt = tf.keras.utils.deserialize_keras_object(
            tf.keras.utils.serialize_keras_object(metric)
        )
        assert rebuilt.name == metric.name
        np.testing.assert_allclose(
            _metric_value(rebuilt, y_true, y_pred),
            _metric_value(metric, y_true, y_pred),
            rtol=1e-6,
        )


def test_penalty_free_drops_a_penalty_even_if_get_config_serializes_it():
    """penalty_free() rebuilds from the base class's config, so a subclass that serializes
    its penalty in its own get_config still gets the "off" default in the copy."""

    class _SerializedPenaltyTreatmentLoss(_BalancePenalizedTreatmentLoss):
        """Serializes its penalty, as a well-behaved Keras subclass would."""

        def get_config(self):
            """Adds lambda_balance to the config."""
            return {**super().get_config(), "lambda_balance": self._lambda_balance}

    original = _SerializedPenaltyTreatmentLoss(
        loss=tf.keras.losses.BinaryCrossentropy(reduction="none"),
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        lambda_balance=5.0,
        reduction="sum_over_batch_size",
    )
    assert original.get_config()["lambda_balance"] == 5.0
    assert original.penalty_free()._lambda_balance == 0.0


def test_outcome_nll_metric_keeps_outcome_likelihood():
    """The outcome metric uses the training loss's outcome_likelihood."""
    y_true, y_pred = _normal_fixture()
    interventional = _normal_outcome_loss(outcome_likelihood="interventional")
    observational = _normal_outcome_loss(outcome_likelihood="observational")
    for chosen, other in ((interventional, observational), (observational, interventional)):
        got = float(metrics.outcome_nll_metric_gen(chosen)(y_true, y_pred))
        np.testing.assert_allclose(got, float(chosen(y_true, y_pred)), rtol=1e-7)
        assert got != pytest.approx(float(other(y_true, y_pred)))


_COL_KWARGS = {"n_outcome_pred_cols": 1, "n_treatment_pred_cols": 1}
_COL_KWARGS_WITH_TRUE = {**_COL_KWARGS, "n_outcome_true_cols": 1}

_METRIC_CLASSES = [
    (
        metrics.PropensityScoreBinaryCrossentropy,
        _COL_KWARGS,
        "propensity_score_binary_crossentropy",
    ),
    (metrics.PropensityScoreAUC, _COL_KWARGS, "propensity_score_auc"),
    (metrics.TreatmentMeanSquaredError, _COL_KWARGS_WITH_TRUE, "treatment_mean_squared_error"),
    (metrics.TreatmentMeanAbsoluteError, _COL_KWARGS_WITH_TRUE, "treatment_mean_absolute_error"),
    (metrics.OutcomeMeanSquaredError, _COL_KWARGS_WITH_TRUE, "outcome_mean_squared_error"),
    (metrics.OutcomeMeanAbsoluteError, _COL_KWARGS_WITH_TRUE, "outcome_mean_absolute_error"),
]
_METRIC_IDS = [cls.__name__ for cls, _, _ in _METRIC_CLASSES]


def _metric_fixture():
    """y_true (outcome, treatment) and a 2-state y_pred with 1 outcome and 1 treatment col."""
    weights = [[0.5, 0.5], [0.3, 0.7], [0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.2, 0.8]]
    outcome_by_state = [[1.0, 2.0], [0.5, -1.0], [3.0, 0.0], [-2.0, 1.0], [0.0, 0.0], [1.5, 2.5]]
    treatment_by_state = [[0.9, 0.1], [0.5, 0.5], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.8, 0.9]]
    y_pred = _make_y_pred(
        outcome_blocks=[outcome_by_state],
        weights=weights,
        treatment_blocks=[treatment_by_state],
    )
    y_true = tf.constant(
        [[1.0, 1.0], [0.0, 0.0], [2.0, 1.0], [-1.0, 0.0], [0.5, 0.0], [2.0, 1.0]],
        dtype=tf.float32,
    )
    return y_true, y_pred, np.asarray(weights), np.asarray(treatment_by_state)


@pytest.mark.parametrize("cls,col_kwargs,default_name", _METRIC_CLASSES, ids=_METRIC_IDS)
def test_metrics_have_explicit_default_names(cls, col_kwargs, default_name):
    """The default name is explicit, so Keras doesn't add a per-session `_N` suffix."""
    assert cls(**col_kwargs).name == default_name
    assert cls(**col_kwargs).name == default_name


@pytest.mark.parametrize("cls,col_kwargs,default_name", _METRIC_CLASSES, ids=_METRIC_IDS)
def test_metrics_forward_name_and_kwargs(cls, col_kwargs, default_name):
    """`name` and other keyword arguments reach the Keras base metric."""
    metric = cls(**col_kwargs, name="custom", dtype="float64")
    assert metric.name == "custom"
    assert metric.dtype == "float64"


@pytest.mark.parametrize("cls,col_kwargs,default_name", _METRIC_CLASSES, ids=_METRIC_IDS)
def test_metric_config_round_trips(cls, col_kwargs, default_name):
    """`from_config(get_config())` rebuilds the metric with the same columns and result."""
    y_true, y_pred, _, _ = _metric_fixture()
    metric = cls(**col_kwargs, name="custom")
    config = metric.get_config()
    for key, value in col_kwargs.items():
        assert config[key] == value

    rebuilt = cls.from_config(config)
    assert rebuilt.name == "custom"
    metric.update_state(y_true, y_pred)
    rebuilt.update_state(y_true, y_pred)
    np.testing.assert_allclose(rebuilt.result().numpy(), metric.result().numpy(), rtol=1e-6)


def test_propensity_score_auc_config_keeps_curve():
    """The AUC's own arguments (`curve`, `num_thresholds`) survive a config round trip."""
    metric = metrics.PropensityScoreAUC(**_COL_KWARGS, curve="PR", num_thresholds=50)
    rebuilt = metrics.PropensityScoreAUC.from_config(metric.get_config())
    assert rebuilt.curve.name == "PR"
    assert rebuilt.num_thresholds == 50


def test_propensity_score_auc_honors_curve():
    """`curve="PR"` computes the PR AUC; it used to be dropped, which gave the ROC AUC."""
    y_true, y_pred, weights, treatment_by_state = _metric_fixture()
    marginal_propensity = (weights * treatment_by_state).sum(axis=1)[:, None]
    treatment_true = y_true[:, -1:]

    expected = {}
    for curve in ("ROC", "PR"):
        reference = tf.keras.metrics.AUC(curve=curve)
        reference.update_state(treatment_true, marginal_propensity)
        expected[curve] = reference.result().numpy()
    # The fixture must tell the two curves apart, or the test proves nothing.
    assert abs(expected["PR"] - expected["ROC"]) > 1e-3

    for curve in ("ROC", "PR"):
        metric = metrics.PropensityScoreAUC(**_COL_KWARGS, curve=curve)
        metric.update_state(y_true, y_pred)
        np.testing.assert_allclose(metric.result().numpy(), expected[curve], rtol=1e-5)
