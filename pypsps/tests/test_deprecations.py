"""The three-int API warns (FutureWarning) and still matches the new path. Delete in v0.5.0."""

import json
import pathlib
import warnings

import numpy as np
import pytest
import tensorflow as tf

from .. import utils
from ..keras import families, losses, metrics, mixture, models, neglogliks, presets
from ..keras.layout import ColumnLayout

_REMOVED = "removed in v0.5.0"
_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "three_int"

N, K = 6, 3
_MIX = mixture.StateMixture(families.Normal(), families.Bernoulli(), n_states=K)


def _y_pred(seed=0):
    """``[N, K * (2 + 1 + 1)]``: loc, scale > 0, state weights on the simplex, probabilities."""
    rng = np.random.default_rng(seed)
    loc = rng.normal(size=(N, K))
    scale = rng.uniform(0.5, 2.0, size=(N, K))
    weights = rng.dirichlet(np.ones(K), size=N)
    prob = rng.uniform(0.1, 0.9, size=(N, K))
    return np.hstack([loc, scale, weights, prob]).astype("float32")


def _y_true(seed=1):
    """``[N, 2]``: outcome, binary treatment."""
    rng = np.random.default_rng(seed)
    return np.hstack([rng.normal(size=(N, 1)), rng.integers(0, 2, size=(N, 1))]).astype("float32")


def _legacy_kwargs():
    return dict(n_outcome_true_cols=1, n_outcome_pred_cols=2, n_treatment_pred_cols=1)


# --- utils helpers ---


def test_split_y_pred_warns_and_matches_decode_pred():
    y = _y_pred()
    with pytest.warns(FutureWarning, match=_REMOVED):
        outcome, weights, treatment = utils.split_y_pred(y, 2, 1)
    parts = _MIX.layout.decode_pred(y)
    np.testing.assert_array_equal(
        outcome, np.hstack([parts.outcome["loc"], parts.outcome["scale"]])
    )
    np.testing.assert_array_equal(weights, parts.state_weights)
    np.testing.assert_array_equal(treatment, parts.treatment["prob"])


def test_split_outcome_pred_warns():
    y = _y_pred()
    with pytest.warns(FutureWarning, match=_REMOVED):
        loc, scale = utils.split_outcome_pred(y[:, : 2 * K], 2)
    np.testing.assert_array_equal(loc, _MIX.layout.decode_pred(y).outcome["loc"])
    np.testing.assert_array_equal(scale, _MIX.layout.decode_pred(y).outcome["scale"])


def test_split_y_true_warns_and_matches_decode_true():
    y = _y_true()
    with pytest.warns(FutureWarning, match=_REMOVED):
        outcome, treatment = utils.split_y_true(y, 1)
    parts = _MIX.layout.decode_true(y)
    np.testing.assert_array_equal(outcome, parts.outcome)
    np.testing.assert_array_equal(treatment, parts.treatment)


def test_get_n_states_and_column_indices_warn():
    with pytest.warns(FutureWarning, match=_REMOVED):
        assert utils.get_n_states(_y_pred(), 2, 1) == K
    with pytest.warns(FutureWarning, match=_REMOVED):
        assert utils.get_state_column_indices(1, K, 2) == [1, 1 + K]


def test_agg_outcome_pred_warns_about_the_raw_parameter_mix():
    """It keeps its old (wrong for loc + scale) semantics and says so."""
    y = _y_pred()
    with pytest.warns(FutureWarning, match="raw outcome parameters") as record:
        agg = utils.agg_outcome_pred(y, 2, 1)
    assert len(record) == 1  # the internal split_y_pred call is silent
    assert agg.shape == (N, 2)


def test_agg_treatment_pred_warns_and_matches_treatment_mean():
    y = _y_pred()
    with pytest.warns(FutureWarning, match=_REMOVED) as record:
        agg = utils.agg_treatment_pred(y, 2, 1)
    assert len(record) == 1
    np.testing.assert_allclose(agg, _MIX.treatment_mean(y), rtol=1e-6)


def test_get_column_layout_warns_and_matches_from_model():
    model = models.build_model(**presets.toy(n_states=K, n_features=4, compile=False))
    with pytest.warns(FutureWarning, match=_REMOVED):
        layout = utils.get_column_layout(model)
    new = ColumnLayout.from_model(model)
    assert layout == utils.ColumnLayout(
        new.n_outcome_pred_cols, new.n_treatment_pred_cols, new.n_outcome_true_cols
    )


# --- per-state NLL helpers ---


def test_negloglik_per_state_and_posterior_warn_and_match_the_mixture():
    y_pred, y_true = _y_pred(), _y_true()
    parts = _MIX.layout.decode_pred(y_pred)
    bce = tf.keras.losses.BinaryCrossentropy(reduction="none")
    with pytest.warns(FutureWarning, match=_REMOVED):
        nll = neglogliks.negloglik_per_state(bce, y_true[:, 1:2], parts.treatment["prob"], K)
    with pytest.warns(FutureWarning, match=_REMOVED):
        gamma = neglogliks.posterior_from_negloglik_per_state(parts.state_weights, nll)
    np.testing.assert_allclose(gamma, _MIX.state_posterior(y_true, y_pred), rtol=1e-5)


def test_prob_state_given_treatment_features_warns_once():
    y_pred, y_true = _y_pred(), _y_true()
    parts = _MIX.layout.decode_pred(y_pred)
    bce = tf.keras.losses.BinaryCrossentropy(reduction="none")
    with pytest.warns(FutureWarning, match=_REMOVED) as record:
        gamma = losses.prob_state_given_treatment_features(
            parts.state_weights, y_true[:, 1:2], parts.treatment["prob"], bce
        )
    assert len(record) == 1
    np.testing.assert_allclose(gamma, _MIX.state_posterior(y_true, y_pred), rtol=1e-5)


# --- loss and metric constructors ---


def test_legacy_loss_constructors_warn_and_match_the_mixture_path():
    y_pred, y_true = _y_pred(), _y_true()
    bce = tf.keras.losses.BinaryCrossentropy(reduction="none")
    with pytest.warns(FutureWarning, match=_REMOVED):
        outcome = losses.OutcomeLoss(
            loss=neglogliks.NegloglikNormal(reduction="none"),
            treatment_loss=bce,
            **_legacy_kwargs(),
        )
    with pytest.warns(FutureWarning, match=_REMOVED):
        treatment = losses.TreatmentLoss(loss=bce, **_legacy_kwargs())
    np.testing.assert_allclose(outcome(y_true, y_pred), losses.OutcomeLoss(_MIX)(y_true, y_pred))
    np.testing.assert_allclose(
        treatment(y_true, y_pred), losses.TreatmentLoss(_MIX)(y_true, y_pred)
    )


@pytest.mark.parametrize(
    "cls",
    [
        metrics.PropensityScoreBinaryCrossentropy,
        metrics.PropensityScoreAUC,
        metrics.TreatmentMeanSquaredError,
        metrics.OutcomeMeanSquaredError,
    ],
)
def test_legacy_metric_constructors_warn(cls):
    kwargs = dict(n_outcome_pred_cols=2, n_treatment_pred_cols=1)
    if cls in (metrics.TreatmentMeanSquaredError, metrics.OutcomeMeanSquaredError):
        kwargs["n_outcome_true_cols"] = 1
    with pytest.warns(FutureWarning, match=_REMOVED):
        legacy = cls(**kwargs)
    new = cls(_MIX)
    y_pred, y_true = _y_pred(), _y_true()
    legacy.update_state(y_true, y_pred)
    new.update_state(y_true, y_pred)
    np.testing.assert_allclose(float(legacy.result()), float(new.result()), rtol=1e-6)


@pytest.mark.parametrize("gen", [metrics.predictive_state_df_gen, metrics.uniform_entropy_gen])
def test_legacy_generator_counts_warn(gen):
    with pytest.warns(FutureWarning, match=_REMOVED):
        legacy = gen(2, 1)
    new = gen(_MIX.layout)
    y_pred, y_true = _y_pred(), _y_true()
    legacy.update_state(y_true, y_pred)
    new.update_state(y_true, y_pred)
    np.testing.assert_allclose(float(legacy.result()), float(new.result()), rtol=1e-6)


# --- saved three-int configs load silently ---


@pytest.mark.parametrize("name", ["toy", "binary_exponential", "continuous_weibull"])
def test_loading_three_int_configs_does_not_warn(name):
    """Old files load without FutureWarnings, though they go through the legacy path."""
    loss_config = json.loads((_FIXTURES / f"{name}_loss.json").read_text())
    metric_configs = json.loads((_FIXTURES / f"{name}_metrics.json").read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        loss = tf.keras.utils.deserialize_keras_object(loss_config)
        loss._outcome_loss.penalty_free()
        for config in metric_configs:
            tf.keras.utils.deserialize_keras_object(config)


def test_the_builder_workflow_raises_no_future_warning(tmp_path):
    """Build, fit, save, load and run inference without touching the deprecated API."""
    from .. import inference

    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 4)).astype("float32")
    t = (np.arange(32) % 2).astype("float32")[:, None]
    y = np.hstack([rng.normal(size=(32, 1)), t]).astype("float32")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model = models.build_model(**presets.toy(n_states=K, n_features=4))
        model.fit([x, t], y, epochs=1, verbose=0)
        model.save(str(tmp_path / "m.keras"))
        reloaded = tf.keras.models.load_model(str(tmp_path / "m.keras"))
        reloaded.evaluate([x, t], y, verbose=0)
        inference.predict_ute_binary(reloaded, x)
