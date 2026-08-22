"""Test utils"""

import random

import numpy as np
import tensorflow as tf

from .. import datasets, utils
from ..keras import models


def test_split_y_does_not_drop_columns():
    """test split_y"""
    np.random.seed(13)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=1000)
    tf.random.set_seed(13)
    model = models.build_toy_model(n_states=5, n_features=ks_data.features.shape[1], compile=True)
    inputs, outputs = ks_data.to_keras_inputs_outputs()
    preds = model.predict(inputs)

    assert preds.shape[0] == ks_data.n_samples

    outcome_params_pred, weights, treatment_params_pred = utils.split_y_pred(
        preds, n_outcome_pred_cols=2, n_treatment_pred_cols=1
    )
    preds_comb = np.hstack([outcome_params_pred, weights, treatment_params_pred])
    print(outcome_params_pred.shape)
    print(weights.shape)
    print(treatment_params_pred.shape)
    np.testing.assert_allclose(preds, preds_comb)


def test_get_column_layout_normal_outcome():
    """get_column_layout reads (2, 1, 1) off a compiled Normal-outcome (loc+scale) model"""
    tf.random.set_seed(10)
    model = models.build_toy_model(n_states=3, n_features=4, compile=True)
    layout = utils.get_column_layout(model)
    assert layout == utils.ColumnLayout(
        n_outcome_pred_cols=2, n_treatment_pred_cols=1, n_outcome_true_cols=1
    )


def test_get_column_layout_exponential_outcome():
    """get_column_layout reads (1, 1, 2) off a compiled exponential/survival (log_rate) model"""
    tf.random.set_seed(10)
    model = models.build_model_binary_exponential(
        n_states=3,
        n_features=4,
        compile=True,
        predictive_state_hidden_layers=[(10, "selu")],
        outcome_hidden_layers=[(10, "selu")],
        log_rate_layer=(10, "selu"),
    )
    layout = utils.get_column_layout(model)
    assert layout == utils.ColumnLayout(
        n_outcome_pred_cols=1, n_treatment_pred_cols=1, n_outcome_true_cols=2
    )


def test_agg_outcome_preds_works():
    """test aggregating by state works"""
    tf.random.set_seed(0)
    random.seed(0)
    np.random.seed(0)
    ks_data = datasets.KangSchafer(true_ate=10, seed=0).sample(n_samples=1000)
    tf.random.set_seed(13)
    n_states = 5
    model = models.build_toy_model(
        n_states=n_states, n_features=ks_data.features.shape[1], compile=True
    )
    inputs, outputs = ks_data.to_keras_inputs_outputs()

    # NOTE: needs more epochs than before the OutcomeLoss posterior-weighting fix: the
    # corrected loss couples the outcome and treatment heads through the posterior, which
    # is mathematically correct but converges a bit slower for this small toy model/lr.
    _ = model.fit(
        inputs,
        outputs,
        epochs=25,
        batch_size=64,
        verbose=2,
        validation_split=0.2,
    )

    preds = model.predict(inputs)
    assert preds.shape[0] == ks_data.n_samples

    outcome_params_pred, weights, _ = utils.split_y_pred(
        preds, n_outcome_pred_cols=2, n_treatment_pred_cols=1
    )
    avg_outcome = utils.agg_outcome_pred(preds, n_outcome_pred_cols=2, n_treatment_pred_cols=1)
    assert avg_outcome.shape[0] == ks_data.n_samples
    assert avg_outcome.shape[1] == 2

    avg_outcome_mean = avg_outcome[:, 0]
    np.testing.assert_allclose(
        avg_outcome_mean, (outcome_params_pred[:, :n_states] * weights).sum(axis=1)
    )
    cor_pred_true = np.corrcoef(avg_outcome_mean, outputs[:, 0])
    print(cor_pred_true)
    # Threshold has margin for numeric drift across TF/Keras versions (fixed seeds still
    # give a deterministic value per environment, but not identical across versions).
    assert cor_pred_true[0, 1] > 0.6


def test_prepare_keras_inputs_outputs():
    """test kersa inputs/outputs works"""
    random.seed(0)
    np.random.seed(0)
    ks_data = datasets.KangSchafer(true_ate=10, seed=0).sample(n_samples=1000)

    res = utils.prepare_keras_inputs_outputs(ks_data.features, ks_data.treatments, ks_data.outcomes)

    res_direct = ks_data.to_keras_inputs_outputs()

    assert len(res) == 2
    assert len(res[0]) == 2

    np.testing.assert_allclose(res[0][0], ks_data.features.values.astype("float32"))
    np.testing.assert_allclose(res[0][1], ks_data.treatments.values)
    np.testing.assert_allclose(res[1][:, 0:1], ks_data.outcomes.values.astype("float32"))

    np.testing.assert_allclose(res[0][0], res_direct[0][0].astype("float32"))
    np.testing.assert_allclose(res[1].astype("float32"), res_direct[1].astype("float32"))
