"""Test utils"""

import numpy as np
import tensorflow as tf
import random

from .. import datasets
from ..keras import models
from .. import utils


def test_split_y_does_not_drop_columns():
    np.random.seed(13)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=1000)
    tf.random.set_seed(13)
    model = models.build_toy_model(
        n_states=5, n_features=ks_data.features.shape[1], compile=True
    )
    inputs, outputs = ks_data.to_keras_inputs_outputs()
    preds = model.predict(inputs)

    assert preds.shape[0] == ks_data.n_samples

    outcome_pred, scale_pred, weights, prop_score = utils.split_y_pred(preds)
    preds_comb = np.hstack([outcome_pred, scale_pred, weights, prop_score])
    np.testing.assert_allclose(preds, preds_comb)


def test_agg_outcome_preds_works():
    tf.random.set_seed(0)
    random.seed(0)
    np.random.seed(0)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=1000)
    tf.random.set_seed(13)
    model = models.build_toy_model(
        n_states=5, n_features=ks_data.features.shape[1], compile=True
    )
    inputs, outputs = ks_data.to_keras_inputs_outputs()

    _ = model.fit(
        inputs,
        outputs,
        epochs=2,
        batch_size=64,
        verbose=2,
        validation_split=0.2,
    )

    preds = model.predict(inputs)
    assert preds.shape[0] == ks_data.n_samples

    outcome_pred, _, weights, _ = utils.split_y_pred(preds)
    avg_outcome = utils.agg_outcome_pred(preds)
    assert avg_outcome.shape[0] == ks_data.n_samples

    np.testing.assert_allclose(avg_outcome, (outcome_pred * weights).sum(axis=1))
    cor_pred_true = np.corrcoef(avg_outcome, outputs[:, 0])
    print(cor_pred_true)
    assert cor_pred_true[0, 1] > 0.4


def test_prepare_keras_inputs_outputs():
    random.seed(0)
    np.random.seed(0)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=1000)

    res = utils.prepare_keras_inputs_outputs(
        ks_data.features, ks_data.treatments, ks_data.outcomes
    )

    res_direct = ks_data.to_keras_inputs_outputs()

    assert len(res) == 2
    assert len(res[0]) == 2

    np.testing.assert_allclose(res[0][0], ks_data.features.values.astype("float32"))
    np.testing.assert_allclose(res[0][1], ks_data.treatments.values)
    np.testing.assert_allclose(
        res[1][:, 0:1], ks_data.outcomes.values.astype("float32")
    )

    np.testing.assert_allclose(res[0][0], res_direct[0][0].astype("float32"))
    np.testing.assert_allclose(
        res[1].astype("float32"), res_direct[1].astype("float32")
    )
