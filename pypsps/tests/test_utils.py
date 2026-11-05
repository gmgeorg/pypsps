"""Test utils"""

import random

import numpy as np
import pytest
import tensorflow as tf

from .. import datasets, utils
from ..keras import models
from ..keras.layout import ColumnLayout


def test_get_n_cols_np_array():
    """get_n_cols works on a plain numpy array"""
    y = np.zeros((4, 3))
    assert utils.get_n_cols(y) == 3


def test_get_n_cols_eager_tensor():
    """get_n_cols works on an eager tf.Tensor"""
    y = tf.zeros((4, 3))
    assert utils.get_n_cols(y) == 3


def test_get_n_cols_graph_mode_tensor():
    """get_n_cols works on a tensor traced inside a tf.function (graph mode)"""

    @tf.function
    def _call(y):
        return utils.get_n_cols(y)

    n_cols = _call(tf.zeros((4, 3)))
    assert int(n_cols) == 3


def test_get_n_cols_keras_symbolic_tensor():
    """get_n_cols works on a symbolic KerasTensor (e.g. from a functional-API Input)"""
    y = tf.keras.layers.Input(shape=(3,))
    assert utils.get_n_cols(y) == 3


def test_toy_model_fits_in_eager_and_graph_mode():
    """The default posterior-weighted causal loss traces get_n_cols through nested
    CausalLoss -> OutcomeLoss calls; this must work both eagerly and in graph mode
    (Keras 3's default compile_loss tracing), which is what originally broke with
    tf.Tensor.get_shape().as_list().
    """
    np.random.seed(13)
    ks_data = datasets.KangSchafer(true_ate=10).sample(n_samples=200)
    inputs, outputs = ks_data.to_keras_inputs_outputs()

    for run_eagerly in (True, False):
        tf.random.set_seed(13)
        model = models.build_toy_model(
            n_states=3, n_features=ks_data.features.shape[1], compile=True
        )
        model.run_eagerly = run_eagerly
        history = model.fit(inputs, outputs, epochs=1, batch_size=64, verbose=0)
        loss = history.history["loss"][0]
        assert np.isfinite(loss), f"non-finite loss with run_eagerly={run_eagerly}"


def test_column_layout_from_model_normal_outcome():
    """get_column_layout reads (2, 1, 1) off a compiled Normal-outcome (loc+scale) model"""
    tf.random.set_seed(10)
    model = models.build_toy_model(n_states=3, n_features=4, compile=True)
    layout = ColumnLayout.from_model(model)
    assert layout.n_outcome_pred_cols == 2
    assert layout.n_treatment_pred_cols == 1
    assert layout.n_outcome_true_cols == 1


def test_column_layout_from_model_uncompiled_model_reads_the_output_layer():
    """An uncompiled builder model has a CausalOutput layer, so the layout is known."""
    model = models.build_toy_model(n_states=3, n_features=4, compile=False)
    assert ColumnLayout.from_model(model).n_outcome_pred_cols == 2
    assert ColumnLayout.from_model(model).n_treatment_pred_cols == 1
    assert ColumnLayout.from_model(model).n_outcome_true_cols == 1


def test_column_layout_from_model_plain_keras_model_raises():
    """A model with no CausalOutput layer and no pypsps loss gives a clear RuntimeError."""
    inp = tf.keras.Input(shape=(4,))
    model = tf.keras.Model(inp, tf.keras.layers.Dense(9)(inp))
    with pytest.raises(RuntimeError, match="[Cc]ompile"):
        ColumnLayout.from_model(model)


def test_column_layout_from_model_exponential_outcome():
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
    layout = ColumnLayout.from_model(model)
    assert layout.n_outcome_pred_cols == 1
    assert layout.n_treatment_pred_cols == 1
    assert layout.n_outcome_true_cols == 2


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
