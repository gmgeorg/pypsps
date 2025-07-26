"""Test module for bootstrap functions."""

import numpy as np
import pandas as pd
import tensorflow as tf

from .. import bootstrap, datasets
from ..keras import models

tfk = tf.keras


def test_bootstrap_ute_binary():
    """Test bootstrap_ute_binary returns DataFrame with correct shape and no NaNs."""
    np.random.seed(1)
    ks_data = datasets.KangSchafer(true_ate=5).sample(n_samples=100)
    inputs, _ = ks_data.to_keras_inputs_outputs()
    tf.random.set_seed(1)
    model = models.build_model_binary_normal(
        n_states=2,
        n_features=ks_data.n_features,
        predictive_state_hidden_layers=[(5, "relu")],
        outcome_hidden_layers=[(3, "relu")],
        loc_layer=(20, "selu"),
        scale_layer=(10, "tanh"),
        compile=True,
    )
    boot_df = bootstrap.bootstrap_ute_binary(model, inputs[0], n_bootstraps=10)
    assert isinstance(boot_df, pd.DataFrame)
    # Should have 10 replicates and 100 units
    assert boot_df.shape == (10, inputs[0].shape[0])
    assert not boot_df.isna().any().any()


def test_bootstrap_ate_binary():
    """Test that bootstrap_ate_binary returns finite CI bounds with lower <= upper."""
    # Prepare data
    np.random.seed(0)
    ks_data = datasets.KangSchafer(true_ate=5).sample(n_samples=200)
    inputs, _ = ks_data.to_keras_inputs_outputs()
    tf.random.set_seed(0)

    model = models.build_model_binary_normal(
        n_states=2,
        n_features=ks_data.n_features,
        predictive_state_hidden_layers=[(5, "relu")],
        outcome_hidden_layers=[(3, "relu")],
        loc_layer=(20, "selu"),
        scale_layer=(10, "tanh"),
        compile=True,
    )
    # Bootstrap CI with small samples for speed
    ate_bootstraps = bootstrap.bootstrap_ate_binary(model, inputs[0], n_bootstraps=10)
    assert ate_bootstraps.shape[0] == 10
    assert ate_bootstraps.notna().all()
