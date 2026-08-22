"""Test module for pypsps keras layers."""

import numpy as np
import tensorflow as tf

from ..keras import layers


def test_bias_only_forward_pass():
    """BiasOnly maps any input to the same trainable constant, broadcast per row."""
    layer = layers.BiasOnly(units=3)
    x = tf.zeros((5, 4))
    out = layer(x)
    assert out.shape == (5, 3)
    np.testing.assert_allclose(out.numpy(), np.zeros((5, 3)))


def test_bias_only_get_config_roundtrip_preserves_units_and_regularizer():
    """get_config/from_config must round-trip `units` and `bias_regularizer` exactly.

    Previously `units` was dropped from get_config entirely (silently reverting to 1 on
    from_config), and `bias_regularizer` was stored raw instead of via
    tf.keras.regularizers.serialize/deserialize.
    """
    layer = layers.BiasOnly(units=4, bias_regularizer=tf.keras.regularizers.l2(0.1))
    layer.build(input_shape=(None, 2))

    config = layer.get_config()
    assert config["units"] == 4
    assert isinstance(config["bias_regularizer"], dict)

    restored = layers.BiasOnly.from_config(config)
    assert restored._units == 4
    assert isinstance(restored._bias_regularizer, tf.keras.regularizers.L2)
    np.testing.assert_allclose(restored._bias_regularizer.l2, 0.1)


def test_bias_only_get_config_roundtrip_no_regularizer():
    """from_config must handle the default bias_regularizer=None case."""
    layer = layers.BiasOnly(units=1)
    config = layer.get_config()
    assert config["bias_regularizer"] is None

    restored = layers.BiasOnly.from_config(config)
    assert restored._units == 1
    assert restored._bias_regularizer is None
