"""Test module for pypsps keras layers."""

import numpy as np
import pytest
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


def test_state_column_gate_selects_its_own_column_in_a_loop():
    """Gates built in a loop each select their own column."""
    n_states = 3
    inp = tf.keras.Input(shape=(n_states,))
    outs = [layers.StateColumnGate(i)(inp) for i in range(n_states)]
    model = tf.keras.Model(inp, tf.keras.layers.Concatenate()(outs[::-1]))
    probe = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    np.testing.assert_array_equal(model.predict(probe, verbose=0), probe[:, ::-1])


def test_state_column_gate_survives_save_and_load(tmp_path):
    """A model of gates round-trips through model.save / load_model."""
    inp = tf.keras.Input(shape=(3,))
    outs = [layers.StateColumnGate(i)(inp) for i in (2, 0)]
    model = tf.keras.Model(inp, tf.keras.layers.Concatenate()(outs))
    path = str(tmp_path / "gates.keras")
    model.save(path)
    reloaded = tf.keras.models.load_model(path)
    probe = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    np.testing.assert_array_equal(reloaded.predict(probe, verbose=0), [[30.0, 10.0]])
    assert [g.index for g in reloaded.layers if isinstance(g, layers.StateColumnGate)] == [2, 0]


def test_state_column_gate_config_and_validation():
    """get_config carries the index; invalid indices raise ValueError."""
    assert layers.StateColumnGate(4).get_config()["index"] == 4
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="index"):
            layers.StateColumnGate(bad)


def test_soft_clip_stays_inside_bounds_with_nonzero_gradient():
    """Outputs lie in (min, max) and the gradient stays positive well beyond the bounds."""
    clip = layers.SoftClip(-2.0, 3.0)
    x = tf.constant([[-10.0], [-5.0], [0.0], [5.0], [10.0]])
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = clip(x)
    grad = tape.gradient(y, x).numpy()
    assert np.all(y.numpy() >= -2.0) and np.all(y.numpy() <= 3.0)
    np.testing.assert_allclose(y.numpy()[2], 0.5, rtol=1e-6)
    assert np.all(grad > 0.0)


def test_soft_clip_inverse_and_config():
    """inverse() maps back through the layer; the bounds round-trip through the config."""
    clip = layers.SoftClip(-2.0, 3.0)
    for value in (-1.9, 0.0, 2.5):
        np.testing.assert_allclose(clip(tf.constant([[clip.inverse(value)]])), [[value]], atol=1e-5)
    with pytest.raises(ValueError, match="lie in"):
        clip.inverse(3.0)
    clone = layers.SoftClip.from_config(clip.get_config())
    assert (clone.min_value, clone.max_value) == (-2.0, 3.0)
    for bad in ((1.0, 1.0), (2.0, 1.0), (0.0, np.inf)):
        with pytest.raises(ValueError, match="min_value < max_value"):
            layers.SoftClip(*bad)
