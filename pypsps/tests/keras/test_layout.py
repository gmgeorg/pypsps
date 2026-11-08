"""Tests for the column layout on TensorFlow tensors and Keras serialization.

The framework-free behaviour (order, widths, validation) is tested in
``tests/core/test_layout.py``.
"""

import numpy as np
import pytest
import tensorflow as tf

from ...keras.layout import ColumnLayout


def _y_pred(layout: ColumnLayout, n_rows: int = 4) -> np.ndarray:
    """A y_pred with distinct values, so any misplaced column shows up."""
    width = layout.y_pred_width()
    return np.arange(n_rows * width, dtype="float32").reshape(n_rows, width)


def test_decode_works_for_numpy_eager_graph_and_keras_tensors():
    """Decoding is plain slicing, so every tensor kind works and keeps static shapes."""
    layout = ColumnLayout(("loc", "scale"), ("prob",), 1, n_states=3)
    y = _y_pred(layout)
    expected = layout.decode_pred(y).outcome["scale"]

    np.testing.assert_array_equal(layout.decode_pred(tf.constant(y)).outcome["scale"], expected)

    @tf.function(input_signature=[tf.TensorSpec([None, layout.y_pred_width()], tf.float32)])
    def graph(t):
        """Decodes inside a traced function."""
        scale = layout.decode_pred(t).outcome["scale"]
        assert scale.shape.as_list() == [None, 3]
        return scale

    np.testing.assert_array_equal(graph(tf.constant(y)), expected)

    symbolic = tf.keras.Input(shape=(layout.y_pred_width(),))
    scale = layout.decode_pred(symbolic).outcome["scale"]
    assert tuple(scale.shape) == (None, 3)


def test_unknown_static_width_raises():
    """The width is read statically, so an unknown last dimension raises."""
    layout = ColumnLayout(("loc",), ("prob",), 1, n_states=2)

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
    def graph(t):
        """Decodes a tensor of unknown width."""
        return layout.decode_pred(t).state_weights

    with pytest.raises(ValueError, match="static last dimension"):
        graph.get_concrete_function()


def test_encode_of_tensors_gives_a_tensor():
    """A tensor among the blocks makes the encoded y_pred a tensor."""
    layout = ColumnLayout(("loc", "scale"), ("prob",), 1, n_states=2)
    block = np.zeros((3, 2))
    out = layout.encode({"loc": block, "scale": block}, tf.constant(block), block)
    assert isinstance(out, tf.Tensor) and out.shape == (3, 8)


@pytest.mark.parametrize("n_states", [1, 4])
def test_config_round_trip_through_keras(n_states):
    """The layout serializes through Keras to plain lists and rebuilds equal."""
    layout = ColumnLayout(("log_scale", "log_shape"), ("loc", "scale"), 2, n_states)
    config = tf.keras.utils.serialize_keras_object(layout)
    assert config["config"]["outcome_params"] == ["log_scale", "log_shape"]
    assert tf.keras.utils.deserialize_keras_object(config) == layout


def test_encode_true_keeps_tensors():
    """A TensorFlow outcome and treatment give a tensor y_true in the same column order."""
    layout = ColumnLayout(("log_scale", "log_shape"), ("prob",), 2, n_states=2)
    outcome = tf.constant([[1.0, 1.0], [2.0, 0.0]])
    treatment = tf.constant([0.0, 1.0])

    y = layout.encode_true(outcome, treatment)

    assert isinstance(y, tf.Tensor) and tuple(y.shape) == (2, 3)
    np.testing.assert_array_equal(y.numpy(), [[1, 1, 0], [2, 0, 1]])
    with pytest.raises(ValueError, match="outcome has 1 column"):
        layout.encode_true(outcome[:, :1], treatment)
