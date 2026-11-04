"""Tests for the y_pred / y_true column layout."""

import itertools

import numpy as np
import pytest
import tensorflow as tf

from ..keras.layout import ColumnLayout

_OUTCOME_PARAMS = [("loc", "scale"), ("log_scale",), ("log_scale", "log_shape")]
_TREATMENT_PARAMS = [("prob",), ("loc", "scale")]


def _y_pred(layout: ColumnLayout, n_rows: int = 4) -> np.ndarray:
    """A y_pred with distinct values, so any misplaced column shows up."""
    width = layout.y_pred_width()
    return np.arange(n_rows * width, dtype="float32").reshape(n_rows, width)


@pytest.mark.parametrize(
    "n_states, outcome_params, treatment_params, n_true",
    list(itertools.product([1, 2, 5], _OUTCOME_PARAMS, _TREATMENT_PARAMS, [1, 2])),
)
def test_encode_decode_round_trip(n_states, outcome_params, treatment_params, n_true):
    """decode_pred(encode(parts)) == parts and encode(decode_pred(y)) == y."""
    layout = ColumnLayout(outcome_params, treatment_params, n_true, n_states)
    y = _y_pred(layout)
    parts = layout.decode_pred(y)
    assert tuple(parts.outcome) == outcome_params
    assert tuple(parts.treatment) == treatment_params
    for block in [*parts.outcome.values(), parts.state_weights, *parts.treatment.values()]:
        assert block.shape == (4, n_states)

    encoded = layout.encode(parts.outcome, parts.state_weights, parts.treatment)
    assert isinstance(encoded, np.ndarray)
    np.testing.assert_array_equal(encoded, y)

    again = layout.decode_pred(encoded)
    for name in outcome_params:
        np.testing.assert_array_equal(again.outcome[name], parts.outcome[name])
    np.testing.assert_array_equal(again.state_weights, parts.state_weights)


def test_decode_follows_the_interleaved_per_state_order():
    """Each param block holds one column per state; weights sit between the blocks."""
    layout = ColumnLayout(("loc", "scale"), ("prob",), 1, n_states=2)
    y = np.array([[10.0, 11.0, 20.0, 21.0, 0.3, 0.7, 0.4, 0.6]])
    parts = layout.decode_pred(y)
    np.testing.assert_array_equal(parts.outcome["loc"], [[10.0, 11.0]])
    np.testing.assert_array_equal(parts.outcome["scale"], [[20.0, 21.0]])
    np.testing.assert_array_equal(parts.state_weights, [[0.3, 0.7]])
    np.testing.assert_array_equal(parts.treatment["prob"], [[0.4, 0.6]])


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


def test_decode_true_reads_exactly_one_treatment_column_and_ignores_trailing():
    """The treatment is the one column right after the outcome columns (1a80698)."""
    y_true = np.array([[5.0, 1.0, 0.0, 99.0, 98.0]])
    parts = ColumnLayout(("log_scale",), ("prob",), n_outcome_true_cols=2).decode_true(y_true)
    np.testing.assert_array_equal(parts.outcome, [[5.0, 1.0]])
    np.testing.assert_array_equal(parts.treatment, [[0.0]])

    with pytest.raises(ValueError, match="at least 3"):
        ColumnLayout(("log_scale",), ("prob",), 2).decode_true(y_true[:, :2])


def test_width_mismatch_raises():
    """A known K pins the width; the error names the layout."""
    layout = ColumnLayout(("loc", "scale"), ("prob",), 1, n_states=3)
    with pytest.raises(ValueError, match="y_pred has 8 columns.*expects 12"):
        layout.decode_pred(np.zeros((2, 8)))


def test_non_divisible_width_raises_when_n_states_unknown():
    """Without K, the width must be a multiple of the columns per state."""
    layout = ColumnLayout(("loc", "scale"), ("prob",), 1)
    assert layout.resolve_n_states(np.zeros((2, 12))) == 3
    with pytest.raises(ValueError, match="not a multiple of 4"):
        layout.resolve_n_states(np.zeros((2, 13)))


def test_unknown_static_width_raises():
    """The width is read statically, so an unknown last dimension raises."""
    layout = ColumnLayout(("loc",), ("prob",), 1)

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
    def graph(t):
        """Decodes a tensor of unknown width."""
        return layout.decode_pred(t).state_weights

    with pytest.raises(ValueError, match="static last dimension"):
        graph.get_concrete_function()


def test_encode_rejects_missing_or_misshaped_param():
    """encode checks the keys and every block's shape."""
    layout = ColumnLayout(("loc", "scale"), ("prob",), 1, n_states=2)
    block = np.zeros((3, 2))
    with pytest.raises(KeyError, match="outcome keys"):
        layout.encode({"loc": block}, block, {"prob": block})
    with pytest.raises(ValueError, match=r"outcome\['scale'\] must be \[N, 2\]"):
        layout.encode({"loc": block, "scale": np.zeros((3, 1))}, block, {"prob": block})
    with pytest.raises(ValueError, match="K=2"):
        layout.encode({"loc": block, "scale": block}, np.zeros((3, 3)), {"prob": block})
    with pytest.raises(ValueError, match="treatment block must have 2 columns"):
        layout.encode({"loc": block, "scale": block}, block, np.zeros((3, 4)))
    # The whole treatment block also works, and tensors give a tensor back.
    out = layout.encode({"loc": block, "scale": block}, tf.constant(block), block)
    assert isinstance(out, tf.Tensor) and out.shape == (3, 8)


def test_names_are_validated():
    """Parameter names must be non-empty and unique."""
    with pytest.raises(ValueError, match="unique"):
        ColumnLayout(("loc", "loc"), ("prob",), 1)
    with pytest.raises(ValueError, match="empty"):
        ColumnLayout((), ("prob",), 1)


@pytest.mark.parametrize("n_states", [None, 4])
def test_config_round_trip(n_states):
    """The layout serializes to plain lists and rebuilds equal."""
    layout = ColumnLayout(("log_scale", "log_shape"), ("loc", "scale"), 2, n_states)
    config = tf.keras.utils.serialize_keras_object(layout)
    assert config["config"]["outcome_params"] == ["log_scale", "log_shape"]
    assert tf.keras.utils.deserialize_keras_object(config) == layout
