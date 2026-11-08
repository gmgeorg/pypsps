"""Both backends compute every portable hidden-layer activation alike."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("torch")

import tensorflow as tf  # noqa: E402
import torch  # noqa: E402

from ...core.spec import HIDDEN_ACTIVATIONS  # noqa: E402
from ...torch import layers as torch_layers  # noqa: E402


def test_the_torch_table_is_the_portable_set():
    """The torch activation table has exactly the names core calls portable."""
    assert set(torch_layers._ACTIVATIONS) == set(HIDDEN_ACTIVATIONS)


@pytest.mark.parametrize("name", HIDDEN_ACTIVATIONS)
def test_backends_compute_every_portable_activation_alike(name):
    """Same name, same function (``leaky_relu``, for one, differs: slope 0.2 vs 0.01)."""
    x = np.linspace(-6.0, 6.0, 201).astype("float32")
    keras_out = tf.keras.activations.get(name)(tf.constant(x)).numpy()
    torch_out = torch_layers.activation(name)(torch.tensor(x)).numpy()
    np.testing.assert_allclose(torch_out, keras_out, rtol=0, atol=1e-6)
