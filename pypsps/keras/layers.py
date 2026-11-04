"""Module for models & layers for pypsps."""

from typing import Optional

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="pypsps")
class BiasOnly(tf.keras.layers.Layer):
    """Bias-only layer (intercept only model).

    A trainable constant only layer for mapping features to a constant (trainable) value:
    BiasOnly()(features) --> constant
    """

    def __init__(
        self,
        units: int = 1,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        **kwargs,
    ):
        super(BiasOnly, self).__init__(**kwargs)
        self._bias_regularizer = bias_regularizer
        self._units = units

    def build(self, input_shape):
        """Builds the layer based on input_shape."""
        self._constant = self.add_weight(
            name="constant",
            shape=[
                self._units,
            ],
            initializer="zeros",
            regularizer=self._bias_regularizer,
            trainable=True,
        )

    def call(self, x):
        """Apply layer on a tensor."""
        return tf.expand_dims(tf.ones_like(x[:, 0]), 1) * self._constant

    def get_config(self):
        """Returns the layer config, including `units` (dropped before, silently reverting to 1
        on `from_config`) and a properly serialized `bias_regularizer`."""
        config = super().get_config().copy()
        config.update(
            {
                "units": self._units,
                "bias_regularizer": tf.keras.regularizers.serialize(self._bias_regularizer)
                if self._bias_regularizer is not None
                else None,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Deserializes `bias_regularizer` back into a Regularizer instance."""
        config = config.copy()
        config["bias_regularizer"] = tf.keras.regularizers.deserialize(config["bias_regularizer"])
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="pypsps")
class StateColumnGate(tf.keras.layers.Layer):
    """Selects column ``index`` of a 2D tensor as a ``(batch, 1)`` tensor.

    A serializable replacement for ``Lambda(lambda x: x[:, i : i + 1])`` in a per-state loop.
    """

    def __init__(self, index: int, **kwargs):
        """Stores the column ``index`` to select."""
        super().__init__(**kwargs)
        if isinstance(index, bool) or int(index) != index or index < 0:
            raise ValueError(f"index must be a non-negative integer, got {index!r}")
        self.index = int(index)

    def call(self, inputs):
        """Returns ``inputs[:, index : index + 1]``."""
        return inputs[:, self.index : self.index + 1]

    def get_config(self):
        """Returns the layer config, including ``index``."""
        config = super().get_config()
        config.update({"index": self.index})
        return config
