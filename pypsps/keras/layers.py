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
        config = super().get_config().copy()
        config.update(
            {
                "bias_regularizer": self._bias_regularizer,
            }
        )
        return config
