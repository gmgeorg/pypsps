"""Module for models & layers for pypsps."""

from typing import Any, Optional, Union

import numpy as np
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
        initializer: Union[str, tf.keras.initializers.Initializer] = "zeros",
        **kwargs: Any,
    ):
        """Initializes the layer.

        Args:
          units: width of the constant output.
          bias_regularizer: optional regularizer on the constant.
          initializer: initializer of the constant, e.g. ``Constant(raw_value)``.
          **kwargs: passed to the Keras Layer.
        """
        super(BiasOnly, self).__init__(**kwargs)
        self._bias_regularizer = bias_regularizer
        self._units = units
        self._initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape: tuple):
        """Builds the layer based on input_shape."""
        self._constant = self.add_weight(
            name="constant",
            shape=[
                self._units,
            ],
            initializer=self._initializer,
            regularizer=self._bias_regularizer,
            trainable=True,
        )

    def call(self, x: tf.Tensor):
        """Apply layer on a tensor."""
        return tf.expand_dims(tf.ones_like(x[:, 0]), 1) * self._constant

    def get_config(self):
        """Returns the layer config, including `units` (dropped before, silently reverting to 1
        on `from_config`), a properly serialized `bias_regularizer`, and the `initializer`."""
        config = super().get_config().copy()
        config.update(
            {
                "units": self._units,
                "bias_regularizer": tf.keras.regularizers.serialize(self._bias_regularizer)
                if self._bias_regularizer is not None
                else None,
                "initializer": tf.keras.initializers.serialize(self._initializer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict):
        """Deserializes `bias_regularizer` and `initializer`; older configs have no initializer."""
        config = config.copy()
        config["bias_regularizer"] = tf.keras.regularizers.deserialize(config["bias_regularizer"])
        if config.get("initializer") is not None:
            config["initializer"] = tf.keras.initializers.deserialize(config["initializer"])
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="pypsps")
class StateColumnGate(tf.keras.layers.Layer):
    """Selects column ``index`` of a 2D tensor as a ``(batch, 1)`` tensor.

    A serializable replacement for ``Lambda(lambda x: x[:, i : i + 1])`` in a per-state loop.
    """

    def __init__(self, index: int, **kwargs: Any):
        """Stores the column ``index`` to select."""
        super().__init__(**kwargs)
        if isinstance(index, bool) or int(index) != index or index < 0:
            raise ValueError(f"index must be a non-negative integer, got {index!r}")
        self.index = int(index)

    def call(self, inputs: tf.Tensor):
        """Returns ``inputs[:, index : index + 1]``."""
        return inputs[:, self.index : self.index + 1]

    def get_config(self):
        """Returns the layer config, including ``index``."""
        config = super().get_config()
        config.update({"index": self.index})
        return config


@tf.keras.utils.register_keras_serializable(package="pypsps")
class SoftClip(tf.keras.layers.Layer):
    """Maps inputs smoothly into ``(min_value, max_value)``: ``min + (max - min) * sigmoid(x)``.

    Unlike a hard clip, the gradient does not drop to zero at the bounds, so a parameter near a
    bound can still move back.
    """

    def __init__(self, min_value: float, max_value: float, **kwargs: Any):
        """Stores the bounds; requires finite ``min_value < max_value``."""
        super().__init__(**kwargs)
        min_value, max_value = float(min_value), float(max_value)
        if not (np.isfinite(min_value) and np.isfinite(max_value) and min_value < max_value):
            raise ValueError(
                f"need finite min_value < max_value, got ({min_value!r}, {max_value!r})"
            )
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs: tf.Tensor):
        """Returns ``min_value + (max_value - min_value) * sigmoid(inputs)``."""
        return self.min_value + (self.max_value - self.min_value) * tf.sigmoid(inputs)

    def inverse(self, value: float) -> float:
        """Returns the input that maps to ``value``, which must lie strictly inside the bounds."""
        if not self.min_value < value < self.max_value:
            raise ValueError(
                f"value must lie in ({self.min_value}, {self.max_value}), got {value!r}"
            )
        frac = (value - self.min_value) / (self.max_value - self.min_value)
        return float(np.log(frac / (1.0 - frac)))

    def get_config(self):
        """Returns the layer config, including the bounds."""
        config = super().get_config()
        config.update({"min_value": self.min_value, "max_value": self.max_value})
        return config


@tf.keras.utils.register_keras_serializable(package="pypsps")
class CausalOutput(tf.keras.layers.Layer):
    """The output layer of a pypsps model: encodes the heads into ``y_pred`` via the mixture.

    It replaces the final ``Concatenate`` and produces the same columns, ``[outcome params |
    state weights | treatment params]`` (see `pypsps.keras.layout.ColumnLayout.encode`). It has
    no weights. It carries the model's `pypsps.keras.mixture.StateMixture` in its config, so a
    model loaded with ``compile=False`` still knows its families and layout, and
    `pypsps.inference` runs on it (`StateMixture.from_model`).

    Call it on a dict ``{"outcome": ..., "state_weights": [N, K], "treatment": ...}``, where
    the outcome and treatment entries are ``{param name: [N, K]}`` or a whole parameter block.
    """

    def __init__(self, mixture: Any, **kwargs: Any):
        """Stores the `StateMixture` (or its serialized config)."""
        super().__init__(**kwargs)
        if isinstance(mixture, dict):
            mixture = tf.keras.utils.deserialize_keras_object(mixture)
        if mixture.n_states is None:
            raise ValueError("CausalOutput needs a StateMixture with n_states set")
        self._mixture = mixture

    @property
    def mixture(self) -> Any:
        """The model's `pypsps.keras.mixture.StateMixture`."""
        return self._mixture

    def call(self, inputs: dict):
        """Returns ``y_pred``, the encoded heads."""
        return self._mixture.layout.encode(
            inputs["outcome"], inputs["state_weights"], inputs["treatment"]
        )

    def compute_output_shape(self, input_shape: Any) -> tuple:
        """``(batch, layout width)``."""
        batch = input_shape["state_weights"][0]
        return (batch, self._mixture.layout.y_pred_width())

    def get_config(self):
        """Returns the layer config, including the serialized mixture."""
        config = super().get_config()
        config.update({"mixture": tf.keras.utils.serialize_keras_object(self._mixture)})
        return config
