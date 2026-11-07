"""Column layout of a pypsps model's ``y_pred`` and ``y_true`` tensors, registered with Keras.

The layout itself is framework-free and lives in `pypsps.core.layout`; see there for the
column order. This module only registers it with Keras (as ``pypsps>ColumnLayout``) and adds
`ColumnLayout.from_model`.
"""

import dataclasses

import tensorflow as tf

from ..core import layout as core_layout
from ..core.layout import Array, PredParts, TrueParts  # noqa: F401  (re-exported)


@tf.keras.utils.register_keras_serializable(package="pypsps")
@dataclasses.dataclass(frozen=True)
class ColumnLayout(core_layout.ColumnLayout):
    """`pypsps.core.layout.ColumnLayout`, serializable by Keras."""

    @classmethod
    def from_model(cls, model: tf.keras.Model) -> "ColumnLayout":
        """The layout of a pypsps model; see `pypsps.keras.mixture.StateMixture.from_model`."""
        # Lazy: the mixture module imports this one.
        from .mixture import StateMixture

        return StateMixture.from_model(model).layout
