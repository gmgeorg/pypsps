"""pypsps Keras components.

Importing this package registers every pypsps loss, metric, layer, family, state mixture and column layout
with Keras, plus the pypress layers and regularizers the builders put into a model, so
``tf.keras.models.load_model`` can find them. Submodules must import siblings with
``from . import x``, never this package, or the import would see it half-initialized.
"""

# Imported for their side effect: registering classes with Keras.
import pypress.keras.layers  # noqa: F401
import pypress.keras.regularizers  # noqa: F401

from . import families, layers, layout, losses, metrics, mixture, neglogliks

__all__ = [
    "families",
    "layers",
    "layout",
    "losses",
    "metrics",
    "mixture",
    "neglogliks",
]
