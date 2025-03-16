from typing import List

import tensorflow as tf


def recommended_callbacks(monitor="val_loss") -> List[tf.keras.callbacks.Callback]:
    """Return a list of recommended callbacks.

    This list is subject to change w/o notice. Do not rely on this in production.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=10),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    return callbacks
