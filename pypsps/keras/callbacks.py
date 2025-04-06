from typing import List

import tensorflow as tf


class VerboseNEpochs(tf.keras.callbacks.Callback):
    """Class to show epoch info after N epochs only."""

    def __init__(self, n: int = 10):
        """
        Callback to print logs every n epochs.
        :param n: int, number of epochs between log prints.
        """
        super().__init__()
        self.n = n

    def on_epoch_end(self, epoch, logs=None):
        """call at end of epoch"""
        # logs is a dictionary containing metric names and values.
        if (epoch == 0) or ((epoch + 1) % self.n == 0):
            logs = logs or {}
            log_str = f"Epoch {epoch + 1}: " + ", ".join(
                f"{key}={value:.4f}" for key, value in logs.items()
            )
            print(log_str)


def recommended_callbacks(
    monitor="val_loss", patience: int = 50, mode="min"
) -> List[tf.keras.callbacks.Callback]:
    """Return a list of recommended callbacks.

    This list is subject to change w/o notice. Do not rely on this in production.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True, mode=mode
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=patience // 3),
        tf.keras.callbacks.TerminateOnNaN(),
        VerboseNEpochs(n=20),
    ]
    return callbacks
