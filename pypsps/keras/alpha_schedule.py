"""The half-Gaussian alpha schedule; it lives in `pypsps.core.alpha_schedule` (no TensorFlow).

Re-exported here so ``pypsps.keras.alpha_schedule`` keeps working.
"""

from ..core.alpha_schedule import (  # noqa: F401
    DEFAULT_DECAY_CONST,
    DEFAULT_WARMUP_EPSILON,
    AlphaSchedule,
    gaussian_alpha,
    schedule_from_params,
    warmup_epochs,
)
