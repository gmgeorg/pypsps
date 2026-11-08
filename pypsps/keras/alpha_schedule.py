"""The alpha schedules; they live in `pypsps.core.alpha_schedule` (no TensorFlow).

Re-exported here so ``pypsps.keras.alpha_schedule`` keeps working.
"""

from ..core.alpha_schedule import (  # noqa: F401
    DEFAULT_DECAY_CONST,
    DEFAULT_WARMUP_EPSILON,
    AlphaSchedule,
    MidFitResumeError,
    ScheduleRun,
    SigmoidAlphaSchedule,
    gaussian_alpha,
    plot_schedule,
    schedule_from_params,
    warmup_epochs,
)
