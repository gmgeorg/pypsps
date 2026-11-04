"""Half-Gaussian schedule for the treatment-loss weight alpha.

    alpha(e) = alpha_end + (alpha_start - alpha_end) * exp(-decay_const * e**2)

for the zero-based epoch ``e``. It starts exactly at ``alpha_start`` with zero slope and
approaches ``alpha_end`` without a clamp. See ``docs/model_building.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

DEFAULT_DECAY_CONST = 0.25
DEFAULT_WARMUP_EPSILON = 1e-3


def gaussian_alpha(epoch: int, alpha_start: float, alpha_end: float, decay_const: float) -> float:
    """Alpha at zero-based ``epoch``; inputs are not validated (see `AlphaSchedule`)."""
    return float(alpha_end + (alpha_start - alpha_end) * math.exp(-decay_const * epoch * epoch))


def warmup_epochs(
    alpha_start: float,
    alpha_end: float,
    decay_const: float,
    epsilon: float = DEFAULT_WARMUP_EPSILON,
) -> int:
    """First zero-based epoch whose distance to ``alpha_end`` is at most ``epsilon`` of the range.

    Equals ``ceil(sqrt(log(1 / epsilon) / decay_const))``, or 0 for a constant schedule.
    """
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon!r}")
    if alpha_start == alpha_end:
        return 0
    return math.ceil(math.sqrt(math.log(1.0 / epsilon) / decay_const))


def _finite(name: str, value: Any) -> float:
    """``value`` as a finite float, or a ``ValueError`` naming ``name``."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


@dataclass(frozen=True)
class AlphaSchedule:
    """A validated half-Gaussian alpha schedule.

    Args:
        alpha_start: Alpha at epoch 0. Finite and > 0.
        alpha_end: Alpha the schedule approaches. Finite and > 0.
        decay_const: ``c`` in ``exp(-c * e**2)``. Finite and > 0.
        epsilon: Relative tolerance that defines `warmup_epochs`, in (0, 1).
    """

    alpha_start: float = 10.0
    alpha_end: float = 1.0
    decay_const: float = DEFAULT_DECAY_CONST
    epsilon: float = DEFAULT_WARMUP_EPSILON

    def __post_init__(self) -> None:
        """Validate the parameters."""
        for name in ("alpha_start", "alpha_end", "decay_const"):
            if not _finite(name, getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")
        warmup_epochs(1.0, 1.0, 1.0, self.epsilon)  # validates epsilon

    @property
    def warmup_epochs(self) -> int:
        """First zero-based epoch at which alpha has settled (see `warmup_epochs`)."""
        return warmup_epochs(
            float(self.alpha_start), float(self.alpha_end), float(self.decay_const), self.epsilon
        )

    def alpha(self, epoch: int) -> float:
        """Alpha during zero-based ``epoch``."""
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0:
            raise ValueError(f"epoch must be a non-negative integer, got {epoch!r}")
        return gaussian_alpha(
            int(epoch), float(self.alpha_start), float(self.alpha_end), float(self.decay_const)
        )

    def check_epoch_cap(self, epochs: int) -> None:
        """Raise ``ValueError`` if ``epochs`` ends before the first settled epoch."""
        if int(epochs) <= self.warmup_epochs:
            raise ValueError(
                f"epochs={epochs} ends before alpha settles: the first settled zero-based "
                f"epoch is {self.warmup_epochs}, so at least {self.warmup_epochs + 1} epochs "
                "are needed"
            )


def schedule_from_params(params: Any) -> AlphaSchedule:
    """Build an `AlphaSchedule` from a mapping or an object with the same field names.

    Missing fields take the `AlphaSchedule` defaults.
    """
    get = params.get if hasattr(params, "get") else (lambda k, d: getattr(params, k, d))
    defaults = AlphaSchedule()
    return AlphaSchedule(
        alpha_start=get("alpha_start", defaults.alpha_start),
        alpha_end=get("alpha_end", defaults.alpha_end),
        decay_const=get("decay_const", defaults.decay_const),
        epsilon=get("epsilon", defaults.epsilon),
    )
