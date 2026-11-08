"""Schedules of the treatment-loss weight alpha, shared by both backends.

`AlphaSchedule` is the half-Gaussian schedule

    alpha(e) = alpha_end + (alpha_start - alpha_end) * exp(-decay_const * e**2)

for the zero-based epoch ``e``. It starts exactly at ``alpha_start`` with zero slope and
approaches ``alpha_end`` without a clamp. `SigmoidAlphaSchedule` is the logistic schedule of the
default callbacks. `ScheduleRun` drives either through a fit, and the backends' callbacks only
write its alpha into the loss. See ``docs/model_building.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

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


class MidFitResumeError(RuntimeError):
    """A scheduled fit was started at an epoch other than 0."""


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

    @property
    def half_life_epoch(self) -> int:
        """Epoch at which ``exp(-c * e**2)`` crosses 1/2, rounded: ``round(sqrt(ln 2 / c))``.

        Only a label for `plot_schedule`.
        """
        return round(math.sqrt(math.log(2.0) / float(self.decay_const)))

    @property
    def is_constant(self) -> bool:
        """True if ``alpha_start == alpha_end``."""
        return self.alpha_start == self.alpha_end

    @property
    def is_decay(self) -> bool:
        """True if ``alpha_start > alpha_end``."""
        return self.alpha_start > self.alpha_end

    @property
    def is_increase(self) -> bool:
        """True if ``alpha_start < alpha_end``."""
        return self.alpha_start < self.alpha_end

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


@dataclass(frozen=True)
class SigmoidAlphaSchedule:
    """A logistic schedule from ``alpha_start`` to ``alpha_end``, centered at ``half_life_epoch``.

    ``alpha(e) = alpha_end + (alpha_start - alpha_end) / (1 + exp(steepness * (e - half_life)))``,
    set to ``alpha_end`` once it is within ``clamp_threshold`` of it. Decay (start > end),
    increase (start < end) and constant schedules all work.

    Args:
        alpha_start: Alpha far before the half-life epoch.
        alpha_end: Alpha far after it.
        half_life_epoch: Epoch at which alpha is midway between the two. >= 0.
        steepness: Steepness of the logistic. > 0.
        clamp_threshold: Distance from ``alpha_end`` below which alpha is set to ``alpha_end``.
            >= 0. None means 10% of the range (0 for a constant schedule); the field then holds
            the resolved value.
    """

    alpha_start: float = 10.0
    alpha_end: float = 1.0
    half_life_epoch: int = 10
    steepness: float = 0.1
    clamp_threshold: float | None = None

    def __post_init__(self) -> None:
        """Validate the parameters and resolve the default clamp threshold."""
        if self.steepness <= 0:
            raise ValueError(f"steepness must be positive, got {self.steepness}")
        if self.half_life_epoch < 0:
            raise ValueError(f"half_life_epoch must be non-negative, got {self.half_life_epoch}")
        if self.clamp_threshold is None:
            alpha_range = abs(self.alpha_start - self.alpha_end)
            object.__setattr__(
                self, "clamp_threshold", 0.1 * alpha_range if alpha_range > 0 else 0.0
            )
        elif self.clamp_threshold < 0:
            raise ValueError(f"clamp_threshold must be non-negative, got {self.clamp_threshold}")

    @property
    def is_constant(self) -> bool:
        """True if ``alpha_start == alpha_end``."""
        return self.alpha_start == self.alpha_end

    @property
    def is_decay(self) -> bool:
        """True if ``alpha_start > alpha_end``."""
        return self.alpha_start > self.alpha_end

    @property
    def is_increase(self) -> bool:
        """True if ``alpha_start < alpha_end``."""
        return self.alpha_start < self.alpha_end

    def alpha(self, epoch: int) -> float:
        """Alpha during zero-based ``epoch``; exactly ``alpha_end`` for a very late epoch."""
        z = self.steepness * (epoch - self.half_life_epoch)
        if z >= 0:  # exp(-z) <= 1 cannot overflow; exp(z) would.
            e = math.exp(-z)
            factor = e / (1.0 + e)
        else:
            factor = 1.0 / (1.0 + math.exp(z))
        alpha = self.alpha_end + (self.alpha_start - self.alpha_end) * factor
        if abs(alpha - self.alpha_end) < self.clamp_threshold:
            alpha = self.alpha_end
        return float(alpha)


class _Schedule(Protocol):
    """What `ScheduleRun` and `plot_schedule` need of a schedule."""

    alpha_start: float
    alpha_end: float

    def alpha(self, epoch: int) -> float:
        """Alpha during zero-based ``epoch``."""
        ...


class ScheduleRun:
    """Drives a schedule through a fit.

    Call `begin` when training starts and `alpha_for` at the start of each epoch; the alpha is
    appended to ``history``.

    Args:
        schedule: `AlphaSchedule` or `SigmoidAlphaSchedule`.
        require_start_at_zero: If True, the first epoch of a fit must be 0, else
            `MidFitResumeError` is raised (resuming mid-fit would skip the warm-up), and `begin`
            resets ``history``. If False, any first epoch is fine and ``history`` accumulates
            across fits.
    """

    def __init__(self, schedule: _Schedule, require_start_at_zero: bool = False):
        """Store the schedule and clear the history."""
        self.schedule = schedule
        self.require_start_at_zero = bool(require_start_at_zero)
        self.history: list[float] = []
        self._started = False

    def begin(self) -> None:
        """Start a fit."""
        if self.require_start_at_zero:
            self.history = []
            self._started = False

    def alpha_for(self, epoch: int) -> float:
        """Alpha for zero-based ``epoch``, recorded in ``history``."""
        if self.require_start_at_zero and not self._started:
            self._started = True
            if int(epoch) != 0:
                raise MidFitResumeError(
                    f"training starts at epoch {epoch}, not 0; the schedule would skip its "
                    "warmup. Restart the fit from epoch 0."
                )
        alpha = self.schedule.alpha(int(epoch))
        self.history.append(alpha)
        return alpha


def plot_schedule(schedule: Any, max_epochs: int = 200, figsize: tuple[float, float] = (10, 6)):
    """The schedule over ``max_epochs`` epochs as a matplotlib figure.

    Args:
        schedule: `AlphaSchedule` or `SigmoidAlphaSchedule`.
        max_epochs: Number of epochs to plot.
        figsize: Figure size.

    Returns:
        The `matplotlib.figure.Figure`.
    """
    import matplotlib.pyplot as plt

    epochs = list(range(max_epochs))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, [schedule.alpha(e) for e in epochs], linewidth=2.5, color="#2E86AB")
    ax.axhline(
        y=schedule.alpha_end,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Target α = {schedule.alpha_end}",
    )
    ax.axhline(
        y=schedule.alpha_start,
        color="green",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Start α = {schedule.alpha_start}",
    )
    ax.axvline(
        x=schedule.half_life_epoch,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label=f"Half-life (epoch {schedule.half_life_epoch})",
    )
    if schedule.is_constant:
        kind = "Constant"
    elif schedule.is_decay:
        kind = "Decay"
    else:
        kind = "Increase"
    ax.set_title(f"Alpha {kind} Schedule", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Alpha (Treatment Loss Weight)", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
    return fig
