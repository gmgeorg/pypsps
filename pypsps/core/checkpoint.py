"""Best-epoch selection and the default callback description, shared by both backends.

`CheckpointSelector` owns the rules of early stopping with weight restoration: which epochs are
eligible, how ties and improvements are decided, when to stop and what to select. A backend's
callback only snapshots the weights when an epoch improved and restores the selected epoch's
weights at the end.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class NoEligibleCheckpointError(RuntimeError):
    """No epoch at or after ``start_from_epoch`` produced a finite monitored value."""


@dataclass(frozen=True)
class Step:
    """What the selector decided after one epoch.

    Attributes:
        improved: This epoch is the best eligible one so far; snapshot the weights.
        stop: ``patience`` eligible-or-later epochs passed without improvement; stop training.
    """

    improved: bool
    stop: bool


class CheckpointSelector:
    """Minimise ``monitor`` over the epochs of a fit and select the best one.

    An epoch is eligible if it is at or after ``start_from_epoch`` and its monitored value is
    finite. An eligible epoch improves on the best so far if it is lower by more than
    ``min_delta``, so ties go to the earliest epoch. Training stops after ``patience`` epochs
    at or after ``start_from_epoch`` without an improvement.

    Call `begin` when training starts, `observe` at the end of each epoch and `finish` when
    training ends.

    Attributes:
        trace: One dict per observed epoch: ``epoch``, ``monitored``, ``eligible`` and ``alpha``.
        best, best_epoch: Value and epoch of the best eligible epoch so far (None before any).
        wait: Epochs since the last improvement.
        stopped_epoch: Epoch at which `observe` said stop (0 if it never did).
        selected_epoch, selected_value: Set by `finish`.
    """

    def __init__(
        self, monitor: str, patience: int, start_from_epoch: int = 0, min_delta: float = 0.0
    ):
        """Validate the arguments."""
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if start_from_epoch < 0:
            raise ValueError(f"start_from_epoch must be >= 0, got {start_from_epoch}")
        self.monitor = monitor
        self.patience = int(patience)
        self.start_from_epoch = int(start_from_epoch)
        self.min_delta = abs(float(min_delta))
        self.begin()

    def begin(self) -> None:
        """Clear all per-fit state."""
        self.trace: list[dict] = []
        self.wait = 0
        self.best: float | None = None
        self.best_epoch: int | None = None
        self.stopped_epoch = 0
        self.selected_epoch: int | None = None
        self.selected_value: float | None = None

    def observe(
        self, epoch: int, logs: Mapping[str, Any] | None, alpha: float | None = None
    ) -> Step:
        """Record zero-based ``epoch`` with its ``logs`` and decide whether to snapshot or stop.

        Args:
            epoch: The epoch that just ended.
            logs: The epoch's logs; must contain ``monitor``.
            alpha: The loss weight alpha during the epoch, recorded in ``trace``.

        Raises:
            KeyError: if ``monitor`` is not in ``logs``.
        """
        logs = logs or {}
        if self.monitor not in logs:
            raise KeyError(f"monitor {self.monitor!r} is not in the epoch logs: {sorted(logs)}")
        current = float(logs[self.monitor])
        eligible = epoch >= self.start_from_epoch and math.isfinite(current)
        self.trace.append(
            {
                "epoch": int(epoch),
                "monitored": current,
                "eligible": bool(eligible),
                "alpha": alpha,
            }
        )
        if epoch < self.start_from_epoch:
            return Step(improved=False, stop=False)
        if eligible and (self.best is None or current < self.best - self.min_delta):
            self.best = current
            self.best_epoch = int(epoch)
            self.wait = 0
            return Step(improved=True, stop=False)
        self.wait += 1
        stop = self.wait >= self.patience
        if stop:
            self.stopped_epoch = int(epoch)
        return Step(improved=False, stop=stop)

    def finish(self) -> int:
        """The selected epoch: the best eligible one.

        Raises:
            NoEligibleCheckpointError: if no epoch was eligible.
        """
        if self.best_epoch is None:
            raise NoEligibleCheckpointError(
                f"no epoch >= {self.start_from_epoch} produced a finite {self.monitor!r}; "
                f"monitored values: {[t['monitored'] for t in self.trace]}"
            )
        self.selected_epoch = self.best_epoch
        self.selected_value = self.best
        return self.best_epoch


def default_callback_args(
    monitor: str,
    patience: int,
    verbose: bool = False,
    alpha_start: float = 2.0,
    alpha_end: float = 1.0,
    half_life_epoch: int = 10,
    steepness: float = 0.1,
) -> dict[str, dict]:
    """The keyword arguments of the three default callbacks of every backend.

    Args:
        monitor: Metric to monitor for early stopping and learning-rate reduction.
        patience: Epochs without improvement before stopping; the learning rate is reduced
            after ``patience // 2``.
        verbose: Whether the callbacks print.
        alpha_start, alpha_end, half_life_epoch, steepness: The sigmoid alpha schedule.

    Returns:
        ``{"early_stopping": {...}, "reduce_lr": {...}, "alpha_schedule": {...}}``: the arguments
        of the backend's ``EarlyStopping``, ``ReduceLROnPlateau`` and ``AlphaScheduleCallback``.
    """
    return {
        "early_stopping": {"monitor": monitor, "patience": patience, "verbose": int(verbose)},
        "reduce_lr": {
            "monitor": monitor,
            "factor": 0.2,
            "patience": int(patience // 2),
            "min_lr": 1e-6,
            "verbose": int(verbose),
        },
        "alpha_schedule": {
            "alpha_start": alpha_start,
            "alpha_end": alpha_end,
            "half_life_epoch": half_life_epoch,
            "steepness": steepness,
            "verbose": verbose,
        },
    }
