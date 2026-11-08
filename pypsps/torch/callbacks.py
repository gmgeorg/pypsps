"""Training callbacks for `pypsps.torch.training.fit`, mirroring `pypsps.keras.callbacks`.

A callback has the Keras hooks (``on_train_begin``, ``on_epoch_begin(epoch, logs)``,
``on_epoch_end(epoch, logs)``, ``on_train_end``) and, while ``fit`` runs, the attributes
``model``, ``loss`` (the `pypsps.torch.losses.CausalLoss`) and ``optimizer``. To stop
training, set ``self.model.stop_training = True``.
"""

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..core import alpha_schedule, checkpoint


class Callback:
    """Base class: every hook does nothing."""

    model: Any = None
    loss: Any = None
    optimizer: Any = None

    def set_context(self, model: torch.nn.Module, loss: Any, optimizer: Any) -> None:
        """Called by ``fit`` before training."""
        self.model, self.loss, self.optimizer = model, loss, optimizer

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        """Called once, before the first epoch."""

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Called at the start of each (zero-based) epoch."""

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Called at the end of each epoch with the epoch's logs (``logs`` may be modified)."""

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        """Called once, after the last epoch."""


class VerboseNEpochs(Callback):
    """Prints the epoch logs after the first epoch and then every ``n`` epochs."""

    def __init__(self, n: int = 10):
        """``n`` is the number of epochs between prints."""
        self.n = n

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Prints the logs."""
        if epoch == 0 or (epoch + 1) % self.n == 0:
            logs = logs or {}
            print(f"Epoch {epoch + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))


def _check_causal_loss(loss: Any) -> None:
    """Raises unless ``loss`` is a pypsps `CausalLoss` (it has the settable ``alpha``)."""
    from .losses import CausalLoss

    if not isinstance(loss, CausalLoss):
        raise AttributeError(
            f"the training loss must be a pypsps CausalLoss with an alpha; got {type(loss).__name__}"
        )


class AlphaScheduleCallback(Callback):
    """Moves alpha (the treatment-loss weight) along a sigmoid between two values.

    ``alpha(epoch) = alpha_end + (alpha_start - alpha_end) * sigmoid_factor(epoch)``, a logistic
    centered at ``half_life_epoch``. Decay (start > end) and increase (start < end) both work.
    The values taken are in ``alpha_history``.
    """

    def __init__(
        self,
        alpha_start: float = 10.0,
        alpha_end: float = 1.0,
        half_life_epoch: int = 10,
        steepness: float = 0.1,
        clamp_threshold: Optional[float] = None,
        verbose: bool = True,
    ):
        """Validates the schedule; ``clamp_threshold`` defaults to 10% of the alpha range."""
        self._init_run(
            alpha_schedule.SigmoidAlphaSchedule(
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                half_life_epoch=half_life_epoch,
                steepness=steepness,
                clamp_threshold=clamp_threshold,
            ),
            require_start_at_zero=False,
        )
        self.verbose = verbose

    def _init_run(self, schedule: Any, require_start_at_zero: bool) -> None:
        """Sets the schedule and the run that drives it (shared with `GaussianAlphaSchedule`)."""
        self.schedule = schedule
        self._run = alpha_schedule.ScheduleRun(schedule, require_start_at_zero)

    @property
    def alpha_start(self) -> float:
        """Alpha at the start of the schedule."""
        return self.schedule.alpha_start

    @property
    def alpha_end(self) -> float:
        """Alpha the schedule approaches."""
        return self.schedule.alpha_end

    @property
    def half_life_epoch(self) -> int:
        """Epoch at which alpha is midway between start and end."""
        return self.schedule.half_life_epoch

    @property
    def steepness(self) -> float:
        """Steepness of the sigmoid."""
        return self.schedule.steepness

    @property
    def clamp_threshold(self) -> float:
        """Distance from alpha_end below which alpha is set to alpha_end."""
        return self.schedule.clamp_threshold

    @property
    def alpha_history(self) -> List[float]:
        """The alphas set so far."""
        return self._run.history

    def _compute_alpha(self, epoch: int) -> float:
        """Alpha for the zero-based ``epoch``."""
        return self.schedule.alpha(int(epoch))

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        """Starts the run (a Gaussian schedule resets its history here)."""
        self._run.begin()

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Sets the loss's alpha for this epoch."""
        _check_causal_loss(self.loss)
        new_alpha = self._run.alpha_for(epoch)
        old_alpha = self.loss.alpha
        self.loss.alpha = new_alpha
        if self.verbose and not self.is_constant and (epoch % 10 == 0 or epoch < 5):
            direction = "↓" if self.is_decay else "↑"
            print(f"\n[Epoch {epoch}] Alpha: {old_alpha:.4f} {direction} {new_alpha:.4f}")

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        """Prints the final alpha."""
        if self.verbose and self.alpha_history:
            print(f"\n[Training Complete] Final Alpha: {self.alpha_history[-1]:.4f}")

    def plot_schedule(self, max_epochs: int = 200, figsize: Tuple[float, float] = (10, 6)) -> Any:
        """The schedule over ``max_epochs`` as a matplotlib figure."""
        return alpha_schedule.plot_schedule(self.schedule, max_epochs, figsize)

    @property
    def is_constant(self) -> bool:
        """True if ``alpha_start == alpha_end``."""
        return self.schedule.is_constant

    @property
    def is_decay(self) -> bool:
        """True if ``alpha_start > alpha_end``."""
        return self.schedule.is_decay

    @property
    def is_increase(self) -> bool:
        """True if ``alpha_start < alpha_end``."""
        return self.schedule.is_increase


MidFitResumeError = alpha_schedule.MidFitResumeError
NoEligibleCheckpointError = checkpoint.NoEligibleCheckpointError


class GaussianAlphaSchedule(AlphaScheduleCallback):
    """`AlphaScheduleCallback` with the half-Gaussian ramp-down of `pypsps.core.alpha_schedule`.

    Raises `MidFitResumeError` if training does not start at epoch 0.
    """

    def __init__(
        self,
        alpha_start: float = 10.0,
        alpha_end: float = 1.0,
        decay_const: float = alpha_schedule.DEFAULT_DECAY_CONST,
        epsilon: float = alpha_schedule.DEFAULT_WARMUP_EPSILON,
        verbose: bool = True,
    ):
        """Validates the schedule; ``warmup_epochs`` is its first settled epoch."""
        schedule = alpha_schedule.AlphaSchedule(
            alpha_start=alpha_start, alpha_end=alpha_end, decay_const=decay_const, epsilon=epsilon
        )
        self._init_run(schedule, require_start_at_zero=True)
        self.verbose = verbose
        self.decay_const = float(decay_const)

    @property
    def steepness(self) -> None:
        """Not a sigmoid: None."""
        return None

    @property
    def clamp_threshold(self) -> float:
        """A Gaussian schedule is never clamped."""
        return 0.0

    @property
    def warmup_epochs(self) -> int:
        """First zero-based epoch at which alpha has settled."""
        return self.schedule.warmup_epochs


class EligibleBestCheckpoint(Callback):
    """Early stopping that restores the best epoch at or after ``start_from_epoch``.

    An adapter over `pypsps.core.checkpoint.CheckpointSelector`, which owns the rules: an epoch
    is eligible if it is at or after ``start_from_epoch`` and its monitored value is finite. Ties
    go to the earliest epoch. Raises `NoEligibleCheckpointError` at the end of training if no
    epoch was eligible. ``trace`` records each epoch; ``selected_epoch`` and ``selected_value``
    are what was restored (the model's ``state_dict`` is restored, including batch-norm
    statistics).
    """

    def __init__(
        self,
        monitor: str,
        patience: int,
        start_from_epoch: int = 0,
        min_delta: float = 0.0,
        verbose: int = 0,
    ):
        """Minimizes ``monitor``; stops after ``patience`` eligible epochs without improvement."""
        self._selector = checkpoint.CheckpointSelector(
            monitor, patience, start_from_epoch=start_from_epoch, min_delta=min_delta
        )
        self.verbose = verbose
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None

    monitor = property(lambda self: self._selector.monitor)
    patience = property(lambda self: self._selector.patience)
    start_from_epoch = property(lambda self: self._selector.start_from_epoch)
    min_delta = property(lambda self: self._selector.min_delta)
    trace = property(lambda self: self._selector.trace)
    wait = property(lambda self: self._selector.wait)
    best = property(lambda self: self._selector.best)
    best_epoch = property(lambda self: self._selector.best_epoch)
    stopped_epoch = property(lambda self: self._selector.stopped_epoch)
    selected_epoch = property(lambda self: self._selector.selected_epoch)
    selected_value = property(lambda self: self._selector.selected_value)

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        """Resets the per-fit state."""
        self._selector.begin()
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Records the epoch; snapshots the weights if it improved; stops if patience ran out."""
        alpha = None if self.loss is None else float(getattr(self.loss, "alpha", np.nan))
        step = self._selector.observe(epoch, logs, alpha=alpha)
        if step.improved:
            self.best_weights = copy.deepcopy(self.model.state_dict())
        if step.stop:
            self.model.stop_training = True

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        """Restores the best eligible weights, or raises if no epoch was eligible."""
        self._selector.finish()
        self.model.load_state_dict(self.best_weights)
        if self.verbose:
            print(
                f"\nRestored epoch {self.selected_epoch} ({self.monitor}={self.selected_value:.4f})"
            )


class EarlyStopping(EligibleBestCheckpoint):
    """Stops after ``patience`` epochs without improvement of ``monitor`` and restores the best.

    `EligibleBestCheckpoint` with ``start_from_epoch=0``.
    """

    def __init__(self, monitor: str, patience: int, min_delta: float = 0.0, verbose: int = 0):
        """Minimizes ``monitor``."""
        super().__init__(monitor, patience, 0, min_delta, verbose)


class ReduceLROnPlateau(Callback):
    """Multiplies the learning rate by ``factor`` when ``monitor`` stops improving.

    Does nothing before ``start_from_epoch``. The new rate is at least ``min_lr``; ``cooldown``
    epochs after a reduction are not counted. Logs ``learning_rate`` into the epoch logs.
    """

    def __init__(
        self,
        monitor: str,
        factor: float = 0.1,
        patience: int = 10,
        min_delta: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0.0,
        start_from_epoch: int = 0,
        verbose: int = 0,
    ):
        """Minimizes ``monitor``."""
        if not 0.0 < factor < 1.0:
            raise ValueError(f"factor must be in (0, 1), got {factor}")
        self.monitor = monitor
        self.factor = factor
        self.patience = int(patience)
        self.min_delta = abs(float(min_delta))
        self.cooldown = int(cooldown)
        self.min_lr = float(min_lr)
        self.start_from_epoch = int(start_from_epoch)
        self.verbose = verbose
        self._reset()

    def _reset(self) -> None:
        self.best = math.inf
        self.wait = 0
        self.cooldown_counter = 0

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        """Resets the plateau state."""
        self._reset()

    def _learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Checks for a plateau and reduces the learning rate."""
        if logs is None:
            logs = {}
        logs["learning_rate"] = self._learning_rate()
        if epoch < self.start_from_epoch:
            return
        if self.monitor not in logs:
            raise KeyError(f"monitor {self.monitor!r} is not in the epoch logs: {sorted(logs)}")
        current = float(logs[self.monitor])
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        elif self.cooldown_counter <= 0:
            self.wait += 1
            if self.wait >= self.patience:
                for group in self.optimizer.param_groups:
                    group["lr"] = max(group["lr"] * self.factor, self.min_lr)
                if self.verbose:
                    print(f"\nEpoch {epoch + 1}: reducing learning rate to {self._learning_rate()}")
                self.cooldown_counter = self.cooldown
                self.wait = 0


def get_default_callbacks(
    monitor: str,
    patience: int,
    verbose: bool = False,
    alpha_start: float = 2.0,
    alpha_end: float = 1.0,
    half_life_epoch: int = 10,
    steepness: float = 0.1,
) -> List[Callback]:
    """Default callbacks: early stopping with best weights, LR reduction and an alpha schedule.

    Args:
      monitor: metric to monitor. No default on purpose; strongly recommended:
        ``"val_causal_loss_metric"``.
      patience: epochs without improvement before stopping (the learning rate is reduced after
        ``patience // 2``). Choose it larger than the steep part of the alpha warm-up, about
        ``half_life_epoch + 1 / steepness``.
      verbose: whether to print callback output.
      alpha_start, alpha_end, half_life_epoch, steepness: the `AlphaScheduleCallback` schedule.
    """
    args = checkpoint.default_callback_args(
        monitor,
        patience,
        verbose=verbose,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        half_life_epoch=half_life_epoch,
        steepness=steepness,
    )
    return [
        EarlyStopping(**args["early_stopping"]),
        ReduceLROnPlateau(**args["reduce_lr"]),
        AlphaScheduleCallback(**args["alpha_schedule"]),
    ]
