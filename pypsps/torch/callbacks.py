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

from ..core import alpha_schedule


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
        if steepness <= 0:
            raise ValueError(f"steepness must be positive, got {steepness}")
        if half_life_epoch < 0:
            raise ValueError(f"half_life_epoch must be non-negative, got {half_life_epoch}")
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.half_life_epoch = half_life_epoch
        self.steepness = steepness
        self.verbose = verbose
        self.alpha_history: List[float] = []
        self._is_constant = alpha_start == alpha_end
        self._is_decay = alpha_start > alpha_end
        alpha_range = abs(alpha_start - alpha_end)
        if clamp_threshold is None:
            self.clamp_threshold = 0.1 * alpha_range if alpha_range > 0 else 0.0
        elif clamp_threshold < 0:
            raise ValueError(f"clamp_threshold must be non-negative, got {clamp_threshold}")
        else:
            self.clamp_threshold = clamp_threshold

    def _compute_alpha(self, epoch: int) -> float:
        """Alpha for the zero-based ``epoch``."""
        factor = 1.0 / (1.0 + math.exp(self.steepness * (epoch - self.half_life_epoch)))
        alpha = self.alpha_end + (self.alpha_start - self.alpha_end) * factor
        if abs(alpha - self.alpha_end) < self.clamp_threshold:
            alpha = self.alpha_end
        return float(alpha)

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Sets the loss's alpha for this epoch."""
        _check_causal_loss(self.loss)
        new_alpha = self._compute_alpha(epoch)
        old_alpha = self.loss.alpha
        self.loss.alpha = new_alpha
        self.alpha_history.append(new_alpha)
        if self.verbose and not self._is_constant and (epoch % 10 == 0 or epoch < 5):
            direction = "↓" if self._is_decay else "↑"
            print(f"\n[Epoch {epoch}] Alpha: {old_alpha:.4f} {direction} {new_alpha:.4f}")

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        """Prints the final alpha."""
        if self.verbose and self.alpha_history:
            print(f"\n[Training Complete] Final Alpha: {self.alpha_history[-1]:.4f}")

    def plot_schedule(self, max_epochs: int = 200, figsize: Tuple[float, float] = (10, 6)) -> Any:
        """The schedule over ``max_epochs`` as a matplotlib figure."""
        import matplotlib.pyplot as plt

        epochs = np.arange(max_epochs)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(epochs, [self._compute_alpha(int(e)) for e in epochs], linewidth=2.5)
        ax.axhline(self.alpha_end, color="red", linestyle="--", label=f"alpha_end={self.alpha_end}")
        ax.axhline(
            self.alpha_start, color="green", linestyle="--", label=f"alpha_start={self.alpha_start}"
        )
        ax.axvline(self.half_life_epoch, color="orange", linestyle=":", label="half-life")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Alpha (treatment loss weight)")
        ax.legend()
        fig.tight_layout()
        return fig

    @property
    def is_constant(self) -> bool:
        """True if ``alpha_start == alpha_end``."""
        return self._is_constant

    @property
    def is_decay(self) -> bool:
        """True if ``alpha_start > alpha_end``."""
        return self._is_decay

    @property
    def is_increase(self) -> bool:
        """True if ``alpha_start < alpha_end``."""
        return not self._is_decay and not self._is_constant


class MidFitResumeError(RuntimeError):
    """A scheduled fit was started at an epoch other than 0."""


class NoEligibleCheckpointError(RuntimeError):
    """No epoch at or after ``start_from_epoch`` produced a finite monitored value."""


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
        self.schedule = alpha_schedule.AlphaSchedule(
            alpha_start=alpha_start, alpha_end=alpha_end, decay_const=decay_const, epsilon=epsilon
        )
        super().__init__(
            alpha_start=float(alpha_start),
            alpha_end=float(alpha_end),
            half_life_epoch=round(math.sqrt(math.log(2.0) / float(decay_const))),
            clamp_threshold=0.0,
            verbose=verbose,
        )
        self.decay_const = float(decay_const)
        self._started = False

    @property
    def warmup_epochs(self) -> int:
        """First zero-based epoch at which alpha has settled."""
        return self.schedule.warmup_epochs

    def _compute_alpha(self, epoch: int) -> float:
        """Alpha for the zero-based ``epoch``."""
        return self.schedule.alpha(int(epoch))

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        """Resets the history."""
        self.alpha_history = []
        self._started = False

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Refuses a fit that does not start at epoch 0, then sets the epoch's alpha."""
        if not self._started:
            self._started = True
            if int(epoch) != 0:
                raise MidFitResumeError(
                    f"training starts at epoch {epoch}, not 0; the schedule would skip its "
                    "warmup. Restart the fit from epoch 0."
                )
        super().on_epoch_begin(epoch, logs)


class EligibleBestCheckpoint(Callback):
    """Early stopping that restores the best epoch at or after ``start_from_epoch``.

    An epoch is eligible if it is at or after ``start_from_epoch`` and its monitored value is
    finite. Ties go to the earliest epoch. Raises `NoEligibleCheckpointError` at the end of
    training if no epoch was eligible. ``trace`` records each epoch; ``selected_epoch`` and
    ``selected_value`` are what was restored (the model's ``state_dict`` is restored,
    including batch-norm statistics).
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
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if start_from_epoch < 0:
            raise ValueError(f"start_from_epoch must be >= 0, got {start_from_epoch}")
        self.monitor = monitor
        self.patience = int(patience)
        self.start_from_epoch = int(start_from_epoch)
        self.min_delta = abs(float(min_delta))
        self.verbose = verbose
        self._reset()

    def _reset(self) -> None:
        """Clears all per-fit state."""
        self.trace: List[Dict[str, Any]] = []
        self.wait = 0
        self.best: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None
        self.stopped_epoch = 0
        self.selected_epoch: Optional[int] = None
        self.selected_value: Optional[float] = None

    def on_train_begin(self, logs: Optional[dict] = None) -> None:
        """Resets the per-fit state."""
        self._reset()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Records the epoch and updates the best eligible epoch."""
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
                "alpha": None if self.loss is None else float(getattr(self.loss, "alpha", np.nan)),
            }
        )
        if epoch < self.start_from_epoch:
            return
        if eligible and (self.best is None or current < self.best - self.min_delta):
            self.best = current
            self.best_epoch = int(epoch)
            self.best_weights = copy.deepcopy(self.model.state_dict())
            self.wait = 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = int(epoch)
            self.model.stop_training = True

    def on_train_end(self, logs: Optional[dict] = None) -> None:
        """Restores the best eligible weights, or raises if no epoch was eligible."""
        if self.best_weights is None:
            raise NoEligibleCheckpointError(
                f"no epoch >= {self.start_from_epoch} produced a finite {self.monitor!r}; "
                f"monitored values: {[t['monitored'] for t in self.trace]}"
            )
        self.model.load_state_dict(self.best_weights)
        self.selected_epoch = self.best_epoch
        self.selected_value = self.best
        if self.verbose:
            print(f"\nRestored epoch {self.selected_epoch} ({self.monitor}={self.best:.4f})")


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
    return [
        EarlyStopping(monitor, patience, verbose=int(verbose)),
        ReduceLROnPlateau(
            monitor, factor=0.2, patience=int(patience // 2), min_lr=1e-6, verbose=int(verbose)
        ),
        AlphaScheduleCallback(
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            half_life_epoch=half_life_epoch,
            steepness=steepness,
            verbose=verbose,
        ),
    ]
