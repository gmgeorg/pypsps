"""Module for pypsps training callbacks."""

import math
from typing import Any, Optional

import numpy as np
import tensorflow as tf

from pypsps.keras import alpha_schedule, losses


def _alpha_variable(model: tf.keras.Model) -> tf.Variable | None:
    """The alpha variable of the model's `CausalLoss`, or None if the loss isn't one.

    An isinstance check, not duck typing on ``.alpha``: Keras's own focal losses have a float
    ``alpha`` attribute that is not the causal-loss alpha.
    """
    loss = getattr(model, "loss", None)
    return loss.alpha if isinstance(loss, losses.CausalLoss) else None


class VerboseNEpochs(tf.keras.callbacks.Callback):
    """Class to show epoch info after N epochs only."""

    def __init__(self, n: int = 10):
        """
        Callback to print logs every n epochs.
        :param n: int, number of epochs between log prints.
        """
        super().__init__()
        self.n = n

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """call at end of epoch"""
        # logs is a dictionary containing metric names and values.
        if (epoch == 0) or ((epoch + 1) % self.n == 0):
            logs = logs or {}
            log_str = f"Epoch {epoch + 1}: " + ", ".join(
                f"{key}={value:.4f}" for key, value in logs.items()
            )
            print(log_str)


class AlphaScheduleCallback(tf.keras.callbacks.Callback):
    """
    Callback to dynamically adjust the alpha parameter (treatment loss weight)
    during training using a sigmoid schedule.

    Supports both decay (alpha_start > alpha_end) and increase (alpha_start < alpha_end)
    schedules, following a sigmoid-shaped curve controlled by a half-life parameter.

    Alpha schedule:
        alpha(epoch) = alpha_end + (alpha_start - alpha_end) * sigmoid_factor(epoch)

    Where sigmoid_factor uses a logistic function centered at the half-life epoch.

    Example Usage:
        ```python
        # Decay example: Start high, decay to 1.0
        alpha_decay = AlphaScheduleCallback(
            alpha_start=10.0,      # Start with high focus on treatment loss
            alpha_end=1.0,         # Decay to balanced joint likelihood
            half_life_epoch=50,    # Reach midpoint by epoch 50
            steepness=0.1,         # Smooth transition
            verbose=True
        )

        # Increase example: Start low, increase to target
        alpha_increase = AlphaScheduleCallback(
            alpha_start=0.1,       # Start with low treatment loss weight
            alpha_end=5.0,         # Increase to higher weight
            half_life_epoch=30,    # Reach midpoint by epoch 30
            steepness=0.15,        # Moderate transition speed
            verbose=True
        )

        # Preview the schedule before training
        fig = alpha_decay.plot_schedule(max_epochs=200)
        plt.show()

        # Use in model.fit()
        history = model.fit(
            [X_train, t_train],
            y_train,
            epochs=200,
            callbacks=[alpha_decay, other_callbacks],
            validation_data=([X_val, t_val], y_val)
        )

        # Access alpha history after training
        print(alpha_decay.alpha_history)
        ```
    """

    def __init__(
        self,
        alpha_start: float = 10.0,
        alpha_end: float = 1.0,
        half_life_epoch: int = 10,
        steepness: float = 0.1,
        clamp_threshold: float | None = None,
        verbose: bool = True,
    ):
        """
        Initialize the AlphaScheduleCallback.

        Args:
            alpha_start: Initial alpha value (default: 10.0)
            alpha_end: Final alpha value to approach (default: 1.0)
                      Can be less than alpha_start (decay) or greater (increase)
            half_life_epoch: Epoch at which alpha reaches the midpoint between start and end
            steepness: Controls the steepness of the sigmoid curve (default: 0.1)
                      Larger values = steeper transition
            clamp_threshold: Distance from alpha_end at which to clamp to alpha_end.
                           When |alpha - alpha_end| < clamp_threshold, alpha is set to alpha_end.
                           Default: None (auto-computed as 10% of |alpha_start - alpha_end|)
            verbose: Whether to print alpha updates (default: True)

        Raises:
            ValueError: If steepness is not positive or half_life_epoch is negative
        """
        super().__init__()

        # Validation
        if steepness <= 0:
            raise ValueError(f"steepness must be positive, got {steepness}")
        if half_life_epoch < 0:
            raise ValueError(f"half_life_epoch must be non-negative, got {half_life_epoch}")

        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.half_life_epoch = half_life_epoch
        self.steepness = steepness
        self.verbose = verbose
        self.alpha_history = []

        # Determine direction: decay (start > end), increase (start < end), or constant
        self._is_constant = alpha_start == alpha_end
        self._is_decay = alpha_start > alpha_end
        self._alpha_range = abs(alpha_start - alpha_end)

        # Auto-compute clamp threshold if not provided (10% of range, or 0 if constant)
        if clamp_threshold is None:
            self.clamp_threshold = 0.1 * self._alpha_range if self._alpha_range > 0 else 0.0
        else:
            if clamp_threshold < 0:
                raise ValueError(f"clamp_threshold must be non-negative, got {clamp_threshold}")
            self.clamp_threshold = clamp_threshold

    def _compute_alpha(self, epoch: int):
        """
        Compute alpha value for the current epoch using sigmoid schedule.

        The sigmoid function provides smooth transition from alpha_start to alpha_end,
        with the half_life_epoch determining the center of the transition.
        Works for both decay (start > end) and increase (start < end) schedules.
        """
        # Sigmoid factor: 1 / (1 + exp(steepness * (epoch - half_life)))
        # This gives 1.0 at epoch 0 (for small steepness) and approaches 0.0 as epoch -> inf
        sigmoid_factor = 1.0 / (1.0 + np.exp(self.steepness * (epoch - self.half_life_epoch)))

        # Scale: alpha = alpha_end + (alpha_start - alpha_end) * sigmoid_factor
        # At epoch 0 (sigmoid_factor ≈ high): alpha ≈ alpha_start
        # At epoch -> inf (sigmoid_factor ≈ 0): alpha ≈ alpha_end
        alpha = self.alpha_end + (self.alpha_start - self.alpha_end) * sigmoid_factor

        # Clamp to alpha_end when close enough (works for both decay and increase)
        if abs(alpha - self.alpha_end) < self.clamp_threshold:
            alpha = self.alpha_end

        return alpha

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        """Update alpha at the beginning of each epoch."""
        # Compute new alpha value
        new_alpha = self._compute_alpha(epoch)

        alpha = _alpha_variable(self.model)
        if alpha is None:
            loss_type = type(getattr(self.model, "loss", None)).__name__
            raise AttributeError(
                f"model.loss must be a pypsps CausalLoss with an alpha variable; got {loss_type}"
            )
        old_alpha = float(alpha.numpy())
        # ASSIGN, do not rebind: the already-traced training graph (model.fit() runs with
        # run_eagerly=False by default) reads the variable, never a rebound Python attribute.
        alpha.assign(new_alpha)
        self.alpha_history.append(new_alpha)

        # Print update if verbose (skip for constant schedules)
        if self.verbose and not self._is_constant and (epoch % 10 == 0 or epoch < 5):
            direction = "↓" if self._is_decay else "↑"
            print(f"\n[Epoch {epoch}] Alpha: {old_alpha:.4f} {direction} {new_alpha:.4f}")

    def on_train_end(self, logs: Optional[dict] = None):
        """Print final alpha value at the end of training."""
        if self.verbose and len(self.alpha_history) > 0:
            print(f"\n[Training Complete] Final Alpha: {self.alpha_history[-1]:.4f}")

    def plot_schedule(self, max_epochs: int = 200, figsize: tuple[float, float] = (10, 6)):
        """
        Plot the alpha schedule for visualization.

        Args:
            max_epochs: Maximum number of epochs to plot (default: 200)
            figsize: Figure size (default: (10, 6))

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        import matplotlib.pyplot as plt

        epochs = np.arange(max_epochs)
        alphas = [self._compute_alpha(epoch) for epoch in epochs]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(epochs, alphas, linewidth=2.5, color="#2E86AB")
        ax.axhline(
            y=self.alpha_end,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Target α = {self.alpha_end}",
        )
        ax.axhline(
            y=self.alpha_start,
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Start α = {self.alpha_start}",
        )
        ax.axvline(
            x=self.half_life_epoch,
            color="orange",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"Half-life (epoch {self.half_life_epoch})",
        )

        if self._is_constant:
            schedule_type = "Constant"
        elif self._is_decay:
            schedule_type = "Decay"
        else:
            schedule_type = "Increase"
        ax.set_title(f"Alpha {schedule_type} Schedule", fontsize=18, fontweight="bold", pad=15)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Alpha (Treatment Loss Weight)", fontsize=14)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=12)
        ax.tick_params(axis="both", labelsize=12)

        plt.tight_layout()
        return fig

    @property
    def is_constant(self) -> bool:
        """Returns True if this is a constant schedule (alpha_start == alpha_end)."""
        return self._is_constant

    @property
    def is_decay(self) -> bool:
        """Returns True if this is a decay schedule (alpha_start > alpha_end)."""
        return self._is_decay

    @property
    def is_increase(self) -> bool:
        """Returns True if this is an increase schedule (alpha_start < alpha_end)."""
        return not self._is_decay and not self._is_constant


class MidFitResumeError(RuntimeError):
    """A scheduled fit was started at an epoch other than 0."""


class NoEligibleCheckpointError(RuntimeError):
    """No epoch at or after ``start_from_epoch`` produced a finite monitored value."""


class GaussianAlphaSchedule(AlphaScheduleCallback):
    """`AlphaScheduleCallback` with a half-Gaussian ramp-down (see `alpha_schedule`).

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
        """Validate the schedule; ``warmup_epochs`` is its first settled epoch."""
        self.schedule = alpha_schedule.AlphaSchedule(
            alpha_start=alpha_start, alpha_end=alpha_end, decay_const=decay_const, epsilon=epsilon
        )
        # half_life_epoch only labels plot_schedule: exp(-c * e**2) = 1/2 at sqrt(ln 2 / c).
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

    def _compute_alpha(self, epoch: int):
        """Alpha for zero-based ``epoch``."""
        return self.schedule.alpha(int(epoch))

    def on_train_begin(self, logs: Optional[dict] = None):
        """Reset the alpha history."""
        self.alpha_history = []
        self._started = False

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        """Refuse a fit that does not start at epoch 0, then assign the epoch's alpha."""
        if not self._started:
            self._started = True
            if int(epoch) != 0:
                raise MidFitResumeError(
                    f"training starts at epoch {epoch}, not 0; the schedule would skip its "
                    "warmup. Restart the fit from epoch 0."
                )
        super().on_epoch_begin(epoch, logs)


def _live_alpha(model: tf.keras.Model) -> float | None:
    """The alpha the model's loss holds now, or None if it has none."""
    alpha = _alpha_variable(model)
    return None if alpha is None else float(alpha.numpy())


class EligibleBestCheckpoint(tf.keras.callbacks.Callback):
    """Early stopping that restores the best epoch at or after ``start_from_epoch``.

    An epoch is eligible if it is at or after ``start_from_epoch`` and its monitored value is
    finite. Ties go to the earliest epoch. Raises `NoEligibleCheckpointError` at the end of
    training if no epoch was eligible. ``trace`` records each epoch; ``selected_epoch`` and
    ``selected_value`` are what was restored.
    """

    def __init__(
        self,
        monitor: str,
        patience: int,
        start_from_epoch: int = 0,
        min_delta: float = 0.0,
        verbose: int = 0,
    ):
        """Minimise ``monitor``; stop after ``patience`` eligible epochs without improvement."""
        super().__init__()
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

    def _reset(self):
        """Clear all per-fit state."""
        self.trace: list[dict] = []
        self.wait = 0
        self.best: float | None = None
        self.best_epoch: int | None = None
        self.best_weights = None
        self.stopped_epoch = 0
        self.selected_epoch: int | None = None
        self.selected_value: float | None = None

    def on_train_begin(self, logs: Optional[dict] = None):
        """Reset the per-fit state."""
        self._reset()

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Record the epoch and update the best eligible epoch."""
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
                "alpha": _live_alpha(self.model),
            }
        )
        if epoch < self.start_from_epoch:
            return
        if eligible and (self.best is None or current < self.best - self.min_delta):
            self.best = current
            self.best_epoch = int(epoch)
            self.best_weights = self.model.get_weights()
            self.wait = 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = int(epoch)
            self.model.stop_training = True

    def on_train_end(self, logs: Optional[dict] = None):
        """Restore the best eligible weights, or raise if no epoch was eligible."""
        if self.best_weights is None:
            raise NoEligibleCheckpointError(
                f"no epoch >= {self.start_from_epoch} produced a finite {self.monitor!r}; "
                f"monitored values: {[t['monitored'] for t in self.trace]}"
            )
        self.model.set_weights(self.best_weights)
        self.selected_epoch = self.best_epoch
        self.selected_value = self.best
        if self.verbose:
            print(f"\nRestored epoch {self.selected_epoch} ({self.monitor}={self.best:.4f})")


class WarmupReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    """`ReduceLROnPlateau` that does nothing before ``start_from_epoch``."""

    def __init__(self, *args: Any, start_from_epoch: int = 0, **kwargs: Any):
        """``start_from_epoch`` is the first epoch at which the plateau check runs."""
        super().__init__(*args, **kwargs)
        self.start_from_epoch = int(start_from_epoch)

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Skip the plateau check before ``start_from_epoch``; still log the learning rate."""
        if epoch < self.start_from_epoch:
            if logs is not None:
                logs["learning_rate"] = float(np.asarray(self.model.optimizer.learning_rate))
            return
        super().on_epoch_end(epoch, logs)


def get_default_callbacks(
    monitor: str,
    patience: int,
    verbose: bool = False,
    alpha_start: float = 2.0,
    alpha_end: float = 1.0,
    half_life_epoch: int = 10,
    steepness: float = 0.1,
):
    """Create default training callbacks for early stopping and learning rate reduction.

    Args:
        monitor: Metric to monitor for early stopping / LR reduction. No default on
            purpose: callers must pick a metric that exists on their compiled model.
            Strongly recommended: "val_causal_loss_metric".
        patience: Number of epochs with no improvement before stopping/reducing LR
        verbose: Whether to print verbose callback output
        alpha_start: Initial alpha value for the schedule (default: 2.0)
        alpha_end: Final alpha value to approach (default: 1.0)
            Can be less than alpha_start (decay) or greater (increase)
        half_life_epoch: Epoch at which alpha reaches midpoint (default: 10)
        steepness: Controls steepness of sigmoid curve (default: 0.1)

    Returns:
        List of callbacks
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.2,
            patience=int(patience // 2),
            min_lr=1e-6,
            mode="min",
        ),
    ]

    alpha_schedule = AlphaScheduleCallback(
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        half_life_epoch=half_life_epoch,
        steepness=steepness,
        verbose=verbose,
    )

    return callbacks + [alpha_schedule]
