"""Module for pypsps training callbacks."""

from typing import Any, Optional

import numpy as np
import tensorflow as tf

from pypsps.core import checkpoint
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
        """Set the schedule and the run that drives it (shared with `GaussianAlphaSchedule`)."""
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
    def alpha_history(self) -> list[float]:
        """The alphas assigned so far."""
        return self._run.history

    def _compute_alpha(self, epoch: int) -> float:
        """Alpha for the zero-based ``epoch``."""
        return self.schedule.alpha(int(epoch))

    def on_train_begin(self, logs: Optional[dict] = None):
        """Start the run (a Gaussian schedule resets its history here)."""
        self._run.begin()

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        """Update alpha at the beginning of each epoch."""
        alpha = _alpha_variable(self.model)
        if alpha is None:
            loss_type = type(getattr(self.model, "loss", None)).__name__
            raise AttributeError(
                f"model.loss must be a pypsps CausalLoss with an alpha variable; got {loss_type}"
            )
        new_alpha = self._run.alpha_for(epoch)
        old_alpha = float(alpha.numpy())
        # ASSIGN, do not rebind: the already-traced training graph (model.fit() runs with
        # run_eagerly=False by default) reads the variable, never a rebound Python attribute.
        alpha.assign(new_alpha)

        # Print update if verbose (skip for constant schedules)
        if self.verbose and not self.is_constant and (epoch % 10 == 0 or epoch < 5):
            direction = "↓" if self.is_decay else "↑"
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
        return alpha_schedule.plot_schedule(self.schedule, max_epochs, figsize)

    @property
    def is_constant(self) -> bool:
        """Returns True if this is a constant schedule (alpha_start == alpha_end)."""
        return self.schedule.is_constant

    @property
    def is_decay(self) -> bool:
        """Returns True if this is a decay schedule (alpha_start > alpha_end)."""
        return self.schedule.is_decay

    @property
    def is_increase(self) -> bool:
        """Returns True if this is an increase schedule (alpha_start < alpha_end)."""
        return self.schedule.is_increase


MidFitResumeError = alpha_schedule.MidFitResumeError
NoEligibleCheckpointError = checkpoint.NoEligibleCheckpointError


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
        schedule = alpha_schedule.AlphaSchedule(
            alpha_start=alpha_start, alpha_end=alpha_end, decay_const=decay_const, epsilon=epsilon
        )
        super().__init__(verbose=verbose)
        self._init_run(schedule, require_start_at_zero=True)
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


def _live_alpha(model: tf.keras.Model) -> float | None:
    """The alpha the model's loss holds now, or None if it has none."""
    alpha = _alpha_variable(model)
    return None if alpha is None else float(alpha.numpy())


class EligibleBestCheckpoint(tf.keras.callbacks.Callback):
    """Early stopping that restores the best epoch at or after ``start_from_epoch``.

    An adapter over `pypsps.core.checkpoint.CheckpointSelector`, which owns the rules: an epoch
    is eligible if it is at or after ``start_from_epoch`` and its monitored value is finite. Ties
    go to the earliest epoch. Raises `NoEligibleCheckpointError` at the end of training if no
    epoch was eligible. ``trace`` records each epoch; ``selected_epoch`` and ``selected_value``
    are what was restored.
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
        self._selector = checkpoint.CheckpointSelector(
            monitor, patience, start_from_epoch=start_from_epoch, min_delta=min_delta
        )
        self.verbose = verbose
        self.best_weights = None

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

    def on_train_begin(self, logs: Optional[dict] = None):
        """Reset the per-fit state."""
        self._selector.begin()
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None):
        """Record the epoch; snapshot the weights if it improved; stop if patience ran out."""
        step = self._selector.observe(epoch, logs, alpha=_live_alpha(self.model))
        if step.improved:
            self.best_weights = self.model.get_weights()
        if step.stop:
            self.model.stop_training = True

    def on_train_end(self, logs: Optional[dict] = None):
        """Restore the best eligible weights, or raise if no epoch was eligible."""
        self._selector.finish()
        self.model.set_weights(self.best_weights)
        if self.verbose:
            print(
                f"\nRestored epoch {self.selected_epoch} ({self.monitor}={self.selected_value:.4f})"
            )


class EarlyStopping(EligibleBestCheckpoint):
    """Stops after ``patience`` epochs without improvement of ``monitor`` and restores the best.

    `EligibleBestCheckpoint` with ``start_from_epoch=0``. Unlike
    ``tf.keras.callbacks.EarlyStopping`` it raises `KeyError` if ``monitor`` is not in the epoch
    logs and `NoEligibleCheckpointError` if it is never finite.
    """

    def __init__(self, monitor: str, patience: int, min_delta: float = 0.0, verbose: int = 0):
        """Minimise ``monitor``."""
        super().__init__(monitor, patience, 0, min_delta, verbose)


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
        ``[EarlyStopping, ReduceLROnPlateau, AlphaScheduleCallback]``. The early stopping is
        pypsps's own `EarlyStopping`: it raises `KeyError` if ``monitor`` is not in the epoch
        logs (e.g. a ``val_`` metric without validation data) and `NoEligibleCheckpointError`
        if it is never finite, instead of carrying on.

    The alpha schedule moves the training loss only. ``causal_loss_metric`` is fixed at alpha
    1 and has no penalties, so during the alpha warm-up it can stall, or briefly rise, while
    the loss falls. Choose ``patience`` larger than the steep part of the warm-up (about
    ``half_life_epoch + 1 / steepness``, 20 epochs at the defaults). A model trained at a
    constant alpha other than 1 is still selected at alpha 1.
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
        tf.keras.callbacks.ReduceLROnPlateau(mode="min", **args["reduce_lr"]),
        AlphaScheduleCallback(**args["alpha_schedule"]),
    ]
