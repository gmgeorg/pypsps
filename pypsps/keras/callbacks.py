"""Module for pypsps training callbacks."""

import numpy as np
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
                           Default: None (auto-computed as 25% of |alpha_start - alpha_end|)
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

        # Auto-compute clamp threshold if not provided (25% of range, or 0 if constant)
        if clamp_threshold is None:
            self.clamp_threshold = 0.25 * self._alpha_range if self._alpha_range > 0 else 0.0
        else:
            if clamp_threshold < 0:
                raise ValueError(f"clamp_threshold must be non-negative, got {clamp_threshold}")
            self.clamp_threshold = clamp_threshold

    def _compute_alpha(self, epoch):
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

    def on_epoch_begin(self, epoch, logs=None):
        """Update alpha at the beginning of each epoch."""
        # Compute new alpha value
        new_alpha = self._compute_alpha(epoch)

        # Update the model's loss function alpha parameter
        if hasattr(self.model, "loss") and hasattr(self.model.loss, "_alpha"):
            alpha_ref = self.model.loss._alpha
            old_alpha = float(alpha_ref.numpy()) if hasattr(alpha_ref, "assign") else alpha_ref

            # ASSIGN, do not rebind. `_alpha` is a tf.Variable so that the new value reaches
            # the already-traced training graph (model.fit() runs with run_eagerly=False by
            # default); rebinding the attribute would only update this eager Python object,
            # which the compiled training function never reads again.
            if hasattr(alpha_ref, "assign"):
                alpha_ref.assign(new_alpha)
            else:
                # Backward compatibility with a loss whose `_alpha` is still a plain float.
                self.model.loss._alpha = new_alpha

            # Store in history
            self.alpha_history.append(new_alpha)

            # Print update if verbose (skip for constant schedules)
            if self.verbose and not self._is_constant and (epoch % 10 == 0 or epoch < 5):
                direction = "↓" if self._is_decay else "↑"
                print(f"\n[Epoch {epoch}] Alpha: {old_alpha:.4f} {direction} {new_alpha:.4f}")
        else:
            raise AttributeError(
                "Model's loss function does not have an '_alpha' attribute. "
                "Ensure the model is compiled with a CausalLoss."
            )

    def on_train_end(self, logs=None):
        """Print final alpha value at the end of training."""
        if self.verbose and len(self.alpha_history) > 0:
            print(f"\n[Training Complete] Final Alpha: {self.alpha_history[-1]:.4f}")

    def plot_schedule(self, max_epochs=200, figsize=(10, 6)):
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
