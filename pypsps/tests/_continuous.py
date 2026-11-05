"""Shared test helpers: synthetic continuous-treatment survival data and a small model."""

import numpy as np

from ..keras import models, presets

LOG_SCALE_BOUNDS = (-3.0, 6.0)
LOG_SHAPE_BOUNDS = (-2.0, 2.0)


def synthetic_continuous_survival(n_samples=512, seed=0):
    """Continuous treatment confounded by x0; Weibull times with scale exp(1 + 0.3 a + 0.2 x1).

    Returns (inputs, y_true, true ATE function); y_true is [time, event, treatment].
    """
    rng = np.random.RandomState(seed)
    features = rng.normal(size=(n_samples, 3)).astype("float32")
    treatment = (0.5 * features[:, 0] + rng.normal(size=n_samples)).astype("float32")
    scale = np.exp(1.0 + 0.3 * treatment + 0.2 * features[:, 1])
    event_time = scale * rng.weibull(1.5, size=n_samples)
    censor_time = rng.exponential(3.0 * scale.mean(), size=n_samples)
    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype("float32")
    y_true = np.column_stack([time, event, treatment]).astype("float32")

    # E[T | a] = E[exp(0.2 x1)] * exp(1 + 0.3 a) * Gamma(1 + 1 / 1.5).
    mean_at = lambda a: np.exp(0.02 + 1.0 + 0.3 * a) * 0.9027452929509336  # noqa: E731

    return [features, treatment[:, None]], y_true, lambda a, base: mean_at(a) - mean_at(base)


def continuous_model(distribution="weibull", **kwargs):
    """Small continuous survival model with the test bounds, from `presets.continuous_survival`."""
    if distribution == "weibull":
        kwargs.setdefault("log_shape_bounds", LOG_SHAPE_BOUNDS)
    return models.build_model(
        **presets.continuous_survival(
            distribution,
            n_states=3,
            n_features=3,
            predictive_state_hidden_layers=[(16, "relu")],
            outcome_hidden_layers=[(16, "relu")],
            log_scale_bounds=LOG_SCALE_BOUNDS,
            **kwargs,
        )
    )
