# Building Models in pypsps

## Keras Integration

All components are fully compatible with the standard `tf.keras` API.

## Core Components (`pypsps.keras`)

* **`models`**: Provides model builders like `build_toy_model(n_states,
  n_features, compile=True, alpha=10.)` for binary treatment and continuous
  outcome.
* **`layers`**: Custom PSPS layer implementations.
* **`losses`**: Causal loss functions optimized for predictive state learning.
* **`metrics`**: Custom evaluation metrics for causal estimation.
* **`callbacks`**: Training helpers (e.g., `get_default_callbacks()`).
* **`neglogliks`**: Negative log-likelihood functions.

## Recommended Training Pattern

```python
from pypsps.keras.models import build_toy_model
from pypsps.keras.callbacks import get_default_callbacks

model = build_toy_model(n_states=5, n_features=10)
callbacks = get_default_callbacks(monitor="val_causal_loss_metric", patience=25)

# Fit using standard Keras API with a validation split
model.fit(X, y, validation_split=0.2, callbacks=callbacks)
```

## Half-Gaussian Alpha Schedule

`GaussianAlphaSchedule` (in `callbacks`) moves the treatment-loss weight alpha along
`alpha(e) = alpha_end + (alpha_start - alpha_end) * exp(-c * e**2)` for the zero-based
epoch `e`. It starts exactly at `alpha_start` with zero slope and has no clamp. Its
`warmup_epochs` is the first epoch at which alpha is within `epsilon` (default `1e-3`) of
its range from `alpha_end`. Pass it to `EligibleBestCheckpoint` and `WarmupReduceLROnPlateau`
as `start_from_epoch` so that model selection and learning-rate reduction only compare
epochs at which the loss weight has settled. `EligibleBestCheckpoint` also treats NaN or
infinite epochs as ineligible and raises if no epoch was eligible.

```python
from pypsps.keras import callbacks

schedule = callbacks.GaussianAlphaSchedule(alpha_start=10.0, alpha_end=1.0, decay_const=0.25)
w = schedule.warmup_epochs  # 6
cbs = [
    schedule,
    callbacks.EligibleBestCheckpoint("val_loss", patience=25, start_from_epoch=w),
    callbacks.WarmupReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=12, min_lr=1e-6, start_from_epoch=w
    ),
]
```
