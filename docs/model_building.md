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
