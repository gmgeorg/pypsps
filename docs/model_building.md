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

## Selecting on the Outcome Likelihood

`metrics.outcome_nll_metric_gen(outcome_loss)` reports the outcome negative
log-likelihood alone, with penalties stripped and no alpha. Monitoring
`val_outcome_nll_metric` therefore selects every model by the same rule, whatever
alpha schedule it was trained with.

## Per-State Column Selection

`layers.StateColumnGate(i)` returns column `i` of a 2D tensor, such as one state's
probability column. Use it instead of a `Lambda` built inside a per-state loop.
Unlike the `Lambda`, it serializes with `model.save()`.

## Outcome Likelihood

`losses.OutcomeLoss(..., outcome_likelihood=...)` chooses which outcome likelihood the
loss evaluates, through the state weights that mix the per-state outcome likelihoods:

* `"interventional"` (default, `OutcomeLikelihood.INTERVENTIONAL`): -log p(Y | do(A = a), X),
  mixing with the prior state weights P(S | X), which equal P(S | X, do(A)) because the
  states precede treatment. These are the weights at treatment assignment time, before
  the treatment is known, and the same weights the ATE estimates in `pypsps.inference`
  use. With `TreatmentLoss` it does not sum to the joint likelihood of (Y, A | X). In
  practice we observe better accuracy when estimating UTEs and ATEs; research comparing
  the two is forthcoming.
* `"observational"` (`OutcomeLikelihood.OBSERVATIONAL`): -log p(Y | A = a, X), mixing
  with the posterior state weights P(S | X, A). With `TreatmentLoss` the two sum to the
  exact joint likelihood of (Y, A | X). This was the only behavior from v0.1.0 through
  v0.1.10.

Metrics built from the loss (`causal_loss_metric_gen`, `outcome_nll_metric_gen`) use the
same likelihood.

To make `causal_loss_metric_gen` follow an alpha schedule, pass it the training loss's
variable, `alpha=causal_loss._alpha`. A float alpha stays fixed.
