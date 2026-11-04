# Building Models in pypsps

## Keras Integration

All components are fully compatible with the standard `tf.keras` API.

## Core Components (`pypsps.keras`)

* **`models`**: Provides model builders like `build_toy_model(n_states,
  n_features, compile=True, alpha=10.)` for binary treatment and continuous
  outcome, and `build_model_continuous_survival` for a continuous treatment and an
  exponential or Weibull time-to-event outcome.
* **`layers`**: Custom PSPS layer implementations.
* **`losses`**: Causal loss functions optimized for predictive state learning.
* **`metrics`**: Custom evaluation metrics for causal estimation.
* **`callbacks`**: Training helpers (e.g., `get_default_callbacks()`).
* **`neglogliks`**: Negative log-likelihood functions.
* **`layout`**: `ColumnLayout`, the one place that knows the column order of `y_pred`
  (`[outcome params | state weights | treatment params]`, each block one column per state)
  and `y_true` (`[outcome | treatment | ignored]`). `decode_pred` splits a prediction into
  named `[N, K]` blocks and `encode` puts them back together.
* **`mixture`**: `StateMixture(outcome_family, treatment_family, n_states)` holds the two
  families and the layout, and computes everything that mixes over predictive states: the
  per-state NLLs, the state posterior P(s | x, a), the outcome and treatment mixture NLLs,
  and the outcome and treatment means (mixing each state's family mean). `OutcomeLoss` and
  `TreatmentLoss` evaluate it; `CausalLoss.from_mixture(mixture, alpha=...)` builds a causal
  loss whose two parts share one mixture. The three-int loss constructors still work.
* **`families`**: one class per within-state distribution: `Normal`, `Bernoulli`,
  `Exponential(parameterization="log_scale" | "log_rate" | "rate")` and `Weibull`. Each
  family knows its parameter names, its head activations, its per-state NLL
  (`nll(y_true, params) -> [N, K]`) and its mean. `families.get("weibull", ...)` looks one
  up by name.

```python
from pypsps.keras import families
from pypsps.keras.layout import ColumnLayout

layout = ColumnLayout.from_families(n_states=3, outcome_family=families.Normal(),
                                    treatment_family=families.Bernoulli())
parts = layout.decode_pred(y_pred)          # parts.outcome["loc"] is [N, 3]
means = families.Normal().mean(parts.outcome)   # E[Y | state k], [N, 3]
```

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
infinite epochs as ineligible and raises if no epoch was eligible. See the
[Training Guide](training_guide.md#2-alpha-schedules).

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
alpha schedule it was trained with. See the
[Training Guide](training_guide.md#3-selecting-on-the-outcome-term).

## Per-State Column Selection

`layers.StateColumnGate(i)` returns column `i` of a 2D tensor, such as one state's
probability column. Use it instead of a `Lambda` built inside a per-state loop.
Unlike the `Lambda`, it serializes with `model.save()`. See the
[Training Guide](training_guide.md#1-per-state-layers-in-a-loop) for why.

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
same likelihood. See the
[Training Guide](training_guide.md#4-interventional-and-observational-outcome-likelihood).

`causal_loss_metric` (`metrics.CausalLossMetric`, or `causal_loss_metric_gen`) is
`outcome_loss_weight` × outcome NLL + 1 × treatment NLL: alpha is fixed at 1 and no
penalties are included, so it doesn't follow the alpha schedule. The builders pass it the
training loss's `outcome_loss_weight`. The training loss's alpha is `causal_loss.alpha`, a
`tf.Variable`; `causal_loss.alpha = x` assigns it.

## Continuous Treatment, Time-to-Event Outcome

`models.build_model_continuous_survival(distribution, ...)`, with the wrappers
`build_model_continuous_exponential` and `build_model_continuous_weibull`, models a Normal
treatment density per state and an exponential or Weibull outcome. `y_true` is
`[time, event indicator, treatment]`, and `inference.predict_ate_continuous` reads the
fitted model directly.

```python
from pypsps.keras import models

model = models.build_model_continuous_weibull(
    n_states=3,
    n_features=10,
    predictive_state_hidden_layers=[(16, "relu")],
    outcome_hidden_layers=[(16, "relu")],
    log_scale_bounds=(-3.0, 6.0),  # log time units
    log_shape_bounds=(-2.0, 2.0),
    max_time=y_true[:, 0].max(),
)
```

The outcome parameters are soft-clipped (`layers.SoftClip`) to `log_scale_bounds` and,
for Weibull, `log_shape_bounds`. Both are required, because sensible values depend on the
time unit. With `max_time`, the builder raises if
`exp(log_shape_max) * (log(max_time) - log_scale_min)` exceeds 50, well below where the
Weibull negative log-likelihood overflows float32 (near 88.7). See the
[Training Guide](training_guide.md#5-time-to-event-parameter-bounds) for choosing the
bounds.

## Saving and Loading

A model compiled by any `build_*` function saves and reloads with its loss, metrics and
current alpha:

```python
import pypsps.keras  # registers the pypsps (and pypress) classes with Keras
import tensorflow as tf

model.save("model.keras")
model = tf.keras.models.load_model("model.keras")
```

* Import `pypsps.keras` before `load_model`, or Keras raises `Could not locate class`.
  A bare `import pypsps` is not enough.
* The training loss reloads with the alpha it had when saved, so a model saved in the
  middle of an alpha schedule keeps that alpha. `causal_loss_metric` has no alpha.
* A model loaded with `compile=False` can `predict`, but `pypsps.inference` and
  `utils.get_column_layout` need the compiled loss: load with `compile=True` (the default).
* A custom elementwise NLL must be registered with
  `tf.keras.utils.register_keras_serializable` to load. `neglogliks.NegloglikLoss`, which
  holds a distribution constructor, isn't serializable.
* Models saved before v0.3.0 don't load with `compile=True`: their saved loss config is
  empty. Load them with `load_model(path, compile=False)`, rebuild the same architecture
  with the matching `build_*(..., compile=True)`, and copy the weights with
  `built.set_weights(loaded.get_weights())`. Saving `built` produces a file that reloads.
