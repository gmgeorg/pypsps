# Training Guide

Practical notes for building and training pypsps models: the reasoning behind
`StateColumnGate`, the alpha schedule and its callbacks, model selection, the outcome
likelihood option, safe bounds for time-to-event parameters, and the predictive-state
regularizer. The API itself is
described in [Model Architecture](model_building.md).

## 1. Per-State Layers in a Loop

PSPS builders create one outcome head per state in a Python loop, and each head often
needs its own column of the predictive-state weights. The obvious way to write that is
wrong:

```python
gates = [tf.keras.layers.Lambda(lambda x: x[:, i : i + 1])(states) for i in range(k)]
```

Python closures bind `i` late: every lambda reads `i` when it runs, after the loop has
finished, so every gate selects the last column. With columns `[10, 20, 30]` the gates
return `[30, 30, 30]`. Nothing raises; the model trains, but every state's head is gated
by the same state.

Binding the index with a default argument, `lambda x, i=i: ...`, gives the right values
in memory, but the model then cannot be loaded again. `load_model` refuses to deserialize
a Python lambda by default, and with `safe_mode=False` the model still fails to
deserialize.

`layers.StateColumnGate(i)` avoids both problems. It stores `i` in its config, so each
gate keeps its own index and the model round-trips through `model.save` and `load_model`.
The same reasoning applies to any per-state computation: prefer a small serializable layer
with explicit arguments over a `Lambda` that captures loop variables. `build_model`'s
`"gated"` architecture, the default, already gates each head with `StateColumnGate`.

## 2. Alpha Schedules

`CausalLoss` adds the outcome loss and `alpha` times the treatment loss. With `alpha = 1`
the total is the joint negative log-likelihood. A larger alpha early in training makes
the predictive states fit the treatment first, which gives the outcome heads balanced
states to learn from.

`callbacks.GaussianAlphaSchedule` lowers alpha along a half-Gaussian curve,

```text
alpha(e) = alpha_end + (alpha_start - alpha_end) * exp(-c * e**2)
```

for the zero-based epoch `e`. Unlike `exp(-c * e)`, the curve has zero slope at epoch 0,
so alpha falls slowly at first and fastest around `e = 1 / sqrt(2 c)`. It approaches
`alpha_end` without a clamp.

`schedule.warmup_epochs` is the first epoch at which alpha is within `epsilon` (default
`1e-3`) of its range from `alpha_end`, `ceil(sqrt(log(1 / epsilon) / c))`. For the default
`c = 0.25` that is epoch 6.

Losses from epochs with different alphas are not comparable, so model selection and
learning-rate reduction should only look at epochs after the warmup:

* `callbacks.EligibleBestCheckpoint(monitor, patience, start_from_epoch=w)` stops early
  and restores the best epoch at or after `w`. Epochs with a NaN or infinite monitored
  value are never selected, ties go to the earliest epoch, and it raises
  `NoEligibleCheckpointError` if no epoch was eligible.
* `callbacks.WarmupReduceLROnPlateau(..., start_from_epoch=w)` holds the learning rate
  until epoch `w`.

`GaussianAlphaSchedule` raises `MidFitResumeError` if a fit does not start at epoch 0,
because resuming would skip the warmup.

## 3. Selecting on the Outcome Term

Even after the warmup, the total loss depends on alpha and on any predictive-state
regularizer, so two runs with different settings cannot be compared on it.
`metrics.outcome_nll_metric_gen(outcome_loss)` reports the outcome negative
log-likelihood alone, with penalties stripped and no alpha. Monitoring
`val_outcome_nll_metric` selects every model by the same rule. Every model compiled by
`compile_model` with the default metrics logs it.

`causal_loss_metric`, which every model logs, is the joint version: the
`outcome_loss_weight`-weighted outcome NLL plus the treatment NLL, at alpha = 1 and without
penalties. It doesn't move with the alpha schedule either, so `val_causal_loss_metric` is
comparable across epochs and across schedules, and keeping `start_from_epoch` at the
warmup only skips epochs whose weights were still fit under a large alpha. Models compiled
this way save and reload with their losses and metrics; see
[Saving and Loading](model_building.md#saving-and-loading).

**Early stopping during the alpha warm-up.** The metric stays at alpha = 1 while the loss
follows the schedule, so early in training the optimizer trades outcome fit for treatment
fit more than the metric rewards. `val_causal_loss_metric` can then stall, or briefly rise,
while the loss falls, and `EarlyStopping` counts those epochs. With
`callbacks.get_default_callbacks`, whose sigmoid schedule takes alpha from about 1.73 at
epoch 0 to 1.05 at epoch 40, choose `patience` larger than the steep part of the warm-up,
about `half_life_epoch + 1 / steepness` (20 epochs at the defaults). The predictive-state
penalty is never in the metric either, so epochs where the optimizer pays NLL to lower the
penalty look like regressions on it. A model trained at a constant alpha other than 1 is
still selected at alpha = 1, the joint likelihood.

## 4. Interventional and Observational Outcome Likelihood

The predictive states S are determined by the features X, before treatment, so
`P(S | X, do(A)) = P(S | X)`. `OutcomeLoss(outcome_likelihood=...)` chooses which outcome
likelihood the loss evaluates:

* `"interventional"` (default): `-log p(Y | do(A = a), X) = -log sum_k P(s_k | X)
  p(Y | A = a, X, s_k)`, mixing the per-state outcome likelihoods with the prior state
  weights. These are the weights at treatment assignment time, before the treatment is
  known, and the same weights the ATE estimates in `pypsps.inference` use.
* `"observational"`: `-log p(Y | A = a, X)`, mixing with the posterior weights
  `P(S | X, A)`. Together with `TreatmentLoss` this is the exact joint likelihood of
  `(Y, A | X)`.

The interventional form is not the exact likelihood of the data conditional on the
observed treatment. It is a causal likelihood alternative that treats treatment
assignment and treatment observation differently in the likelihood. In practice we
observe better accuracy when estimating UTEs and ATEs.
[Goerg and Shuryak (2026)][goerg2026] compare the two for a continuous treatment (dose) and a
right-censored outcome.

## 5. Time-to-Event Parameter Bounds

The survival builders predict `log_scale` (and `log_shape` for Weibull) per state and
pass them through `layers.SoftClip`, which maps any input into `(min, max)` with a
sigmoid. A hard clip (`tf.clip_by_value`) has zero gradient once a prediction is outside
the bounds, so a state whose parameter crosses a bound can stay there for the rest of
training. The soft clip keeps a nonzero gradient near the bounds.

The bounds are required arguments, because sensible values depend on the time unit:

* `log_scale_bounds` should contain the log of every scale the data can plausibly need,
  in the unit of `time`, with room on both sides. The scale is the mean for exponential
  and close to it for Weibull.
* `log_shape_bounds` (Weibull) bounds `k = exp(log_shape)`: `k < 1` is a decreasing
  hazard, `k = 1` a constant one, and `k > 1` an increasing one. `(-2, 2)` allows `k`
  from about 0.14 to 7.4.

Wide bounds are not free. The Weibull negative log-likelihood contains
`(t / scale) ** k = exp(k * (log t - log_scale))`, which overflows float32 when the
exponent passes about 88.7, and the loss becomes infinite or NaN. The largest exponent
the bounds allow is

```text
exp(log_shape_max) * (log(max_time) - log_scale_min)
```

which depends on the time unit only through the ratio of `max_time` to the smallest scale.
Pass the largest observed time as `max_time` and the builder raises a `ValueError` when
this exceeds 50, a wide margin below the overflow. If it raises, raise the lower log-scale
bound, lower the upper log-shape bound, or rescale time. For the exponential outcome the
same rule holds with shape 1.

## 6. Example: Continuous Treatment, Weibull Outcome

[Goerg and Shuryak (2026)][goerg2026] study this setting, continuous dose–response estimation
under right-censoring.

The data below have a treatment confounded by `x0` and Weibull event times whose scale
grows with the treatment, with random censoring. The model is trained with the alpha
schedule, selected on the outcome term after the warmup, and then used for the ATE.

```python
import numpy as np

from pypsps import inference
from pypsps.keras import callbacks, models, presets

rng = np.random.default_rng(0)
n = 2000
x = rng.normal(size=(n, 3)).astype("float32")
a = (0.5 * x[:, 0] + rng.normal(size=n)).astype("float32")
scale = np.exp(1.0 + 0.3 * a + 0.2 * x[:, 1])
t_event = scale * rng.weibull(1.5, size=n)
t_censor = rng.exponential(3.0 * scale.mean(), size=n)
time = np.minimum(t_event, t_censor)
event = (t_event <= t_censor).astype("float32")
y = np.column_stack([time, event, a]).astype("float32")  # [time, event, treatment]

model = models.build_model(
    **presets.continuous_weibull(
        n_states=3,
        n_features=3,
        predictive_state_hidden_layers=[(16, "relu")],
        outcome_hidden_layers=[(16, "relu")],
        log_scale_bounds=(-3.0, 6.0),
        log_shape_bounds=(-2.0, 2.0),
        max_time=float(time.max()),
    )
)

schedule = callbacks.GaussianAlphaSchedule(alpha_start=10.0, alpha_end=1.0, verbose=False)
w = schedule.warmup_epochs
checkpoint = callbacks.EligibleBestCheckpoint(
    "val_outcome_nll_metric", patience=10, start_from_epoch=w
)
model.fit(
    [x, a[:, None]],
    y,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    verbose=0,
    callbacks=[
        schedule,
        checkpoint,
        callbacks.WarmupReduceLROnPlateau(
            monitor="val_outcome_nll_metric", factor=0.5, patience=5, start_from_epoch=w
        ),
    ],
)

# ATE of treatment levels -1 and 1 against 0.
ate = inference.predict_ate_continuous(model, x, treatment_grid=[-1.0, 0.0, 1.0])
```

The true ATEs here are about -0.65 and 0.88, and the fitted model gives values close to
them, for example `[-0.59, 0.0, 0.90]`.

## 7. Tuning the Predictive-State Regularizer

`compile_model(predictive_states_regularizer=...)` takes any Keras regularizer, which
`CausalLoss` applies to the batch's state weights. The presets choose one from
`df_penalty_l2`:

* `presets.toy` and `presets.binary_normal`:
  `pypress.keras.regularizers.DegreesOfFreedom(l2=df_penalty_l2, target=max(1, log(n_states / 2)))`.
* `presets.binary_exponential`: `pypress.keras.regularizers.Uniform(l2=df_penalty_l2)`.
* The continuous presets: the `predictive_states_regularizer` you pass, none by default.

The degrees-of-freedom target is a starting point, not a recommendation. It is 1.0 for up to
5 states and 1.6 for 10. The degrees of freedom of the state weights start at exactly 1 for
any `n_states` and grow as the states differentiate, so a target well below `n_states`
limits that from the first step. Tune `target` and `l2` for the data by replacing the
regularizer:

```python
from pypress.keras import regularizers
from pypsps.keras import models, presets

args = presets.toy(n_states=6, n_features=10)
reg = regularizers.DegreesOfFreedom(l2=0.1, target=4.0)
model = models.build_model(**(args | {"predictive_states_regularizer": reg}))
```

Compare the candidates on `val_causal_loss_metric` or `val_outcome_nll_metric`. Neither
includes the penalty, so models trained with different regularizers are comparable.

The regularizer's `l2` and `target` can also follow a pypress `ScheduledValue`, advanced by
`pypress.keras.callbacks.RegularizerScheduler`. Over-provisioned states are cheap early, and a
state whose weight collapses to 0 rarely recovers, so either warm `l2` up from 0 or anneal
`target` down from about `n_states`:

```python
from pypress.keras import callbacks as press_callbacks
from pypress.keras import schedules

target = schedules.ScheduledValue(start=6.0, end=3.0, duration_epochs=20)
reg = regularizers.DegreesOfFreedom(l2=0.1, target=target)
model = models.build_model(**(args | {"predictive_states_regularizer": reg}))
model.fit(..., callbacks=[press_callbacks.RegularizerScheduler(), ...])
```

Without the callback a `ScheduledValue` stays at its `end` value, so the penalty is applied
fully from the first epoch. This needs pypress 0.3.2 or later, which finds scheduled values
in the loss's regularizer.

[goerg2026]: https://openreview.net/forum?id=LtfivZ5XJ9
