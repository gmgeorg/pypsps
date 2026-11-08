# Building Models in pypsps

## Keras Integration

All components are fully compatible with the standard `tf.keras` API.

## Core Components (`pypsps.keras`)

* **`models`**: `build_model` builds a PSPS model for any outcome and treatment family, and
  `compile_model` compiles it. See [Building a Model](#building-a-model).
* **`presets`**: named `build_model` argument sets: `toy`, `binary_normal`,
  `binary_exponential`, `continuous_survival`, `continuous_exponential` and
  `continuous_weibull`. See [Presets](#presets).
* **`layers`**: Custom PSPS layer implementations.
* **`losses`**: Causal loss functions optimized for predictive state learning.
* **`metrics`**: Custom evaluation metrics for causal estimation. The propensity,
  treatment and outcome metrics take the model's mixture (`metrics.PropensityScoreAUC(mixture)`,
  `metrics.OutcomeMeanSquaredError(mixture)`); `predictive_state_df_gen` and
  `uniform_entropy_gen` take its layout (`mixture.layout`). See
  [Outcome and Treatment Metrics](#outcome-and-treatment-metrics).
* **`callbacks`**: Training helpers (e.g., `get_default_callbacks()`).
* **`neglogliks`**: Negative log-likelihood functions.
* **`layout`**: `ColumnLayout`, the one place that knows the column order of `y_pred`
  (`[outcome params | state weights | treatment params]`, each block one column per state)
  and `y_true` (`[outcome | treatment | ignored]`). `decode_pred` splits a prediction into
  named `[N, K]` blocks and `encode` puts them back together. `decode_true` splits `y_true`
  and `encode_true(outcome, treatment)` builds it, checking that the outcome has exactly the
  number of columns the outcome family reads.
* **`layers.CausalOutput(mixture)`**: the output layer of every builder. It encodes the heads
  (`{"outcome": ..., "state_weights": ..., "treatment": ...}`) into `y_pred` and stores the
  mixture in the saved model.
* **`mixture`**: `StateMixture(outcome_family, treatment_family, n_states)` holds the two
  families and the layout, and computes everything that mixes over predictive states: the
  per-state NLLs, the state posterior P(s | x, a), the outcome and treatment mixture NLLs,
  and the outcome and treatment means (mixing each state's family mean). `OutcomeLoss` and
  `TreatmentLoss` evaluate it; `CausalLoss.from_mixture(mixture, alpha=...)` builds a causal
  loss whose two parts share one mixture. The losses and metrics take only a mixture (or a
  layout); the column-count constructors were removed in v0.5.0 (see
  [Removed in v0.5.0](#removed-in-v050)).
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

## Building a Model

`models.build_model` builds a PSPS model for any outcome family and any treatment family.
Its output layer is a `layers.CausalOutput` that holds the model's `StateMixture`, and by
default it compiles the model with `models.compile_model`.

```python
from pypsps.keras import families, models

model = models.build_model(
    n_states=4,
    n_features=10,
    outcome_family=families.Weibull(log_scale_bounds=(-2.0, 4.0), log_shape_bounds=(-1.0, 1.0)),
    treatment_family=families.Normal(),
    architecture="gated",  # or "shared_trunk"
    outcome_heads={"log_shape": models.HeadSpec(per_state=False)},
    outcome_init={"log_scale": 1.6},
    max_time=100.0,  # checks the bounds against float32 overflow
    alpha=1.0,  # any further keyword goes to compile_model
)
```

| Outcome or treatment | Family | Parameters (one column per state each) |
|---|---|---|
| Continuous | `families.Normal()` | `loc`, `scale` |
| Binary | `families.Bernoulli()` | `prob` |
| Time to event | `families.Exponential(parameterization="log_scale" \| "log_rate" \| "rate")` | one of `log_scale`, `log_rate`, `rate` |
| Time to event | `families.Weibull(log_scale_bounds, log_shape_bounds)` | `log_scale`, `log_shape` |
| Any elementwise Keras loss | `families.LossFamily(loss, n_params)` | `param_0`, ... |

A time-to-event `y_true` is `[time, event indicator, treatment]`; otherwise it is
`[outcome, treatment]`.

* **`architecture`**: `"shared_trunk"` runs one outcome trunk shared by every parameter
  and state. `"gated"` (the default) gives each outcome parameter its own trunk, and gates the
  treatment input of each state's head by that state's weight.
* **`outcome_heads`**: one `HeadSpec(hidden_layer=(16, "swish"), constant=False,
  per_state=True)` per outcome parameter. `constant=True` makes the parameter one trainable
  constant per state. `per_state=False` shares one head across all states. `hidden_layer=None`
  makes the head linear.
* **`treatment_embedding_units`**: the width of a Dense embedding of the treatment before it
  joins the features (default 8); `None` uses the batch-normalized treatment directly.
* **`outcome_init` / `treatment_init`**: start values on the parameter scale, as a scalar or one
  value per state. The bias of each head is set so the parameter starts there. Bounded treatment
  parameters cannot take an init.
* **`compile=False`** returns the model uncompiled; it then takes no compile keywords.

**Hidden-layer activations.** The `(units, activation)` entries of `predictive_state_hidden_layers`,
`outcome_hidden_layers` and `HeadSpec.hidden_layer` take any Keras activation name. The ones that
`pypsps.torch` builds alike are `pypsps.core.spec.HIDDEN_ACTIVATIONS`: `elu`, `gelu`, `linear`,
`relu`, `selu`, `sigmoid`, `silu`, `softplus`, `swish` and `tanh`. Any other Keras name (say
`leaky_relu`) builds and trains, but `build_model` warns with
`pypsps.core.spec.NonPortableActivationWarning` (once per name) because
`PSPSModel.from_keras` can't convert the model; filter the warning if you never convert. A name
Keras does not know raises a `ValueError` before any layer is created.

Every model has one treatment head, `treatment_params` (`layers.PredictiveStateParams`): one
treatment distribution per state, independent of the features. For a binary treatment it
holds each state's propensity logit.

### Compiling

`compile_model(model, alpha=..., outcome_loss_weight=..., outcome_likelihood=...,
predictive_states_regularizer=..., optimizer=..., learning_rate=0.01, clipnorm=1.0,
metrics=..., extra_metrics=...)` builds the `CausalLoss` from the model's mixture and compiles
with Nadam by default. It works for any network that ends in `layers.CausalOutput`, including
one you wire by hand. `metrics=None` uses `default_metrics(causal_loss)`, chosen by the families:

| Condition | Logged keys |
|---|---|
| always | `causal_loss_metric`, `outcome_nll_metric`, `predictive_state_df`, `uniform_entropy` |
| binary treatment | `propensity_score_binary_crossentropy`, `propensity_score_auroc`, `propensity_score_aupr` |
| any other treatment | `treatment_mean_squared_error` |
| any outcome family but `LossFamily` | `outcome_mean_squared_error` (time to event: over units with an event) |

`extra_metrics` adds to the defaults; `metrics` replaces them.

### Layer Names

The layer names are part of the interface. `{p}` is a family parameter name, and `{s}` is
`state_{k}`, or `shared` for a head with `per_state=False`.

| Block | Names |
|---|---|
| inputs | `features`, `treatment`, `features_bn`, `treatment_bn`, `treatment_embedding`, `features_and_treatment` |
| state network | `predictive_state_dense_i`, `predictive_state_dropout_i`, `predictive_state_bn_i`, `predictive_state_skip`, `state_weights` |
| treatment head | `treatment_params` |
| outcome trunk (shared trunk) | `outcome_trunk_dense_i`, `outcome_trunk_dropout_i`, `outcome_trunk_bn_i`, `outcome_trunk_skip` |
| outcome trunk and gates (gated) | `{p}_trunk_dense_i`, ..., `{p}_gate_{s}`, `{p}_gated_treatment_{s}`, `{p}_input_{s}` |
| outcome heads | `{p}_hidden_{s}`, `{p}_raw_{s}`, `{p}_{s}`, `{p}_all_states` |
| output | `output_tensor` (`layers.CausalOutput`) |

## Presets

A preset is a function that returns the `build_model` arguments of a common model as a new,
plain dict. It creates no layers, so it can be printed, compared and edited. Each takes the
arguments of the `build_*` function it replaces.

```python
from pypsps.keras import models, presets

model = models.build_model(**presets.toy(n_states=4, n_features=10))
```

| Preset | Outcome / treatment | Network |
|---|---|---|
| `toy(n_states, n_features)` | Normal / Bernoulli | shared trunk; state layers `[(10, "relu"), (10, "selu")]`, outcome `[(10, "tanh")]`, `loc` head `(5, "selu")`, constant `scale` |
| `binary_normal(...)` | Normal / Bernoulli | shared trunk; `loc_layer` head, `scale_layer` head or a constant scale |
| `binary_exponential(...)` | Exponential (`log_rate`) / Bernoulli | shared trunk; `log_rate_layer` head |
| `continuous_survival(distribution, ...)` | Exponential (`log_scale`) or Weibull / Normal | gated; treatment embedding |
| `continuous_exponential(...)`, `continuous_weibull(...)` | wrappers of `continuous_survival` | |

The toy and binary presets compile with Nadam without gradient clipping. The toy and
binary-normal presets penalize the states with
`DegreesOfFreedom(l2=df_penalty_l2, target=max(1, log(n_states / 2)))`, the binary-exponential
preset with `Uniform(l2=df_penalty_l2)`, and the continuous presets with the
`predictive_states_regularizer` you pass (none by default). See the
[Training Guide](training_guide.md#7-tuning-the-predictive-state-regularizer).

To change an entry, merge a dict into the preset. The merge is explicit, so nothing is
silently shadowed; passing a key twice (`build_model(**presets.toy(3, 10), alpha=2.0)`) is a
`TypeError`.

```python
args = presets.binary_normal(n_states=3, n_features=10, predictive_state_hidden_layers=[(16, "relu")],
                             outcome_hidden_layers=[(16, "relu")], loc_layer=(8, "selu"))
model = models.build_model(**(args | {"outcome_loss_weight": 2.0, "architecture": "gated"}))
```

With `compile=False` a preset leaves out its compile keywords. Merging `{"compile": False}`
into a compiling preset raises a `TypeError` that lists those keywords; pass `compile=False`
to the preset instead.

## Removed in v0.5.0

v0.5.0 removes everything that v0.3.4 to v0.4.0 deprecated. Each removed name has a
one-line replacement:

| Removed | Use instead |
|---|---|
| `models.build_toy_model(...)` | `models.build_model(**presets.toy(...))` |
| `models.build_model_binary_normal(...)` | `models.build_model(**presets.binary_normal(...))` |
| `models.build_model_binary_exponential(...)` | `models.build_model(**presets.binary_exponential(...))` |
| `models.build_model_continuous_survival(...)` | `models.build_model(**presets.continuous_survival(...))` |
| `models.build_model_continuous_exponential(...)` | `models.build_model(**presets.continuous_exponential(...))` |
| `models.build_model_continuous_weibull(...)` | `models.build_model(**presets.continuous_weibull(...))` |
| `losses.OutcomeLoss(loss, treatment_loss, <counts>)`, `losses.TreatmentLoss(loss, <counts>)` | `OutcomeLoss(mixture)`, `TreatmentLoss(mixture)`, or `CausalLoss.from_mixture(mixture, ...)` |
| Metric classes built from counts, e.g. `OutcomeMeanSquaredError(n_outcome_pred_cols=2, ...)` | `OutcomeMeanSquaredError(mixture)` |
| `predictive_state_df_gen(2, 1)`, `uniform_entropy_gen(2, 1)` | `predictive_state_df_gen(mixture.layout)` |
| `metrics.causal_loss_metric_gen(..., alpha=...)` | drop `alpha` (the metric always uses 1); `outcome_loss_weight` is keyword-only |
| `utils.split_y_pred`, `split_outcome_pred`, `split_y_true`, `get_n_states`, `get_state_column_indices` | `ColumnLayout.decode_pred` / `decode_true` |
| `utils.agg_outcome_pred` (mixed raw parameters, not means) | `StateMixture.outcome_mean` |
| `utils.agg_treatment_pred` | `StateMixture.treatment_mean` |
| `utils.get_column_layout`, `utils.ColumnLayout` | `ColumnLayout.from_model(model)` |
| `neglogliks.negloglik_per_state`, `posterior_from_negloglik_per_state`, `losses.prob_state_given_treatment_features` | `StateMixture.nll_per_state`, `mixture.posterior`, `StateMixture.state_posterior` |
| `families.from_negloglik`, `ColumnLayout.from_counts` | a built-in family (or `families.LossFamily`), `ColumnLayout.from_families` |
| `models.copy_legacy_weights` | migrate with pypsps v0.4.x (below) |

Also:

* `StateMixture` and `ColumnLayout` require `n_states`; K is no longer inferred from the
  `y_pred` width.
* `CausalLoss` raises `ValueError` if its outcome and treatment losses hold different
  mixtures. In v0.4.x the column-count constructors only warned.

### Models saved before v0.5.0

| Saved by | `.keras` file in v0.5.0 | Migration |
|---|---|---|
| v0.3.4 to v0.4.x | loads as before | none |
| v0.3.0 to v0.3.3 (and earlier) | loads only with `compile=False` | rebuild and copy the weights |

A file saved by v0.3.0 to v0.3.3 stores its loss and metrics with column counts, so
`load_model` raises `ValueError` with the default `compile=True`. It also has no
`CausalOutput` layer, so `pypsps.inference` and `compile_model` can't find its mixture.
Build the matching preset model and copy the weights into it:

* **Continuous models**: the weight layout is unchanged, so this works in v0.5.0:

  ```python
  import pypsps.keras  # registers the pypsps classes with Keras
  import tensorflow as tf
  from pypsps.keras import models, presets

  old = tf.keras.models.load_model("old.keras", compile=False)
  new = models.build_model(**presets.continuous_weibull(...))  # the old model's arguments
  new.set_weights(old.get_weights())
  new.save("new.keras")
  ```

* **Binary models**: v0.4.0 replaced the per-state propensity layers with one
  `treatment_params` layer. Migrate with pypsps v0.4.x, whose `models.copy_legacy_weights`
  maps the old layers onto the new ones, and save a `.keras` file that v0.5.0 loads:

  ```python
  # with pypsps==0.4.*
  old = tf.keras.models.load_model("old.keras", compile=False)
  new = models.build_model(**presets.binary_normal(...))  # the old model's arguments
  models.copy_legacy_weights(old, new)
  new.save("new.keras")
  ```

  If you only have a `.weights.h5` file, load it into the old model with the pypsps release
  that saved it, save a `.keras` file, and migrate that file.
* **Toy models**: v0.4.0 added one BatchNormalization layer, so old toy weights don't copy.
  Retrain; the toy model is a demo.

> **Warning:** `load_weights` on a binary `.weights.h5` file saved before v0.4.0 raises
> `ValueError`, but only after it has overwritten part of the model. Discard a model whose
> `load_weights` raised, and build a new one.

## Recommended Training Pattern

```python
from pypsps.keras import models, presets
from pypsps.keras.callbacks import get_default_callbacks

model = models.build_model(**presets.toy(n_states=5, n_features=10))
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

## Outcome and Treatment Metrics

`OutcomeMeanSquaredError`, `OutcomeMeanAbsoluteError`, `TreatmentMeanSquaredError` and
`TreatmentMeanAbsoluteError` score the mixed mean `sum_k P(s_k | x) * E[. | s_k]`, where
each state's mean comes from its family: `loc` for Normal, `exp(log_scale) * Gamma(1 + 1 /
shape)` for Weibull. The propensity metrics score `sum_k P(s_k | x) * p_k`.

```python
from pypsps.keras import metrics

mixture = model.loss.mixture
model.compile(loss=model.loss, optimizer="adam",
              metrics=[metrics.OutcomeMeanSquaredError(mixture)])
```

For a time-to-event outcome, the outcome MSE and MAE use only units with an observed
event. A censored unit's time is only a lower bound, so it adds no error and no weight. A
batch without events therefore leaves the metric unchanged. If no event has been seen since
the last reset, the metric reports `NaN`, not 0. Don't monitor `val_outcome_mean_squared_error`
on a heavily censored validation set: if the first epoch is `NaN`, `EarlyStopping` keeps it
as the best value and stops after `patience` epochs. `TerminateOnNaN` reads only the loss,
so it is not affected.

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
  practice we observe better accuracy when estimating UTEs and ATEs; see
  [Goerg and Shuryak (2026)][goerg2026] for a comparison with a continuous treatment and a
  right-censored outcome.
* `"observational"` (`OutcomeLikelihood.OBSERVATIONAL`): -log p(Y | A = a, X), mixing
  with the posterior state weights P(S | X, A). With `TreatmentLoss` the two sum to the
  exact joint likelihood of (Y, A | X). This was the only behavior from v0.1.0 through
  v0.1.10.

Metrics built from the loss (`causal_loss_metric_gen`, `outcome_nll_metric_gen`) use the
same likelihood. See the
[Training Guide](training_guide.md#4-interventional-and-observational-outcome-likelihood).

`causal_loss_metric` (`metrics.CausalLossMetric`, or `causal_loss_metric_gen`) is
`outcome_loss_weight` × outcome NLL + 1 × treatment NLL: alpha is fixed at 1 and no
penalties are included, so it doesn't follow the alpha schedule. `compile_model` passes it
the training loss's `outcome_loss_weight`. The training loss's alpha is `causal_loss.alpha`, a
`tf.Variable`; `causal_loss.alpha = x` assigns it.

### Sample Weights

`OutcomeLoss`, `TreatmentLoss` and `CausalLoss` compute one loss per unit, and Keras
weights each unit by its `sample_weight` before applying `reduction`. An integer weight
counts a unit that many times. `CausalLoss` adds the `predictive_states_regularizer` once,
after the reduction: it penalizes the batch's state weights, so `sample_weight` does not
scale it. `causal_loss_metric` and `outcome_nll_metric` report the weighted mean over units,
sum(w × loss) / sum(w), across all batches.

## Continuous Treatment, Time-to-Event Outcome

`presets.continuous_survival(distribution, ...)`, with the wrappers
`presets.continuous_exponential` and `presets.continuous_weibull`, returns the `build_model`
arguments of a model with a Normal treatment density per state and an exponential or Weibull
outcome. `y_true` is
`[time, event indicator, treatment]`, and `inference.predict_ate_continuous` reads the
fitted model directly. [Goerg and Shuryak (2026)][goerg2026] apply this setup to continuous
dose–response estimation under right-censoring.

```python
from pypsps.keras import models, presets

model = models.build_model(
    **presets.continuous_weibull(
        n_states=3,
        n_features=10,
        predictive_state_hidden_layers=[(16, "relu")],
        outcome_hidden_layers=[(16, "relu")],
        log_scale_bounds=(-3.0, 6.0),  # log time units
        log_shape_bounds=(-2.0, 2.0),
        max_time=y_true[:, 0].max(),
    )
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

A model compiled by `build_model` or `compile_model` saves and reloads with its loss, metrics and
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
* Every model ends in a `layers.CausalOutput` layer (named `output_tensor`), which carries the
  model's `StateMixture`. So a model loaded with `compile=False` still runs `pypsps.inference`,
  and `StateMixture.from_model(model)` and `ColumnLayout.from_model(model)` read it. Models
  built before v0.3.4 have no such layer and, since v0.5.0, load only with `compile=False`;
  see [Models saved before v0.5.0](#models-saved-before-v050).
* The same layer carries the model's build arguments, `pypsps.core.spec.Spec.get_config`
  (including `max_time`), and `models.spec_of(model)` returns the `Spec` (without warning again about a Keras-only
  activation). `PSPSModel.from_keras` reads it. Models saved before v0.6.3 have none (`spec_of` gives `None`); they still load and
  convert, with the build arguments read from the layer names. A `.keras` file saved by v0.6.3
  or later can't be loaded by an older pypsps.
* A custom elementwise NLL must be registered with
  `tf.keras.utils.register_keras_serializable` to load. `neglogliks.NegloglikLoss`, which
  holds a distribution constructor, isn't serializable.

[goerg2026]: https://openreview.net/forum?id=LtfivZ5XJ9
