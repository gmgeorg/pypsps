# Changelog

All notable changes to `pypsps` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## pypsps v0.5.1 - Nov 5, 2026

### Dependencies

* pypress v0.5.4 (was v0.3.2). The predictive-state penalty keeps its strength: since
  pypress v0.5.2, calling a pypress regularizer returns its penalty times the batch size (to
  cancel Keras's division of activity losses). `CausalLoss` adds the regularizer to the loss
  itself, never as an activity regularizer, so it now adds `regularizer.penalty(weights)`:
  the same amount as before, at most `l2`, at any batch size. Other regularizers (e.g. a
  built-in Keras one) are still called.
* `ema_decay` on a pypress regularizer (recommended by pypress v0.5.3 for small batches)
  works inside `CausalLoss`: its running averages advance on training batches only.

### Changed

* `metrics.predictive_state_df_gen` and `uniform_entropy_gen` use
  `pypress.keras.statistics` instead of the deprecated `pypress.utils.tr_kernel` and
  `safe_xlogx`. The values are unchanged.

## pypsps v0.5.0 - Nov 4, 2026

Breaking: removes everything deprecated in v0.3.4 to v0.4.0. `docs/model_building.md`
("Removed in v0.5.0") has a replacement for each removed name and the migration for old
model files.

Model files: `.keras` files saved by v0.3.4 to v0.4.x load as before. Files saved by
v0.3.0 to v0.3.3 (and earlier) load only with `compile=False`: their loss and metric configs
use column counts, and loading them with `compile=True` raises `ValueError`, naming the
migration. Rebuild such a model from its preset and copy the weights. Continuous models can
use `new.set_weights(old.get_weights())`. Binary models must be migrated with
`models.copy_legacy_weights` of pypsps v0.4.x, because v0.5.0 removes it.

### Removed

* The `build_*` functions: `build_toy_model`, `build_model_binary_normal`,
  `build_model_binary_exponential`, `build_model_continuous_survival`,
  `build_model_continuous_exponential` and `build_model_continuous_weibull`. Use
  `models.build_model(**presets.<name>(...))`.
* `models.copy_legacy_weights`.
* The column-count constructors of `OutcomeLoss`, `TreatmentLoss` and the metric classes,
  and the count form of `predictive_state_df_gen` and `uniform_entropy_gen`. They take a
  `StateMixture` (or a `ColumnLayout`) and raise `TypeError` for anything else.
* `metrics.causal_loss_metric_gen(alpha=...)`; `outcome_loss_weight` is now keyword-only.
* `utils.split_y_pred`, `split_outcome_pred`, `split_y_true`, `get_n_states`,
  `get_state_column_indices`, `agg_outcome_pred`, `agg_treatment_pred`, `get_column_layout`
  and `utils.ColumnLayout`.
* `neglogliks.negloglik_per_state`, `neglogliks.posterior_from_negloglik_per_state` and
  `losses.prob_state_given_treatment_features`.
* `families.from_negloglik` and `ColumnLayout.from_counts`.

### Changed

* `StateMixture` and `ColumnLayout` require `n_states`; K is no longer inferred from the
  `y_pred` width. A saved mixture without `n_states` raises `ValueError`.
* `CausalLoss` raises `ValueError` when its outcome and treatment losses hold different
  mixtures (in v0.4.x the column-count constructors only warned).
* The notebooks use `ColumnLayout` and `StateMixture` instead of the removed `utils`
  helpers, and are re-run.

## pypsps v0.4.0 - Nov 4, 2026

Breaking: every model now uses one treatment head, `treatment_params` (a
`PredictiveStateParams` layer). The binary builders' `propensity_logit_state_<k>` and
`propensity_score` layers are gone, and binary-model `.weights.h5` files saved by earlier
versions don't load. **Warning:** `load_weights` on such a file raises `ValueError` only after
it has overwritten part of the model (the old propensity logits land in the constant scale
heads). Discard that model, and migrate from the old `.keras` file with
`models.copy_legacy_weights`. Several other layer names change (see "Changed"), and the toy
model gets one more BatchNormalization layer, so toy-model `.weights.h5` files don't load
either. `.keras` files load as before.

Deprecated: every `build_*` function. Use `models.build_model(**presets.<name>(...))` with the
same arguments. The `build_*` functions are removed in v0.5.0. `docs/model_building.md` has a
migration section.

### Added

* `models.copy_legacy_weights(old, new)` copies the weights of a model built before v0.4.0
  into the matching preset model. For binary models it maps each layer by its role and moves
  the per-state propensity logits into `treatment_params`; the outcome and state weights then
  match exactly, and the propensity up to float32 rounding. For continuous models it is
  `new.set_weights(old.get_weights())`. It raises `ValueError` for a toy model and for models
  that don't match. It is removed in v0.5.0.
* `models.default_metrics` logs `propensity_score_auroc` for a binary treatment, as the binary
  builders did, next to `propensity_score_aupr`.

### Changed

* `build_toy_model`, `build_model_binary_normal` and `build_model_binary_exponential` build
  their preset's model:
  * The propensity is one `treatment_params` layer, like the continuous-treatment models.
  * Layer names follow one scheme (listed in `docs/model_building.md`): `features` and
    `treatment` for the inputs (were auto-named in the binary-normal model), `features_bn`,
    `features_and_treatment` (toy: was `features_bn_and_treatment`),
    `predictive_state_dense_<i>` (was `features_processing_for_propensity_score` and auto
    names), `state_weights` (was `predictive_state_simplex_<n>`), `outcome_trunk_dense_<i>`
    (was `inputs_processing`), and `{param}_hidden_state_<k>` / `{param}_raw_state_<k>` /
    `{param}_state_<k>` (were `{param}_feat_eng_state_<k>` / `{param}_pred_state_<k>` /
    `scale_logit_<k>`). `output_tensor` and `{param}_all_states` are unchanged.
  * The toy model's state network uses the standard hidden stack, which adds one
    BatchNormalization layer.
  * Metrics: the toy and binary-normal models add `outcome_nll_metric`,
    `predictive_state_df` and `uniform_entropy`; the binary-exponential model adds
    `outcome_nll_metric`, `uniform_entropy` and `outcome_mean_squared_error` (over units with
    an event). `causal_loss_metric` is unchanged.
* `build_model` and `compile_model` are no longer experimental.
* The toy and binary-normal degrees-of-freedom target stays `max(1, log(n_states / 2))`; the
  training guide's new section on the predictive-state regularizer explains how to tune it.
* The `get_default_callbacks` docstring and the training guide explain why
  `val_causal_loss_metric` can stall during the alpha warm-up, and how to choose `patience`.

### Deprecated

* `models.build_toy_model`, `build_model_binary_normal` and `build_model_binary_exponential`
  warn with `FutureWarning` and are removed in v0.5.0.

### Fixed

* `build_model_binary_normal(loc_layer=None)` and `build_model_binary_exponential(log_rate_layer=None)`
  raised `TypeError`; `None` now gives a linear head per state. Empty hidden-layer lists raised
  `IndexError`; they now give an identity stack.
* `outcome_loss_weight` can be set for the binary-normal and toy models (through the preset
  dict); the builders ignored it.
* The binary models no longer pass `input_dim` to `PredictiveStateSimplex`, which made Keras
  warn.

### Removed

* `CausalLoss._alpha` (deprecated in v0.3.0); use `CausalLoss.alpha`.
* The private `models._build_binary_normal_causal_loss` and
  `_build_binary_exponential_causal_loss` helpers.

### Dependencies

* pypress v0.3.2. Its `RegularizerScheduler` now advances a `ScheduledValue` in the
  predictive-state regularizer, which pypsps holds on the loss; with v0.3.1 the value stayed
  at its `end`. The training guide shows how to warm up `l2` or anneal `target`.

### Development

* `pyproject.toml` configures isort with the black profile and 100 columns, so isort and
  ruff-format no longer rewrite each other's imports.

## pypsps v0.3.8 - Nov 4, 2026

### Fixed

* `models.get_propensity_state_conditional_means(model)` works for every pypsps model. It
  used to look up the binary builders' `propensity_logit_state_<k>` layers, so it raised for
  continuous and `build_model` models. It now reads the treatment block of `y_pred` through
  the model's mixture and returns each state's mean treatment, `(n_states, 1)`: the
  propensity score for a binary treatment, the `loc` for a Normal one. Models saved before
  v0.3.4 and loaded with `compile=False` still use the layer names (until v0.5.0).

## pypsps v0.3.7 - Nov 4, 2026

### Added

* `pypsps.keras.presets`: one function per `build_*` builder (`toy`, `binary_normal`,
  `binary_exponential`, `continuous_survival`, `continuous_exponential`,
  `continuous_weibull`), with the builder's arguments. Each returns a plain dict of
  `models.build_model` arguments, so `models.build_model(**presets.continuous_weibull(...))`
  builds the model, and a dict merge (`presets.x(...) | {"alpha": v}`) replaces an entry.

### Deprecated

* `models.build_model_continuous_survival`, `build_model_continuous_exponential` and
  `build_model_continuous_weibull` warn with `FutureWarning` and are removed in v0.5.0. Use
  `models.build_model(**presets.continuous_*(...))` with the same arguments. The model is the
  same: its weights are in the same order, so `.weights.h5` files load, and predictions,
  loss and gradients are identical.

### Changed

For models from the continuous builders (and presets):

* Layer names: the treatment head `treatment_loc_scale` is now `treatment_params`, and the
  state simplex is `state_weights`. `outcome_params_all_states` is replaced by
  `log_scale_all_states` (and `log_shape_all_states` for Weibull). Code that calls
  `model.get_layer` with an old name needs the new one; weight files are unaffected.
* Metrics: `treatment_mean_squared_error` and `outcome_mean_squared_error` (over units with
  an event) are added.

### Fixed

* `treatment_init_values=(loc, scale)` crashed, because pypress wants one row per parameter
  and one column per state. A flat `(loc, scale)` and two per-state rows both work now.
* `log_shape_bounds` that exclude 0 crashed (the start value was clipped onto a bound). The
  log shape now starts at 0 when the bounds contain it, and at their midpoint otherwise.
* `init_scale <= 0` gave a NaN start value; it raises `ValueError` now.

## pypsps v0.3.6 - Nov 4, 2026

### Added

* `models.build_model` (experimental): one builder for any outcome family and any treatment
  family, with a `"shared_trunk"` or `"gated"` architecture, a `HeadSpec` per outcome parameter
  (hidden layer, constant, or shared across states), start values for outcome and treatment
  parameters, and the `max_time` overflow check for time-to-event outcomes. It validates every
  argument before it creates any layer.
* `models.compile_model`: compiles any network that ends in `layers.CausalOutput`, with the
  `CausalLoss` built from the model's mixture. `models.default_metrics` gives the metrics it
  uses for each pair of families.
* `layers.BiasOnly(initializer=...)` sets the start value of the constant. Configs saved without
  the argument still load.

## pypsps v0.3.5 - Nov 4, 2026

### Fixed

* `OutcomeLoss`, `TreatmentLoss` and `CausalLoss` weight each unit by its `sample_weight`.
  They used to reduce the batch first, so a weight vector `w` only scaled the batch loss by
  `mean(w)`. Each `call` now returns the per-unit loss `[N]`, and Keras weights and reduces
  it. This also makes `reduction="mean_with_sample_weight"` work.
* `causal_loss_metric` and `outcome_nll_metric` average over units, weighted by
  `sample_weight`, instead of averaging the batch means.

### Changed

* Loss and metric values change on weighted data. Without `sample_weight` the losses are
  unchanged. The two metrics change slightly when the last batch is smaller, because they
  now count units, as Keras's own `loss` does.
* `CausalLoss` adds the `predictive_states_regularizer` once, after the reduction, and
  `sample_weight` does not scale it. Without `sample_weight` this is the previous value
  under every reduction.
* An `OutcomeLoss` or `TreatmentLoss` subclass that adds a penalty in `call` now adds it per
  unit, before the reduction. Under `"sum_over_batch_size"` the value is unchanged; under
  `"sum"` the penalty is counted once per unit.

## pypsps v0.3.4 - Nov 4, 2026

### Added

* `layers.CausalOutput(mixture)`: the output layer of every builder. It encodes the heads
  into `y_pred` (same columns as before, no weights, still named `output_tensor`) and stores
  the model's `StateMixture` in the saved model.
* `StateMixture.from_model(model)` and `ColumnLayout.from_model(model)` read the mixture off
  the `CausalOutput` layer, falling back to the compiled loss for models built earlier.

### Changed

* `pypsps.inference` and `utils.get_column_layout` read the outcome family from the model,
  so they also work on a model loaded with `compile=False`. `inference` mixes the family
  means through `StateMixture.outcome_mean`.
* Models saved by v0.3.4 contain a `pypsps>CausalOutput` layer and don't load in earlier
  releases. `.weights.h5` files saved by v0.3.3 still load into models built by v0.3.4.

### Deprecated

* The column-count API raises a `FutureWarning` and is removed in v0.5.0: the three-int
  constructors of the losses and metrics, the two-int form of `predictive_state_df_gen` and
  `uniform_entropy_gen`, `utils.split_y_pred`, `split_outcome_pred`, `split_y_true`,
  `get_n_states`, `get_state_column_indices`, `agg_outcome_pred`, `agg_treatment_pred`,
  `get_column_layout` and `ColumnLayout`, `neglogliks.negloglik_per_state` and
  `posterior_from_negloglik_per_state`, and `losses.prob_state_given_treatment_features`.
  Loading a file saved with column-count configs doesn't warn. `docs/model_building.md`
  lists the replacements and the v0.5.0 migration.

## pypsps v0.3.3 - Nov 4, 2026

### Fixed

* `OutcomeMeanSquaredError` and `OutcomeMeanAbsoluteError` mixed the raw per-state
  parameters, so a Normal outcome was also scored against the mixed `scale` column. They
  now score the mixed family mean `sum_k P(s_k | x) * E[Y | s_k]`. **Their values change**
  for the toy and binary-Normal builders. `TreatmentMeanSquaredError` and
  `TreatmentMeanAbsoluteError` get the same fix: a Normal treatment uses `loc` only.
* For a time-to-event outcome, the outcome MSE and MAE use only units with an observed
  event, and report `NaN` (not 0) when no event has been seen since the last reset.
* The propensity metrics read the treatment column through the column layout instead of
  the last `y_true` column.

### Changed

* The propensity, treatment and outcome metrics take the model's `StateMixture`, e.g.
  `metrics.OutcomeMeanSquaredError(model.loss.mixture)`. `predictive_state_df_gen` and
  `uniform_entropy_gen` take its `ColumnLayout`. The metric configs store the mixture (or
  the layout).
* The column-count constructors still work, and configs saved by v0.3.0 to v0.3.2 still
  load. They infer the family from the counts: 2 parameters and 1 `y_true` column is
  Normal, and 1 parameter is used as the mean directly. Any other count raises a
  `ValueError` asking for the mixture.
* The builders create one `StateMixture` and pass it to `CausalLoss.from_mixture` and to
  every metric. Builder signatures are unchanged.

## pypsps v0.3.2 - Nov 4, 2026

### Added

* `pypsps.keras.mixture.StateMixture`: the per-state NLLs, the state posterior, the outcome
  and treatment mixture NLLs, `causal_nll`, and the outcome and treatment means, from one
  outcome family, one treatment family and the column layout.
* `OutcomeLoss(mixture, outcome_likelihood=...)`, `TreatmentLoss(mixture)` and
  `CausalLoss.from_mixture(mixture, ...)`. `CausalLoss.mixture` and `CausalLoss.layout`.

### Changed

* The losses compute through `StateMixture`. Values and gradients equal the previous
  ones to float32 rounding. The three-int constructors still work: they map their
  elementwise NLLs onto families.
* Loss configs store a `"mixture"` instead of the nested NLLs and the three ints. Configs
  saved by v0.3.0 and v0.3.1 still load.
* `CausalLoss` raises a `ValueError` if losses built from a mixture don't share the same
  mixture. With the three-int constructors, a treatment NLL or column-count mismatch between
  the outcome and the treatment loss gives a `FutureWarning`; it will raise from v0.5.0.
* `OutcomeLikelihood` now lives in `pypsps.keras.mixture`; `losses.OutcomeLikelihood` is
  the same object.
* An unknown loss `reduction` raises `ValueError` instead of `NotImplementedError`.

## pypsps v0.3.1 - Nov 4, 2026

### Added

* `pypsps.keras.layout.ColumnLayout`: decodes `y_pred` into named per-state parameter
  blocks and the state weights, decodes `y_true` into outcome and treatment, and encodes
  blocks back into `y_pred`. It checks widths and serializes with Keras.
* `pypsps.keras.families`: `Normal`, `Bernoulli`, `Exponential` and `Weibull`, each with
  its parameters, head activations, per-state NLL and mean. `LossFamily` wraps any
  elementwise Keras loss. `families.get(name, ...)` looks a family up by name, and
  `families.from_negloglik(loss, n_params, n_true_cols)` maps the `neglogliks` losses
  onto families.

### Changed

* `utils.split_y_pred`, `split_y_true`, `split_outcome_pred`, `get_n_states` and
  `get_state_column_indices` now delegate to `ColumnLayout`. `get_n_states` raises a
  `ValueError` when the `y_pred` width is not a multiple of the columns per state;
  previously it silently truncated.
* `inference` reads each state's outcome mean from the family of the model's outcome
  loss. Values are unchanged up to float32 rounding.
* The continuous survival builders' bound and `max_time` checks now live in the
  `Exponential` and `Weibull` families. The overflow error message is reworded.

## pypsps v0.3.0 - Nov 4, 2026

Breaking: pypsps requires Keras 3 (`tensorflow>=2.16`). Keras 2 (`TF_USE_LEGACY_KERAS=1`)
is no longer supported.

Breaking: the metric generators (`causal_loss_metric_gen`, `outcome_nll_metric_gen`,
`predictive_state_df_gen`, `uniform_entropy_gen`) return Keras metric objects instead of
functions. Inside `model.compile` their names are unchanged.

Value change: `causal_loss_metric` of the continuous builders now uses alpha = 1
(`outcome_loss_weight` × outcome NLL + treatment NLL, no penalties). It used to follow the
scheduled alpha. Their `outcome_loss_weight` applies as before. The toy and binary
builders' values don't change.

### Added

* A model compiled by any `build_*` function now reloads with
  `tf.keras.models.load_model(path)`, including its loss, metrics and current alpha, and
  `pypsps.inference` works on the reloaded model. See "Saving and Loading" in
  docs/model_building.md.
* `import pypsps.keras` registers every pypsps layer, loss and metric, and the pypress
  layers and regularizers the builders use, with Keras.
* `metrics.CausalLossMetric(outcome_loss, treatment_loss, outcome_loss_weight=1.0)`:
  `outcome_loss_weight` × outcome NLL + 1 × treatment NLL, without the predictive-state
  regularizer or any embedded penalty, at alpha = 1 whatever alpha the training loss uses.
  Values are comparable across epochs and alpha schedules. `outcome_loss_weight=0` is
  allowed. Every builder compiles it as `causal_loss_metric`, with the training loss's
  `outcome_loss_weight`.
* `CausalLoss.alpha`: the alpha variable. `loss.alpha = x` assigns in place, so alpha
  schedules and every object that reads `loss.alpha` see the new value.
* `get_config`/`from_config` on `OutcomeLoss`, `TreatmentLoss` and `CausalLoss`. The config
  stores the current alpha.

### Changed

* Dependencies: `tensorflow>=2.16`; `tf-keras` removed; `tensorflow_probability` is no
  longer a runtime dependency. pypsps doesn't import it. Install it yourself to pass a
  `tfp.distributions` constructor to `neglogliks.NegloglikLoss`.
* `import pypsps.keras` no longer imports TensorFlow Probability, which saves about 1 s
  of import time.
* The metric generators return Keras metric objects (`CausalLossMetric` or
  `tf.keras.metrics.MeanMetricWrapper`). Calling one directly now returns a running mean; use
  `reset_state()` between calls.
* `AlphaScheduleCallback` (and `GaussianAlphaSchedule`) require the model's loss to be a
  `CausalLoss`.
* `penalty_free()` rebuilds the loss from its base class's config; the constructor arguments
  are listed once.

### Deprecated

* `CausalLoss._alpha`: use `CausalLoss.alpha`. It warns with `FutureWarning` and is removed
  in v0.4.0.
* The `alpha` argument of `causal_loss_metric_gen`. It warns with `FutureWarning`, is
  ignored, and is removed in v0.5.0.

### Removed

* `pypsps.keras.metrics._penalty_free_copy` (private pass-through to `penalty_free()`).

### Fixed

* `load_model` on a compiled pypsps model raised `TypeError`.
* `AlphaScheduleCallback` silently did nothing for a loss whose alpha was a plain float; that
  fallback is removed.

## pypsps v0.2.2 - Nov 4, 2026

Breaking (logged metric keys): four metric classes log under new default names, and the
binary builders log the propensity AUC as `propensity_score_auroc` and `propensity_score_aupr`.

### Fixed

* `PropensityScoreAUC(curve="PR")` computed the ROC AUC: every metric class in
  `pypsps.keras.metrics` ignored `name` and other keyword arguments. The binary builders
  (`build_toy_model`, `build_model_binary_normal`, `build_model_binary_exponential`) now
  report the PR AUC they asked for, and the ROC AUC alongside it.

### Changed

* Default metric names: `outcome_mean_squared_error`, `outcome_mean_absolute_error`,
  `treatment_mean_squared_error`, `treatment_mean_absolute_error` (previously
  `mean_squared_error` / `mean_absolute_error`), and `propensity_score_auc` without a
  per-session `_N` suffix. `val_mean_squared_error` becomes
  `val_outcome_mean_squared_error`.
* The binary builders compile two propensity AUC metrics with explicit names:
  `propensity_score_auroc` (`curve="ROC"`) and `propensity_score_aupr` (`curve="PR"`).
  `val_propensity_score_auc_N`, which was the ROC AUC despite asking for PR, becomes
  `val_propensity_score_auroc`.

### Added

* `get_config` on the six metric classes, so they round-trip through `from_config`.

## pypsps v0.2.1 - Nov 4, 2026

### Added

* `pypsps.keras.models.build_model_continuous_survival` for a continuous (Normal)
  treatment and an exponential or Weibull outcome, with the wrappers
  `build_model_continuous_exponential` and `build_model_continuous_weibull`. Each state's
  outcome head sees the shared feature trunk and the treatment embedding gated by that
  state's probability; Weibull scale and shape have separate hidden layers. The outcome
  parameters are soft-clipped to required `log_scale_bounds` (and `log_shape_bounds`); an
  optional `max_time` rejects bounds that would overflow the loss in float32.
* `pypsps.keras.layers.SoftClip`: a serializable sigmoid squash into `(min, max)`.

### Fixed

* `inference.predict_ate_continuous` raised `AttributeError` for array features; it now
  returns an ndarray as documented, and a Series named `"ate"` for a DataFrame.

## pypsps v0.2.0 - Nov 4, 2026

Breaking: `OutcomeLoss` now defaults to the interventional outcome likelihood, so the
default loss values, gradients and trained models differ from v0.1.10. Pass
`outcome_likelihood="observational"` to keep the previous behavior exactly.

### Added

* `OutcomeLoss(outcome_likelihood=...)` with `pypsps.keras.losses.OutcomeLikelihood`:
  `"interventional"` (default, -log p(Y | do(A), X) with prior weights
  P(S | X) = P(S | X, do(A))) or `"observational"` (-log p(Y | A, X) with posterior
  weights P(S | X, A), the v0.1.10 behavior, bit-identical).
  Strings convert to the enum. `penalty_free()` keeps the setting.
* `CausalLoss(alpha=...)` and `causal_loss_metric_gen(alpha=...)` accept a float32
  `tf.Variable`, which is shared rather than copied, so a metric can follow an alpha
  schedule.

## pypsps v0.1.10 - Nov 4, 2026

### Added

* `pypsps.keras.layers.StateColumnGate`: a serializable layer that selects one column of a
  2D tensor, for per-state loops where a `Lambda` closure would bind late.
* `pypsps.keras.metrics.outcome_nll_metric_gen`: a metric of the outcome negative
  log-likelihood alone, without penalties or alpha, for use as a selection criterion.
* `pypsps.datasets.utils.identify_binary_features`: finds strictly binary and quasi-binary
  columns of a DataFrame.

## pypsps v0.1.9 - Nov 4, 2026

### Added

* `pypsps.keras.alpha_schedule`: a half-Gaussian schedule for the treatment-loss weight,
  `alpha(e) = alpha_end + (alpha_start - alpha_end) * exp(-c * e**2)`, with validation
  (`AlphaSchedule`) and the first settled epoch (`warmup_epochs`).
* `pypsps.keras.callbacks.GaussianAlphaSchedule`, an `AlphaScheduleCallback` that follows it.
* `pypsps.keras.callbacks.EligibleBestCheckpoint`: early stopping that selects and restores
  the best epoch at or after `start_from_epoch`, skips NaN and infinite epochs, and raises
  `NoEligibleCheckpointError` if no epoch was eligible.
* `pypsps.keras.callbacks.WarmupReduceLROnPlateau`: `ReduceLROnPlateau` that is inactive
  before `start_from_epoch`.

### Changed

* Bumped the `pypress` dependency to
  [v0.3.1](https://github.com/gmgeorg/pypress/releases/tag/v0.3.1).

## pypsps v0.1.8 - Nov 2, 2026

### Fixed

* `pypsps.utils.get_column_layout` now raises a clear `RuntimeError` telling the caller to
  compile the model when `model.loss` is not a `CausalLoss`, instead of an opaque
  `AttributeError` on `model.loss._outcome_loss`.

## pypsps v0.1.6 - Aug 27, 2026

### Fixed

* Bumped the `pypress` dependency to
  [v0.2.4](https://github.com/gmgeorg/pypress/releases/tag/v0.2.4), which fixes
  `TargetEntropy`/`Uniform` (`pypress.keras.regularizers`) not being scale-invariant in `K`
  (the number of predictive states): the squared deviation from the target entropy had a
  dynamic range that grew as `log(K)^2`, so a fixed `l2` implied a different effective
  regularization strength depending on `K`. `pypress` v0.2.4 normalizes the deviation by
  `log(K)` before squaring, bounding the penalty in `[0, l2]` regardless of `K`.
* `pypress` v0.2.4 also changed `DegreesOfFreedom` from an L1 to a squared L2 penalty,
  renaming its `l1`/`dof_l1` constructor params to `l2`/`dof_l2`. Updated all `pypsps` call
  sites accordingly, including renaming the public `df_penalty_l1` parameter (in
  `pypsps/keras/models.py`'s `build_toy_model`, `build_model_binary_normal`,
  `build_model_binary_exponential`, etc.) to `df_penalty_l2` to match. This is a breaking
  rename for any caller passing `df_penalty_l1` by keyword.

## pypsps v0.1.5 - Aug 23, 2026

### Fixed

* `pypsps/keras/neglogliks.py`: bare, no-`axis` `tf.squeeze()` calls in
  `NegloglikLoss.call`, `NegloglikExponential.call`, `NegloglikExponentialScale.call`, and
  `NegloglikWeibull.call` decided at trace time which size-1 dimensions to drop; when a
  tensor's static shape had an unknown (`None`) dimension -- which happens whenever
  `train_step` is retraced for a batch shaped differently from the one used to first trace
  the graph (e.g. a partial/remainder final batch, or a batch size larger than a
  cross-validation fold) -- `tf.squeeze()` couldn't statically prove that dimension wasn't 1,
  degrading the entire output shape to unknown rank. That unknown-rank tensor then broke
  shape inference inside `posterior_from_negloglik_per_state`, raising `TypeError:
  unsupported operand type(s) for -: 'NoneType' and 'int'` during `model.fit()` in graph
  mode. Fixed by deleting the now-redundant squeezes (the squeezed values are already
  rank-1 from column slicing) and pinning the loss tensor's static shape explicitly via
  `tf.ensure_shape(losses, [None])`. `NegloglikNormal.call` had the same pattern and is now
  fixed for real upstream (a prior fix only ever landed in a downstream fork).

## pypsps v0.1.4 - Aug 23, 2026

### Fixed

* `get_n_cols` (`pypsps/utils.py`) used the old TF1-era `y.get_shape().as_list()[1]` for
  non-`np.ndarray` inputs. Under Keras 3, the tensor flowing through the nested
  `CausalLoss` -> `OutcomeLoss` call chain during graph-mode loss tracing is a `KerasTensor`,
  which has no `get_shape()` method at all, raising `AttributeError` and crashing training
  with the default (posterior-weighted) loss. Fixed by using `y.shape[1]` unconditionally,
  which works for `np.ndarray`, eager `tf.Tensor`, graph-traced tensors, and `KerasTensor`
  alike.

## pypsps v0.1.3 - Aug 22, 2026

### Fixed

* `split_y_true` (`pypsps/utils.py`) sliced `treatment_true` as "all remaining columns after
  the outcome columns" (`y_true[:, n_outcome_true_cols:]`) instead of the single treatment
  column. Every committed loss/model only ever has one treatment column, so this was correct
  by accident; passing a `y_true` with extra trailing columns (e.g. survival event indicators
  or auxiliary features appended after treatment) would silently fold them into
  `treatment_true` with no error. Fixed to slice exactly one column
  (`y_true[:, n_outcome_true_cols : n_outcome_true_cols + 1]`).

## pypsps v0.1.2 - Aug 22, 2026

### Added

* `NegloglikExponentialScale` / `NegloglikWeibull` (`pypsps/keras/neglogliks.py`): survival
  losses parameterized by `log_scale` (and `log_shape` for Weibull) instead of `log_rate`,
  for more numerically stable gradients when predicting the mean survival time directly.
  Both formulas verified against `tfp.distributions.Exponential`/`Weibull` `log_prob`/
  `log_survival_function`
* `NegloglikExponential`/`NegloglikExponentialScale`/`NegloglikWeibull` are now registered
  Keras-serializable (`register_keras_serializable`); `NegloglikExponential` also gained a
  `get_config()` override, since without it its `log_rate` flag silently reset to `False` on
  save/load
* `_state_conditional_outcome_mean` (`inference.py`) now converts `NegloglikExponentialScale`
  (`exp(log_scale)`) and `NegloglikWeibull` (`exp(log_scale) * Gamma(1 + 1/shape)`) raw
  parameters to their distribution mean, so `predict_ute_binary`/`predict_ute_continuous`/
  `predict_ate_binary`/`predict_ate_continuous` work for these two outcome losses instead of
  raising `NotImplementedError`

## pypsps v0.1.1 - Aug 22, 2026

### Fixed

* `causal_loss_metric_gen` (`pypsps/keras/metrics.py`) reused the exact `outcome_loss`/
  `treatment_loss` instances it was given, so any penalty term folded into either loss's own
  `call()` (e.g. a within-state balance penalty) would leak into the metric used for
  `EarlyStopping`/`ReduceLROnPlateau`/checkpoint selection -- silently turning "pure held-out
  likelihood" into "likelihood plus whatever penalty this candidate happened to draw",
  systematically rewarding weaker penalties regardless of fit quality. No such penalty is
  wired into committed `OutcomeLoss`/`TreatmentLoss` yet, so this was a structural gap rather
  than an active bug. Fixed by adding an explicit `penalty_free()` method to
  `OutcomeLoss`/`TreatmentLoss` that reconstructs the loss from a hand-written list of its own
  non-penalty constructor arguments (not by scanning attribute names for a naming convention);
  `causal_loss_metric_gen` now always calls it before wrapping the losses in its internal
  `CausalLoss`. A future subclass that adds a penalty either gets it correctly zeroed
  automatically (if it defaults to "off") or `penalty_free()` raises until that subclass
  explicitly overrides it -- it can never leak through unnoticed.

## pypsps v0.1.0 - Aug 21, 2026

Breaking: the propensity head's output shape changed (state-conditional instead of
pre-mixed), so models trained/saved with `<0.1.0` are not compatible with this release.

### Fixed

* `OutcomeLoss` mixed predictive states using the prior weights `P(s_k|x)` instead of
  the posterior `P(s_k|x,a)`; since treatment `a` is observed, only the posterior
  makes `TreatmentLoss + OutcomeLoss` telescope exactly into `-log p(a,y|x)`
* `log(weights + eps)` in the state-mixture losses clamped log-weights at `log(eps)`,
  distorting sharp/near-degenerate predictive states; replaced with a `safe_log` that
  only substitutes for exact-zero entries
* `CausalLoss`'s `alpha` (treatment-loss weight) was a plain Python float, so
  `AlphaScheduleCallback`'s per-epoch schedule was baked into a constant the first time
  `model.fit()` traced the training graph; rebinding it afterward updated only an eager
  Python attribute the compiled training function never read again, so the schedule
  never reached the optimizer even though it reported the intended curve. `alpha` is now
  a `tf.Variable`, updated via `.assign(...)`
* `predict_ute_binary`/`predict_ute_continuous` (`inference.py`) hardcoded
  `n_outcome_pred_cols=2` (Normal-only), silently mis-slicing columns -- and computing
  the wrong number of states -- for any other outcome distribution (e.g. the
  exponential/survival model), returning a plausible-looking but meaningless number
  with no error. Now uses `utils.get_column_layout(model)` and converts each outcome
  distribution's raw parameters to its actual mean (Normal: `loc`; Exponential:
  `exp(-log_rate)`), raising `NotImplementedError` for any distribution it doesn't know
  how to convert
* `BiasOnly.get_config()` dropped `units` (silently reverting to `1` on reload) and
  stored `bias_regularizer` raw instead of via `tf.keras.regularizers.serialize`/
  `deserialize`
* the three `bootstrap_*` functions seeded their resampling RNG inconsistently
  (`RandomState(0)` vs `RandomState(n_samples)`); all three now take an explicit
  `random_state: int = 0`
* `build_toy_model`/`build_model_binary_normal` didn't compile `causal_loss_metric`
  (only `build_model_binary_exponential` did), so `get_default_callbacks`'s
  documented recommended monitor, `"val_causal_loss_metric"`, didn't exist for
  those two builders; Keras only warns (not raises) on a missing monitored metric,
  so `EarlyStopping`/`ReduceLROnPlateau` silently never triggered
* README.md's and `docs/inference.md`'s code examples called a nonexistent
  `inference.predict_ate(...)` (the real functions are `predict_ate_binary`/
  `predict_ate_continuous`) and called `utils.split_y_pred(preds)` with the wrong
  number of arguments while unpacking its result into the wrong number of
  variables -- anyone copy-pasting the README example hit an immediate error

### Changed

* **Breaking**: the propensity head is now state-conditional (one output per state,
  like the outcome head) instead of pre-mixed into a single marginal column, so that
  the posterior above can be computed
* `recommended_callbacks` renamed to `get_default_callbacks`; `monitor` is now a
  required argument instead of defaulting to `"val_loss"`
* Bumped `pypress` dependency from `v0.0.6` to `v0.2.3`; `Uniform` regularizer's `l1`
  penalty argument renamed to `l2` to track pypress's L1 -> L2 entropy penalty change

### Added

* `utils.get_column_layout(model)` / `utils.ColumnLayout`: reads
  `n_outcome_pred_cols`/`n_treatment_pred_cols`/`n_outcome_true_cols` off a compiled
  model's `CausalLoss` instead of hardcoding them at each call site -- the root cause
  of a real shape-mismatch bug found in downstream code
* `uniform_entropy_gen`: metric for the mean Shannon entropy of predictive state
  weights across a batch
* `get_propensity_state_conditional_means(model)`: since the propensity head is no
  longer a single `pypress.keras.layers.PredictiveStateMeans` layer, this reconstructs
  the same per-state constants (post-sigmoid) from the model's `propensity_logit_state_<k>`
  layers -- the replacement for the old `model.layers[-2].state_conditional_means`
  sanity check, usable both before and after training

## pypsps v0.0.13 - Aug 20, 2026

### Fixed

* `OutcomeLoss` computed a weighted sum of per-state negative log-likelihoods instead
  of the exact NLL of the state mixture (`logsumexp`)
* `PropensityScoreBinaryCrossentropy`/`PropensityScoreAUC` read the treatment label
  from a hardcoded column slice of `y_true`, breaking for outcomes with more than one
  column (e.g. survival's `event_time` + `event_indicator`)
* `TreatmentMeanAbsoluteError` referenced the wrong attribute for the outcome column
  count

### Added

* `docs/` directory with architecture, datasets, development, inference, and model
  building guides; `CLAUDE.md` contributor guidelines for AI assistants

## pypsps v0.0.12 - Dec 21, 2025

### Changed

* Version/packaging cleanup

## pypsps v0.0.11 - Jul 26, 2025

### Added

* `bootstrap.py`: bootstrap resampling for uncertainty estimates around causal effects

### Changed

* Cleaned up the demo notebook; updated architecture diagram in the README

## pypsps v0.0.10 - May 18, 2025

### Changed

* Raised minimum supported Python version

## pypsps v0.0.9 - Apr 6, 2025

### Added

* `datasets.binary_survival`: example dataset and notebook for binary survival outcomes
* Model and metrics support for survival-style outcomes (`event_time` + `event_indicator`)

## pypsps v0.0.8 - Mar 22, 2025

### Added

* Exponential distribution support in `neglogliks` and losses
* Causal loss metric

## pypsps v0.0.7 - Mar 18, 2025

### Changed

* Raised minimum supported Python version to >=3.12 and updated the `pypress`
  dependency accordingly
* Expanded test coverage for metrics

### Added

* `VerboseNEpochs` callback to print training logs every N epochs

## pypsps v0.0.6 - Mar 17, 2025

### Fixed

* Broadcasting and binary-loss errors in `losses.py`

### Added

* `pypsps.inference`: ATE estimation routines
* `pypsps.keras.callbacks` module

## pypsps v0.0.5 - Apr 26, 2024

### Changed

* Moved to relative imports throughout the package
* Added default loss functions for model builders

## pypsps v0.0.4 - Apr 24, 2024

### Added

* `neglogliks.py`: negative log-likelihood calculations for general distributions
* `build_model_binary_normal`: configurable model builder with additional
  architecture arguments
* Average-outcome metrics
* Unit test coverage for `utils`

## pypsps v0.0.3 - Apr 24, 2024

### Changed

* Migrated packaging from `setup.py`/`requirements.txt` to Poetry
* Added a GitHub Actions Python test workflow

## pypsps v0.0.2 - Apr 23, 2024

### Changed

* Cleanup of loss functions and general code cleanup
* Added model/layer serialization support
* New metrics
* Renamed `pypsps/version.py` to `pypsps/_version.py`

## pypsps v0.0.1 - Feb 28, 2022

Initial release of `pypsps`.

## TEMPLATE: pypsps vX.Y.Z

### Added

* ...

### Changed

* ...

### Deprecated

* ...

### Removed

* ...

### Fixed

* ...
