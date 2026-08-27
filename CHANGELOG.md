# Changelog

All notable changes to `pypsps` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
