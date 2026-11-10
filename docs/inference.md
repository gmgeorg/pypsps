# Causal Inference & Utilities

## Core Inference Functions

* **`pypsps.inference.predict_ate_binary(model, features)`** /
  **`predict_ate_continuous(model, features, treatment_grid, ...)`**: Predicts
  the Average Treatment Effect (ATE) for binary or continuous treatment, given a
  trained model and input features. Unit-level effects are available via the
  corresponding `predict_ute_binary`/`predict_ute_continuous`.
  They read the outcome family from the model
  (`pypsps.keras.mixture.StateMixture.from_model`), so they also work on a model loaded
  with `compile=False`.
* **`weights="prior" | "posterior"`** (on `predict_ate_*` and `predict_ute_*`, binary and
  continuous treatment): which state weights mix the per-state outcome means. `"prior"`
  (default) uses `P(S | X)`, which does not depend on the treatment; these are the weights the
  `"interventional"` outcome likelihood mixes with in training. `"posterior"` uses
  `P(S | X, A = a)`, by Bayes' rule from the per-state treatment density, so the effect is the
  difference of the model's conditional means `E[Y | a, x]`; these are the weights the
  `"observational"` outcome likelihood mixes with in training. Averaged over units, the posterior
  curve is the g-formula for `E[Y | a, X]`, which is the do(a) effect only if `X` alone removes
  the confounding. See
  [Training Guide](training_guide.md#4-interventional-and-observational-outcome-likelihood).
* **`StateMixture.from_model(model)`**: the model's mixture. Its `layout.decode_pred(preds)`
  splits predictions into named per-state blocks (outcome parameters, state weights,
  treatment parameters); `outcome_mean(preds)` and `treatment_mean(preds)` give the mixed
  outcome mean and the marginal propensity. They replace the column-count helpers that
  `pypsps.utils` had until v0.4.x (`split_y_pred`, `agg_outcome_pred`, ...).
* **`pypsps.bootstrap`**: Offers non-parametric bootstrap estimation methods for
  measuring uncertainty and generating confidence intervals for ATE estimates.

## Example Notebooks

* `notebooks/pypsps_minimal_working_example.ipynb`: Basic workflow for ATE
  estimation.
* `notebooks/pypsps_minimal_working_example_torch.ipynb`: The same workflow with the
  PyTorch backend.
* `notebooks/pypsps_demo.ipynb`: In-depth end-to-end demonstrations covering
  multiple edge cases.
