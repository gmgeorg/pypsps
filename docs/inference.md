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
* **`StateMixture.from_model(model)`**: the model's mixture. Its `layout.decode_pred(preds)`
  splits predictions into named per-state blocks (outcome parameters, state weights,
  treatment parameters); `outcome_mean(preds)` and `treatment_mean(preds)` give the mixed
  outcome mean and the marginal propensity. The column-count helpers in `pypsps.utils`
  (`split_y_pred`, `split_y_true`, `split_outcome_pred`, `get_n_states`,
  `get_state_column_indices`, `agg_outcome_pred`, `agg_treatment_pred`,
  `get_column_layout`) are deprecated and removed in v0.5.0.
* **`pypsps.bootstrap`**: Offers non-parametric bootstrap estimation methods for
  measuring uncertainty and generating confidence intervals for ATE estimates.

## Example Notebooks

* `notebooks/pypsps_minimal_working_example.ipynb`: Basic workflow for ATE
  estimation.
* `notebooks/pypsps_demo.ipynb`: In-depth end-to-end demonstrations covering
  multiple edge cases.
