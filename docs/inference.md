# Causal Inference & Utilities

## Core Inference Functions

* **`pypsps.inference.predict_ate(model, features)`**: Predicts the Average
  Treatment Effect (ATE) given a trained model and input features.
* **`pypsps.utils.split_y_pred(preds)`**: Splits combined neural network
  predictions into propensity state outputs and outcome predictions.
* **`pypsps.bootstrap`**: Offers non-parametric bootstrap estimation methods for
  measuring uncertainty and generating confidence intervals for ATE estimates.

## Example Notebooks

* `notebooks/pypsps_minimal_working_example.ipynb`: Basic workflow for ATE
  estimation.
* `notebooks/pypsps_demo.ipynb`: In-depth end-to-end demonstrations covering
  multiple edge cases.
