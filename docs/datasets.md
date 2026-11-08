# Datasets

## Base Dataset Class

All datasets in `pypsps.datasets` inherit from `CausalDataset` defined in
`pypsps/datasets/base.py`.

### Common Methods

* `to_keras_inputs_outputs(layout=None)`: Formats raw dataset structures into
  Keras-compatible input/target tuples. The targets are `[outcome | treatment]`. Pass the
  model's layout (`pypsps.keras.layout.ColumnLayout.from_model(model)` for Keras,
  `model.mixture.layout` for a torch `PSPSModel`) as `layout=` to build them with
  `ColumnLayout.encode_true`, which raises if the outcome does not have the number of
  columns the outcome family reads (for example, a `[time, event]` outcome for a Normal
  model). Without a layout, nothing is checked. `pypsps.utils.prepare_keras_inputs_outputs(features, treatments, outcomes,
  layout=None)` does the same for DataFrames or arrays.
* `naive_ate()`: Computes the unadjusted observational ATE for baseline
  evaluation.

## Utilities

* `datasets.utils.identify_binary_features(X, threshold=0.95)`: returns the names of
  strictly binary columns (at most 2 distinct non-null values) and quasi-binary
  columns (numeric or categorical, whose two most frequent values cover at least
  `threshold` of the non-null rows).

## Available Implementations

* **`kang_schafer.py`**: Kang-Schafer simulation benchmark.
* **`lalonde.py`**: Classic LaLonde observational dataset.
* **`lunceford_davidian.py`**: Lunceford & Davidian simulation setup.
* **`binary_survival.py`**: Binary outcome survival dataset implementations.
* **`bites.py`**: BITES benchmark dataset interface.
