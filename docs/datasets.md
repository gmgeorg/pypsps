# Datasets

## Base Dataset Class

All datasets in `pypsps.datasets` inherit from `CausalDataset` defined in
`pypsps/datasets/base.py`.

### Common Methods

* `to_keras_inputs_outputs()`: Formats raw dataset structures into
  Keras-compatible input/target tuples.
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
