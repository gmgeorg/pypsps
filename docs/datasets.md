# Datasets

## Base Dataset Class

All datasets in `pypsps.datasets` inherit from `CausalDataset` defined in
`pypsps/datasets/base.py`.

### Common Methods

* `to_keras_inputs_outputs()`: Formats raw dataset structures into
  Keras-compatible input/target tuples.
* `naive_ate()`: Computes the unadjusted observational ATE for baseline
  evaluation.

## Available Implementations

* **`kang_schafer.py`**: Kang-Schafer simulation benchmark.
* **`lalonde.py`**: Classic LaLonde observational dataset.
* **`lunceford_davidian.py`**: Lunceford & Davidian simulation setup.
* **`binary_survival.py`**: Binary outcome survival dataset implementations.
* **`bites.py`**: BITES benchmark dataset interface.
