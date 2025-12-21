# CLAUDE.md - AI Assistant Context for pypsps

This document provides context for AI assistants working on the pypsps codebase.

## Project Overview

**pypsps** is a Python implementation of Predictive State Propensity Subclassification (PSPS), a causal deep learning algorithm for observational (non-randomized) data. The algorithm was originally proposed by Kelly, Kong, and Goerg (2022) for causal inference with data-driven propensity score stratification.

### Key Characteristics

- **Framework**: Built on TensorFlow/Keras with custom layers, metrics, and loss functions
- **Generality**: Supports any treatment type (binary, continuous, multi-class) and any outcome type (univariate, multivariate, binary, continuous, etc.)
- **API Compatibility**: Fully compatible with tf.keras API
- **License**: MIT
- **Python Version**: >=3.12,<4.0.0

## Architecture

PSPS decomposes the joint distribution Pr(outcome, treatment | features) by conditioning on intermediate predictive states from Pr(treatment | features). These predictive state representations are trained simultaneously with outcome models to provide principled propensity score strata that guarantee balancedness within strata.

## Project Structure

```
pypsps/
├── __init__.py              # Package initialization with version
├── _version.py              # Version management (uses importlib.metadata)
├── bootstrap.py             # Bootstrap estimation methods
├── inference.py             # Causal inference utilities (ATE prediction, etc.)
├── utils.py                 # Utility functions
├── datasets/                # Dataset implementations
│   ├── base.py             # Base CausalDataset class
│   ├── kang_schafer.py     # Kang-Schafer simulation dataset
│   ├── lalonde.py          # LaLonde dataset
│   ├── lunceford_davidian.py
│   ├── bites.py
│   ├── binary_survival.py  # Binary survival models
├── keras/                   # Keras/TensorFlow components
│   ├── layers.py           # Custom PSPS layers
│   ├── losses.py           # Causal loss functions
│   ├── metrics.py          # Causal metrics
│   ├── models.py           # Model builders (build_toy_model, etc.)
│   ├── callbacks.py        # Training callbacks
│   ├── neglogliks.py       # Negative log-likelihood implementations
└── tests/                   # Test suite
    ├── test_datasets.py
    ├── test_inference.py
    ├── test_losses.py
    ├── test_metrics.py
    ├── test_models.py
    ├── test_neglogliks.py
    ├── test_bootstrap.py
    └── test_utils.py
```

## Key Dependencies

- **tensorflow** (>=2.11.0)
- **tensorflow_probability** (>=0.18.0)
- **tf-keras** (>=2.14.1)
- **numpy** (>=1.11.0)
- **pandas** (>=1.0.0)
- **pypress** (custom dependency from GitHub)
- **tqdm** (>=4.62) - for progress bars

### Dev Dependencies

- pytest, flake8, ruff, pre-commit
- seaborn, scikit-learn, optuna, pydot, scikit-survival

## Version Management

**IMPORTANT**: The project uses a programmatic versioning approach:
- Single source of truth: `pyproject.toml` (version field)
- `_version.py` uses `importlib.metadata.version("pypsps")` to read the installed package version
- Never hardcode versions in `_version.py` - it should always read from package metadata

## Development Workflow

1. **Testing**: Run tests with `pytest`
2. **Code Quality**: Uses ruff (line-length=100) and flake8
3. **Pre-commit hooks**: Configured for code quality checks
4. **Installation**: `pip install git+https://github.com/gmgeorg/pypsps.git`

## Core Functionality

### Building Models

The `pypsps.keras.models` module provides:
- `build_toy_model(n_states, n_features, compile=True, alpha=10.)` - Template for binary treatment, continuous outcome

### Datasets

All datasets should inherit from `datasets.base.CausalDataset()`:
- Provides consistent interface for causal datasets
- Methods: `to_keras_inputs_outputs()`, `naive_ate()`, etc.
- Examples: `KangSchafer`, `LaLonde`, survival datasets

### Inference

- `inference.predict_ate(model, features)` - Predict Average Treatment Effect
- `utils.split_y_pred(preds)` - Split predictions into components

## Common Patterns

1. **Dataset Creation**: Wrap custom data in `CausalDataset` class
2. **Model Training**: Use `recommended_callbacks()` from models module
3. **ATE Estimation**: Compare true ATE vs naive ATE vs PSPS-predicted ATE
4. **Validation**: Use `validation_split` during training

## Important Notes

- This is NOT an official Google implementation - it's a re-implementation with extensions
- Original paper: [Kelly, Kong, and Goerg (2022)](https://proceedings.mlr.press/v177/kelly22a.html)
- The architecture supports more general scenarios than the original paper

## Git Workflow

- Main branch: `main`
- Uses conventional commits style (feat:, chore:, fix:, etc.)
- PRs are welcome; always open new branches and PRs. Do not work on `main` or push to `main`.

## Example Notebooks

- `notebooks/pypsps_minimal_working_example.ipynb` - Basic ATE estimation
- `notebooks/pypsps_demo.ipynb` - Comprehensive examples

## Testing Philosophy

- Comprehensive test coverage in `tests/` directory
- Tests for all major components: datasets, inference, losses, metrics, models
- Bootstrap functionality is tested separately
- Use pytest fixtures for common setup

## Code Style

- Line length: 100 characters (ruff configuration)
- Auto-fix enabled in ruff
- Follow PEP 8 conventions
- Type hints encouraged but not strictly enforced

## When Contributing

1. Update tests when adding new features
2. Keep version only in `pyproject.toml`
3. Follow existing patterns in dataset and model implementations
4. Add examples to notebooks when appropriate
5. Ensure compatibility with Keras API
6. Document any new loss functions or metrics clearly
