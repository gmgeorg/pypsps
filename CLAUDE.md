# pypsps AI Assistant Guidelines & Codebase Context

This document serves as the primary instructions and contextual map for AI
assistants working on the pypsps codebase. Detailed module documentation and
implementation guides are stored in the `docs/` directory.

## Core Mandates

* **Git Workflow**: Never commit or push directly to `main`. Create feature
  branches and open Pull Requests using Conventional Commits (`feat:`, `fix:`,
  `chore:`).
* **Single Version Source**: Manage versioning exclusively in `pyproject.toml`.
  Do not hardcode version numbers in `pypsps/_version.py` or other files.
* **Code Standards**: Adhere to PEP 8 with a 100-character line length managed
  via `ruff`.

## Project Overview

**pypsps** is a Python implementation of Predictive State Propensity
Subclassification (PSPS), a causal deep learning framework for observational
(non-randomized) data based on Kelly, Kong, and Goerg (2022). It uses `tf.keras`
to build propensity score strata that ensure within-stratum feature balance
across various treatment and outcome types.

* **Python Requirements**: `>=3.12, <4.0.0`
* **License**: MIT
* **Note**: This repository is an independent re-implementation and extension,
  not an official Google product.

## Module Documentation Index

* **[Architecture & Theory](docs/architecture.md)**: Mathematical principles,
  predictive state representations, and joint distribution modeling.
* **[Development & Setup](docs/development.md)**: Dependency specs, `pytest`
  testing rules, formatting workflows, and git standards.
* **[Model Architecture](docs/model_building.md)**: Keras API builders, custom
  layers, loss functions, metrics, and training callbacks.
* **[Datasets API](docs/datasets.md)**: `CausalDataset` base class, data
  formatting methods, and benchmark datasets (Kang-Schafer, LaLonde, etc.).
* **[Inference & Post-Processing](docs/inference.md)**: ATE estimation routines,
  prediction output splitting, and bootstrap sampling.

## Key Directory Structure

```text
pypsps/
├── _version.py              # Dynamic version resolution (via importlib.metadata)
├── bootstrap.py             # Uncertainty and bootstrap estimation
├── inference.py             # Causal inference and ATE estimation
├── utils.py                 # Shared helper functions
├── datasets/                # Inherit from datasets.base.CausalDataset
├── keras/                   # Custom layers, losses, metrics, models, callbacks
└── tests/                   # Pytest suite
```
