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
* **Backends**: TensorFlow (`pypsps[tf]`) and PyTorch (`pypsps[torch]`) are optional
  extras; `pypsps.core`, `inference` and `bootstrap` import neither.
* **License**: MIT
* **Note**: This repository is an independent re-implementation and extension,
  not an official Google product.

## Module Documentation Index

* **[Domain Glossary](docs/CONTEXT.md)**: Canonical terms (unit, predictive state,
  stratum, state weights, propensity, alpha, UTE/ATE) and which synonyms to avoid.
* **[Architecture & Theory](docs/architecture.md)**: Mathematical principles,
  predictive state representations, and joint distribution modeling.
* **[Development & Setup](docs/development.md)**: Dependency specs, `pytest`
  testing rules, formatting workflows, and git standards.
* **[Model Architecture](docs/model_building.md)**: Keras API builders, custom
  layers, loss functions, metrics, and training callbacks.
* **[Training Guide](docs/training_guide.md)**: Per-state layers, alpha schedules,
  model selection, the outcome likelihood option, and time-to-event bounds, with an
  example.
* **[PyTorch Backend](docs/torch.md)**: `pypsps.torch` (`PSPSModel`, `CausalLoss`, `fit`,
  metrics, callbacks, `io`), converting Keras models, and the differences from Keras.
* **[Datasets API](docs/datasets.md)**: `CausalDataset` base class, data
  formatting methods, and benchmark datasets (Kang-Schafer, LaLonde, etc.).
* **[Inference & Post-Processing](docs/inference.md)**: ATE estimation routines,
  prediction output splitting, and bootstrap sampling.

## Key Directory Structure

```text
pypsps/
├── _version.py              # Dynamic version resolution (via importlib.metadata)
├── _backend.py              # Lazy Keras / PyTorch dispatch for inference and bootstrap
├── bootstrap.py             # Uncertainty and bootstrap estimation (framework-free)
├── inference.py             # Causal inference and ATE estimation (framework-free)
├── utils.py                 # Shared helper functions (framework-free)
├── core/                    # Framework-free: layout, spec, alpha schedule, checkpoint, presets
├── datasets/                # Inherit from datasets.base.CausalDataset
├── keras/                   # TensorFlow backend (pypsps[tf]): layers, losses, metrics, models
│   ├── models.py            # build_model, compile_model, default_metrics
│   └── presets.py           # Named build_model argument sets (toy, binary_*, continuous_*)
├── torch/                   # PyTorch backend (pypsps[torch]): the same model, no TensorFlow
│   ├── models.py            # PSPSModel, build_model
│   ├── training.py          # fit, predict, evaluate
│   └── convert.py           # PSPSModel.from_keras
└── tests/                   # Pytest suite: core/ keras/ torch/ parity/ (each skips w/o its framework)
```

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
