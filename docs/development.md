# Development & Contribution Guide

## Version Management

The single source of truth for package versioning is `pyproject.toml`.

- `pypsps/_version.py` programmatically reads the version using
  `importlib.metadata.version("pypsps")`.
- **Never** hardcode version strings in Python files.

## Environment & Dependencies

See `pyproject.toml`.

## Git Workflow

- Main branch: `main`
- Always create a new feature/fix branch and submit a PR. **Do not push directly
  to `main`**.
- Commit messages must adhere to Conventional Commits (`feat:`, `fix:`,
  `chore:`, etc.).

## Code Style & Quality

- **Line length**: 100 characters (enforced by `ruff`).
- **Formatting**: Auto-fix enabled in `ruff`.
- **Standards**: PEP 8 compliance; type hints are recommended.

## Testing Strategy

- Tests reside in `pypsps/tests/`. Run via `pytest`. They are split by backend:
  `core/` (no framework), `keras/` (needs TensorFlow), `torch/` (needs PyTorch) and
  `parity/` (needs both, compares the two backends). A directory is skipped when its
  framework is not installed, so `pip install -e ".[torch]"` runs everything but the Keras
  and parity tests. CI runs the `tf`, `torch` and `tf,torch` installs.
- A test of framework-free behaviour (the layout, family specs, `make_spec`, preset
  arguments, datasets, `inference`, `bootstrap`) belongs in `tests/core`, so every CI job runs
  it. It must run on NumPy alone: `test_core_suite_runs_without_any_framework` runs the
  directory with TensorFlow and PyTorch unimportable. Assertions about tensors, Keras
  serialization or tensor math stay in the backend directory.
- Install CPU-only PyTorch with
  `pip install -e ".[torch]" --extra-index-url https://download.pytorch.org/whl/cpu`.
- Ensure new features have accompanying unit tests for models, datasets, losses,
  metrics, or inference.
