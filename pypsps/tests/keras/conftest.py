"""Skips the Keras tests when TensorFlow is not installed (the ``pypsps[tf]`` extra)."""

import importlib.util


def _missing(name: str) -> bool:
    """True if the module cannot be imported."""
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


# Collect nothing from this directory unless every framework it tests is installed.
collect_ignore_glob = ["*"] if any(_missing(m) for m in ["tensorflow"]) else []
