"""Skips the PyTorch tests when PyTorch or pypress.torch is not installed."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("pypress.torch")
