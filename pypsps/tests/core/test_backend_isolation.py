"""The core, inference and each backend import only the framework they need."""

import pathlib
import subprocess
import sys
import textwrap

import pytest


def _run(code: str) -> str:
    """Runs ``code`` in a fresh interpreter and returns its stdout; fails on a non-zero exit."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip()


@pytest.mark.parametrize(
    "module",
    [
        "pypsps",
        "pypsps.core",
        "pypsps.inference",
        "pypsps.bootstrap",
        "pypsps.utils",
        "pypsps.datasets",
    ],
)
def test_framework_free_modules_import_neither_framework(module):
    """Importing the core never imports TensorFlow or PyTorch."""
    out = _run(
        f"""
        import sys, {module}
        print(sorted(m for m in ("tensorflow", "torch", "keras") if m in sys.modules))
        """
    )
    assert out == "[]"


def test_torch_backend_does_not_import_tensorflow():
    """``import pypsps.torch`` works without TensorFlow: it is never imported."""
    pytest.importorskip("torch")
    out = _run(
        """
        import sys
        sys.modules["tensorflow"] = None  # make `import tensorflow` fail
        import pypsps.torch
        from pypsps.torch import models, families
        models.build_model(2, 3, families.Normal(), families.Bernoulli())
        print("ok")
        """
    )
    assert out == "ok"


def test_keras_backend_without_tensorflow_explains_the_extra():
    """``import pypsps.keras`` without TensorFlow names the ``pypsps[tf]`` extra."""
    result = subprocess.run(
        [sys.executable, "-c", 'import sys; sys.modules["tensorflow"] = None; import pypsps.keras'],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "pypsps[tf]" in result.stderr


def test_torch_backend_without_torch_explains_the_extra():
    """``import pypsps.torch`` without PyTorch names the ``pypsps[torch]`` extra."""
    result = subprocess.run(
        [sys.executable, "-c", 'import sys; sys.modules["torch"] = None; import pypsps.torch'],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "pypsps[torch]" in result.stderr


def test_datasets_work_without_scipy():
    """``pypsps.datasets`` needs only declared dependencies: scipy is not one of them."""
    out = _run(
        """
        import sys
        sys.modules["scipy"] = None  # make `import scipy` fail, as on a clean install
        from pypsps import datasets
        df = datasets.CancerSurvivalSimulator(seed=1).sample(n_samples=50).to_data_frame()
        print(len(df))
        """
    )
    assert out == "50"


def test_core_suite_runs_without_any_framework():
    """Every other test in ``tests/core`` passes with TensorFlow and PyTorch unimportable.

    This keeps the core suite framework-free, so a test that needs a framework belongs in
    ``tests/keras``, ``tests/torch`` or ``tests/parity``.
    """
    # A finder rather than ``sys.modules[name] = None``: libraries such as scipy probe for a
    # framework by name, and a None entry breaks them in a way a missing package does not.
    code = """
        import sys

        class Missing:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("tensorflow", "keras", "torch"):
                    raise ModuleNotFoundError(f"No module named {name!r}")

        sys.meta_path.insert(0, Missing())
        import pytest
        sys.exit(pytest.main([
            "pypsps/tests/core", "-k", "not backend_isolation", "-q", "-p", "no:cacheprovider",
        ]))
    """
    root = pathlib.Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
