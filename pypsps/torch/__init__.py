"""pypsps on PyTorch: the PSPS model as an ``nn.Module`` with a plain callable loss.

Needs PyTorch and pypress, which are optional dependencies of pypsps. Install them with
``pip install "pypsps[torch] @ git+https://github.com/gmgeorg/pypsps.git"``; TensorFlow is not
needed.

    from pypsps.torch import families, models, training

    model = models.build_model(3, n_features, families.Normal(), families.Bernoulli())
    history = training.fit(model, features, treatment, outcome, epochs=50)
    y_pred = training.predict(model, features, treatment)

Modules: `families`, `mixture` (the likelihoods), `losses` (`CausalLoss`), `layers`, `models`
(`build_model`, `PSPSModel`), `presets`, `metrics`, `callbacks`, `training` (`fit`, `predict`)
and `io` (`save`, `load`). It mirrors `pypsps.keras`: the same names, the same ``y_pred``
column layout and the same history keys.
"""

try:
    import torch  # noqa: F401
except ImportError as error:  # pragma: no cover - exercised in a subprocess
    raise ImportError(
        "pypsps.torch needs PyTorch, which is not installed. Install it with: "
        "pip install 'pypsps[torch] @ git+https://github.com/gmgeorg/pypsps.git'"
    ) from error
try:
    import pypress.torch  # noqa: F401
except ImportError as error:  # pragma: no cover - exercised in a subprocess
    raise ImportError(
        "pypsps.torch needs pypress with its PyTorch backend. Install it with: "
        "pip install 'pypsps[torch] @ git+https://github.com/gmgeorg/pypsps.git'"
    ) from error

from . import callbacks, families, io, layers, losses, metrics, mixture, models, presets, training

__all__ = [
    "callbacks",
    "families",
    "io",
    "layers",
    "losses",
    "metrics",
    "mixture",
    "models",
    "presets",
    "training",
]
