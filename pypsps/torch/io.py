"""Saving and loading a PyTorch pypsps model: `save` and `load`.

The file holds the model's build arguments (`PSPSModel.get_config`, plain Python values) and
its ``state_dict``, so `load` needs no pickled code and reads the file with
``torch.load(weights_only=True)``. The format is PyTorch's own and is not the Keras
``.keras`` format; to move a Keras model to PyTorch use `PSPSModel.from_keras`.
"""

import os
from typing import Union

import torch

from . import models

_FORMAT = "pypsps-torch"
_VERSION = 1


def save(model: models.PSPSModel, path: Union[str, "os.PathLike[str]"]) -> None:
    """Writes ``model``'s configuration and weights (including batch-norm statistics) to ``path``.

    Raises:
      TypeError: if ``model`` is not a `PSPSModel`.
      NotImplementedError: if the model has a `LossFamily` (a callable cannot be saved).
    """
    if not isinstance(model, models.PSPSModel):
        raise TypeError(f"save takes a PSPSModel, got {type(model).__name__}")
    torch.save(
        {
            "format": _FORMAT,
            "version": _VERSION,
            "config": model.get_config(),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load(
    path: Union[str, "os.PathLike[str]"], map_location: Union[str, torch.device] = "cpu"
) -> models.PSPSModel:
    """Rebuilds the model saved at ``path`` with its weights, in eval mode.

    Raises:
      ValueError: if the file was not written by `save`, or by a newer format version.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
        raise ValueError(f"{os.fspath(path)!r} is not a pypsps PyTorch model file")
    if payload.get("version", 0) > _VERSION:
        raise ValueError(
            f"{os.fspath(path)!r} has format version {payload['version']}; this pypsps reads "
            f"up to version {_VERSION}. Upgrade pypsps."
        )
    model = models.PSPSModel.from_config(payload["config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(map_location).eval()
