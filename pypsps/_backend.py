"""Backend dispatch for `pypsps.inference` and `pypsps.bootstrap`.

Inference only needs a model's ``y_pred`` (a NumPy array) and each state's outcome mean, so it
is written once, in NumPy. This module is the one place that knows which framework a model
belongs to. It imports TensorFlow or PyTorch lazily: a PyTorch model is recognized without
TensorFlow installed, and every other model is treated as a Keras model.
"""

import sys
from typing import Any

import numpy as np


def is_torch_model(model: Any) -> bool:
    """True if ``model`` is a PyTorch module.

    A model can only be one if PyTorch is already imported, so this never imports it.
    """
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(model, torch.nn.Module)


def predict(model: Any, features: Any, treatment: Any) -> np.ndarray:
    """``y_pred`` of a pypsps model for ``features`` and ``treatment``, as a NumPy array."""
    if is_torch_model(model):
        from .torch import training

        return training.predict(model, features, treatment)
    return model.predict([features, treatment], verbose=0)


def state_means_and_weights(
    model: Any, features: Any, treatment: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Each state's outcome mean E[Y | a, x, s_k] and the state weights, both ``[N, K]``.

    Raises:
      NotImplementedError: for an outcome family without a mean (a `LossFamily`).
    """
    y_pred = predict(model, features, treatment)
    if is_torch_model(model):
        import torch

        from .torch import mixture as mixture_lib

        mixture = mixture_lib.StateMixture.from_model(model)
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(y_pred), dtype=torch.float32)
            means = mixture.outcome_mean(tensor, per_state=True).numpy()
        weights = np.asarray(mixture.layout.decode_pred(y_pred).state_weights)
        return means, weights
    from .keras import mixture as mixture_lib

    mixture = mixture_lib.StateMixture.from_model(model)
    means = np.asarray(mixture.outcome_mean(y_pred, per_state=True))
    weights = np.asarray(mixture.layout.decode_pred(y_pred).state_weights)
    return means, weights
