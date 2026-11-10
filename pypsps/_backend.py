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


WEIGHTS = ("prior", "posterior")


def check_weights(weights: str) -> str:
    """Returns ``weights`` if it is ``"prior"`` or ``"posterior"``; raises ValueError otherwise."""
    if weights not in WEIGHTS:
        raise ValueError(f"weights must be one of {WEIGHTS}, got {weights!r}")
    return weights


def _treatment_column(treatment: Any) -> np.ndarray:
    """``treatment`` as a float32 ``[N, 1]`` array."""
    return np.asarray(treatment, dtype=np.float32).reshape(-1, 1)


def state_means_and_weights(
    model: Any, features: Any, treatment: Any, weights: str = "prior"
) -> tuple[np.ndarray, np.ndarray]:
    """Each state's outcome mean E[Y | a, x, s_k] and the state weights, both ``[N, K]``.

    Args:
      weights: ``"prior"`` returns the model's state weights P(s_k | x), which do not depend on
        the treatment. ``"posterior"`` returns P(s_k | x, a) for the given treatment, by Bayes'
        rule from the per-state treatment density (the weights the observational outcome
        likelihood mixes with).

    Raises:
      NotImplementedError: for an outcome family without a mean (a `LossFamily`).
      ValueError: if ``weights`` is not ``"prior"`` or ``"posterior"``.
    """
    check_weights(weights)
    y_pred = predict(model, features, treatment)
    if is_torch_model(model):
        import torch

        from .torch import mixture as mixture_lib

        mixture = mixture_lib.StateMixture.from_model(model)
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(y_pred), dtype=torch.float32)
            means = mixture.outcome_mean(tensor, per_state=True).numpy()
            if weights == "posterior":
                y_true = _posterior_y_true(mixture, treatment)
                state_weights = mixture.state_posterior(torch.as_tensor(y_true), tensor).numpy()
        if weights == "prior":
            state_weights = np.asarray(mixture.layout.decode_pred(y_pred).state_weights)
        return means, state_weights
    from .keras import mixture as mixture_lib

    mixture = mixture_lib.StateMixture.from_model(model)
    means = np.asarray(mixture.outcome_mean(y_pred, per_state=True))
    if weights == "posterior":
        state_weights = np.asarray(
            mixture.state_posterior(_posterior_y_true(mixture, treatment), y_pred)
        )
    else:
        state_weights = np.asarray(mixture.layout.decode_pred(y_pred).state_weights)
    return means, state_weights


def _posterior_y_true(mixture: Any, treatment: Any) -> np.ndarray:
    """A ``y_true`` for `state_posterior`: zero outcome columns, then the treatment column.

    The posterior reads only the treatment column.
    """
    a = _treatment_column(treatment)
    zeros = np.zeros((a.shape[0], int(mixture.layout.n_outcome_true_cols)), dtype=np.float32)
    return np.concatenate([zeros, a], axis=1)
