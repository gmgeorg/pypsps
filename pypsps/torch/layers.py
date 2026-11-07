"""Layers of the PyTorch backend: the same building blocks as `pypsps.keras.layers`.

The numeric conventions match Keras, so a model converts one-to-one
(`pypsps.torch.models.PSPSModel.from_keras`): dense kernels start Glorot-uniform with zero
bias, and batch normalization uses Keras's ``momentum=0.99`` and ``epsilon=1e-3`` (which is
``momentum=0.01`` in PyTorch's complementary convention).
"""

from typing import Optional

import numpy as np
import torch
from torch import nn

# Keras BatchNormalization(momentum=0.99, epsilon=1e-3); PyTorch's momentum is 1 - Keras's.
BN_MOMENTUM = 0.01
BN_EPS = 1e-3

_ACTIVATIONS = {
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
}


def activation(name: str) -> nn.Module:
    """The activation module called ``name``.

    One of ``linear``, ``relu``, ``selu``, ``elu``, ``gelu``, ``tanh``, ``swish`` (``silu``),
    ``softplus`` and ``sigmoid``.

    Raises:
      ValueError: for any other name.
    """
    if not isinstance(name, str) or name not in _ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; known: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


class Dense(nn.Linear):
    """``nn.Linear`` with Keras's initialization and an optional activation.

    The kernel is Glorot-uniform and the bias zero (PyTorch's default is Kaiming-uniform).
    Parameters are ``weight`` and ``bias`` as in ``nn.Linear``, so a state dict names this layer
    like the Keras ``Dense`` layer of the same name.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_name: str = "linear",
        bias_init: float = 0.0,
    ):
        """Creates the layer; ``bias_init`` is the constant the bias starts at."""
        super().__init__(in_features, out_features)
        self.activation_name = activation_name
        self.activation = activation(activation_name)
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, float(bias_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``activation(x W^T + b)``."""
        return self.activation(super().forward(x))


def batch_norm(num_features: int) -> nn.BatchNorm1d:
    """``nn.BatchNorm1d`` with Keras's momentum and epsilon."""
    return nn.BatchNorm1d(num_features, eps=BN_EPS, momentum=BN_MOMENTUM)


class BiasOnly(nn.Module):
    """Bias-only layer (intercept only model): a trainable constant per output.

    ``BiasOnly(units)(x)`` is the ``[N, units]`` constant, whatever ``x`` is.
    """

    def __init__(self, units: int = 1, init: float = 0.0):
        """Creates the layer; the constant starts at ``init``."""
        super().__init__()
        self.constant = nn.Parameter(torch.full((int(units),), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The constant, repeated for every row of ``x``."""
        return self.constant.expand(x.shape[0], -1)


class SoftClip(nn.Module):
    """Maps inputs smoothly into ``(min_value, max_value)``: ``min + (max - min) * sigmoid(x)``.

    Unlike a hard clip, the gradient does not drop to zero at the bounds, so a parameter near a
    bound can still move back.
    """

    def __init__(self, min_value: float, max_value: float):
        """Stores the bounds; requires finite ``min_value < max_value``."""
        super().__init__()
        min_value, max_value = float(min_value), float(max_value)
        if not (np.isfinite(min_value) and np.isfinite(max_value) and min_value < max_value):
            raise ValueError(
                f"need finite min_value < max_value, got ({min_value!r}, {max_value!r})"
            )
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``min_value + (max_value - min_value) * sigmoid(x)``."""
        return self.min_value + (self.max_value - self.min_value) * torch.sigmoid(x)

    def inverse(self, value: float) -> float:
        """The input that maps to ``value``, which must lie strictly inside the bounds."""
        if not self.min_value < value < self.max_value:
            raise ValueError(
                f"value must lie in ({self.min_value}, {self.max_value}), got {value!r}"
            )
        frac = (value - self.min_value) / (self.max_value - self.min_value)
        return float(np.log(frac / (1.0 - frac)))

    def extra_repr(self) -> str:
        """The bounds, for ``print(module)``."""
        return f"min_value={self.min_value}, max_value={self.max_value}"


def l1_penalty(modules: "list[nn.Module]", l1: float) -> Optional[torch.Tensor]:
    """``l1 * sum |kernel|`` over the kernels of ``modules``; None for no modules or ``l1 = 0``."""
    if l1 <= 0 or not modules:
        return None
    return l1 * sum(m.weight.abs().sum() for m in modules)
