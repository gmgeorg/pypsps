"""Copy a trained `pypsps.keras` model into a `pypsps.torch.models.PSPSModel`.

The build arguments are read from the `pypsps.core.spec.Spec` the Keras model carries on its
`CausalOutput` layer (v0.6.3 and later). A model saved before that has no Spec; its build
arguments are then inferred from the layer names and configs, which
`pypsps.keras.models.build_model` fixes. Every weight is copied by layer name:

* dense kernels are transposed (Keras ``[in, out]``, torch ``[out, in]``);
* batch normalization maps ``gamma, beta, moving_mean, moving_variance`` to
  ``weight, bias, running_mean, running_var``;
* the state-weights layer, the treatment head and the constant (``BiasOnly``) heads map to
  their pypress / pypsps torch counterparts.

In eval mode the copy's ``forward`` equals the Keras ``predict`` up to float32 rounding.
"""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..core.spec import HeadSpec, Spec
from . import families as torch_families
from . import mixture as torch_mixture
from . import models


def _layers_by_name(keras_model: Any) -> Dict[str, Any]:
    """The Keras model's layers keyed by name."""
    return {layer.name: layer for layer in keras_model.layers}


def _hidden(by_name: Dict[str, Any], prefix: str) -> List[Tuple[int, str]]:
    """``(units, activation)`` of the layers ``{prefix}_dense_0, {prefix}_dense_1, ...``."""
    found = []
    i = 0
    while f"{prefix}_dense_{i}" in by_name:
        layer = by_name[f"{prefix}_dense_{i}"]
        found.append((int(layer.units), layer.get_config()["activation"]))
        i += 1
    return found


def _mixture(keras_model: Any) -> torch_mixture.StateMixture:
    """The torch counterpart of the Keras model's state mixture."""
    from ..keras import mixture as keras_mixture

    source = keras_mixture.StateMixture.from_model(keras_model)
    configs = {}
    for role, family in (
        ("outcome_family", source.outcome_family),
        ("treatment_family", source.treatment_family),
    ):
        name = type(family).__name__
        if name == "LossFamily":
            raise NotImplementedError(
                "a model with a LossFamily wraps a callable and has no PyTorch counterpart; "
                "rebuild it with pypsps.torch.models.build_model and a torch LossFamily"
            )
        configs[role] = torch_families.from_config(name, family.get_config())
    return torch_mixture.StateMixture(n_states=source.n_states, **configs)


def _head_specs(
    by_name: Dict[str, Any], mixture: torch_mixture.StateMixture
) -> Dict[str, HeadSpec]:
    """The `HeadSpec` of each outcome parameter, from the head layer names."""
    specs = {}
    for name in mixture.outcome_family.param_names:
        per_state = f"{name}_raw_state_0" in by_name
        raw = by_name.get(f"{name}_raw_state_0" if per_state else f"{name}_raw_shared")
        if raw is None:
            raise ValueError(f"not a pypsps model: no output head for parameter {name!r}")
        suffix = "state_0" if per_state else "shared"
        hidden = by_name.get(f"{name}_hidden_{suffix}")
        specs[name] = HeadSpec(
            hidden_layer=(
                None if hidden is None else (int(hidden.units), hidden.get_config()["activation"])
            ),
            constant=type(raw).__name__ == "BiasOnly",
            per_state=per_state,
        )
    return specs


def infer_build_args(keras_model: Any) -> Dict[str, Any]:
    """The `pypsps.torch.models.build_model` arguments that rebuild ``keras_model``."""
    by_name = _layers_by_name(keras_model)
    for required in ("features_bn", "state_weights", "treatment_params"):
        if required not in by_name:
            raise ValueError(
                f"not a pypsps.keras.models.build_model model: it has no layer {required!r}"
            )
    mixture = _mixture(keras_model)
    state_layers = _hidden(by_name, "predictive_state")
    shared = (
        "predictive_state_skip" in by_name
        or "outcome_trunk_skip" in by_name
        or "outcome_trunk_dense_0" in by_name
    )
    heads = _head_specs(by_name, mixture)
    # Without any hidden layer the two architectures differ only in what a per-state head sees:
    # the gated one builds a gate layer for it, the shared trunk does not.
    shared = shared or any(
        h.per_state and not h.constant and f"{name}_gate_state_0" not in by_name
        for name, h in heads.items()
    )
    # a constant head has no trunk in the gated architecture, so read the trunk of a learned one
    first = next(
        (name for name, h in heads.items() if not h.constant), mixture.outcome_family.param_names[0]
    )
    outcome_layers = _hidden(by_name, "outcome_trunk" if shared else f"{first}_trunk")
    dropout = [by_name[n].rate for n in by_name if re.fullmatch(r".*_dropout_\d+", n)]
    first_dense = by_name.get("predictive_state_dense_0") or by_name.get(
        "outcome_trunk_dense_0" if shared else f"{first}_trunk_dense_0"
    )
    l1 = 0.0
    if first_dense is not None and first_dense.kernel_regularizer is not None:
        l1 = float(first_dense.kernel_regularizer.get_config().get("l1", 0.0))
    embedding = by_name.get("treatment_embedding")
    return {
        "n_states": mixture.n_states,
        "n_features": int(by_name["features_bn"].input.shape[-1]),
        "outcome_family": mixture.outcome_family,
        "treatment_family": mixture.treatment_family,
        "architecture": "shared_trunk" if shared else "gated",
        "predictive_state_hidden_layers": state_layers,
        "outcome_hidden_layers": outcome_layers,
        "outcome_heads": heads,
        "treatment_embedding_units": None if embedding is None else int(embedding.units),
        "dropout_rate": float(dropout[0]) if dropout else 0.0,
        "l1": l1,
    }


def _copy_dense(layer: Any, module: torch.nn.Linear) -> None:
    """Keras Dense ``[in, out]`` kernel and bias into a torch Linear."""
    kernel, bias = layer.get_weights()
    module.weight.data.copy_(torch.as_tensor(np.ascontiguousarray(kernel.T)))
    module.bias.data.copy_(torch.as_tensor(bias))


def _copy_bn(layer: Any, module: torch.nn.BatchNorm1d) -> None:
    """Keras BatchNormalization into a torch BatchNorm1d."""
    gamma, beta, mean, var = layer.get_weights()
    module.weight.data.copy_(torch.as_tensor(gamma))
    module.bias.data.copy_(torch.as_tensor(beta))
    module.running_mean.copy_(torch.as_tensor(mean))
    module.running_var.copy_(torch.as_tensor(var))


def _carried_spec_config(keras_model: Any) -> Dict[str, Any] | None:
    """The `Spec.get_config` stored on the model's output layer, or None (older models)."""
    for layer in keras_model.layers:
        if hasattr(layer, "spec_config"):
            return layer.spec_config
    return None


def copy_weights(keras_model: Any, model: "models.PSPSModel") -> None:
    """Copies every weight of ``keras_model`` into ``model`` by layer name.

    ``model`` must have the architecture of ``keras_model`` (as `from_keras` builds it, or
    `pypsps.torch.models.build_model` with the same arguments).
    """
    for layer in keras_model.layers:
        name, kind = layer.name, type(layer).__name__
        weights = layer.get_weights()
        if not weights:
            continue
        if kind == "BatchNormalization":
            _copy_bn(layer, getattr(model, name))
        elif kind == "Dense":
            _copy_dense(layer, getattr(model, name))
        elif kind == "BiasOnly":
            getattr(model, name).constant.data.copy_(torch.as_tensor(weights[0]).reshape(-1))
        elif kind == "PredictiveStateSimplex":
            _copy_dense(layer, model.state_weights.linear)
        elif kind == "PredictiveStateParams":
            logits = torch.as_tensor(weights[0])
            model.treatment_params.logits.data.copy_(
                logits.reshape(model.treatment_params.logits.shape)
            )
        else:
            raise ValueError(f"cannot copy the weights of layer {name!r} ({kind})")


def from_keras(keras_model: Any) -> "models.PSPSModel":
    """A `PSPSModel` with the architecture and weights of ``keras_model``, in eval mode.

    The architecture is the model's carried `Spec`; a model without one (saved before v0.6.3)
    is read from its layer names (`infer_build_args`).

    Raises:
      NotImplementedError: for a model with a `LossFamily`.
      ValueError: before anything is built, if a hidden layer uses an activation that is not
        in `pypsps.core.spec.HIDDEN_ACTIVATIONS`.
    """
    config = _carried_spec_config(keras_model)
    if config is None:
        model = models.build_model(**infer_build_args(keras_model))
    else:
        model = models.PSPSModel(Spec.from_config(config, _mixture(keras_model)))
    copy_weights(keras_model, model)
    return model.eval()
