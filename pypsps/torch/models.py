"""Model builders for pypsps on PyTorch: `build_model` and `PSPSModel`.

`build_model` takes the arguments of `pypsps.keras.models.build_model` minus the compile
keywords (those go to `pypsps.torch.training.fit`) and validates them through the same shared
code. It returns a `PSPSModel`, an ``nn.Module`` whose ``forward(features, treatment)`` gives
the flat ``y_pred`` of `pypsps.core.layout.ColumnLayout`: ``[outcome params | state weights |
treatment params]``.

The submodule names mirror the Keras layer names (``features_bn``, ``treatment_embedding``,
``predictive_state_dense_0``, ``state_weights``, ``treatment_params``, ``outcome_trunk_dense_0``
or ``{param}_trunk_dense_0``, ``{param}_hidden_state_0``, ``{param}_raw_state_0``, ...), so
`PSPSModel.from_keras` copies a trained Keras model by name.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pypress.torch import layers as pr_layers
from torch import nn

from ..core import spec as core_spec
from ..core.spec import HeadSpec
from . import families, layers
from . import mixture as mixture_lib

Architecture = core_spec.Architecture
HiddenLayers = core_spec.HiddenLayers


def _make_spec(*args: Any, **kwargs: Any) -> core_spec.Spec:
    """`pypsps.core.spec.make_spec` with the PyTorch state mixture."""
    return core_spec.make_spec(mixture_lib.StateMixture, *args, **kwargs)


def _add_stack(
    model: nn.Module,
    in_features: int,
    hidden_layers: Tuple[Tuple[int, str], ...],
    dropout_rate: float,
    name: str,
) -> Tuple[int, List[nn.Module]]:
    """Registers a Dense -> Dropout -> BN stack on ``model``; returns (width, modules in order)."""
    modules: List[nn.Module] = []
    width = in_features
    for i, (units, act) in enumerate(hidden_layers):
        dense = layers.Dense(width, units, act)
        model.add_module(f"{name}_dense_{i}", dense)
        dropout = nn.Dropout(dropout_rate)
        model.add_module(f"{name}_dropout_{i}", dropout)
        bn = layers.batch_norm(units)
        model.add_module(f"{name}_bn_{i}", bn)
        modules += [dense, dropout, bn]
        width = units
    return width, modules


def _run(modules: List[nn.Module], x: torch.Tensor) -> torch.Tensor:
    """Applies ``modules`` in order."""
    for module in modules:
        x = module(x)
    return x


class PSPSModel(nn.Module):
    """A PSPS network: ``forward(features, treatment) -> y_pred``.

    The network has four parts: a state network from the features to the state weights
    (``state_weights``); a treatment head with one feature-independent constant per state and
    treatment parameter (``treatment_params``); one head per outcome parameter and state, built
    as ``outcome_heads`` says; and the `StateMixture` (``mixture``) that interprets ``y_pred``.

    Build one with `build_model`. ``forward`` takes ``features`` ``[N, n_features]`` and
    ``treatment`` ``[N]`` or ``[N, 1]`` and returns ``y_pred`` ``[N, layout.y_pred_width()]``.
    ``train()`` / ``eval()`` switch dropout and batch normalization as usual.

    Attributes:
      mixture: the `pypsps.torch.mixture.StateMixture`.
      layout: its `pypsps.core.layout.ColumnLayout`.
      spec: the validated `pypsps.core.spec.Spec` the model was built from.
    """

    def __init__(self, spec: core_spec.Spec):
        """Builds the layers of ``spec`` (use `build_model` to validate arguments)."""
        super().__init__()
        self.spec = spec
        self.mixture = spec.mixture
        self.layout = spec.mixture.layout
        n_states = spec.n_states
        n_features = spec.n_features

        self.features_bn = layers.batch_norm(n_features)
        if spec.treatment_embedding_units is None:
            self._t_width = 1
            self._embed = False
        else:
            self._embed = True
            self.treatment_bn = layers.batch_norm(1)
            self.treatment_embedding = layers.Dense(1, spec.treatment_embedding_units, "swish")
            self._t_width = spec.treatment_embedding_units
        ft_width = n_features + self._t_width  # features_and_treatment

        state_width, self._state_stack = _add_stack(
            self, n_features, spec.predictive_state_hidden_layers, spec.dropout_rate,
            "predictive_state",
        )  # fmt: skip
        self._state_skip = spec.architecture == "shared_trunk" and bool(
            spec.predictive_state_hidden_layers
        )
        if self._state_skip:
            state_width += n_features
        self.state_weights = pr_layers.PredictiveStateSimplex(
            n_states=n_states, in_features=state_width
        )
        treatment_params = spec.mixture.treatment_family.params
        self.treatment_params = pr_layers.PredictiveStateParams(
            n_states=n_states,
            n_params_per_state=len(treatment_params),
            activations=[
                p.activation if p.bounds is None else layers.SoftClip(*p.bounds)
                for p in treatment_params
            ],
            init_values=spec.treatment_init,
            flatten_output=True,  # param-major, as in ColumnLayout
        )

        # name -> list of head module names, one per state
        self._heads: Dict[str, List[Tuple[str, Optional[str]]]] = {}
        self._trunks: Dict[str, List[nn.Module]] = {}
        self._l1_modules: List[nn.Module] = list(self._state_stack_dense())
        if spec.architecture == "shared_trunk":
            self._build_shared_trunk(ft_width)
        else:
            self._build_gated(ft_width)

    # --- construction ---

    def _state_stack_dense(self) -> List[nn.Module]:
        """The Dense layers of the state network."""
        return [m for m in self._state_stack if isinstance(m, layers.Dense)]

    def _add_head(
        self, p: Any, head: HeadSpec, head_width: int, ft_width: int, suffix: str
    ) -> Tuple[str, Optional[str]]:
        """Registers one outcome parameter's head; returns (raw name, hidden name or None)."""
        init = 0.0 if p.init is None else p.raw_value(p.init)
        raw_name = f"{p.name}_raw_{suffix}"
        hidden_name = None
        if head.constant:
            self.add_module(raw_name, layers.BiasOnly(init=init))
        else:
            width = head_width
            if head.hidden_layer is not None:
                units, act = head.hidden_layer
                hidden_name = f"{p.name}_hidden_{suffix}"
                self.add_module(hidden_name, layers.Dense(width, units, act))
                width = units
            self.add_module(raw_name, layers.Dense(width, 1, bias_init=init))
        activation = families.head_activation(p)
        self.add_module(f"{p.name}_{suffix}", activation)
        return raw_name, hidden_name

    def _build_shared_trunk(self, ft_width: int) -> None:
        """One outcome trunk with a skip connection; heads are built state by state."""
        spec = self.spec
        trunk_width, stack = _add_stack(
            self, ft_width, spec.outcome_hidden_layers, spec.dropout_rate, "outcome_trunk"
        )
        self._trunks["outcome_trunk"] = stack
        self._l1_modules += [m for m in stack if isinstance(m, layers.Dense)]
        self._outcome_skip = bool(spec.outcome_hidden_layers)
        if self._outcome_skip:
            trunk_width += ft_width
        shared: Dict[str, Tuple[str, Optional[str]]] = {}
        for k in range(spec.n_states):
            for p in spec.outcome_params:
                head = spec.heads[p.name]
                entry_list = self._heads.setdefault(p.name, [])
                if head.per_state:
                    entry_list.append(
                        self._add_head(p, head, trunk_width, ft_width, f"state_{k}")
                        + (f"state_{k}",)
                    )
                    continue
                if p.name not in shared:
                    shared[p.name] = self._add_head(p, head, trunk_width, ft_width, "shared")
                entry_list.append(shared[p.name] + ("shared",))

    def _build_gated(self, ft_width: int) -> None:
        """One trunk per outcome parameter; each state's head sees the treatment gated by its
        state weight."""
        spec = self.spec
        for p in spec.outcome_params:
            head = spec.heads[p.name]
            entry_list = self._heads.setdefault(p.name, [])
            if head.constant:
                if head.per_state:
                    for k in range(spec.n_states):
                        entry_list.append(
                            self._add_head(p, head, 0, ft_width, f"state_{k}") + (f"state_{k}",)
                        )
                else:
                    entry = self._add_head(p, head, 0, ft_width, "shared") + ("shared",)
                    entry_list += [entry] * spec.n_states
                continue
            trunk_width, stack = _add_stack(
                self, ft_width, spec.outcome_hidden_layers, spec.dropout_rate, f"{p.name}_trunk"
            )
            self._trunks[p.name] = stack
            self._l1_modules += [m for m in stack if isinstance(m, layers.Dense)]
            if not head.per_state:
                entry = self._add_head(p, head, trunk_width, ft_width, "shared") + ("shared",)
                entry_list += [entry] * spec.n_states
                continue
            for k in range(spec.n_states):
                entry_list.append(
                    self._add_head(p, head, trunk_width + self._t_width, ft_width, f"state_{k}")
                    + (f"state_{k}",)
                )

    # --- forward ---

    def _head_value(
        self,
        p_name: str,
        entry: Tuple[str, Optional[str], str],
        head_input: Optional[torch.Tensor],
        features_and_treatment: torch.Tensor,
    ) -> torch.Tensor:
        """One outcome parameter for one state (or shared), on the parameter's support."""
        raw_name, hidden_name, suffix = entry
        raw_module = getattr(self, raw_name)
        if isinstance(raw_module, layers.BiasOnly):
            raw = raw_module(features_and_treatment)
        else:
            x = head_input
            if hidden_name is not None:
                x = getattr(self, hidden_name)(x)
            raw = raw_module(x)
        return getattr(self, f"{p_name}_{suffix}")(raw)

    def forward(self, features: torch.Tensor, treatment: torch.Tensor) -> torch.Tensor:
        """``y_pred`` ``[N, layout width]`` for ``features`` ``[N, F]`` and ``treatment``."""
        spec = self.spec
        features = features.to(torch.float32)
        treatment = treatment.to(torch.float32).reshape(features.shape[0], 1)
        features_bn = self.features_bn(features)
        if self._embed:
            treatment_repr = self.treatment_embedding(self.treatment_bn(treatment))
        else:
            treatment_repr = treatment
        features_and_treatment = torch.cat([features_bn, treatment_repr], dim=1)

        state_hidden = _run(self._state_stack, features_bn)
        if self._state_skip:
            state_hidden = torch.cat([state_hidden, features_bn], dim=1)
        state_weights = self.state_weights(state_hidden)
        treatment_block = self.treatment_params(state_weights)

        columns: Dict[str, List[torch.Tensor]] = {p.name: [] for p in spec.outcome_params}
        if spec.architecture == "shared_trunk":
            trunk = _run(self._trunks["outcome_trunk"], features_and_treatment)
            if self._outcome_skip:
                trunk = torch.cat([trunk, features_and_treatment], dim=1)
            values: Dict[Tuple[str, str], torch.Tensor] = {}
            for p in spec.outcome_params:
                for entry in self._heads[p.name]:
                    key = (p.name, entry[0])
                    if key not in values:
                        values[key] = self._head_value(p.name, entry, trunk, features_and_treatment)
                    columns[p.name].append(values[key])
        else:
            for p in spec.outcome_params:
                head = spec.heads[p.name]
                entries = self._heads[p.name]
                if head.constant:
                    values_p = {}
                    for entry in entries:
                        if entry[0] not in values_p:
                            values_p[entry[0]] = self._head_value(
                                p.name, entry, None, features_and_treatment
                            )
                        columns[p.name].append(values_p[entry[0]])
                    continue
                trunk = _run(self._trunks[p.name], features_and_treatment)
                if not head.per_state:
                    value = self._head_value(p.name, entries[0], trunk, features_and_treatment)
                    columns[p.name] += [value] * spec.n_states
                    continue
                for k, entry in enumerate(entries):
                    gate = state_weights[:, k : k + 1]
                    head_input = torch.cat([trunk, treatment_repr * gate], dim=1)
                    columns[p.name].append(
                        self._head_value(p.name, entry, head_input, features_and_treatment)
                    )
        outcome = {name: torch.cat(values, dim=1) for name, values in columns.items()}
        return self.layout.encode(outcome, state_weights, treatment_block)

    # --- regularization and helpers ---

    def kernel_l1_penalty(self) -> torch.Tensor:
        """``l1 * sum |kernel|`` over the hidden-layer kernels (the Keras ``l1`` argument)."""
        penalty = layers.l1_penalty(self._l1_modules, self.spec.l1)
        if penalty is None:
            return torch.zeros((), device=self.features_bn.weight.device)
        return penalty

    def propensity_state_conditional_means(self) -> torch.Tensor:
        """Each predictive state's mean treatment, shape ``(n_states, 1)``.

        For a binary treatment this is the propensity score of each state; for a Normal
        treatment, each state's ``loc``. The treatment block of ``y_pred`` does not depend on
        the features, so this reads it off the treatment head directly.
        """
        table = self.treatment_params.state_conditional_params  # [K, P]
        names = self.layout.treatment_params
        params = {n: table[:, i][None, :] for i, n in enumerate(names)}
        return self.mixture.treatment_family.mean(params).T

    # --- configuration ---

    def get_config(self) -> dict:
        """The build arguments as JSON-friendly values; `from_config` inverts it."""
        spec = self.spec
        return {
            "n_features": spec.n_features,
            "mixture": spec.mixture.get_config(),
            "architecture": spec.architecture,
            "predictive_state_hidden_layers": [
                list(h) for h in spec.predictive_state_hidden_layers
            ],
            "outcome_hidden_layers": [list(h) for h in spec.outcome_hidden_layers],
            "outcome_heads": {
                name: {
                    "hidden_layer": None if h.hidden_layer is None else list(h.hidden_layer),
                    "constant": h.constant,
                    "per_state": h.per_state,
                }
                for name, h in spec.heads.items()
            },
            "treatment_embedding_units": spec.treatment_embedding_units,
            "dropout_rate": spec.dropout_rate,
            "l1": spec.l1,
            "outcome_init": {p.name: p.init for p in spec.outcome_params if p.init is not None},
            "treatment_init": (
                None if spec.treatment_init is None else np.asarray(spec.treatment_init).tolist()
            ),
        }

    @classmethod
    def from_config(cls, config: dict) -> "PSPSModel":
        """Rebuilds an untrained model (fresh weights) from `get_config`."""
        mixture = mixture_lib.StateMixture.from_config(config["mixture"])
        treatment_init = config.get("treatment_init")
        treatment_family = mixture.treatment_family
        return build_model(
            mixture.n_states,
            config["n_features"],
            mixture.outcome_family,
            treatment_family,
            architecture=config["architecture"],
            predictive_state_hidden_layers=[
                tuple(h) for h in config["predictive_state_hidden_layers"]
            ],
            outcome_hidden_layers=[tuple(h) for h in config["outcome_hidden_layers"]],
            outcome_heads={
                name: HeadSpec(
                    hidden_layer=None if h["hidden_layer"] is None else tuple(h["hidden_layer"]),
                    constant=h["constant"],
                    per_state=h["per_state"],
                )
                for name, h in config["outcome_heads"].items()
            },
            treatment_embedding_units=config["treatment_embedding_units"],
            dropout_rate=config["dropout_rate"],
            l1=config["l1"],
            outcome_init=config.get("outcome_init") or None,
            treatment_init=(
                None
                if treatment_init is None
                else dict(zip(treatment_family.param_names, treatment_init))
            ),
        )

    @classmethod
    def from_keras(cls, keras_model: Any) -> "PSPSModel":
        """A PyTorch copy of a (trained) `pypsps.keras.models.build_model` model.

        The build arguments are read off the Keras model (its output layer's mixture and its
        layer names and configs) and every weight is copied by layer name, with dense kernels
        transposed and batch-normalization statistics mapped. In eval mode the copy's
        ``forward`` equals the Keras ``predict`` up to float32 rounding.

        Raises:
          NotImplementedError: for a model with a `LossFamily` (a callable has no PyTorch
            counterpart).
          ValueError: if the model is not one `pypsps.keras.models.build_model` makes.
        """
        from . import convert  # lazy: needs TensorFlow only when converting

        return convert.from_keras(keras_model)

    def extra_repr(self) -> str:
        """Architecture and sizes, for ``print(model)``."""
        spec = self.spec
        return (
            f"architecture={spec.architecture!r}, n_states={spec.n_states}, "
            f"n_features={spec.n_features}"
        )


def build_model(
    n_states: int,
    n_features: int,
    outcome_family: families.Family,
    treatment_family: families.Family,
    *,
    architecture: Architecture = "gated",
    predictive_state_hidden_layers: HiddenLayers = ((16, "relu"),),
    outcome_hidden_layers: HiddenLayers = ((16, "relu"),),
    outcome_heads: Optional[Mapping[str, HeadSpec]] = None,
    treatment_embedding_units: Optional[int] = 8,
    dropout_rate: float = 0.2,
    l1: float = 0.0,
    outcome_init: Optional[Mapping[str, float]] = None,
    treatment_init: Optional[Mapping[str, Union[float, Sequence[float]]]] = None,
    max_time: Optional[float] = None,
) -> PSPSModel:
    """Builds a pypsps model for any outcome family and treatment family.

    The arguments are those of `pypsps.keras.models.build_model` minus ``compile`` and the
    compile keywords: the optimizer, learning rate, alpha, regularizer and metrics are
    arguments of `pypsps.torch.training.fit`.

    Args:
      n_states: number of predictive states, K >= 1.
      n_features: number of numeric features, >= 1.
      outcome_family: the within-state outcome distribution, e.g. ``families.Weibull(...)``.
      treatment_family: the within-state treatment distribution, e.g. ``families.Bernoulli()``
        or ``families.Normal()``.
      architecture: ``"gated"`` (one trunk per outcome parameter; each state's head sees the
        treatment gated by its state weight) or ``"shared_trunk"`` (one outcome trunk and a
        skip connection, shared by every head).
      predictive_state_hidden_layers: ``(units, activation)`` per layer of the state network;
        may be empty. Activations: see `pypsps.torch.layers.activation`.
      outcome_hidden_layers: ``(units, activation)`` per layer of the outcome trunk; may be
        empty.
      outcome_heads: optional `HeadSpec` per outcome parameter; missing ones get `HeadSpec()`.
      treatment_embedding_units: width of a learned treatment embedding; None uses the raw
        treatment column.
      dropout_rate: dropout rate of the hidden layers.
      l1: L1 penalty on the hidden-layer kernels (added to the loss by ``fit``).
      outcome_init: optional initial value per outcome parameter, on its support.
      treatment_init: optional initial value per treatment parameter, on its support: one
        value for every state, or one per state. Give all parameters or none.
      max_time: optional largest time in the data, for survival outcome families: raises a
        ValueError when the family's bounds let the loss overflow float32 at that time.

    Returns:
      A `PSPSModel` in training mode.

    Raises:
      ValueError, KeyError, TypeError: for invalid arguments, before any layer is created.
    """
    spec = _make_spec(
        n_states,
        n_features,
        outcome_family,
        treatment_family,
        architecture,
        predictive_state_hidden_layers,
        outcome_hidden_layers,
        outcome_heads,
        treatment_embedding_units,
        dropout_rate,
        l1,
        outcome_init,
        treatment_init,
        max_time,
    )
    return PSPSModel(spec)
