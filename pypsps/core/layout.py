"""Column layout of a pypsps model's ``y_pred`` and ``y_true`` tensors.

A pypsps model predicts one tensor per unit, laid out as

    y_pred = [outcome params | state weights | treatment params]

where each parameter block holds one column per predictive state (K columns), parameter by
parameter: ``[p0_s0, ..., p0_s{K-1}, p1_s0, ..., p1_s{K-1}, ...]``. So
``y_pred`` has ``K * (P_o + 1 + P_t)`` columns for ``P_o`` outcome and ``P_t`` treatment
parameters per state. ``y_true`` is ``[outcome (C columns) | treatment (1 column) | ...]``;
`ColumnLayout.decode_true` ignores trailing columns, and `ColumnLayout.encode_true` builds
``y_true`` with exactly C + 1 columns.

`ColumnLayout` is the one module that knows this order. It decodes a tensor into named
``[N, K]`` blocks and encodes them back. It knows nothing about distributions: the families
in `pypsps.core.families` say which parameters exist, and `ColumnLayout.from_families`
reads their names.

Every method accepts NumPy arrays and framework tensors (TensorFlow eager and graph tensors,
KerasTensors, PyTorch tensors) and returns the same kind (decoding is plain slicing). Widths
are read from the static last dimension, so a layout is safe to capture in a traced function.
This module imports neither TensorFlow nor PyTorch; `pypsps.keras.layout` registers the layout
with Keras.
"""

import dataclasses
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:  # annotation only
    from .families import Family

Array = Any  # a NumPy array or a TensorFlow / PyTorch tensor


class PredParts(NamedTuple):
    """A decoded ``y_pred``: named per-state parameter blocks and the state weights."""

    outcome: dict[str, Array]  # {param name: [N, K]}, in family parameter order
    state_weights: Array  # [N, K], P(S | X); rows sum to 1
    treatment: dict[str, Array]  # {param name: [N, K]}


class TrueParts(NamedTuple):
    """A decoded ``y_true``."""

    outcome: Array  # [N, C]; C = 1 for Normal, 2 for survival (time, event)
    treatment: Array  # [N, 1]


def _static_width(y: Array, what: str) -> int:
    """Returns the static last dimension of ``y``; raises if it is unknown."""
    width = y.shape[-1]
    if width is None:
        raise ValueError(f"{what} must have a static last dimension, got shape {y.shape}")
    return int(width)


def _split_params(block: Array, names: tuple[str, ...], n_states: int) -> dict[str, Array]:
    """Splits a ``[N, K * P]`` parameter block into ``{name: [N, K]}``."""
    return {n: block[:, i * n_states : (i + 1) * n_states] for i, n in enumerate(names)}


def _state_column_indices(state_idx: int, n_states: int, n_params: int) -> list[int]:
    """Column indices of state ``state_idx`` in a ``[N, K * P]`` parameter block."""
    return [state_idx + i * n_states for i in range(n_params)]


def _concat(blocks: list[Array]) -> Array:
    """Concatenates along the last axis; NumPy in, NumPy out, tensors in, tensor out.

    The framework is picked from the blocks' type, so neither one is imported unless used.
    """
    if all(isinstance(b, np.ndarray) for b in blocks):
        return np.concatenate(blocks, axis=-1)
    root = type(next(b for b in blocks if not isinstance(b, np.ndarray))).__module__.split(".")[0]
    if root == "torch":
        import torch

        return torch.cat([torch.as_tensor(b) for b in blocks], dim=-1)
    import tensorflow as tf

    return tf.concat(blocks, axis=-1)


def _as_array(x: Array) -> Array:
    """Returns ``x`` as a NumPy array if it is a pandas object or a list; tensors pass through."""
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray(x)
    return x


def _as_columns(y: Array, what: str) -> Array:
    """``y`` as ``[N, C]``: a 1-D ``[N]`` becomes one column; any other rank raises."""
    ndim = len(y.shape)
    if ndim == 1:
        return y[:, None]
    if ndim != 2:
        raise ValueError(f"{what} must be [N] or [N, C], got shape {tuple(y.shape)}")
    return y


def _check_names(names: Iterable[str], what: str) -> tuple[str, ...]:
    """Returns ``names`` as a tuple of str; raises if empty or not unique."""
    names = tuple(str(n) for n in names)
    if not names:
        raise ValueError(f"{what} must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{what} must be unique, got {names}")
    return names


@dataclasses.dataclass(frozen=True)
class ColumnLayout:
    """Order and width of every block in a pypsps model's ``y_pred`` and ``y_true``.

    Attributes:
      outcome_params: outcome parameter names per state, e.g. ``("loc", "scale")``.
      treatment_params: treatment parameter names per state, e.g. ``("prob",)``.
      n_outcome_true_cols: C, the number of outcome columns in ``y_true``.
      n_states: K, the number of predictive states.
    """

    outcome_params: tuple[str, ...]
    treatment_params: tuple[str, ...]
    n_outcome_true_cols: int
    n_states: int

    def __post_init__(self):
        """Normalizes the names to tuples and validates every field."""
        object.__setattr__(
            self, "outcome_params", _check_names(self.outcome_params, "outcome_params")
        )
        object.__setattr__(
            self, "treatment_params", _check_names(self.treatment_params, "treatment_params")
        )
        if int(self.n_outcome_true_cols) < 1:
            raise ValueError(f"n_outcome_true_cols must be >= 1, got {self.n_outcome_true_cols}")
        object.__setattr__(self, "n_outcome_true_cols", int(self.n_outcome_true_cols))
        if self.n_states is None or int(self.n_states) < 1:
            raise ValueError(f"n_states must be an int >= 1, got {self.n_states}")
        object.__setattr__(self, "n_states", int(self.n_states))

    # --- construction ---

    @classmethod
    def from_families(
        cls, n_states: int, outcome_family: "Family", treatment_family: "Family"
    ) -> "ColumnLayout":
        """The layout of a model with these outcome and treatment families."""
        return cls(
            outcome_params=outcome_family.param_names,
            treatment_params=treatment_family.param_names,
            n_outcome_true_cols=outcome_family.n_true_cols,
            n_states=n_states,
        )

    # --- widths ---

    @property
    def n_outcome_pred_cols(self) -> int:
        """Outcome parameters per state (P_o)."""
        return len(self.outcome_params)

    @property
    def n_treatment_pred_cols(self) -> int:
        """Treatment parameters per state (P_t)."""
        return len(self.treatment_params)

    @property
    def _cols_per_state(self) -> int:
        """Columns per state across all blocks: ``P_o + 1 + P_t``."""
        return self.n_outcome_pred_cols + 1 + self.n_treatment_pred_cols

    def y_pred_width(self) -> int:
        """``K * (P_o + 1 + P_t)``."""
        return self.n_states * self._cols_per_state

    def resolve_n_states(self, y_pred: Array) -> int:
        """Returns K, checking the ``y_pred`` width against the layout."""
        width = _static_width(y_pred, "y_pred")
        if width != self.y_pred_width():
            raise ValueError(
                f"y_pred has {width} columns; layout ({', '.join(self.outcome_params)} | "
                f"K={self.n_states} | {', '.join(self.treatment_params)}) expects "
                f"{self.y_pred_width()}"
            )
        return self.n_states

    # --- decode / encode ---

    def _split_blocks(self, y_pred: Array) -> tuple[Array, Array, Array]:
        """``(outcome block [N, K*P_o], state weights [N, K], treatment block [N, K*P_t])``."""
        k = self.resolve_n_states(y_pred)
        w0 = self.n_outcome_pred_cols * k
        t0 = w0 + k
        return y_pred[:, :w0], y_pred[:, w0:t0], y_pred[:, t0:]

    def decode_pred(self, y_pred: Array) -> PredParts:
        """Splits ``y_pred`` into named ``[N, K]`` blocks and the state weights."""
        k = self.resolve_n_states(y_pred)
        outcome, weights, treatment = self._split_blocks(y_pred)
        return PredParts(
            outcome=_split_params(outcome, self.outcome_params, k),
            state_weights=weights,
            treatment=_split_params(treatment, self.treatment_params, k),
        )

    def decode_true(self, y_true: Array) -> TrueParts:
        """Splits ``y_true`` into the outcome columns and the one treatment column after them.

        Columns after the treatment column are ignored.
        """
        c = self.n_outcome_true_cols
        width = _static_width(y_true, "y_true")
        if width < c + 1:
            raise ValueError(
                f"y_true has {width} columns; the layout needs at least {c + 1} "
                f"({c} outcome + 1 treatment)"
            )
        return TrueParts(outcome=y_true[:, :c], treatment=y_true[:, c : c + 1])

    def encode_true(self, outcome: Array, treatment: Array) -> Array:
        """Inverse of `decode_true`: ``[outcome (C columns) | treatment (1 column)]``.

        This is the one place that builds ``y_true`` and checks its width. ``decode_true``
        ignores trailing columns, because a Keras model gets ``y_true`` as the caller made it;
        this method does not, so an outcome of the wrong width (a ``[time, event]`` survival
        outcome for a Normal family, say) raises here instead of being read as an outcome and
        a treatment.

        Args:
          outcome: ``[N, C]``, or ``[N]`` when C == 1. C is the outcome family's number of
            ``y_true`` columns.
          treatment: ``[N]`` or ``[N, 1]``.

        Returns:
          ``y_true`` with ``C + 1`` columns. pandas objects and lists are converted to NumPy;
          a NumPy array in gives a NumPy array out, and a tensor in gives a tensor out. The
          dtypes are not changed, so tensors must already share one.

        Raises:
          ValueError: if the outcome does not have exactly C columns, the treatment has more
            than one column, either has a rank above 2, or the row counts differ.
        """
        outcome = _as_columns(_as_array(outcome), "outcome")
        treatment = _as_columns(_as_array(treatment), "treatment")
        width = _static_width(outcome, "outcome")
        if width != self.n_outcome_true_cols:
            raise ValueError(
                f"outcome has {width} column(s) but the outcome family reads "
                f"{self.n_outcome_true_cols}"
            )
        treatment_width = _static_width(treatment, "treatment")
        if treatment_width != 1:
            raise ValueError(
                f"treatment has {treatment_width} column(s); y_true has exactly one treatment column"
            )
        n_outcome, n_treatment = outcome.shape[0], treatment.shape[0]
        if n_outcome is not None and n_treatment is not None and n_outcome != n_treatment:
            raise ValueError(f"outcome has {n_outcome} rows but treatment has {n_treatment}")
        return _concat([outcome, treatment])

    def encode(
        self,
        outcome: Mapping[str, Array] | Array,
        state_weights: Array,
        treatment: Mapping[str, Array] | Array,
    ) -> Array:
        """Inverse of `decode_pred`: concatenates named ``[N, K]`` blocks into ``y_pred``.

        Args:
          outcome: ``{param name: [N, K]}`` with exactly the layout's outcome names, or the whole
            ``[N, K * P_o]`` outcome block.
          state_weights: ``[N, K]``.
          treatment: ``{param name: [N, K]}``, or the whole ``[N, K * P_t]`` treatment block.

        Returns:
          ``y_pred``; a NumPy array if every input is one, else a tensor.
        """
        k = _static_width(state_weights, "state_weights")
        if self.n_states is not None and k != self.n_states:
            raise ValueError(f"state_weights has {k} columns; the layout has K={self.n_states}")

        def blocks(parts: Mapping[str, Array], names: tuple[str, ...], what: str):
            """Returns ``parts`` in ``names`` order, checking keys and shapes."""
            if set(parts) != set(names):
                raise KeyError(f"{what} keys must be {list(names)}, got {sorted(parts)}")
            out = []
            for name in names:
                block = parts[name]
                if len(block.shape) != 2 or _static_width(block, f"{what}[{name!r}]") != k:
                    raise ValueError(
                        f"{what}[{name!r}] must be [N, {k}], got shape {tuple(block.shape)}"
                    )
                out.append(block)
            return out

        def as_blocks(part: Mapping[str, Array] | Array, names: tuple[str, ...], what: str):
            """A mapping in ``names`` order, or a whole block checked for its width."""
            if isinstance(part, Mapping):
                return blocks(part, names, what)
            expected = k * len(names)
            if _static_width(part, what) != expected:
                raise ValueError(
                    f"{what} block must have {expected} columns, got shape {tuple(part.shape)}"
                )
            return [part]

        return _concat(
            as_blocks(outcome, self.outcome_params, "outcome")
            + [state_weights]
            + as_blocks(treatment, self.treatment_params, "treatment")
        )

    # --- serialization ---

    def get_config(self) -> dict:
        """Plain lists and ints; `from_config` inverts it."""
        return {
            "outcome_params": list(self.outcome_params),
            "treatment_params": list(self.treatment_params),
            "n_outcome_true_cols": self.n_outcome_true_cols,
            "n_states": self.n_states,
        }

    @classmethod
    def from_config(cls, config: dict) -> "ColumnLayout":
        """Rebuilds a layout from `get_config`."""
        return cls(**config)
