"""Column layout of a pypsps model's ``y_pred`` and ``y_true`` tensors.

A pypsps model predicts one tensor per unit, laid out as

    y_pred = [outcome params | state weights | treatment params]

where each parameter block holds one column per predictive state (K columns), parameter by
parameter: ``[p0_s0, ..., p0_s{K-1}, p1_s0, ..., p1_s{K-1}, ...]``. So
``y_pred`` has ``K * (P_o + 1 + P_t)`` columns for ``P_o`` outcome and ``P_t`` treatment
parameters per state. ``y_true`` is ``[outcome (C columns) | treatment (1 column) | ...]``;
trailing columns are ignored.

`ColumnLayout` is the one module that knows this order. It decodes a tensor into named
``[N, K]`` blocks and encodes them back. It knows nothing about distributions: the families
in `pypsps.keras.families` say which parameters exist, and `ColumnLayout.from_families`
reads their names.

Every method accepts NumPy arrays, eager tensors, graph tensors and KerasTensors, and returns
the same kind (decoding is plain slicing). Widths are read from the static last dimension, so
a layout is safe to capture in a traced function.
"""

import dataclasses
from collections.abc import Iterable
from typing import TYPE_CHECKING, Mapping, NamedTuple, Optional, Union

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:  # annotation only, so the layout depends on tf alone
    from .families import Family

Array = Union[np.ndarray, tf.Tensor]


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
    """Concatenates along the last axis; NumPy in, NumPy out."""
    if all(isinstance(b, np.ndarray) for b in blocks):
        return np.concatenate(blocks, axis=-1)
    return tf.concat(blocks, axis=-1)


def _check_names(names: Iterable[str], what: str) -> tuple[str, ...]:
    """Returns ``names`` as a tuple of str; raises if empty or not unique."""
    names = tuple(str(n) for n in names)
    if not names:
        raise ValueError(f"{what} must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"{what} must be unique, got {names}")
    return names


@tf.keras.utils.register_keras_serializable(package="pypsps")
@dataclasses.dataclass(frozen=True)
class ColumnLayout:
    """Order and width of every block in a pypsps model's ``y_pred`` and ``y_true``.

    Attributes:
      outcome_params: outcome parameter names per state, e.g. ``("loc", "scale")``.
      treatment_params: treatment parameter names per state, e.g. ``("prob",)``.
      n_outcome_true_cols: C, the number of outcome columns in ``y_true``.
      n_states: K. ``None`` infers it from the ``y_pred`` width; only the legacy three-int
        helpers use that.
    """

    outcome_params: tuple[str, ...]
    treatment_params: tuple[str, ...]
    n_outcome_true_cols: int
    n_states: Optional[int] = None

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
        if self.n_states is not None:
            if int(self.n_states) < 1:
                raise ValueError(f"n_states must be >= 1, got {self.n_states}")
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

    @classmethod
    def from_counts(
        cls,
        n_outcome_pred_cols: int,
        n_treatment_pred_cols: int,
        n_outcome_true_cols: int,
        n_states: Optional[int] = None,
    ) -> "ColumnLayout":
        """A layout from parameter counts alone (legacy adapter).

        Parameters are named ``outcome_param_0, ...`` and ``treatment_param_0, ...``.
        """
        return cls(
            outcome_params=tuple(f"outcome_param_{i}" for i in range(int(n_outcome_pred_cols))),
            treatment_params=tuple(
                f"treatment_param_{i}" for i in range(int(n_treatment_pred_cols))
            ),
            n_outcome_true_cols=n_outcome_true_cols,
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

    def y_pred_width(self, n_states: Optional[int] = None) -> int:
        """``K * (P_o + 1 + P_t)``; ``n_states`` defaults to the layout's own."""
        k = self.n_states if n_states is None else int(n_states)
        if k is None:
            raise ValueError("n_states is unknown; pass it or set it on the layout")
        return k * self._cols_per_state

    def resolve_n_states(self, y_pred: Array) -> int:
        """Returns K for ``y_pred``, checking its width against the layout."""
        width = _static_width(y_pred, "y_pred")
        if self.n_states is not None:
            if width != self.y_pred_width():
                raise ValueError(
                    f"y_pred has {width} columns; layout ({', '.join(self.outcome_params)} | "
                    f"K={self.n_states} | {', '.join(self.treatment_params)}) expects "
                    f"{self.y_pred_width()}"
                )
            return self.n_states
        if width % self._cols_per_state != 0:
            raise ValueError(
                f"y_pred has {width} columns, which is not a multiple of "
                f"{self._cols_per_state} (= {self.n_outcome_pred_cols} outcome + 1 weight + "
                f"{self.n_treatment_pred_cols} treatment columns per state)"
            )
        return width // self._cols_per_state

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

    def encode(
        self,
        outcome: Mapping[str, Array],
        state_weights: Array,
        treatment: Union[Mapping[str, Array], Array],
    ) -> Array:
        """Inverse of `decode_pred`: concatenates named ``[N, K]`` blocks into ``y_pred``.

        Args:
          outcome: ``{param name: [N, K]}`` with exactly the layout's outcome names.
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

        if isinstance(treatment, Mapping):
            treatment_blocks = blocks(treatment, self.treatment_params, "treatment")
        else:
            expected = k * self.n_treatment_pred_cols
            if _static_width(treatment, "treatment") != expected:
                raise ValueError(
                    f"treatment block must have {expected} columns, got shape "
                    f"{tuple(treatment.shape)}"
                )
            treatment_blocks = [treatment]
        return _concat(
            blocks(outcome, self.outcome_params, "outcome") + [state_weights] + treatment_blocks
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
