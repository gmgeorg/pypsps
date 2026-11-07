"""The validated families and column layout of a state mixture; no tensor math.

`pypsps.keras.mixture.StateMixture` and `pypsps.torch.mixture.StateMixture` subclass
`MixtureSpec` and add the likelihoods and means for their framework.
"""

from . import families as families_lib
from .layout import ColumnLayout

REDUCTIONS = ("none", "sum", "sum_over_batch_size")


def reduction_key(reduction: str | None) -> str:
    """The normalized reduction: ``"none"``, ``"sum"`` or ``"sum_over_batch_size"``.

    ``None`` means ``"none"``; the legacy ``"auto"`` means ``"sum_over_batch_size"``.

    Raises:
      ValueError: for any other value.
    """
    key = "none" if reduction is None else str(getattr(reduction, "value", reduction)).lower()
    if key == "auto":
        key = "sum_over_batch_size"
    if key not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
    return key


class MixtureSpec:
    """The outcome and treatment families of a pypsps model, and their column layout.

    Args:
      outcome_family: distribution of the outcome within a state.
      treatment_family: distribution of the treatment within a state; must read exactly one
        ``y_true`` column.
      n_states: K, the number of predictive states.

    Subclasses set ``_family_base`` to the backend's `Family` class, which both families
    must be instances of, and ``_layout_cls`` to the backend's `ColumnLayout`.
    """

    _family_base: type = families_lib.Family
    _layout_cls: type = ColumnLayout

    def __init__(
        self,
        outcome_family: families_lib.Family,
        treatment_family: families_lib.Family,
        n_states: int,
    ):
        """Validates the families and derives the column layout."""
        for name, family in (
            ("outcome_family", outcome_family),
            ("treatment_family", treatment_family),
        ):
            if not isinstance(family, self._family_base):
                raise TypeError(f"{name} must be a pypsps Family, got {type(family).__name__}")
        if treatment_family.n_true_cols != 1:
            raise ValueError(
                "treatment_family must read exactly one y_true column, got "
                f"{type(treatment_family).__name__} with n_true_cols={treatment_family.n_true_cols}"
            )
        self._outcome_family = outcome_family
        self._treatment_family = treatment_family
        self._layout = self._layout_cls.from_families(n_states, outcome_family, treatment_family)

    @property
    def outcome_family(self) -> families_lib.Family:
        """The outcome family."""
        return self._outcome_family

    @property
    def treatment_family(self) -> families_lib.Family:
        """The treatment family."""
        return self._treatment_family

    @property
    def n_states(self) -> int:
        """K, the number of predictive states."""
        return self._layout.n_states

    @property
    def layout(self) -> ColumnLayout:
        """The column layout of ``y_pred`` and ``y_true``."""
        return self._layout

    def __eq__(self, other: object) -> bool:
        """Equal families and ``n_states``."""
        return (
            isinstance(other, MixtureSpec)
            and type(other) is type(self)
            and self._outcome_family == other._outcome_family
            and self._treatment_family == other._treatment_family
            and self.n_states == other.n_states
        )

    def __hash__(self):
        """Consistent with `__eq__`."""
        return hash((self._outcome_family, self._treatment_family, self.n_states))

    def __repr__(self) -> str:
        """Shows the families and K."""
        return (
            f"StateMixture({self._outcome_family!r}, {self._treatment_family!r}, "
            f"n_states={self.n_states})"
        )
