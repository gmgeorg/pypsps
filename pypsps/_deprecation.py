"""Deprecation warnings for the three-int (column count) API, removed in v0.5.0."""

import contextlib
import contextvars
import warnings
from typing import Iterator

_SILENCED = contextvars.ContextVar("pypsps_deprecations_silenced", default=False)


def warn(what: str, use: str, stacklevel: int = 3) -> None:
    """Emits the `FutureWarning` for a deprecated three-int API, unless silenced.

    Args:
      what: the deprecated function, class or argument form.
      use: what to use instead.
      stacklevel: points the warning at the caller of the deprecated API.
    """
    if _SILENCED.get():
        return
    warnings.warn(
        f"{what} is deprecated since v0.3.4 and will be removed in v0.5.0; use {use}",
        FutureWarning,
        stacklevel=stacklevel,
    )


@contextlib.contextmanager
def silenced() -> Iterator[None]:
    """No deprecation warnings inside: loading a saved three-int config must not warn."""
    token = _SILENCED.set(True)
    try:
        yield
    finally:
        _SILENCED.reset(token)
