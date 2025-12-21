"""Version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pypsps")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0+unknown"
