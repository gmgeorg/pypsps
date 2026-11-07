"""The framework-free core of pypsps: no TensorFlow and no PyTorch needed.

* `layout`: the column layout of ``y_pred`` and ``y_true``.
* `families`: parameter specs and the family specs (parameters, bounds, config, validation).
* `mixture`: the validated pair of families and the column layout every backend's state
  mixture builds on.
* `spec`: the validated arguments of ``build_model``.
* `likelihood`: `OutcomeLikelihood`.
* `alpha_schedule`: the schedule of the treatment-loss weight.
* `presets`: the named ``build_model`` argument sets, shared by the backends.

`pypsps.keras` and `pypsps.torch` subclass these and add the tensor math.
"""

from . import alpha_schedule, families, layout, likelihood, mixture, presets, spec

__all__ = ["alpha_schedule", "families", "layout", "likelihood", "mixture", "presets", "spec"]
