"""Training and prediction of a PyTorch pypsps model: `fit` and `predict`.

`fit` is the PyTorch counterpart of ``compile_model`` followed by ``model.fit`` in Keras: it
builds the `pypsps.torch.losses.CausalLoss` and the optimizer, runs the epochs, logs the
same history keys (``loss``, ``val_loss`` and the metrics of `pypsps.torch.metrics`) and calls
the callbacks of `pypsps.torch.callbacks`.

    history = training.fit(model, features, treatment, outcome, epochs=50, alpha=2.0)
    y_pred = training.predict(model, features, treatment)

Conventions that follow Keras (see ``docs/torch.md`` for the differences that remain):

* ``batch_size`` is 32 and the data is reshuffled every epoch;
* ``validation_split`` takes the last fraction of the data, before shuffling;
* the logged ``loss`` is the mean over the units of the epoch, in training mode, including
  the regularizers (the kernel L1 penalty and the predictive-state regularizer);
* the default optimizer is Nadam (learning rate 0.01) clipping the gradient norm of each
  parameter to 1.0, like ``tf.keras.optimizers.Nadam(clipnorm=1.0)``.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..core.likelihood import OutcomeLikelihood
from . import callbacks as callbacks_lib
from . import losses as losses_lib
from . import metrics as metrics_lib

Array = Any  # numpy array, pandas object, list or torch tensor

_DEFAULT_LEARNING_RATE = 0.01
_DEFAULT_CLIPNORM = 1.0
_UNSET: Any = object()


class History:
    """What `fit` returns, like the Keras ``History``.

    Attributes:
      history: metric name -> list of per-epoch values (``loss``, ``val_loss``, ...).
      epoch: the zero-based epochs that ran.
      loss: the `CausalLoss` used (its ``alpha`` is the last epoch's).
      optimizer: the optimizer used.
    """

    def __init__(self, loss: losses_lib.CausalLoss, optimizer: torch.optim.Optimizer):
        """Creates an empty history."""
        self.history: Dict[str, List[float]] = {}
        self.epoch: List[int] = []
        self.loss = loss
        self.optimizer = optimizer

    def _append(self, epoch: int, logs: Dict[str, float]) -> None:
        self.epoch.append(epoch)
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)


def _tensor(x: Array, name: str) -> torch.Tensor:
    """``x`` as a float32 tensor; a pandas object or list becomes a NumPy array first."""
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32)
    array = np.asarray(x, dtype="float32")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} has non-finite values (NaN or inf)")
    return torch.from_numpy(np.ascontiguousarray(array))


def _y_true(outcome: Array, treatment: torch.Tensor, n_outcome_cols: int) -> torch.Tensor:
    """``[outcome (C columns) | treatment]`` as the loss reads it."""
    y = _tensor(outcome, "outcome")
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[1] != n_outcome_cols:
        raise ValueError(
            f"outcome has {y.shape[1]} column(s) but the outcome family reads {n_outcome_cols}"
        )
    return torch.cat([y, treatment.reshape(-1, 1)], dim=1)


def _prepare(
    features: Array,
    treatment: Array,
    outcome: Array,
    sample_weight: Optional[Array],
    n_features: int,
    n_outcome_cols: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Validated float32 tensors: features ``[N, F]``, treatment ``[N]``, y_true, weights."""
    x = _tensor(features, "features")
    if x.ndim != 2 or x.shape[1] != n_features:
        raise ValueError(f"features must have shape [N, {n_features}], got {tuple(x.shape)}")
    a = _tensor(treatment, "treatment").reshape(-1)
    if a.shape[0] != x.shape[0]:
        raise ValueError(f"features has {x.shape[0]} rows but treatment has {a.shape[0]}")
    y = _y_true(outcome, a, n_outcome_cols)
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"features has {x.shape[0]} rows but outcome has {y.shape[0]}")
    weight = None
    if sample_weight is not None:
        weight = _tensor(sample_weight, "sample_weight").reshape(-1)
        if weight.shape[0] != x.shape[0]:
            raise ValueError(
                f"features has {x.shape[0]} rows but sample_weight has {weight.shape[0]}"
            )
    return x, a, y, weight


def default_optimizer(
    model: torch.nn.Module,
    learning_rate: float = _DEFAULT_LEARNING_RATE,
) -> torch.optim.Optimizer:
    """Nadam with the Keras defaults (betas 0.9 / 0.999, epsilon 1e-7)."""
    return torch.optim.NAdam(model.parameters(), lr=learning_rate, eps=1e-7)


def _clip_each(parameters: Sequence[torch.nn.Parameter], clipnorm: float) -> None:
    """Clips the gradient norm of every parameter on its own, as Keras ``clipnorm`` does."""
    for p in parameters:
        if p.grad is not None:
            torch.nn.utils.clip_grad_norm_([p], clipnorm)


def _total_loss(
    model: torch.nn.Module,
    loss: losses_lib.CausalLoss,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    weight: Optional[torch.Tensor],
) -> torch.Tensor:
    """The causal loss (with the state regularizer) plus the model's kernel L1 penalty."""
    total = loss(y_true, y_pred, weight)
    l1_penalty = getattr(model, "kernel_l1_penalty", None)
    if l1_penalty is not None:
        total = total + l1_penalty().to(total.dtype)
    return total


def _batches(
    n: int, batch_size: int, order: Optional[torch.Tensor], merge_single_tail: bool = False
):
    """Index tensors of the batches over ``order`` (or ``arange(n)``).

    With ``merge_single_tail`` a trailing batch of one unit is folded into the previous batch,
    since batch normalization cannot train on a single unit.
    """
    index = torch.arange(n) if order is None else order
    starts = list(range(0, n, batch_size))
    if merge_single_tail and len(starts) > 1 and n - starts[-1] == 1:
        starts.pop()
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else n
        yield index[start:stop]


def _evaluate(
    model: torch.nn.Module,
    loss: losses_lib.CausalLoss,
    metrics: Sequence[metrics_lib.Metric],
    data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    batch_size: int,
    device: torch.device,
    weighted_metrics: bool = False,
) -> Dict[str, float]:
    """The ``loss`` and metrics over ``data`` in eval mode."""
    x, a, y, weight = data
    model.eval()
    loss.eval()
    for metric in metrics:
        metric.reset()
    total, count = 0.0, 0
    with torch.no_grad():
        for idx in _batches(x.shape[0], batch_size, None):
            xb, ab, yb = x[idx].to(device), a[idx].to(device), y[idx].to(device)
            wb = None if weight is None else weight[idx].to(device)
            y_pred = model(xb, ab)
            total += float(_total_loss(model, loss, yb, y_pred, wb)) * len(idx)
            count += len(idx)
            for metric in metrics:
                metric.update(yb, y_pred, wb if weighted_metrics else None)
    logs = {"loss": total / max(count, 1)}
    logs.update({metric.name: metric.result() for metric in metrics})
    return logs


def fit(
    model: torch.nn.Module,
    features: Array,
    treatment: Array,
    outcome: Array,
    *,
    epochs: int = 1,
    batch_size: int = 32,
    sample_weight: Optional[Array] = None,
    validation_split: float = 0.0,
    validation_data: Optional[Tuple[Any, ...]] = None,
    shuffle: bool = True,
    alpha: float = 1.0,
    outcome_loss_weight: float = 1.0,
    outcome_likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
    predictive_states_regularizer: Optional[Callable] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    learning_rate: float = _UNSET,
    clipnorm: Optional[float] = _UNSET,
    metrics: Optional[Sequence[metrics_lib.Metric]] = None,
    extra_metrics: Sequence[metrics_lib.Metric] = (),
    weighted_metrics: bool = False,
    callbacks: Sequence[callbacks_lib.Callback] = (),
    verbose: int = 0,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> History:
    """Trains a `pypsps.torch.models.PSPSModel`.

    Args:
      model: the model from `pypsps.torch.models.build_model`.
      features: ``[N, n_features]``.
      treatment: ``[N]``.
      outcome: ``[N]`` or ``[N, C]``: the outcome columns the outcome family reads (a survival
        family reads ``[time, event]``).
      epochs: number of passes over the training data.
      batch_size: units per step (Keras default 32); a trailing batch of one unit is merged
        into the previous batch.
      sample_weight: optional ``[N]`` weights multiplying each unit's loss.
      validation_split: fraction of the last rows held out for validation (before shuffling).
      validation_data: ``(features, treatment, outcome[, sample_weight])``; overrides the split.
      shuffle: whether to reshuffle the training units every epoch.
      alpha: weight of the treatment log-likelihood (see `pypsps.torch.losses.CausalLoss`);
        change it during training with `callbacks.AlphaScheduleCallback`.
      outcome_loss_weight: weight of the outcome log-likelihood, >= 0.
      outcome_likelihood: ``"interventional"`` (default) or ``"observational"``.
      predictive_states_regularizer: optional regularizer of the state weights, e.g. a
        `pypress.torch.regularizers` instance.
      optimizer: your own optimizer (over ``model.parameters()``); by default Nadam with
        ``learning_rate`` (0.01) and gradient clipping ``clipnorm`` (1.0; None disables it).
        Pass either the optimizer or those two.
      metrics: the metrics to log; None uses `pypsps.torch.metrics.default_metrics`.
      extra_metrics: metrics appended to ``metrics``.
      weighted_metrics: whether the metrics use ``sample_weight``. The loss always does; the
        metrics do not by default, as in Keras (``compile(metrics=...)``), so that they stay
        comparable across fits with different weights.
      callbacks: `pypsps.torch.callbacks.Callback` objects.
      verbose: 0 is silent, 1 prints one line per epoch.
      seed: seed of the shuffling.
      device: where to train; by default where the model's parameters are.

    Returns:
      A `History`. The model is left in eval mode.

    Raises:
      TypeError: if ``optimizer`` is given with ``learning_rate`` or ``clipnorm``.
      ValueError: for invalid data or arguments.
    """
    if optimizer is not None and (learning_rate is not _UNSET or clipnorm is not _UNSET):
        raise TypeError("pass learning_rate/clipnorm or optimizer, not both")
    if not 0.0 <= validation_split < 1.0:
        raise ValueError(f"validation_split must be in [0, 1), got {validation_split}")
    if batch_size < 1 or epochs < 0:
        raise ValueError("batch_size must be >= 1 and epochs >= 0")
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(dev)
    mixture = model.mixture
    n_outcome_cols = mixture.layout.n_outcome_true_cols
    train = _prepare(
        features, treatment, outcome, sample_weight, model.spec.n_features, n_outcome_cols
    )
    val = None
    if validation_data is not None:
        vf, vt, vo, *vw = validation_data
        val = _prepare(vf, vt, vo, vw[0] if vw else None, model.spec.n_features, n_outcome_cols)
    elif validation_split > 0.0:
        n_total = train[0].shape[0]
        cut = int(n_total * (1.0 - validation_split))
        if cut < 1 or cut >= n_total:
            raise ValueError(
                f"validation_split={validation_split} leaves no rows to train or validate"
            )
        val = tuple(None if t is None else t[cut:] for t in train)
        train = tuple(None if t is None else t[:cut] for t in train)
    x, a, y, weight = train
    n = x.shape[0]

    loss = losses_lib.CausalLoss(
        mixture,
        alpha=alpha,
        outcome_loss_weight=outcome_loss_weight,
        outcome_likelihood=outcome_likelihood,
        predictive_states_regularizer=predictive_states_regularizer,
        reduction="sum_over_batch_size",
    ).to(dev)
    if optimizer is None:
        optimizer = default_optimizer(
            model, _DEFAULT_LEARNING_RATE if learning_rate is _UNSET else learning_rate
        )
        clip = _DEFAULT_CLIPNORM if clipnorm is _UNSET else clipnorm
    else:
        clip = None
    parameters = [p for p in model.parameters() if p.requires_grad]
    chosen = (
        list(metrics_lib.default_metrics(mixture, outcome_loss_weight, outcome_likelihood))
        if metrics is None
        else list(metrics)
    ) + list(extra_metrics)

    history = History(loss, optimizer)
    model.stop_training = False
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    for callback in callbacks:
        callback.set_context(model, loss, optimizer)
        callback.on_train_begin()

    for epoch in range(epochs):
        for callback in callbacks:
            callback.on_epoch_begin(epoch)
        model.train()
        loss.train()
        for metric in chosen:
            metric.reset()
        order = torch.randperm(n, generator=generator) if shuffle else None
        total, count = 0.0, 0
        for idx in _batches(n, batch_size, order, merge_single_tail=True):
            xb, ab, yb = x[idx].to(dev), a[idx].to(dev), y[idx].to(dev)
            wb = None if weight is None else weight[idx].to(dev)
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(xb, ab)
            value = _total_loss(model, loss, yb, y_pred, wb)
            value.backward()
            if clip is not None:
                _clip_each(parameters, clip)
            optimizer.step()
            total += float(value.detach()) * len(idx)
            count += len(idx)
            for metric in chosen:
                metric.update(yb.detach(), y_pred.detach(), wb if weighted_metrics else None)
        logs: Dict[str, float] = {"loss": total / max(count, 1)}
        logs.update({metric.name: metric.result() for metric in chosen})
        if val is not None:
            val_logs = _evaluate(model, loss, chosen, val, batch_size, dev, weighted_metrics)
            logs.update({f"val_{k}": v for k, v in val_logs.items()})
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs)
        history._append(epoch, logs)
        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                + " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())
            )
        if getattr(model, "stop_training", False):
            break
    for callback in callbacks:
        callback.on_train_end()
    model.eval()
    return history


def evaluate(
    model: torch.nn.Module,
    features: Array,
    treatment: Array,
    outcome: Array,
    sample_weight: Optional[Array] = None,
    *,
    alpha: float = 1.0,
    outcome_loss_weight: float = 1.0,
    outcome_likelihood: Union[OutcomeLikelihood, str] = OutcomeLikelihood.INTERVENTIONAL,
    predictive_states_regularizer: Optional[Callable] = None,
    metrics: Optional[Sequence[metrics_lib.Metric]] = None,
    weighted_metrics: bool = False,
    batch_size: int = 1024,
) -> Dict[str, float]:
    """The ``loss`` and the metrics of ``model`` on the data, in eval mode.

    The counterpart of Keras ``model.evaluate(..., return_dict=True)`` with the arguments of
    `fit` that define the loss. The training/eval mode of the model and of the regularizer is
    restored afterwards.
    """
    dev = next(model.parameters()).device
    data = _prepare(
        features,
        treatment,
        outcome,
        sample_weight,
        model.spec.n_features,
        model.mixture.layout.n_outcome_true_cols,
    )
    loss = losses_lib.CausalLoss(
        model.mixture,
        alpha=alpha,
        outcome_loss_weight=outcome_loss_weight,
        outcome_likelihood=outcome_likelihood,
        predictive_states_regularizer=predictive_states_regularizer,
    ).to(dev)
    chosen = (
        list(metrics_lib.default_metrics(model.mixture, outcome_loss_weight, outcome_likelihood))
        if metrics is None
        else list(metrics)
    )
    modes = [
        (module, module.training)
        for module in (model, predictive_states_regularizer)
        if isinstance(module, torch.nn.Module)
    ]
    try:
        return _evaluate(model, loss, chosen, data, batch_size, dev, weighted_metrics)
    finally:
        for module, mode in modes:
            module.train(mode)


def predict(
    model: torch.nn.Module,
    features: Array,
    treatment: Array,
    batch_size: int = 1024,
) -> np.ndarray:
    """``y_pred`` ``[N, width]`` of ``model`` as a NumPy array, in eval mode (no dropout).

    The model's training/eval mode is restored afterwards.
    """
    dev = next(model.parameters()).device
    x = _tensor(features, "features")
    a = _tensor(treatment, "treatment").reshape(-1)
    if x.shape[0] != a.shape[0]:
        raise ValueError(f"features has {x.shape[0]} rows but treatment has {a.shape[0]}")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            parts = [
                model(x[i : i + batch_size].to(dev), a[i : i + batch_size].to(dev)).cpu()
                for i in range(0, x.shape[0], batch_size)
            ]
    finally:
        model.train(was_training)
    return torch.cat(parts).numpy() if parts else np.zeros((0, model.layout.y_pred_width()), "f4")
