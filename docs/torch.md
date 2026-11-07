# PyTorch Backend

`pypsps.torch` computes the same PSPS model as `pypsps.keras` on PyTorch, without
TensorFlow installed: an `nn.Module` model, a plain callable loss, and a small `fit` loop.
It uses [`pypress.torch`](https://github.com/gmgeorg/pypress) for the predictive-state layers
and regularizers.

## Installation

Each backend is an optional extra, so install the one you use (or both):

```bash
pip install "pypsps[torch] @ git+https://github.com/gmgeorg/pypsps.git"   # PyTorch only
pip install "pypsps[tf] @ git+https://github.com/gmgeorg/pypsps.git"      # TensorFlow only
pip install "pypsps[tf,torch] @ git+https://github.com/gmgeorg/pypsps.git" # both
```

Without an extra you get the framework-free parts: `pypsps.core` (column layout, families
specs, alpha schedule, preset logic), `pypsps.inference`, `pypsps.bootstrap` and
`pypsps.datasets`. Importing `pypsps.keras` or `pypsps.torch` without its framework raises an
`ImportError` that names the extra to install.

## Quick start

```python
import torch

from pypsps import datasets, inference
from pypsps.torch import callbacks, models, presets, training

ks_data = datasets.KangSchafer(true_ate=20).sample(n_samples=1000)
torch.manual_seed(10)
preset = presets.toy(n_states=4, n_features=ks_data.n_features, alpha=10.0)
model = models.build_model(**preset)  # the build_model arguments
history = training.fit(
    model,
    ks_data.features.values,
    ks_data.treatments.values,
    ks_data.outcomes.values,
    epochs=250,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks.get_default_callbacks(monitor="val_causal_loss_metric", patience=25),
    **preset.fit_kwargs,  # alpha, learning rate, clipnorm, state regularizer
)
y_pred = training.predict(model, ks_data.features.values, ks_data.treatments.values)
parts = model.layout.decode_pred(y_pred)  # parts.outcome["loc"], parts.state_weights, ...
pred_ate = inference.predict_ate_binary(model, ks_data.features)
```

## What is where

| Keras (`pypsps.keras`) | PyTorch (`pypsps.torch`) |
|---|---|
| `models.build_model(..., compile=True)` | `models.build_model(...)`, a `PSPSModel` (`nn.Module`) |
| `models.compile_model`, `model.fit` | `training.fit(model, features, treatment, outcome, ...)` |
| `model.predict([x, a])` | `training.predict(model, x, a)` |
| `model.evaluate` | `training.evaluate(model, x, a, outcome, ...)` |
| `losses.CausalLoss` (variable `alpha`) | `losses.CausalLoss` (an `nn.Module`; `loss.alpha = x`) |
| `mixture.StateMixture` | `mixture.StateMixture` (same methods on torch tensors) |
| `families.Normal()`, `Weibull(...)`, ... | the same names and arguments |
| `presets.toy(...)`, `binary_normal(...)`, ... | the same names; each returns the `build_model` arguments with `.fit_kwargs` |
| `metrics.*`, `models.default_metrics` | `metrics.*` (accumulating objects), `metrics.default_metrics` |
| `callbacks.*` | `callbacks.*` (same names, Keras-style hooks) |
| `model.save(...)`, `load_model` | `io.save(model, path)`, `io.load(path)` |

`y_pred` has the same column layout as in Keras (`[outcome params | state weights |
treatment params]`, one block of K columns per parameter), so `model.layout`,
`mixture.outcome_mean(y_pred)` and `pypsps.inference` work on either backend.

The `build_model` arguments are those of the Keras builder minus the compile keywords. A
Keras `compile_model(alpha=..., outcome_likelihood=..., learning_rate=...)` becomes the same
keywords of `training.fit`. The optimizer defaults are the Keras ones: Nadam with a learning
rate of 0.01, clipping the gradient norm of each parameter to 1.0 (`clipnorm=None` turns it
off). To use another optimizer, pass `optimizer=` (not together with `learning_rate` or
`clipnorm`).

## Converting a Keras model

`PSPSModel.from_keras(keras_model)` rebuilds the architecture from the Keras layer names
and copies every weight (including batch-normalization statistics). In eval mode the copy
predicts the same `y_pred` up to float32 rounding, which is how the cross-backend tests check
the two implementations against each other. A model with a `LossFamily` has no PyTorch
counterpart (a Keras loss cannot be converted) and raises `NotImplementedError`.

```python
from pypsps.torch import models

torch_model = models.PSPSModel.from_keras(trained_keras_model)
```

## Saving and loading

`io.save(model, path)` writes the build arguments and the `state_dict`; `io.load(path)`
rebuilds the model and returns it in eval mode. The file is read with
`torch.load(weights_only=True)`, so it holds no pickled code. A model with a `LossFamily`
cannot be saved (it wraps a callable). Save the optimizer state yourself if you want to resume
training.

## Conventions shared with Keras

* Batch normalization: Keras momentum 0.99 and epsilon 1e-3 are PyTorch `momentum=0.01`
  and `eps=1e-3`.
* Dense layers use Glorot-uniform kernels and zero biases (PyTorch's default would be
  Kaiming-uniform).
* The loss is `outcome_loss_weight * outcome NLL + alpha * treatment NLL`; `"sum_over_batch_size"`
  divides by the number of units, not by the sum of the weights, and the predictive-state
  regularizer is added once after the reduction, never scaled by `sample_weight`.
* Likelihood numerics: Bernoulli probabilities are clipped to `[1e-7, 1 - 1e-7]`; the Normal
  scale has a floor of 1e-6; the Weibull and exponential logs use `log(t + 1e-6)`.
* `fit` uses a batch size of 32 and reshuffles every epoch; `validation_split` holds out the
  last fraction of the rows before shuffling.
* The logged `loss` is the mean over the units of the epoch in training mode, including the
  regularizers. `val_loss` and the `val_` metrics are computed in eval mode. The history keys
  are the Keras ones (`causal_loss_metric`, `outcome_nll_metric`, `predictive_state_df`,
  `uniform_entropy`, `propensity_score_*`, `outcome_mean_squared_error`, ...), so
  `monitor="val_causal_loss_metric"` works unchanged.
* The metrics ignore `sample_weight` unless you pass `weighted_metrics=True` (as Keras
  `compile(metrics=...)` does), so they stay comparable across weighted fits.

## Differences from Keras

* `training.fit` leaves the model in eval mode; call `model.train()` to continue training
  with your own loop.
* `seed=` seeds the shuffling only. Dropout draws from PyTorch's global generator, so call
  `torch.manual_seed` for a reproducible fit.
* `torch.optim.NAdam` differs slightly from `tf.keras.optimizers.Nadam` in its momentum
  schedule, so the trajectories of the two backends are close but not identical.
* PyTorch's batch-norm running variance uses the unbiased batch variance; Keras uses the
  biased one. The difference is negligible for batch sizes above a few dozen.
* `propensity_score_auroc` is exact, where Keras bins the scores into 200 thresholds, and
  `propensity_score_aupr` is the average precision. They agree with Keras closely.
* `alpha` is a plain float on the loss (`loss.alpha = x`), and the `callbacks` change it
  between epochs. There is no graph-tracing caveat.
* `callbacks.EarlyStopping` and `callbacks.ReduceLROnPlateau` are part of the package (Keras
  has its own); `EarlyStopping` restores the best weights like
  `restore_best_weights=True`.
* The Keras `get_propensity_state_conditional_means(model)` is
  `model.propensity_state_conditional_means()`.

## Tests

`pypsps/tests/torch/` needs only PyTorch. `pypsps/tests/parity/` needs both frameworks and
compares the backends on identical arrays: the per-state and mixture likelihoods, the
posterior, the means and the gradients of the loss; the forward pass of every converted
architecture and preset; and `evaluate` (loss and metrics) on weighted and unweighted data.
