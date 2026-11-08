# Architecture & Theory

## Overview

Predictive State Propensity Subclassification (PSPS) is a causal deep learning
algorithm designed for observational (non-randomized) data.

## Theoretical Framework

PSPS decomposes the joint distribution $\Pr(\text{outcome}, \text{treatment}
\mid \text{features})$ by conditioning on intermediate predictive states derived
from $\Pr(\text{treatment} \mid \text{features})$.

Key features:

* **Balancedness**: Provides data-driven propensity score strata that guarantee
  balancedness within strata.
* **Simultaneous Training**: Predictive state representations are optimized
  jointly with outcome models.
* **Generality**: Supports arbitrary treatment types (binary, continuous,
  multi-class) and outcome types (univariate, multivariate, binary, continuous,
  survival).

## The Spec

`pypsps.core.spec.make_spec` validates every `build_model` argument before any layer exists
(including the hidden-layer activation names, `HIDDEN_ACTIVATIONS`) and returns a `Spec`.
The `Spec` is the model description both backends build from and serialize: `Spec.get_config()`
is a JSON dict of the build arguments (without the mixture, which each backend serializes in
its own format), and `Spec.from_config` validates it again. A Keras model carries its `Spec`
config on its `CausalOutput` layer, and a PyTorch model saves it with its weights, so a model
converts between the backends from the description rather than from layer names.
