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
