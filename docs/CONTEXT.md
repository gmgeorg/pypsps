# pypsps

Causal effect estimation from observational data with Predictive State Propensity
Subclassification (PSPS): units are softly assigned to predictive states learned from the
treatment model, each state is treated as a propensity stratum, and outcomes are modeled within
each state.

## Language

### Data

**Unit**:
One observation — the entity that receives a treatment and has an outcome. Counts of units
follow the sklearn/pandas convention, `n_samples`.
_Avoid_: individual, subject

**Features**:
Pre-treatment covariates of a unit, used to predict both treatment and outcome.
_Avoid_: covariates, X

**Treatment**:
The variable whose causal effect is estimated; binary, continuous, or multi-class.
_Avoid_: exposure, action

**Outcome**:
The response measured after treatment; may be continuous, binary, multivariate, or a
(censored) survival time.
_Avoid_: target, response, y

**Causal Dataset**:
Features, treatment, and outcome for a set of units.

**True UTE / True ATE**:
The known ground-truth effect in a simulated causal dataset, used to evaluate estimates.

### Predictive states

**Predictive State**:
One of `n_states` latent classes learned from the treatment model, within which treatment is
independent of features. The term comes from the predictive state (PRESS) framework.
_Avoid_: cluster, PSS

**Stratum**:
A predictive state viewed as a propensity stratum — the PSPS reading of a predictive state.
Use it only when talking about balance or subclassification; otherwise say predictive state.
_Avoid_: subclass

**State Weights**:
A unit's prior membership probabilities over the predictive states, given its features
(they sum to 1). Hard assignment is the special case of one weight equal to 1.
_Avoid_: weights (unqualified), state probabilities

**State Posterior**:
A unit's membership probabilities over the predictive states, given its features _and_ its
observed treatment.

### Model

**Propensity**:
The distribution of treatment given features, modeled as a mixture over predictive states
weighted by the state weights. Each state has a single, feature-independent treatment
distribution.
_Avoid_: propensity score (when treatment isn't binary)

**State-conditional Outcome Model**:
The model of the outcome given features and treatment, within one predictive state.

**Causal Loss**:
The joint training objective: the outcome likelihood plus `alpha` × the treatment likelihood.

**Alpha**:
The relative weight of the treatment (propensity) likelihood in the causal loss; λ in the
paper.
_Avoid_: propensity penalty, propensity score penalty

### Effects

**Counterfactual Prediction**:
The predicted outcome for a unit with its treatment set by intervention, do(treatment), rather
than observed.

**UTE** (Unit Treatment Effect):
The effect of treatment on a single unit's outcome: the difference between its counterfactual
predictions under two treatment values, mixed over predictive states.
_Avoid_: ITE, CATE

**ATE** (Average Treatment Effect):
The UTE averaged over a population of units.

**Naive ATE**:
The unadjusted difference in mean outcome between treated and untreated units; a baseline that
ignores confounding.
