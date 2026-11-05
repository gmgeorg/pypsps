"""The state-mixture losses equal the frozen v0.3.1 loss math (`_reference_v0_2`).

Values and gradients with respect to ``y_pred`` agree to float32 rounding: TensorFlow's
vectorized float32 log/exp can round an element differently depending on its position, and
the reference loops over [N] columns where the mixture works on [N, K] blocks.
"""

import numpy as np
import pytest
import tensorflow as tf

from ..keras import families, losses, mixture, neglogliks
from . import _reference_v0_2 as ref

# This module also covers the deprecated column-count API on purpose (removed in v0.5.0).
pytestmark = pytest.mark.filterwarnings("ignore:.*removed in v0.5.0:FutureWarning")

_N = 12

_CASES = {
    "normal+bernoulli": (
        families.Normal(),
        families.Bernoulli(),
        neglogliks.NegloglikNormal(reduction="none"),
        tf.keras.losses.BinaryCrossentropy(reduction="none"),
    ),
    "exponential_log_rate+bernoulli": (
        families.Exponential("log_rate"),
        families.Bernoulli(),
        neglogliks.NegloglikExponential(reduction="none", log_rate=True),
        tf.keras.losses.BinaryCrossentropy(reduction="none"),
    ),
    "exponential_log_scale+normal": (
        families.Exponential("log_scale"),
        families.Normal(),
        neglogliks.NegloglikExponentialScale(reduction="none"),
        neglogliks.NegloglikNormal(reduction="none"),
    ),
    "weibull+normal": (
        families.Weibull(),
        families.Normal(),
        neglogliks.NegloglikWeibull(reduction="none"),
        neglogliks.NegloglikNormal(reduction="none"),
    ),
}


def _data(outcome_family, treatment_family, n_states, seed=0):
    """Valid (y_true, y_pred) for the two families."""
    rng = np.random.RandomState(seed)

    def params(family):
        """Values on each parameter's support."""
        out = {}
        for p in family.params:
            if p.activation == "softplus" or p.name == "scale":
                out[p.name] = rng.uniform(0.5, 2.0, (_N, n_states))
            elif p.activation == "sigmoid":
                out[p.name] = rng.uniform(0.05, 0.95, (_N, n_states))
            else:
                out[p.name] = rng.uniform(-1.0, 1.0, (_N, n_states))
        return {k: v.astype("f4") for k, v in out.items()}

    mix = mixture.StateMixture(outcome_family, treatment_family, n_states)
    weights = rng.dirichlet(np.ones(n_states), size=_N).astype("f4")
    y_pred = mix.layout.encode(params(outcome_family), weights, params(treatment_family))
    if isinstance(outcome_family, families.SurvivalFamily):
        y_out = np.stack([rng.uniform(0.5, 5.0, _N), rng.randint(0, 2, _N)], axis=1)
    else:
        y_out = rng.normal(size=(_N, 1))
    if treatment_family.is_binary:
        a = rng.randint(0, 2, (_N, 1))
    else:
        a = rng.normal(size=(_N, 1))
    y_true = np.concatenate([y_out, a], axis=1).astype("f4")
    return mix, tf.constant(y_true), tf.constant(y_pred)


def _value_and_grad(fn, y_pred):
    """fn(y_pred) and d sum(fn) / d y_pred."""
    with tf.GradientTape() as tape:
        tape.watch(y_pred)
        value = fn(y_pred)
        total = tf.reduce_sum(value)
    return value.numpy(), tape.gradient(total, y_pred).numpy()


def _assert_close(new, old):
    """Values and gradients agree to float32 rounding."""
    np.testing.assert_allclose(new[0], old[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(new[1], old[1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_states", [1, 3])
@pytest.mark.parametrize("reduction", ["none", "sum", "sum_over_batch_size"])
@pytest.mark.parametrize("likelihood", ["interventional", "observational"])
@pytest.mark.parametrize("case", list(_CASES))
def test_mixture_matches_reference(case, likelihood, reduction, n_states):
    """StateMixture.outcome_nll / treatment_nll + reduce equal today's loss math."""
    outcome_family, treatment_family, loss, treatment_loss = _CASES[case]
    mix, y_true, y_pred = _data(outcome_family, treatment_family, n_states)
    ints = dict(
        n_outcome_true_cols=outcome_family.n_true_cols,
        n_outcome_pred_cols=len(outcome_family.params),
        n_treatment_pred_cols=len(treatment_family.params),
    )
    _assert_close(
        _value_and_grad(
            lambda y: mixture.reduce(mix.outcome_nll(y_true, y, likelihood), reduction), y_pred
        ),
        _value_and_grad(
            lambda y: ref.outcome_nll(
                loss, treatment_loss, y_true, y, likelihood=likelihood, reduction=reduction, **ints
            ),
            y_pred,
        ),
    )
    _assert_close(
        _value_and_grad(lambda y: mixture.reduce(mix.treatment_nll(y_true, y), reduction), y_pred),
        _value_and_grad(
            lambda y: ref.treatment_nll(treatment_loss, y_true, y, reduction=reduction, **ints),
            y_pred,
        ),
    )


@pytest.mark.parametrize("n_states", [1, 3])
@pytest.mark.parametrize("reduction", ["none", "sum", "sum_over_batch_size"])
@pytest.mark.parametrize("likelihood", ["interventional", "observational"])
@pytest.mark.parametrize("path", ["mixture", "legacy"])
@pytest.mark.parametrize("case", list(_CASES))
def test_losses_match_reference(case, path, likelihood, reduction, n_states):
    """OutcomeLoss / TreatmentLoss, built either way, equal today's loss math."""
    outcome_family, treatment_family, loss, treatment_loss = _CASES[case]
    mix, y_true, y_pred = _data(outcome_family, treatment_family, n_states)
    ints = dict(
        n_outcome_true_cols=outcome_family.n_true_cols,
        n_outcome_pred_cols=len(outcome_family.params),
        n_treatment_pred_cols=len(treatment_family.params),
    )
    if path == "mixture":
        outcome = losses.OutcomeLoss(mix, likelihood, reduction=reduction)
        treatment = losses.TreatmentLoss(mix, reduction=reduction)
    else:
        outcome = losses.OutcomeLoss(
            loss, treatment_loss, outcome_likelihood=likelihood, reduction=reduction, **ints
        )
        treatment = losses.TreatmentLoss(treatment_loss, reduction=reduction, **ints)
    _assert_close(
        _value_and_grad(lambda y: outcome(y_true, y), y_pred),
        _value_and_grad(
            lambda y: ref.outcome_nll(
                loss, treatment_loss, y_true, y, likelihood=likelihood, reduction=reduction, **ints
            ),
            y_pred,
        ),
    )
    _assert_close(
        _value_and_grad(lambda y: treatment(y_true, y), y_pred),
        _value_and_grad(
            lambda y: ref.treatment_nll(treatment_loss, y_true, y, reduction=reduction, **ints),
            y_pred,
        ),
    )
