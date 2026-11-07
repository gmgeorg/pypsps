"""The PyTorch state mixture and causal loss equal the Keras ones on identical arrays."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("torch")

import tensorflow as tf  # noqa: E402
import torch  # noqa: E402

from ...keras import families as kf  # noqa: E402
from ...keras import losses as klosses  # noqa: E402
from ...keras import mixture as kmix  # noqa: E402
from ...torch import families as tf_  # noqa: E402
from ...torch import losses as tlosses  # noqa: E402
from ...torch import mixture as tmix  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:Setting .reduction. of")

_N, _K = 24, 3

_CASES = {
    "normal+bernoulli": (lambda m: m.Normal(), lambda m: m.Bernoulli()),
    "exp_log_rate+bernoulli": (lambda m: m.Exponential("log_rate"), lambda m: m.Bernoulli()),
    "exp_log_scale+normal": (lambda m: m.Exponential("log_scale"), lambda m: m.Normal()),
    "exp_rate+normal": (lambda m: m.Exponential("rate"), lambda m: m.Normal()),
    "weibull+bernoulli": (lambda m: m.Weibull(), lambda m: m.Bernoulli()),
    "bernoulli+normal": (lambda m: m.Bernoulli(), lambda m: m.Normal()),
}


def _data(layout, outcome_name, seed=0):
    """Random ``y_true`` / ``y_pred`` valid for every family (probabilities in (0, 1))."""
    rng = np.random.RandomState(seed)
    width = layout.y_pred_width()
    y_pred = rng.uniform(0.2, 0.9, (_N, width)).astype("f4")
    weights = rng.dirichlet(np.ones(_K), _N).astype("f4")
    first = layout.n_outcome_pred_cols
    y_pred[:, first : first + _K] = weights
    c = layout.n_outcome_true_cols
    outcome = rng.uniform(0.3, 3.0, (_N, c)).astype("f4")
    if c == 2:  # [time, event]
        outcome[:, 1] = rng.randint(0, 2, _N)
    if outcome_name == "bernoulli":
        outcome = rng.randint(0, 2, (_N, c)).astype("f4")
    treatment = rng.randint(0, 2, (_N, 1)).astype("f4")
    y_true = np.concatenate([outcome, treatment], axis=1)
    return y_true, y_pred


def _pair(case):
    out, trt = _CASES[case]
    k = kmix.StateMixture(out(kf), trt(kf), n_states=_K)
    t = tmix.StateMixture(out(tf_), trt(tf_), n_states=_K)
    return k, t


def _np(x):
    return x.detach().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("likelihood", ["interventional", "observational"])
def test_mixture_values_match_keras(case, likelihood):
    """Per-state NLLs, the posterior, the mixture NLLs and the means agree with Keras."""
    k, t = _pair(case)
    assert k.layout.y_pred_width() == t.layout.y_pred_width()
    y_true, y_pred = _data(k.layout, case.split("+")[0])
    kt, kp = tf.constant(y_true), tf.constant(y_pred)
    tt, tp = torch.tensor(y_true), torch.tensor(y_pred)

    kn, tn = k.nll_per_state(kt, kp), t.nll_per_state(tt, tp)
    np.testing.assert_allclose(_np(tn.outcome), kn.outcome.numpy(), rtol=2e-4, atol=1e-4)
    np.testing.assert_allclose(_np(tn.treatment), kn.treatment.numpy(), rtol=2e-4, atol=1e-4)
    np.testing.assert_allclose(
        _np(t.state_posterior(tt, tp)), k.state_posterior(kt, kp).numpy(), rtol=2e-4, atol=1e-5
    )
    for name in ("outcome_nll",):
        np.testing.assert_allclose(
            _np(getattr(t, name)(tt, tp, likelihood)),
            getattr(k, name)(kt, kp, likelihood).numpy(),
            rtol=2e-4,
            atol=1e-4,
        )
    np.testing.assert_allclose(
        _np(t.treatment_nll(tt, tp)), k.treatment_nll(kt, kp).numpy(), rtol=2e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        _np(t.causal_nll(tt, tp, likelihood, 0.5)),
        k.causal_nll(kt, kp, likelihood, 0.5).numpy(),
        rtol=2e-4,
        atol=1e-4,
    )
    for per_state in (False, True):
        np.testing.assert_allclose(
            _np(t.outcome_mean(tp, per_state)),
            k.outcome_mean(kp, per_state).numpy(),
            rtol=2e-4,
            atol=1e-5,
        )
    np.testing.assert_allclose(
        _np(t.treatment_mean(tp)), k.treatment_mean(kp).numpy(), rtol=2e-4, atol=1e-5
    )


@pytest.mark.parametrize("case", ["normal+bernoulli", "weibull+bernoulli"])
@pytest.mark.parametrize("likelihood", ["interventional", "observational"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reduction", ["sum_over_batch_size", "sum", "none"])
def test_causal_loss_and_gradient_match_keras(case, likelihood, weighted, reduction):
    """The reduced loss and its gradient w.r.t. y_pred agree, with and without sample weights."""
    k, t = _pair(case)
    y_true, y_pred = _data(k.layout, case.split("+")[0], seed=1)
    sw = np.random.RandomState(2).uniform(0.2, 2.0, _N).astype("f4") if weighted else None

    kloss = klosses.CausalLoss(
        klosses.OutcomeLoss(k, outcome_likelihood=likelihood, reduction="none"),
        klosses.TreatmentLoss(k, reduction="none"),
        alpha=1.7,
        outcome_loss_weight=0.8,
        reduction=reduction,
    )
    tloss = tlosses.CausalLoss(
        t, alpha=1.7, outcome_loss_weight=0.8, outcome_likelihood=likelihood, reduction=reduction
    )

    kp = tf.Variable(y_pred)
    with tf.GradientTape() as tape:
        kv = kloss(tf.constant(y_true), kp, sample_weight=sw)
        kscalar = tf.reduce_sum(kv)
    kgrad = tape.gradient(kscalar, kp).numpy()

    tp = torch.tensor(y_pred, requires_grad=True)
    tv = tloss(torch.tensor(y_true), tp, sample_weight=None if sw is None else torch.tensor(sw))
    tv.sum().backward()

    np.testing.assert_allclose(_np(tv), kv.numpy(), rtol=2e-4, atol=1e-4)
    np.testing.assert_allclose(tp.grad.numpy(), kgrad, rtol=5e-3, atol=5e-4)


def test_alpha_is_settable_and_validated():
    """``loss.alpha = x`` changes the next call; a non-finite alpha raises."""
    _, t = _pair("normal+bernoulli")
    loss = tlosses.CausalLoss(t, alpha=1.0)
    y_true, y_pred = _data(t.layout, "normal")
    base = loss(torch.tensor(y_true), torch.tensor(y_pred))
    loss.alpha = 3.0
    assert loss(torch.tensor(y_true), torch.tensor(y_pred)) > base
    with pytest.raises(ValueError):
        loss.alpha = float("nan")
