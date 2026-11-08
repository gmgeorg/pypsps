"""PyTorch metrics, evaluation and training agree with Keras."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("torch")


from ...keras import families as kf  # noqa: E402
from ...keras import models as kmodels  # noqa: E402
from ...keras.layout import ColumnLayout  # noqa: E402
from ...torch import models as tmodels  # noqa: E402
from ...torch import training  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

_N = 200


def _data(survival=False, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(_N, 5)).astype("f4")
    a = (rng.rand(_N) < 1 / (1 + np.exp(-x[:, 0]))).astype("f4")
    y = np.exp(0.3 * x[:, 1] + 0.5 * a + 0.3 * rng.normal(size=_N)).astype("f4")
    outcome = np.stack([y, (rng.rand(_N) < 0.7).astype("f4")], 1) if survival else y[:, None]
    return x, a, outcome


_CASES = {
    "normal": (lambda: kf.Normal(), False),
    "exponential": (lambda: kf.Exponential("log_scale"), True),
    "weibull": (
        lambda: kf.Weibull(log_scale_bounds=(-2.0, 4.0), log_shape_bounds=(-1.0, 1.0)),
        True,
    ),
}


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("likelihood", ["interventional", "observational"])
@pytest.mark.parametrize("weighted", [False, True])
def test_evaluate_matches_keras(case, likelihood, weighted):
    """Loss and every shared metric of the converted model equal Keras ``evaluate``."""
    make, survival = _CASES[case]
    x, a, outcome = _data(survival)
    sw = np.random.RandomState(3).uniform(0.5, 2.0, _N).astype("f4") if weighted else None
    keras_model = kmodels.build_model(
        3, 5, make(), kf.Bernoulli(), compile=False, l1=0.01, outcome_init={}
    )
    kmodels.compile_model(
        keras_model, alpha=1.5, outcome_loss_weight=0.7, outcome_likelihood=likelihood
    )
    y_true = ColumnLayout.from_model(keras_model).encode_true(outcome, a)
    expected = keras_model.evaluate(
        [x, a], y_true, sample_weight=sw, batch_size=64, verbose=0, return_dict=True
    )
    model = tmodels.PSPSModel.from_keras(keras_model)
    actual = training.evaluate(
        model,
        x,
        a,
        outcome,
        sw,
        alpha=1.5,
        outcome_loss_weight=0.7,
        outcome_likelihood=likelihood,
        batch_size=64,
    )
    tolerance = {"propensity_score_auroc": 0.02, "propensity_score_aupr": 0.03}
    shared = sorted(set(expected) & set(actual))
    assert {"loss", "causal_loss_metric", "outcome_nll_metric", "predictive_state_df"} <= set(
        shared
    )
    for name in shared:
        assert actual[name] == pytest.approx(
            expected[name], rel=1e-3, abs=tolerance.get(name, 1e-4)
        ), name
    assert set(actual) - set(expected) == set()


def test_same_fit_reaches_a_similar_loss_as_keras():
    """Brief training from the same start lowers the causal loss alike in both backends."""
    x, a, outcome = _data()
    keras_model = kmodels.build_model(
        3, 5, kf.Normal(), kf.Bernoulli(), compile=False, dropout_rate=0.0
    )
    kmodels.compile_model(keras_model, learning_rate=0.01)
    y_true = ColumnLayout.from_model(keras_model).encode_true(outcome, a)
    model = tmodels.PSPSModel.from_keras(keras_model)
    initial = training.evaluate(model, x, a, outcome)["causal_loss_metric"]
    keras_model.fit([x, a], y_true, epochs=40, batch_size=32, verbose=0)
    expected = keras_model.evaluate([x, a], y_true, verbose=0, return_dict=True)
    training.fit(model, x, a, outcome, epochs=40, learning_rate=0.01, seed=0)
    final = training.evaluate(model, x, a, outcome)["causal_loss_metric"]
    assert final < initial
    assert expected["causal_loss_metric"] < initial
    # the optimizers differ in small ways (Nadam), so compare the size of the improvement
    gain_keras = initial - expected["causal_loss_metric"]
    assert (initial - final) == pytest.approx(gain_keras, rel=0.35)
