"""Tests for `pypsps.torch.training`, `metrics`, `callbacks` and `io` (no TensorFlow needed)."""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from ... import bootstrap, inference
from ...torch import callbacks, families, io, losses, metrics, models, training

_N = 160


def _data(seed=0, survival=False):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(_N, 4)).astype("f4")
    a = (rng.rand(_N) < 1 / (1 + np.exp(-x[:, 0]))).astype("f4")
    y = (x[:, 1] + a + 0.3 * rng.normal(size=_N)).astype("f4")
    if survival:
        y = np.stack([np.exp(0.3 * y), (rng.rand(_N) < 0.7).astype("f4")], 1)
    return x, a, y


def _model(**kwargs):
    torch.manual_seed(0)
    return models.build_model(3, 4, families.Normal(), families.Bernoulli(), **kwargs)


def test_fit_logs_the_keras_history_keys_and_lowers_the_loss():
    """History has the Keras metric names, with val_ copies, and the loss goes down."""
    x, a, y = _data()
    history = training.fit(_model(), x, a, y, epochs=8, validation_split=0.25, seed=0)
    keys = set(history.history)
    expected = {
        "loss",
        "causal_loss_metric",
        "outcome_nll_metric",
        "predictive_state_df",
        "uniform_entropy",
        "propensity_score_binary_crossentropy",
        "propensity_score_auroc",
        "propensity_score_aupr",
        "outcome_mean_squared_error",
    }
    assert expected | {f"val_{k}" for k in expected} <= keys
    assert history.epoch == list(range(8))
    assert history.history["loss"][-1] < history.history["loss"][0]


def test_fit_is_reproducible_with_a_seed_and_leaves_the_model_in_eval_mode():
    """Same seed, same weights; fit ends in eval mode."""
    x, a, y = _data()
    first, second = _model(), _model()
    torch.manual_seed(1)  # dropout draws from the global generator; ``seed`` is the shuffling
    training.fit(first, x, a, y, epochs=3, seed=7)
    torch.manual_seed(1)
    training.fit(second, x, a, y, epochs=3, seed=7)
    assert not first.training
    for p, q in zip(first.parameters(), second.parameters()):
        assert torch.equal(p, q)


def test_fit_merges_a_trailing_single_unit_batch():
    """n = k * batch_size + 1 trains: the last unit joins the previous batch, not a batch of one."""
    x, a, y = _data()
    n = 4 * 32 + 1
    model = _model()
    seen = []
    model.features_bn.register_forward_pre_hook(
        lambda module, inputs: seen.append(inputs[0].shape[0]) if module.training else None
    )
    history = training.fit(model, x[:n], a[:n], y[:n], epochs=2, batch_size=32, seed=0)
    assert seen == [32, 32, 32, 33] * 2
    assert np.all(np.isfinite(history.history["loss"]))


def test_validation_split_takes_the_last_rows():
    """validation_split=0.25 validates on the last quarter, as Keras does."""
    x, a, y = _data()
    model = _model()
    split = training.fit(model, x, a, y, epochs=1, validation_split=0.25, seed=0, learning_rate=0.0)
    cut = int(_N * 0.75)
    expected = training.evaluate(model, x[cut:], a[cut:], y[cut:])
    assert split.history["val_causal_loss_metric"][0] == pytest.approx(
        expected["causal_loss_metric"], rel=1e-4
    )


def test_sample_weight_changes_the_fit_and_zero_weight_units_are_ignored():
    """A unit with weight 0 does not affect evaluation; weights change the loss."""
    x, a, y = _data()
    model = _model().eval()
    weights = np.ones(_N, "f4")
    weights[:40] = 0.0
    full = training.evaluate(model, x[40:], a[40:], y[40:])
    masked = training.evaluate(model, x, a, y, weights, weighted_metrics=True)
    assert masked["causal_loss_metric"] == pytest.approx(full["causal_loss_metric"], rel=1e-4)


def test_metrics_ignore_sample_weight_unless_asked():
    """As in Keras, the metrics are unweighted by default; the loss is always weighted."""
    x, a, y = _data()
    model = _model().eval()
    weights = np.linspace(0.1, 3.0, _N).astype("f4")
    plain = training.evaluate(model, x, a, y)
    weighted = training.evaluate(model, x, a, y, weights)
    assert weighted["causal_loss_metric"] == pytest.approx(plain["causal_loss_metric"])
    assert weighted["loss"] != pytest.approx(plain["loss"])
    both = training.evaluate(model, x, a, y, weights, weighted_metrics=True)
    assert both["causal_loss_metric"] != pytest.approx(plain["causal_loss_metric"])


def test_optimizer_and_learning_rate_are_exclusive():
    """Passing an optimizer with learning_rate or clipnorm raises TypeError."""
    x, a, y = _data()
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(TypeError):
        training.fit(model, x, a, y, optimizer=optimizer, learning_rate=0.1)
    training.fit(model, x, a, y, epochs=1, optimizer=optimizer)


def test_clipnorm_clips_each_parameter():
    """Each parameter's gradient norm is clipped on its own."""
    params = [torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.ones(9))]
    params[0].grad = torch.full((4,), 10.0)
    params[1].grad = torch.full((9,), 0.1)
    training._clip_each(params, 1.0)
    assert params[0].grad.norm().item() == pytest.approx(1.0, rel=1e-5)
    assert params[1].grad.norm().item() == pytest.approx(0.3, rel=1e-5)


@pytest.mark.parametrize(
    "bad",
    [{"validation_split": 1.0}, {"batch_size": 0}],
)
def test_fit_rejects_bad_arguments(bad):
    x, a, y = _data()
    with pytest.raises(ValueError):
        training.fit(_model(), x, a, y, **bad)


def test_fit_rejects_mismatched_or_non_finite_data():
    x, a, y = _data()
    with pytest.raises(ValueError, match="rows"):
        training.fit(_model(), x, a[:-1], y)
    with pytest.raises(ValueError, match=r"outcome has 2 column.*reads 1"):
        training.fit(_model(), x, a, np.stack([y, y], 1))
    x_bad = x.copy()
    x_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        training.fit(_model(), x_bad, a, y)


def test_survival_outcome_trains_and_masks_censored_units_in_the_error():
    """A Weibull outcome reads [time, event]; outcome_mean_squared_error counts events only."""
    x, a, y = _data(survival=True)
    model = models.build_model(
        2,
        4,
        families.Weibull(log_scale_bounds=(-2.0, 4.0), log_shape_bounds=(-1.0, 1.0)),
        families.Bernoulli(),
    )
    history = training.fit(model, x, a, y, epochs=2, seed=0)
    assert np.isfinite(history.history["loss"]).all()
    metric = metrics.OutcomeMeanSquaredError(model.mixture)
    y_true = torch.tensor(np.concatenate([y, a[:, None]], 1))
    metric.update(y_true, torch.tensor(training.predict(model, x, a)))
    censored = torch.tensor(np.concatenate([y * [1, 0], a[:, None]], 1))
    metric.reset()
    metric.update(censored, torch.tensor(training.predict(model, x, a)))
    assert np.isnan(metric.result())  # no event seen


def test_normal_treatment_uses_the_treatment_error_metric():
    names = {
        m.name
        for m in metrics.default_metrics(
            models.build_model(2, 4, families.Normal(), families.Normal()).mixture
        )
    }
    assert "treatment_mean_squared_error" in names
    assert "propensity_score_auroc" not in names


def test_auc_matches_the_rank_statistic():
    """The exact ROC AUC equals the Mann-Whitney probability, ties counting one half."""
    model = _model()
    metric = metrics.PropensityScoreAUC(model.mixture)
    rng = np.random.RandomState(1)
    score = np.round(rng.rand(50), 1)
    label = (rng.rand(50) < 0.4).astype("f4")
    pairs = [(s1 > s0) + 0.5 * (s1 == s0) for s1 in score[label == 1] for s0 in score[label == 0]]
    # build a y_pred whose marginal propensity is `score`: one state, prob = score
    one = models.build_model(1, 4, families.Normal(), families.Bernoulli()).mixture
    y_pred = one.layout.encode(
        {"loc": np.zeros((50, 1), "f4"), "scale": np.ones((50, 1), "f4")},
        np.ones((50, 1), "f4"),
        {"prob": score[:, None].astype("f4")},
    )
    y_true = np.stack([np.zeros(50, "f4"), label], 1)
    metric = metrics.PropensityScoreAUC(one)
    metric.update(torch.tensor(y_true), torch.tensor(y_pred))
    assert metric.result() == pytest.approx(np.mean(pairs), abs=1e-6)


def test_alpha_schedule_callback_sets_alpha_each_epoch():
    x, a, y = _data()
    schedule = callbacks.AlphaScheduleCallback(
        alpha_start=5.0, alpha_end=1.0, half_life_epoch=2, steepness=1.0, verbose=False
    )
    history = training.fit(_model(), x, a, y, epochs=6, callbacks=[schedule])
    assert schedule.alpha_history[0] > schedule.alpha_history[-1] >= 1.0
    assert history.loss.alpha == schedule.alpha_history[-1]
    with pytest.raises(ValueError):
        callbacks.AlphaScheduleCallback(steepness=0.0)


def test_gaussian_schedule_requires_epoch_zero():
    schedule = callbacks.GaussianAlphaSchedule(alpha_start=4.0, alpha_end=1.0, verbose=False)
    schedule.set_context(_model(), losses.CausalLoss(_model().mixture), None)
    with pytest.raises(callbacks.MidFitResumeError):
        schedule.on_epoch_begin(3)


def test_eligible_best_checkpoint_restores_the_best_epoch():
    x, a, y = _data()
    model = _model()
    checkpoint = callbacks.EligibleBestCheckpoint(
        "val_causal_loss_metric", patience=2, start_from_epoch=1
    )
    history = training.fit(
        model, x, a, y, epochs=30, validation_split=0.25, callbacks=[checkpoint], seed=0
    )
    assert checkpoint.selected_epoch is not None and checkpoint.selected_epoch >= 1
    best = min(history.history["val_causal_loss_metric"][1:])
    assert checkpoint.selected_value == pytest.approx(best)
    restored = training.evaluate(model, x[-40:], a[-40:], y[-40:])["causal_loss_metric"]
    assert restored == pytest.approx(best, rel=1e-4)
    never = callbacks.EligibleBestCheckpoint("loss", patience=1, start_from_epoch=5)
    with pytest.raises(callbacks.NoEligibleCheckpointError):
        training.fit(_model(), x, a, y, epochs=2, callbacks=[never])


def test_reduce_lr_on_plateau_lowers_the_learning_rate():
    x, a, y = _data()
    plateau = callbacks.ReduceLROnPlateau(
        "val_causal_loss_metric", factor=0.5, patience=1, min_delta=10.0
    )
    history = training.fit(
        _model(), x, a, y, epochs=4, validation_split=0.25, callbacks=[plateau], learning_rate=0.01
    )
    assert history.history["learning_rate"][-1] < 0.01


def test_save_and_load_round_trip(tmp_path):
    x, a, y = _data()
    model = _model(treatment_init={"prob": [0.2, 0.5, 0.8]})
    training.fit(model, x, a, y, epochs=2, seed=0)
    path = tmp_path / "model.pt"
    io.save(model, path)
    loaded = io.load(path)
    np.testing.assert_array_equal(training.predict(model, x, a), training.predict(loaded, x, a))
    torch.save({"format": "other"}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a pypsps"):
        io.load(tmp_path / "other.pt")


def test_load_reads_a_version_1_file_without_max_time(tmp_path):
    """A file written before the Spec carried ``max_time`` (no such key) still loads."""
    x, a, y = _data()
    model = _model()
    training.fit(model, x, a, y, epochs=1, seed=0)
    config = model.get_config()
    del config["max_time"]
    path = tmp_path / "old.pt"
    torch.save(
        {
            "format": "pypsps-torch",
            "version": 1,
            "config": config,
            "state_dict": model.state_dict(),
        },
        path,
    )
    loaded = io.load(path)
    assert loaded.spec.max_time is None
    np.testing.assert_array_equal(training.predict(model, x, a), training.predict(loaded, x, a))


def test_save_and_load_keeps_max_time(tmp_path):
    """``max_time`` survives ``io.save`` and ``io.load``."""
    weibull = families.Weibull(log_scale_bounds=(-2.0, 4.0), log_shape_bounds=(-1.0, 1.0))
    model = models.build_model(2, 4, weibull, families.Bernoulli(), max_time=50.0)
    io.save(model, tmp_path / "weibull.pt")
    assert io.load(tmp_path / "weibull.pt").spec.max_time == 50.0


def test_inference_and_bootstrap_run_on_a_torch_model():
    """The shared inference code accepts a PyTorch model and pandas inputs."""
    x, a, y = _data()
    model = _model()
    training.fit(model, x, a, y, epochs=2, seed=0)
    features = pd.DataFrame(x, columns=list("abcd"))
    ute = inference.predict_ute_binary(model, features)
    assert len(ute) == _N
    assert inference.predict_ate_binary(model, features) == pytest.approx(float(np.mean(ute)))
    samples = bootstrap.bootstrap_ate_binary(model, features, n_bootstraps=3, random_state=0)
    assert samples.notna().all()


def test_predict_batches_and_restores_the_mode():
    x, a, y = _data()
    model = _model().train()
    whole = training.predict(model, x, a, batch_size=1000)
    batched = training.predict(model, x, a, batch_size=7)
    assert model.training
    np.testing.assert_allclose(whole, batched, rtol=1e-5, atol=1e-6)


class _ModeRecorder(torch.nn.Module):
    """A zero regularizer that records whether it was called in training mode."""

    def __init__(self):
        super().__init__()
        self.modes = []

    def forward(self, weights):
        self.modes.append(self.training)
        return weights.sum() * 0.0


def test_validation_and_evaluate_run_the_regularizer_in_eval_mode():
    """Validation batches do not advance the regularizer; evaluate restores its mode."""
    x, a, y = _data()
    recorder = _ModeRecorder()
    training.fit(
        _model(),
        x,
        a,
        y,
        epochs=2,
        batch_size=40,
        validation_split=0.25,
        predictive_states_regularizer=recorder,
    )
    assert recorder.modes == [True] * 3 + [False] + [True] * 3 + [False]

    model = _model()
    recorder = _ModeRecorder().train()
    training.evaluate(model, x, a, y, predictive_states_regularizer=recorder)
    assert recorder.modes == [False]
    assert recorder.training


def _accelerator():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.skipif(_accelerator() is None, reason="needs a CUDA or MPS device")
def test_fit_and_evaluate_with_metrics_on_an_accelerator():
    """The metrics (AUC included) move values to the CPU and need no float64 on the device."""
    x, a, y = _data(survival=True)
    w = np.linspace(0.5, 1.5, _N).astype("f4")
    model = models.build_model(
        3,
        4,
        families.Weibull(log_scale_bounds=(-2.0, 4.0), log_shape_bounds=(-1.0, 1.0)),
        families.Bernoulli(),
    )
    history = training.fit(
        model,
        x,
        a,
        y,
        epochs=1,
        sample_weight=w,
        weighted_metrics=True,
        validation_split=0.25,
        device=_accelerator(),
    )
    assert np.isfinite(history.history["val_propensity_score_auroc"][-1])
    logs = training.evaluate(model, x, a, y, sample_weight=w, weighted_metrics=True)
    assert np.isfinite(logs["outcome_mean_squared_error"])


def test_encode_true_keeps_torch_tensors():
    """A tensor outcome and treatment give a tensor y_true on the same device and dtype."""
    layout = models.build_model(2, 4, families.Weibull(), families.Bernoulli()).layout
    outcome = torch.tensor([[1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
    treatment = torch.tensor([0.0, 1.0, 1.0])

    y = layout.encode_true(outcome, treatment)

    assert isinstance(y, torch.Tensor) and y.shape == (3, 3)
    assert y.device == outcome.device and y.dtype == torch.float32
    np.testing.assert_array_equal(y.numpy(), [[1, 1, 0], [2, 0, 1], [3, 1, 1]])
    with pytest.raises(ValueError, match=r"outcome has 1 column.*reads 2"):
        layout.encode_true(outcome[:, :1], treatment)


def test_steep_schedule_does_not_overflow():
    """A late epoch of a steep schedule is alpha_end, with no OverflowError and no warning."""
    schedule = callbacks.AlphaScheduleCallback(
        alpha_start=10.0, alpha_end=1.0, half_life_epoch=10, steepness=10.0, verbose=False
    )
    schedule.set_context(_model(), losses.CausalLoss(_model().mixture), None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        schedule.on_epoch_begin(100)
    assert schedule.loss.alpha == 1.0
