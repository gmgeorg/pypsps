"""Both backends' callbacks follow the same training policy.

The callbacks are driven with scripted inputs: there is no fit, and the models are stand-ins
that hold a real ``CausalLoss``, weights, or an optimizer.
"""

import math

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("torch")

import tensorflow as tf  # noqa: E402
import torch  # noqa: E402

from ...core import alpha_schedule  # noqa: E402
from ...keras import callbacks as kc  # noqa: E402
from ...keras import families as kf  # noqa: E402
from ...keras import losses as kl  # noqa: E402
from ...keras import mixture as kmix  # noqa: E402
from ...keras import models as kmodels  # noqa: E402
from ...torch import callbacks as tc  # noqa: E402
from ...torch import families as tf_  # noqa: E402
from ...torch import losses as tl  # noqa: E402
from ...torch import metrics as tm  # noqa: E402
from ...torch import mixture as tmix  # noqa: E402

_SIGMOID = {
    "decay": dict(alpha_start=10.0, alpha_end=1.0, half_life_epoch=10, steepness=0.5),
    "increase": dict(alpha_start=1.0, alpha_end=5.0, half_life_epoch=5, steepness=0.2),
    "constant": dict(alpha_start=3.0, alpha_end=3.0),
    "steep": dict(alpha_start=10.0, alpha_end=1.0, half_life_epoch=10, steepness=10.0),
    "no_clamp": dict(alpha_start=10.0, alpha_end=1.0, steepness=0.3, clamp_threshold=0.0),
}


class _KerasModel:
    """A stand-in compiled Keras model: a real loss, and weights that are the last epoch."""

    def __init__(self, alpha=1.5):
        """Hold a CausalLoss and record the restored weights."""
        mixture = kmix.StateMixture(kf.Normal(), kf.Bernoulli(), n_states=2)
        self.loss = kl.CausalLoss.from_mixture(mixture, alpha=alpha)
        self.epoch, self.restored, self.stop_training = -1, None, False

    def get_weights(self):
        """The current epoch."""
        return [np.array(self.epoch)]

    def set_weights(self, weights):
        """Record the restored epoch."""
        self.restored = int(weights[0])


class _TorchModel(torch.nn.Module):
    """A stand-in model whose only parameter is the last epoch."""

    def __init__(self):
        """Hold the epoch as a parameter, so it is in the state dict."""
        super().__init__()
        self.epoch_value = torch.nn.Parameter(torch.tensor(-1.0))
        self.stop_training = False

    @property
    def restored(self):
        """The epoch whose weights are loaded now."""
        return int(self.epoch_value.item())


def _torch_loss(alpha=1.5):
    """A real torch CausalLoss."""
    mixture = tmix.StateMixture(tf_.Normal(), tf_.Bernoulli(), n_states=2)
    return tl.CausalLoss(mixture, alpha=alpha)


def _same(a, b):
    """Equality that treats NaN as equal to NaN."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _trace(cb):
    """The checkpoint trace with NaN replaced by a string, so rows compare equal."""
    return [
        {k: ("nan" if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        for row in cb.trace
    ]


@pytest.mark.parametrize("case", list(_SIGMOID))
def test_sigmoid_schedules_assign_the_same_alphas(case):
    """50 epochs give equal histories, and the live alpha of each loss follows them."""
    kcb = kc.AlphaScheduleCallback(verbose=False, **_SIGMOID[case])
    tcb = tc.AlphaScheduleCallback(verbose=False, **_SIGMOID[case])
    kmodel, tloss = _KerasModel(), _torch_loss()
    kcb.set_model(kmodel)
    tcb.set_context(None, tloss, None)
    for epoch in range(50):
        kcb.on_epoch_begin(epoch)
        tcb.on_epoch_begin(epoch)
        assert float(kmodel.loss.alpha.numpy()) == pytest.approx(tloss.alpha, rel=1e-6)
    assert kcb.alpha_history == tcb.alpha_history
    assert len(kcb.alpha_history) == 50
    assert (kcb.is_decay, kcb.is_increase, kcb.is_constant) == (
        tcb.is_decay,
        tcb.is_increase,
        tcb.is_constant,
    )


def test_gaussian_schedules_assign_the_same_alphas_and_refuse_a_resume():
    """The half-Gaussian schedule agrees, and both refuse a first epoch other than 0."""
    kcb = kc.GaussianAlphaSchedule(alpha_start=6.0, decay_const=0.3, verbose=False)
    tcb = tc.GaussianAlphaSchedule(alpha_start=6.0, decay_const=0.3, verbose=False)
    kmodel, tloss = _KerasModel(), _torch_loss()
    kcb.set_model(kmodel)
    tcb.set_context(None, tloss, None)
    kcb.on_train_begin()
    tcb.on_train_begin()
    for epoch in range(50):
        kcb.on_epoch_begin(epoch)
        tcb.on_epoch_begin(epoch)
    assert kcb.alpha_history == tcb.alpha_history
    assert float(kmodel.loss.alpha.numpy()) == pytest.approx(tloss.alpha, rel=1e-6)
    assert kcb.warmup_epochs == tcb.warmup_epochs
    kcb.on_train_begin()
    tcb.on_train_begin()
    with pytest.raises(alpha_schedule.MidFitResumeError):
        kcb.on_epoch_begin(2)
    with pytest.raises(alpha_schedule.MidFitResumeError):
        tcb.on_epoch_begin(2)


_NAN, _INF = float("nan"), float("inf")
_SERIES = {
    "normal": ([5.0, 4.0, 3.0, 3.5, 2.0, 2.5, 2.6, 2.7, 2.8], 0, 3),
    "ties": ([3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0], 0, 3),
    "start_from_epoch": ([0.1, 0.2, 5.0, 4.0, 4.5, 4.6, 4.7, 1.0], 2, 3),
    "nan_and_inf": ([1.0, _NAN, 3.0, _INF, 2.0, _NAN, 2.0, 2.0], 2, 3),
    "all_nan": ([_NAN, _NAN, _NAN], 0, 2),
    "nan_before_start": ([_NAN, 1.0, 2.0, 2.0, 2.0], 1, 2),
    "none_eligible": ([1.0, 1.0, 1.0], 5, 2),
}


def _run_keras(cb, series, logs=None):
    """Drive a Keras checkpoint callback; return its outcome."""
    model = _KerasModel()
    cb.set_model(model)
    cb.on_train_begin()
    try:
        for e, v in enumerate(series):
            model.epoch = e
            cb.on_epoch_end(e, {"val_m": v} if logs is None else logs)
            if model.stop_training:
                break
        cb.on_train_end()
    except (KeyError, kc.NoEligibleCheckpointError) as exc:
        return type(exc), None
    return None, model.restored


def _run_torch(cb, series, logs=None):
    """Drive a torch checkpoint callback; return its outcome."""
    model = _TorchModel()
    cb.set_context(model, _torch_loss(), None)
    cb.on_train_begin()
    try:
        for e, v in enumerate(series):
            with torch.no_grad():
                model.epoch_value.fill_(float(e))
            cb.on_epoch_end(e, {"val_m": v} if logs is None else logs)
            if model.stop_training:
                break
        cb.on_train_end()
    except (KeyError, tc.NoEligibleCheckpointError) as exc:
        return type(exc), None
    return None, model.restored


@pytest.mark.parametrize("case", list(_SERIES))
@pytest.mark.parametrize("kind", ["eligible", "early_stopping"])
def test_checkpoint_selects_the_same_epoch(case, kind):
    """Trace, selected and stopped epochs, restored weights and errors agree."""
    series, start, patience = _SERIES[case]
    if kind == "early_stopping":
        start = 0
        make = lambda mod: mod.EarlyStopping("val_m", patience=patience)  # noqa: E731
    else:
        make = lambda mod: mod.EligibleBestCheckpoint(  # noqa: E731
            "val_m", patience=patience, start_from_epoch=start
        )
    kcb, tcb = make(kc), make(tc)
    assert _run_keras(kcb, series) == _run_torch(tcb, series)
    assert _trace(kcb) == _trace(tcb)
    for name in ("selected_epoch", "selected_value", "stopped_epoch", "best_epoch", "wait"):
        assert _same(getattr(kcb, name), getattr(tcb, name)), name


def test_a_missing_monitor_raises_the_same_error():
    """The default early stopping raises KeyError in both backends."""
    for cb_pair in (
        (kc.get_default_callbacks("val_m", 2)[0], tc.get_default_callbacks("val_m", 2)[0]),
        (kc.EligibleBestCheckpoint("val_m", 2), tc.EligibleBestCheckpoint("val_m", 2)),
    ):
        assert _run_keras(cb_pair[0], [1.0], logs={"loss": 1.0}) == (KeyError, None)
        assert _run_torch(cb_pair[1], [1.0], logs={"loss": 1.0}) == (KeyError, None)


def test_the_error_classes_are_shared():
    """One class each, for both backends."""
    assert kc.NoEligibleCheckpointError is tc.NoEligibleCheckpointError
    assert kc.MidFitResumeError is tc.MidFitResumeError


def test_default_callbacks_match():
    """The same roles in the same order, with the same arguments read back from the instances."""
    kwargs = dict(
        monitor="val_m",
        patience=10,
        verbose=True,
        alpha_start=8.0,
        alpha_end=2.0,
        half_life_epoch=4,
        steepness=0.2,
    )
    kcbs, tcbs = kc.get_default_callbacks(**kwargs), tc.get_default_callbacks(**kwargs)
    assert [type(c).__name__ for c in kcbs] == [type(c).__name__ for c in tcbs]
    assert [type(c).__name__ for c in kcbs] == [
        "EarlyStopping",
        "ReduceLROnPlateau",
        "AlphaScheduleCallback",
    ]
    stop = ("monitor", "patience", "start_from_epoch", "min_delta", "verbose")
    plateau = ("monitor", "factor", "patience", "min_lr", "min_delta", "cooldown", "verbose")
    sched = ("alpha_start", "alpha_end", "half_life_epoch", "steepness", "clamp_threshold")
    for names, k, t in zip((stop, plateau, sched), kcbs, tcbs):
        for name in names:
            assert getattr(k, name) == pytest.approx(getattr(t, name)), name
    assert kcbs[0].patience == 10 and kcbs[1].patience == 5 and kcbs[1].factor == 0.2
    assert kcbs[2].verbose == tcbs[2].verbose is True


_PLATEAUS = {
    "plain": dict(patience=2, factor=0.5, min_delta=1e-4),
    "cooldown": dict(patience=1, factor=0.5, cooldown=2),
    "min_lr": dict(patience=1, factor=0.1, min_lr=0.003),
    "default_min_delta": dict(patience=2, factor=0.2, min_lr=1e-6),
}
_PLATEAU_LOGS = [5, 4, 4, 4, 4, 3.9, 3.9999, 3.9, 3.9, 3.9, 3.9, 3.9, 2, 2, 2, 2, 2, 2, 2, 2]


@pytest.mark.parametrize("case", list(_PLATEAUS))
def test_reduce_lr_on_plateau_matches(case):
    """The learning rate after each epoch is equal under plateaus, a cooldown and min_lr."""
    kwargs = _PLATEAUS[case]
    kcb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_m", mode="min", **kwargs)
    tcb = tc.ReduceLROnPlateau("val_m", **kwargs)

    kmodel = _KerasModel()
    kmodel.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    kcb.set_model(kmodel)
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.1)
    tcb.set_context(None, None, optimizer)

    kcb.on_train_begin()
    tcb.on_train_begin()
    klr, tlr = [], []
    for e, v in enumerate(_PLATEAU_LOGS):
        kcb.on_epoch_end(e, {"val_m": v})
        tcb.on_epoch_end(e, {"val_m": v})
        klr.append(float(np.asarray(kmodel.optimizer.learning_rate)))
        tlr.append(optimizer.param_groups[0]["lr"])
    np.testing.assert_allclose(klr, tlr, rtol=1e-6)
    assert klr[-1] < 0.1


def _families(module, outcome, treatment):
    """A (outcome, treatment) family pair of ``module`` by name."""
    makers = {
        "Normal": lambda: module.Normal(),
        "Bernoulli": lambda: module.Bernoulli(),
        "Exponential": lambda: module.Exponential("log_scale"),
        "Weibull": lambda: module.Weibull(
            log_scale_bounds=(-2.0, 4.0), log_shape_bounds=(-1.0, 1.0)
        ),
    }
    return makers[outcome](), makers[treatment]()


@pytest.mark.parametrize("treatment", ["Bernoulli", "Normal"])
@pytest.mark.parametrize("outcome", ["Normal", "Exponential", "Weibull"])
def test_default_metric_names_match(outcome, treatment):
    """Both backends log the same default metrics, in the same order."""
    kmixture = kmix.StateMixture(*_families(kf, outcome, treatment), n_states=3)
    tmixture = tmix.StateMixture(*_families(tf_, outcome, treatment), n_states=3)
    keras_names = [m.name for m in kmodels.default_metrics(kl.CausalLoss.from_mixture(kmixture))]
    torch_names = [m.name for m in tm.default_metrics(tmixture)]
    assert keras_names == torch_names
