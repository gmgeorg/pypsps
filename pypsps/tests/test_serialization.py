"""Saving and loading compiled pypsps models, and the loss configs behind it."""

import json
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pypress
import pytest
import tensorflow as tf

from .. import datasets, utils
from ..keras import families, losses, metrics, mixture, models, neglogliks
from .test_models import _continuous_model, _synthetic_continuous_survival


def _small(builder, **extra):
    """A small compiled model from `builder`, plus matching (inputs, outputs)."""
    if builder is models.build_model_binary_exponential:
        data = datasets.CancerSurvivalSimulator(seed=42).sample(n_samples=64)
        extra.setdefault("log_rate_layer", (5, "selu"))
    else:
        np.random.seed(10)
        data = datasets.KangSchafer(true_ate=10).sample(n_samples=64)
        if builder is models.build_model_binary_normal:
            extra.setdefault("loc_layer", (5, "selu"))
            extra.setdefault("scale_layer", (5, "tanh"))
    if builder is not models.build_toy_model:
        extra.setdefault("predictive_state_hidden_layers", [(5, "selu")])
        extra.setdefault("outcome_hidden_layers", [(5, "selu")])
    inputs, outputs = data.to_keras_inputs_outputs()
    tf.random.set_seed(10)
    return builder(n_states=3, n_features=data.n_features, compile=True, **extra), inputs, outputs


def _all_builders():
    """(id, factory) for every builder; each factory returns (model, inputs, outputs)."""

    def continuous(distribution):
        def factory():
            inputs, y_true, _ = _synthetic_continuous_survival(n_samples=64)
            tf.random.set_seed(10)
            return _continuous_model(distribution), inputs, y_true

        return factory

    return [
        ("toy", lambda: _small(models.build_toy_model)),
        ("binary_normal", lambda: _small(models.build_model_binary_normal)),
        ("binary_exponential", lambda: _small(models.build_model_binary_exponential)),
        ("continuous_exponential", continuous("exponential")),
        ("continuous_weibull", continuous("weibull")),
    ]


_BUILDERS = _all_builders()


@pytest.mark.parametrize("factory", [f for _, f in _BUILDERS], ids=[i for i, _ in _BUILDERS])
def test_compiled_model_round_trips_through_load_model(factory, tmp_path):
    """`load_model(path)` restores the loss, its live alpha and every metric."""
    model, inputs, outputs = factory()
    model.loss.alpha = 3.0  # as if saved mid-schedule
    before = model.evaluate(inputs, outputs, verbose=0, return_dict=True)

    path = str(tmp_path / "model.keras")
    model.save(path)
    reloaded = tf.keras.models.load_model(path)

    assert isinstance(reloaded.loss, losses.CausalLoss)
    assert float(reloaded.loss.alpha) == pytest.approx(3.0)
    assert utils.get_column_layout(reloaded) == utils.get_column_layout(model)
    after = reloaded.evaluate(inputs, outputs, verbose=0, return_dict=True)
    assert set(after) == set(before)
    for key, value in before.items():
        np.testing.assert_allclose(after[key], value, rtol=1e-5, err_msg=key)


def test_causal_loss_metric_ignores_the_training_alpha_after_a_reload(tmp_path):
    """The reloaded training loss keeps following alpha; causal_loss_metric still doesn't."""
    model, inputs, outputs = _small(models.build_toy_model)
    path = str(tmp_path / "model.keras")
    model.save(path)
    reloaded = tf.keras.models.load_model(path)

    at_one = reloaded.evaluate(inputs, outputs, verbose=0, return_dict=True)
    reloaded.loss.alpha = 5.0
    at_five = reloaded.evaluate(inputs, outputs, verbose=0, return_dict=True)
    assert at_five["loss"] > at_one["loss"]
    np.testing.assert_allclose(
        at_five["causal_loss_metric"], at_one["causal_loss_metric"], rtol=1e-6
    )


def _causal_loss(alpha=2.0, regularizer=None) -> losses.CausalLoss:
    treatment_nll = tf.keras.losses.BinaryCrossentropy(reduction="none")
    kwargs = dict(
        n_outcome_true_cols=1,
        n_outcome_pred_cols=2,
        n_treatment_pred_cols=1,
        reduction="sum_over_batch_size",
    )
    return losses.CausalLoss(
        outcome_loss=losses.OutcomeLoss(
            loss=neglogliks.NegloglikNormal(reduction="none"),
            treatment_loss=treatment_nll,
            outcome_likelihood="observational",
            **kwargs,
        ),
        treatment_loss=losses.TreatmentLoss(loss=treatment_nll, **kwargs),
        alpha=alpha,
        outcome_loss_weight=0.5,
        predictive_states_regularizer=regularizer,
        reduction="sum_over_batch_size",
    )


def _fixture():
    rng = np.random.RandomState(0)
    weights = tf.nn.softmax(rng.uniform(size=(6, 2)), axis=1).numpy()
    y_pred = np.concatenate(
        [
            rng.uniform(-1, 1, size=(6, 2)),  # loc, per state
            rng.uniform(0.5, 2, size=(6, 2)),  # scale, per state
            weights,
            rng.uniform(0.1, 0.9, size=(6, 2)),  # propensity, per state
        ],
        axis=1,
    ).astype(np.float32)
    y_true = np.stack([rng.normal(size=6), rng.randint(0, 2, size=6)], axis=1).astype(np.float32)
    return tf.constant(y_true), tf.constant(y_pred)


@pytest.mark.parametrize(
    "regularizer",
    [None, pypress.keras.regularizers.DegreesOfFreedom(l2=1.0, target=float(np.log(4.0)))],
    ids=["no_regularizer", "df_regularizer"],
)
def test_causal_loss_config_round_trips(regularizer):
    """The config is JSON; the rebuilt loss has the same settings and value."""
    loss = _causal_loss(regularizer=regularizer)
    config = tf.keras.utils.serialize_keras_object(loss)
    json.dumps(config)
    rebuilt = tf.keras.utils.deserialize_keras_object(config)

    assert type(rebuilt) is losses.CausalLoss
    assert config["config"]["outcome_loss"]["config"]["mixture"]["registered_name"] == (
        "pypsps>StateMixture"
    )
    assert rebuilt._outcome_loss_weight == 0.5
    assert rebuilt._outcome_loss._outcome_likelihood is losses.OutcomeLikelihood.OBSERVATIONAL
    assert (rebuilt._predictive_states_regularizer is None) == (regularizer is None)
    y_true, y_pred = _fixture()
    np.testing.assert_allclose(
        float(rebuilt(y_true, y_pred)), float(loss(y_true, y_pred)), rtol=1e-6
    )


def test_causal_loss_config_stores_the_live_alpha():
    """The config holds alpha's current value; from_config makes a new float32 variable."""
    loss = _causal_loss(alpha=2.0)
    loss.alpha = 5.0
    config = loss.get_config()
    assert config["alpha"] == 5.0

    rebuilt = losses.CausalLoss.from_config(config)
    assert isinstance(rebuilt.alpha, tf.Variable)
    assert rebuilt.alpha.dtype == tf.float32
    assert rebuilt.alpha is not loss.alpha
    assert float(rebuilt.alpha) == 5.0


def test_alpha_property_is_the_variable_and_setter_assigns():
    """`loss.alpha` is the variable; `loss.alpha = x` assigns into it, never rebinds."""
    shared = tf.Variable(1.0, trainable=False, dtype=tf.float32)
    loss = _causal_loss(alpha=shared)
    assert loss.alpha is shared
    loss.alpha = 4.0
    assert loss.alpha is shared
    assert float(shared) == 4.0
    with pytest.raises(TypeError, match="constructor"):
        loss.alpha = tf.Variable(2.0)


def test_underscore_alpha_is_a_deprecated_alias():
    """`_alpha` still reads and writes the variable, with a FutureWarning naming v0.4.0."""
    loss = _causal_loss(alpha=2.0)
    with pytest.warns(FutureWarning, match="v0.4.0"):
        assert loss._alpha is loss.alpha
    with pytest.warns(FutureWarning, match="v0.4.0"):
        loss._alpha = 7.0
    assert float(loss.alpha) == 7.0


def test_load_in_a_fresh_process_after_import_pypsps_keras(tmp_path):
    """A fresh process that only imports pypsps.keras can load a compiled model, and
    TensorFlow Probability is never imported.

    In-process tests can't catch a missing registration, an import cycle, or TFP creeping
    back in, because everything is already imported here.
    """
    model, _, _ = _small(models.build_toy_model)
    path = str(tmp_path / "model.keras")
    model.save(path)
    script = textwrap.dedent(
        f"""
        import sys
        import pypsps.keras
        import tensorflow as tf
        model = tf.keras.models.load_model({path!r})
        assert type(model.loss).__name__ == "CausalLoss", type(model.loss)
        assert "tensorflow_probability" not in sys.modules, "pypsps imported TFP"
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")


_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "three_int"
_FIXTURE_BUILDERS = {
    "toy": _BUILDERS[0][1],
    "binary_normal": _BUILDERS[1][1],
    "binary_exponential": _BUILDERS[2][1],
    "continuous_exponential": _BUILDERS[3][1],
    "continuous_weibull": _BUILDERS[4][1],
}


@pytest.mark.parametrize("name", list(_FIXTURE_BUILDERS))
def test_three_int_configs_still_load(name):
    """A loss config saved by v0.3.0 (three ints, nested NLLs) loads and gives today's value.

    The fixtures were written by v0.3.0's builders with alpha = 2.5. Delete in v0.5.0 with
    the legacy from_config branch.
    """
    config = json.loads((_FIXTURES / f"{name}_loss.json").read_text())
    assert "n_outcome_pred_cols" in config["config"]["outcome_loss"]["config"]
    loaded = tf.keras.utils.deserialize_keras_object(config)
    assert isinstance(loaded, losses.CausalLoss)
    assert float(loaded.alpha) == 2.5
    assert loaded._treatment_loss.mixture is loaded.mixture

    model, inputs, outputs = _FIXTURE_BUILDERS[name]()
    model.loss.alpha = 2.5
    y_pred = tf.constant(model.predict(inputs, verbose=0))
    y_true = tf.constant(outputs, tf.float32)
    np.testing.assert_allclose(
        float(loaded(y_true, y_pred)), float(model.loss(y_true, y_pred)), rtol=1e-6
    )
    # Saving again writes the mixture schema, which loads to the same value.
    again = tf.keras.utils.deserialize_keras_object(tf.keras.utils.serialize_keras_object(loaded))
    assert "mixture" in again._outcome_loss.get_config()
    np.testing.assert_allclose(
        float(again(y_true, y_pred)), float(loaded(y_true, y_pred)), rtol=1e-6
    )


class _PenalizedOutcomeLoss(losses.OutcomeLoss):
    """An OutcomeLoss subclass that adds a penalty (the pattern penalty_free strips)."""

    def __init__(self, *args, penalty: float = 0.0, **kwargs):
        """Stores the penalty."""
        super().__init__(*args, **kwargs)
        self.penalty = penalty

    def get_config(self):
        """Adds the penalty."""
        return {**super().get_config(), "penalty": self.penalty}

    def call(self, y_true, y_pred):
        """The outcome loss plus the penalty."""
        return super().call(y_true, y_pred) + self.penalty


def test_penalty_free_keeps_the_mixture():
    """penalty_free() drops the penalty and keeps an equal mixture."""
    mix = mixture.StateMixture(families.Weibull((-2.0, 5.0), (-1.0, 1.0)), families.Normal(), 2)
    original = _PenalizedOutcomeLoss(mix, "observational", penalty=3.0)
    clean = original.penalty_free()
    assert type(clean) is _PenalizedOutcomeLoss and clean.penalty == 0.0
    assert clean.mixture == original.mixture
    assert clean.outcome_likelihood is losses.OutcomeLikelihood.OBSERVATIONAL


def test_causal_loss_from_config_shares_one_mixture():
    """After a round trip both losses hold the same mixture object, as when built."""
    mix = mixture.StateMixture(families.Normal(), families.Bernoulli(), n_states=2)
    loss = losses.CausalLoss.from_mixture(mix, alpha=2.0)
    rebuilt = tf.keras.utils.deserialize_keras_object(tf.keras.utils.serialize_keras_object(loss))
    assert rebuilt.mixture == mix
    assert rebuilt._outcome_loss.mixture is rebuilt._treatment_loss.mixture


def _metric_value(metric, y_true, y_pred) -> float:
    """One-shot value of a stateful metric: reset, update once, read."""
    metric.reset_state()
    metric.update_state(y_true, y_pred)
    return float(metric.result())


@pytest.mark.parametrize("name", list(_FIXTURE_BUILDERS))
def test_three_int_metric_configs_still_load(name):
    """The compiled metrics saved by v0.3.0 (three ints, int generator kwargs) load, and each
    equals today's metric of the same name.

    The values agree because v0.3.0's builders only used counts that the shape rule maps
    onto the right family: Normal (2, 1) and the 1-parameter identity. Delete in v0.5.0.
    """
    configs = json.loads((_FIXTURES / f"{name}_metrics.json").read_text())
    model, inputs, outputs = _FIXTURE_BUILDERS[name]()
    current = {
        m.name: m
        for m in map(tf.keras.utils.deserialize_keras_object, model.get_compile_config()["metrics"])
    }
    y_pred = tf.constant(model.predict(inputs, verbose=0))
    y_true = tf.constant(outputs, tf.float32)
    assert sorted(c["config"]["name"] for c in configs) == sorted(current)
    for config in configs:
        assert "mixture" not in config["config"]
        loaded = tf.keras.utils.deserialize_keras_object(config)
        if isinstance(loaded, metrics.OutcomeMeanSquaredError):
            assert loaded.mixture.outcome_family == families.Normal()
        np.testing.assert_allclose(
            _metric_value(loaded, y_true, y_pred),
            _metric_value(current[loaded.name], y_true, y_pred),
            rtol=1e-5,
            err_msg=loaded.name,
        )


@pytest.mark.parametrize("builder_id", ["toy", "binary_normal", "binary_exponential"])
def test_propensity_metric_name_aupr(builder_id):
    """The binary builders compile the PR-curve AUC as propensity_score_aupr (decision B1b),
    and every mixture metric shares the loss's mixture."""
    model, _, _ = dict(_BUILDERS)[builder_id]()
    compiled = model._compile_config.config["metrics"]
    aupr = [m for m in compiled if m.name == "propensity_score_aupr"]
    assert len(aupr) == 1 and aupr[0].curve.name == "PR"
    for metric in compiled:
        if isinstance(metric, metrics._MixtureMetricMixin):
            assert metric.mixture is model.loss.mixture
