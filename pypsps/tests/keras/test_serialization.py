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

from ... import datasets, inference
from ...keras import families, losses, metrics, mixture, models, presets
from ...keras.layout import ColumnLayout
from ._continuous import continuous_model, synthetic_continuous_survival


def _small(preset, **extra):
    """A small compiled model from `presets.<preset>`, plus matching (inputs, outputs)."""
    if preset == "binary_exponential":
        data = datasets.CancerSurvivalSimulator(seed=42).sample(n_samples=64)
        extra.setdefault("log_rate_layer", (5, "selu"))
    else:
        np.random.seed(10)
        data = datasets.KangSchafer(true_ate=10).sample(n_samples=64)
        if preset == "binary_normal":
            extra.setdefault("loc_layer", (5, "selu"))
            extra.setdefault("scale_layer", (5, "tanh"))
    if preset != "toy":
        extra.setdefault("predictive_state_hidden_layers", [(5, "selu")])
        extra.setdefault("outcome_hidden_layers", [(5, "selu")])
    inputs, outputs = data.to_keras_inputs_outputs()
    tf.random.set_seed(10)
    arguments = getattr(presets, preset)(
        n_states=3, n_features=data.n_features, compile=True, **extra
    )
    return models.build_model(**arguments), inputs, outputs


def _all_builders():
    """(id, factory) for every builder; each factory returns (model, inputs, outputs)."""

    def continuous(distribution):
        def factory():
            inputs, y_true, _ = synthetic_continuous_survival(n_samples=64)
            tf.random.set_seed(10)
            return continuous_model(distribution), inputs, y_true

        return factory

    return [
        ("toy", lambda: _small("toy")),
        ("binary_normal", lambda: _small("binary_normal")),
        ("binary_exponential", lambda: _small("binary_exponential")),
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
    assert ColumnLayout.from_model(reloaded) == ColumnLayout.from_model(model)
    after = reloaded.evaluate(inputs, outputs, verbose=0, return_dict=True)
    assert set(after) == set(before)
    for key, value in before.items():
        np.testing.assert_allclose(after[key], value, rtol=1e-5, err_msg=key)


def test_causal_loss_metric_ignores_the_training_alpha_after_a_reload(tmp_path):
    """The reloaded training loss keeps following alpha; causal_loss_metric still doesn't."""
    model, inputs, outputs = _small("toy")
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
    """A Normal/Bernoulli K=2 causal loss with the observational likelihood."""
    return losses.CausalLoss.from_mixture(
        mixture.StateMixture(families.Normal(), families.Bernoulli(), n_states=2),
        alpha=alpha,
        outcome_likelihood="observational",
        outcome_loss_weight=0.5,
        predictive_states_regularizer=regularizer,
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


def test_underscore_alpha_is_removed():
    """The deprecated `_alpha` alias is gone in v0.4.0; `alpha` is the variable."""
    loss = _causal_loss(alpha=2.0)
    assert not hasattr(loss, "_alpha")


def test_load_in_a_fresh_process_after_import_pypsps_keras(tmp_path):
    """A fresh process that only imports pypsps.keras can load a compiled model, and
    TensorFlow Probability is never imported.

    In-process tests can't catch a missing registration, an import cycle, or TFP creeping
    back in, because everything is already imported here.
    """
    model, _, _ = _small("toy")
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
_FIXTURE_NAMES = [
    "toy",
    "binary_normal",
    "binary_exponential",
    "continuous_exponential",
    "continuous_weibull",
]


@pytest.mark.parametrize("name", _FIXTURE_NAMES)
def test_three_int_loss_configs_raise(name):
    """A loss config saved by v0.3.0 (three ints, nested NLLs) raises since v0.5.0, naming the
    migration: load with compile=False and compile again."""
    config = json.loads((_FIXTURES / f"{name}_loss.json").read_text())
    assert "n_outcome_pred_cols" in config["config"]["outcome_loss"]["config"]
    with pytest.raises(ValueError, match="no 'mixture'.*compile=False"):
        tf.keras.utils.deserialize_keras_object(config)


@pytest.mark.parametrize("name", _FIXTURE_NAMES)
def test_three_int_metric_configs_raise(name):
    """Every pypsps metric config saved by v0.3.0 raises the same way."""
    configs = json.loads((_FIXTURES / f"{name}_metrics.json").read_text())
    pypsps_configs = [c for c in configs if (c.get("registered_name") or "").startswith("pypsps>")]
    assert pypsps_configs
    for config in pypsps_configs:
        with pytest.raises(ValueError, match="no 'mixture'.*compile=False"):
            tf.keras.utils.deserialize_keras_object(config)


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


@pytest.mark.parametrize("factory", [f for _, f in _BUILDERS], ids=[i for i, _ in _BUILDERS])
def test_compile_false_reload_runs_inference(factory, tmp_path):
    """A `compile=False` reload reads the mixture off its CausalOutput layer for inference."""
    model, inputs, _ = factory()
    features = np.asarray(inputs[0])
    binary = model.loss.mixture.treatment_family.is_binary

    def estimate(m):
        if binary:
            return inference.predict_ute_binary(m, features)
        return inference.predict_ate_continuous(m, features, treatment_grid=[-1.0, 0.0, 1.0])

    path = str(tmp_path / "model.keras")
    model.save(path)
    reloaded = tf.keras.models.load_model(path, compile=False)

    assert reloaded.loss is None
    assert mixture.StateMixture.from_model(reloaded) == model.loss.mixture
    np.testing.assert_allclose(estimate(reloaded), estimate(model), rtol=1e-5, atol=1e-6)


_WEIGHTS_LAYOUT_DIR = pathlib.Path(__file__).parent / "fixtures" / "weights_layout"
_HIDDEN = dict(predictive_state_hidden_layers=[(5, "selu")], outcome_hidden_layers=[(5, "selu")])
# Binary and toy weight files changed in v0.4.0 (one treatment head) and don't load.
_WEIGHTS_LAYOUT_BUILDERS = {
    "continuous_weibull": lambda: models.build_model(
        **presets.continuous_weibull(
            n_states=3,
            n_features=4,
            log_scale_bounds=(-3.0, 6.0),
            log_shape_bounds=(-2.0, 2.0),
            **_HIDDEN,
        )
    ),
}


@pytest.mark.parametrize("name", sorted(_WEIGHTS_LAYOUT_BUILDERS))
def test_weights_file_layout_matches_v0_3_3(name, tmp_path):
    """Continuous `.weights.h5` files saved by v0.3.3 still load: same paths and shapes.

    Keras 3 keys weight files by layer order, so swapping the output layer must not reorder
    the layers that hold weights. The fixtures were written by v0.3.3's builders.
    """
    import h5py

    path = tmp_path / "model.weights.h5"
    _WEIGHTS_LAYOUT_BUILDERS[name]().save_weights(str(path))
    layout = {}
    with h5py.File(path, "r") as f:
        f.visititems(
            lambda p, o: layout.__setitem__(p, list(o.shape))
            if isinstance(o, h5py.Dataset)
            else None
        )
    expected = json.loads((_WEIGHTS_LAYOUT_DIR / f"{name}.json").read_text())
    assert layout == expected
