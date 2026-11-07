"""TensorFlow helpers of the Keras backend."""

import tensorflow as tf


def safe_log(p: tf.Tensor, neg_inf: float = -1e9, name: str | None = None) -> tf.Tensor:
    """Numerically stable log(p) for p in [0, 1], e.g. softmax/simplex outputs.

    Equivalent in spirit to computing log-weights via `log_softmax(logits)` directly
    (no artificial floor on small-but-nonzero values), for cases where only the
    already-softmaxed probabilities are available and not the underlying logits.
    `log(p + eps)` imposes a hard floor at `log(eps)` (e.g. ~ -13.8 for eps=1e-6),
    which distorts sharp/near-degenerate states whose true log-probability is far
    below that floor. This instead only substitutes `neg_inf` for exact zeros
    (float32 softmax underflow), and does so via `tf.where` so no NaN/inf gradient
    flows back through `tf.math.log` at p == 0.
    """
    p = tf.convert_to_tensor(p)
    is_positive = p > 0
    safe_p = tf.where(is_positive, p, tf.ones_like(p))
    return tf.where(
        is_positive,
        tf.math.log(safe_p),
        tf.fill(tf.shape(p), tf.constant(neg_inf, p.dtype)),
        name=name,
    )
