"""Which outcome likelihood the outcome mixture NLL evaluates."""

import enum


class OutcomeLikelihood(str, enum.Enum):
    """Which outcome likelihood the outcome mixture NLL evaluates.

    ``INTERVENTIONAL`` (the default in `OutcomeLoss`): -log p(Y | do(A), X), mixing the
    per-state outcome likelihoods with the prior state weights P(S | X) = P(S | X, do(A)),
    since the states precede treatment; these are the same weights the ATE estimates use.
    ``OBSERVATIONAL``: -log p(Y | A, X), mixing the per-state outcome likelihoods with the
    posterior state weights P(S | X, A); together with `TreatmentLoss` this is the exact
    joint likelihood of (Y, A | X).
    """

    OBSERVATIONAL = "observational"
    INTERVENTIONAL = "interventional"
