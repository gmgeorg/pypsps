"""Test viz."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .. import viz


def test_plot_propensity_eval_returns_fig_and_axes():
    """plot_propensity_eval should build a figure with a histogram and PR-curve axis."""
    np.random.seed(13)
    n_samples = 200
    y_true = np.random.binomial(1, 0.4, size=n_samples)
    y_score = np.clip(y_true * 0.5 + np.random.uniform(size=n_samples), 0, 1)

    fig, (ax_hist, ax_pr) = viz.plot_propensity_eval(y_true, y_score)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2
    assert ax_hist.get_title() == "Propensity Score Distribution"
    assert ax_pr.get_title() == "Precision-Recall Curve"

    hist_legend = ax_hist.get_legend()
    assert hist_legend is not None
    assert hist_legend.get_title().get_text() == "Treatment"
    assert len(hist_legend.get_texts()) > 0

    plt.close(fig)
