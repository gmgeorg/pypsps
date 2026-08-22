"""Module for visualizing propensity scores and evaluation metrics.

This is only available if installed with [dev] extras, which includes seaborn and matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics


def plot_propensity_eval(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    figsize=(12, 5),
    hist_kws=None,
    pr_kws=None,
):
    """Plot propensity-score distributions by treatment and the Precision-Recall curve.

    Args:
      y_true (array-like): binary treatment labels (0/1).
      y_score (array-like): predicted propensity scores in [0,1].
      figsize (tuple): overall figure size.
      hist_kws (dict): kwargs passed to seaborn.histplot().
      pr_kws (dict): kwargs passed to plt.plot() for PR curve.

    Returns:
      fig, (ax_hist, ax_pr): Matplotlib Figure and Axes.
    """
    hist_kws = {} if hist_kws is None else hist_kws
    pr_kws = {} if pr_kws is None else pr_kws

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Prepare data
    df = pd.DataFrame({"treatment": y_true, "propensity": y_score})

    # Compute PR curve & AUPR
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_score)
    aupr = sklearn.metrics.auc(recall, precision)

    # Create subplots
    fig, (ax_hist, ax_pr) = plt.subplots(1, 2, figsize=figsize)

    # 1) Propensity distributions
    default_hist_kws = {"bins": 20, "alpha": 0.5, "linewidth": 1.5}
    default_hist_kws.update(hist_kws)
    sns.histplot(
        data=df,
        x="propensity",
        hue="treatment",
        element="step",
        stat="density",
        common_norm=False,
        common_bins=True,
        palette=["C0", "C1"],
        ax=ax_hist,
        **default_hist_kws,
    )
    ax_hist.grid(True)
    ax_hist.set_title("Propensity Score Distribution")
    ax_hist.set_xlabel("Propensity Score")
    ax_hist.set_ylabel("Density")
    # sns.histplot builds its own legend; ax_hist.legend(...) would drop its handles
    # and leave an empty box, so just retitle the existing one.
    legend = ax_hist.get_legend()
    if legend is not None:
        legend.set_title("Treatment")

    # 2) Precision-Recall curve
    ax_pr.plot(recall, precision, label=f"AUPR = {aupr:.3f}", **pr_kws)
    ax_pr.axhline(y=y_true.mean(), color="red", linestyle="--", label="Baseline")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(True, linestyle="--", alpha=0.5)
    ax_pr.legend(loc="lower left")

    plt.tight_layout()
    return fig, (ax_hist, ax_pr)
