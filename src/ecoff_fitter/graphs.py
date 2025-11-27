import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_mic_distribution(
    low_log,
    high_log,
    weights,
    dilution_factor,
    mu1,
    sigma1,
    mu2=None,
    sigma2=None,
    pi1=None,
    pi2=None,
    log2_ecoff=None,
    global_x_min=None,
    global_x_max=None,
    ax=None,
):
    """
    Plot MIC intervals (bars) with one or two fitted normal components and ECOFF.

    Args:
        low_log, high_log (array-like): lower/upper log2 MIC bounds for each interval.
        weights (array-like): observation weights for each interval.
        mu1, sigma1 (float): parameters for component 1 (wild type if single-component).
        mu2, sigma2 (float or None): parameters for component 2 (if mixture).
        pi1, pi2 (float or None): mixture proportions. Default 0.5 each if not provided.
        log2_ecoff (float or None): ECOFF value to mark (vertical line).
        global_x_min, global_x_max (float or None): x-axis range in log2(MIC).
        ax (matplotlib.axes.Axes or None): optional axis for subplotting.

    Returns:
        matplotlib.axes.Axes: the plot axis used.
    """

    # Create axis if none passed
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))

    # Clip and define x-range
    if global_x_min is None:
        global_x_min = np.nanmin(low_log)
    if global_x_max is None:
        global_x_max = np.nanmax(high_log)
    low_log = np.clip(low_log, global_x_min, global_x_max)
    high_log = np.clip(high_log, global_x_min, global_x_max)

    # Aggregate observed MIC counts
    intervals = list(zip(low_log, high_log))
    unique_intervals = sorted(set(intervals))
    counts = [
        np.sum(weights[np.logical_and(low_log == lo, high_log == hi)])
        for lo, hi in unique_intervals
    ]
    mids = [(lo + hi) / 2 for lo, hi in unique_intervals]
    widths = [hi - lo for lo, hi in unique_intervals]

    # Plot histogram bars
    ax.bar(
        mids,
        counts,
        width=widths,
        align="center",
        edgecolor="darkgrey",
        color="lightgrey",
        label="Observed intervals",
    )

    # ECOFF vertical line
    if log2_ecoff is not None:
        ax.axvline(
            x=log2_ecoff,
            linestyle="--",
            color="blue",
            lw=1.5,
            label="ECOFF",
        )

    # X-values for model curves
    x_values = np.linspace(global_x_min, global_x_max, 600)

    # --- Single component case ---
    if mu2 is None or sigma2 is None:
        y1 = norm.pdf(x_values, mu1, sigma1)
        y1 *= max(counts) / np.max(y1)
        ax.plot(x_values, y1, color="red", lw=2, label="Component 1")
    else:
        # --- Two-component mixture case ---
        # Handle default mixture proportions
        if pi1 is None and pi2 is None:
            pi1 = pi2 = 0.5
        elif pi1 is None or pi2 is None:
            total = (pi1 or 0) + (pi2 or 0)
            pi1 = (pi1 or 0.5) / total
            pi2 = (pi2 or 0.5) / total

        # Weighted component PDFs
        y1 = pi1 * norm.pdf(x_values, mu1, sigma1)
        y2 = pi2 * norm.pdf(x_values, mu2, sigma2)
        y_mix = y1 + y2

        # Scale all to histogram height
        scale_factor = max(counts) / np.max(y_mix)
        y1 *= scale_factor
        y2 *= scale_factor

        # Plot the correctly scaled curves
        ax.plot(x_values, y1, "r-", lw=2, label=f"Component 1")
        ax.plot(x_values, y2, "g-", lw=2, label=f"Component 2")
        ax.plot(x_values, y_mix, "k--", lw=1.2, label="Mixture")

    # Axis labels, title, and cleanup
    ax.set_xlabel("log2(MIC)")
    ax.set_ylabel("Counts")

    ax.set_xlim([global_x_min, global_x_max])
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- NEW: replace x-axis tick labels with true MIC values ---
    # Get tick positions (still in log2 scale)
    xticks = ax.get_xticks()
    # Convert them to MIC concentrations
    mic_labels = [f"{np.round(dilution_factor**x, 2):g}" for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(mic_labels)
    ax.set_xlabel("MIC (mg/L)")


    return ax
