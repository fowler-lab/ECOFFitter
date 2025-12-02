import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_mic_distribution(
    low_log,
    high_log,
    weights,
    dilution_factor,
    mus,
    sigmas,
    pis=None,
    log2_ecoff=None,
    global_x_min=None,
    global_x_max=None,
    ax=None,
):
    """
    Plot MIC intervals with a K-component Gaussian mixture fit.

    Args:
        low_log, high_log (array-like): interval log2 MIC bounds.
        weights (array-like): observation weights.
        dilution_factor (float): MIC dilution factor (typically 2).
        mus (list[float]): means for each mixture component.
        sigmas (list[float]): std devs for each mixture component.
        pis (list[float] | None): mixture weights (must sum to 1). Defaults to uniform.
        log2_ecoff (float | None): ECOFF threshold in log2 scale.
        global_x_min, global_x_max (float | None): x-axis limits.
        ax (matplotlib.axes.Axes | None): optional axis for subplotting.

    Returns:
        matplotlib.axes.Axes
    """

    K = len(mus)
    mus = np.asarray(mus, float)
    sigmas = np.asarray(sigmas, float)

    # Default equal mixture proportions
    if pis is None:
        pis = np.ones(K) / K
    else:
        pis = np.asarray(pis, float)
        pis = pis / pis.sum()

    # Create axis if required
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))

    # Determine x-range
    if global_x_min is None:
        global_x_min = np.nanmin(low_log)
    if global_x_max is None:
        global_x_max = np.nanmax(high_log)

    # Clip intervals
    low_log = np.clip(low_log, global_x_min, global_x_max)
    high_log = np.clip(high_log, global_x_min, global_x_max)

    # Unique MIC intervals + counts
    intervals = list(zip(low_log, high_log))
    unique_intervals = sorted(set(intervals))

    counts = [
        weights[(low_log == lo) & (high_log == hi)].sum()
        for lo, hi in unique_intervals
    ]
    mids = [(lo + hi) / 2 for lo, hi in unique_intervals]
    widths = [hi - lo for lo, hi in unique_intervals]

    # Plot MIC intervals
    ax.bar(
        mids, counts, width=widths,
        align="center",
        edgecolor="darkgrey",
        color="lightgrey",
        label="Observed intervals"
    )

    # ECOFF marker
    if log2_ecoff is not None:
        ax.axvline(
            x=log2_ecoff,
            linestyle="--",
            color="blue",
            lw=1.5,
            label="ECOFF",
        )

    # Mixture component curves
    x_values = np.linspace(global_x_min, global_x_max, 600)

    component_pdfs = [
        pi * norm.pdf(x_values, mu, sigma)
        for pi, mu, sigma in zip(pis, mus, sigmas)
    ]
    mixture_pdf = np.sum(component_pdfs, axis=0)

    # Scale PDFs to histogram height
    scale = max(counts) / np.max(mixture_pdf)

    for i, comp in enumerate(component_pdfs, start=1):
        ax.plot(x_values, comp * scale, lw=2, label=f"Component {i}")

    ax.plot(
        x_values,
        mixture_pdf * scale,
        "k--",
        lw=1.2,
       label="Mixture"
    )

    # Labels
    ax.set_xlabel("log2(MIC)")
    ax.set_ylabel("Counts")
    ax.set_xlim(global_x_min, global_x_max)

    # Convert x axis to actual MIC values
    xticks = ax.get_xticks()
    mic_labels = [f"{dilution_factor ** x:.2g}" for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(mic_labels)
    ax.set_xlabel("MIC (mg/L)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=7, frameon=False)

    return ax
