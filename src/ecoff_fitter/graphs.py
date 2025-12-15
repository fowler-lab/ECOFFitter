import numpy as np
from typing import Optional
from numpy.typing import NDArray
import matplotlib.axes
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_mic_distribution(
    low_log: NDArray[np.float_],
    high_log: NDArray[np.float_],
    weights: NDArray[np.float_],
    dilution_factor: float | int,
    mus: NDArray[np.float_] | list[float],
    sigmas: NDArray[np.float_] | list[float],
    pis: Optional[NDArray[np.float_] | list[float]] = None,
    log2_ecoff: Optional[float] = None,
    global_x_min: Optional[float] = None,
    global_x_max: Optional[float] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """
    Plot MIC intervals with a K-component Gaussian mixture fit.
    Supports left- and right-censoring with visual tail extensions.
    """

    tail_steps = 3.0
    arrow_offset = 0

    mus = np.asarray(mus, float)
    sigmas = np.asarray(sigmas, float)

    if pis is None:
        pis = np.ones_like(mus) / len(mus)
    else:
        pis = np.asarray(pis, float)
        pis = pis / pis.sum()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))

    # ---------------------------------------------------
    # Identify censoring
    # ---------------------------------------------------
    left_censored  = (low_log <= -19) | np.isinf(low_log)
    right_censored = np.isinf(high_log)

    # ---------------------------------------------------
    # Determine finite bounds
    # ---------------------------------------------------
    finite_high_vals = high_log[np.isfinite(high_log)]
    finite_low_vals  = low_log[np.isfinite(low_log)]

    finite_high = finite_high_vals.max() if len(finite_high_vals) else 0
    finite_low  = finite_low_vals.min()  if len(finite_low_vals)  else 0

    # Axis left bound
    if np.any(left_censored):
        plot_xmin = finite_low - tail_steps
    else:
        plot_xmin = finite_low

    # Axis right bound
    if np.any(right_censored):
        plot_xmax = finite_high + tail_steps
    else:
        plot_xmax = finite_high

    if global_x_max is not None and np.isfinite(global_x_max):
        plot_xmax = global_x_max

    # ---------------------------------------------------
    # Construct plotting intervals
    # ---------------------------------------------------
    plot_low  = low_log.copy()
    plot_high = high_log.copy()

    # RIGHT CENSOR → extend above finite_high
    plot_high[right_censored] = finite_high + tail_steps

    # LEFT CENSOR → exactly 3 steps below true high bound
    plot_low[left_censored]  = high_log[left_censored] - tail_steps
    plot_high[left_censored] = high_log[left_censored]

    # Clip to plotting region
    plot_low  = np.clip(plot_low,  plot_xmin, plot_xmax)
    plot_high = np.clip(plot_high, plot_xmin, plot_xmax)

    # ---------------------------------------------------
    # Deduplicate plotting intervals and compute counts
    # ---------------------------------------------------
    plot_intervals = list(zip(plot_low, plot_high))
    unique_intervals = sorted(set(plot_intervals))

    counts = [
        weights[(plot_low == lo) & (plot_high == hi)].sum()
        for lo, hi in unique_intervals
    ]

    mids = [(lo + hi) / 2 for lo, hi in unique_intervals]
    widths = [(hi - lo) for lo, hi in unique_intervals]

    # ---------------------------------------------------
    # Draw histogram bars
    # ---------------------------------------------------
    ax.bar(
        mids,
        counts,
        width=widths,
        align="center",
        edgecolor="darkgrey",
        color="lightgrey",
        label="Observed intervals",
    )

    axis_min = plot_low.min()  
    axis_max = plot_high.max() 

    # ---------------------------------------------------
    # Censoring arrows
    # ---------------------------------------------------
    if np.any(right_censored):
        count_inf = weights[right_censored].sum()
        ax.annotate(
            "",
            xy=(plot_xmax + arrow_offset, count_inf / 2),
            xytext=(plot_xmax - 0.7,     count_inf / 2),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="darkgrey"),
        )

    if np.any(left_censored):
        count_low = weights[left_censored].sum()
        mid_y = count_low / 2

        left_edge = plot_low.min()

        ax.annotate(
            "",
            xy=(left_edge, mid_y),            # arrow head at the bar edge
            xytext=(left_edge + 0.7, mid_y),  # arrow tail pointing rightward
            arrowprops=dict(arrowstyle="->", lw=1.8, color="darkgrey"),
        )

    # ---------------------------------------------------
    # ECOFF line
    # ---------------------------------------------------
    if log2_ecoff is not None:
        ax.axvline(
            x=log2_ecoff,
            linestyle="--",
            color="blue",
            lw=1.5,
            label="ECOFF",
        )

    # ---------------------------------------------------
    # Mixture model PDF curves
    # ---------------------------------------------------
    x_values = np.linspace(plot_xmin, plot_xmax, 600)

    component_pdfs = [
        pi * norm.pdf(x_values, mu, sigma)
        for pi, mu, sigma in zip(pis, mus, sigmas)
    ]

    mixture_pdf = np.sum(component_pdfs, axis=0)

    scale = max(counts) / np.max(mixture_pdf)

    if len(component_pdfs) > 1:
        for i, comp in enumerate(component_pdfs, start=1):
            ax.plot(x_values, comp * scale, lw=2, label=f"Component {i}")
        ax.plot(x_values, mixture_pdf * scale, "k--", lw=1.2, label="Model")
    else:
        ax.plot(x_values, mixture_pdf * scale, "k--", lw=1.2, label="Model")

    # ---------------------------------------------------
    # Axis formatting
    # ---------------------------------------------------
    
    if np.any(left_censored):
        ax.margins(x=0)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylabel("Counts")

    xticks = ax.get_xticks()
    mic_labels = [f"{dilution_factor ** x:.2g}" for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(mic_labels)
    ax.set_xlabel("MIC (mg/L)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, frameon=False)

    return ax
