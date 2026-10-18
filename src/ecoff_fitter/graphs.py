import numpy as np
from typing import Optional
from numpy.typing import NDArray
import matplotlib.axes
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_mic_distribution(
    low_log: NDArray[np.floating],
    high_log: NDArray[np.floating],
    weights: NDArray[np.floating],
    dilution_factor: float | int,
    mus: NDArray[np.floating] | list[float],
    sigmas: NDArray[np.floating] | list[float],
    pis: Optional[NDArray[np.floating] | list[float]] = None,
    log2_ecoff: Optional[float] = None,
    global_x_min: Optional[float] = None,
    global_x_max: Optional[float] = None,
    density: bool = False,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> matplotlib.axes.Axes:
    """
    Plot MIC observation intervals and their fitted normal component curves.

    Supports left- and right-censoring with visual tail extensions.

    If global_x_min and/or global_x_max are supplied, these are used
    as the plotting limits. This allows all distributions belonging
    to the same drug to use the same x-axis range.

    Args:
        low_log (NDArray[np.floating]): Lower interval bounds in log dilution
            space.
        high_log (NDArray[np.floating]): Upper interval bounds in log dilution
            space.
        weights (NDArray[np.floating]): Observation weight for each interval.
        dilution_factor (float | int): Ratio between adjacent MIC steps; used
            to format x-axis labels.
        mus (NDArray[np.floating] | list[float]): Component means.
        sigmas (NDArray[np.floating] | list[float]): Component standard
            deviations.
        pis (Optional[NDArray[np.floating] | list[float]]): Component weights;
            equal weights are used when omitted.
        log2_ecoff (Optional[float]): ECOFF position to mark on the x-axis.
        global_x_min (Optional[float]): Optional left plot bound in log
            dilution space.
        global_x_max (Optional[float]): Optional right plot bound in log
            dilution space.
        density (bool): Whether to show density heights and density curves.
        ax (Optional[matplotlib.axes.Axes]): Axes to draw on; created if omitted.

    Returns:
        matplotlib.axes.Axes: Axes containing the plot.
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
    left_censored = (low_log <= -19) | np.isinf(low_log)
    right_censored = np.isinf(high_log)

    # ---------------------------------------------------
    # Determine finite bounds
    # ---------------------------------------------------
    finite_high_vals = high_log[np.isfinite(high_log)]
    finite_low_vals = low_log[np.isfinite(low_log)]

    finite_high = 0.0
    if finite_high_vals.size > 0:
        finite_high = float(np.max(finite_high_vals))

    finite_low = 0.0
    if finite_low_vals.size > 0:
        finite_low = float(np.min(finite_low_vals))

    # ---------------------------------------------------
    # Determine local plotting bounds
    # ---------------------------------------------------

    # Left bound:
    # extend by tail_steps if this distribution is
    # left-censored.
    if np.any(left_censored):
        plot_xmin = finite_low - tail_steps
    else:
        plot_xmin = finite_low

    # Right bound:
    # extend by tail_steps if this distribution is
    # right-censored.
    if np.any(right_censored):
        plot_xmax = finite_high + tail_steps
    else:
        plot_xmax = finite_high

    # ---------------------------------------------------
    # Override with global drug-level bounds
    # ---------------------------------------------------
    # These should already include the 3-step censoring
    # extension where required at the drug level.
    if global_x_min is not None and np.isfinite(global_x_min):
        plot_xmin = global_x_min

    if global_x_max is not None and np.isfinite(global_x_max):
        plot_xmax = global_x_max

    # ---------------------------------------------------
    # Construct plotting intervals
    # ---------------------------------------------------
    plot_low = low_log.copy()
    plot_high = high_log.copy()

    # RIGHT CENSOR
    #
    # Keep the censored bar as a 3-step extension from the
    # finite boundary for this distribution. The overall
    # axis itself may extend further because another
    # distribution for the same drug is censored.
    plot_high[right_censored] = finite_high + tail_steps

    # LEFT CENSOR
    #
    # Exactly 3 steps below the true high bound.
    plot_low[left_censored] = (
        high_log[left_censored] - tail_steps
    )
    plot_high[left_censored] = high_log[left_censored]

    # Clip intervals to the common plotting region
    plot_low = np.clip(
        plot_low,
        plot_xmin,
        plot_xmax,
    )

    plot_high = np.clip(
        plot_high,
        plot_xmin,
        plot_xmax,
    )

    # ---------------------------------------------------
    # Deduplicate plotting intervals and compute counts
    # ---------------------------------------------------
    plot_intervals = list(zip(plot_low, plot_high))
    unique_intervals = sorted(set(plot_intervals))

    counts = [
        weights[
            (plot_low == lo) &
            (plot_high == hi)
        ].sum()
        for lo, hi in unique_intervals
    ]

    mids = np.array(
        [
            (lo + hi) / 2
            for lo, hi in unique_intervals
        ],
        dtype=float,
    )

    widths = np.array(
        [
            hi - lo
            for lo, hi in unique_intervals
        ],
        dtype=float,
    )

    total_weight = weights.sum()

    print (density)

    if density:
        heights = np.divide(
            np.asarray(counts, dtype=float),
            total_weight * widths,
            out=np.zeros_like(widths, dtype=float),
            where=widths > 0,
        )
        ylabel = "Probability density"
    else:
        heights = np.asarray(
            counts,
            dtype=float,
        )
        ylabel = "Counts"

    # ---------------------------------------------------
    # Draw histogram bars
    # ---------------------------------------------------
    ax.bar(
        mids,
        heights,
        width=widths,
        align="center",
        edgecolor="darkgrey",
        color="lightgrey",
        label="Observed intervals",
    )

    # IMPORTANT:
    # Use the plotting bounds rather than the observed
    # intervals. This preserves the common drug-level
    # x-axis range.
    axis_min = plot_xmin
    axis_max = plot_xmax

    # ---------------------------------------------------
    # Censoring arrows
    # ---------------------------------------------------
    if np.any(right_censored):

        count_inf = weights[right_censored].sum()

        # Arrow belongs to the end of this distribution's
        # censored bar, rather than necessarily the global
        # axis maximum.
        right_edge = min(
            finite_high + tail_steps,
            plot_xmax,
        )

        ax.annotate(
            "",
            xy=(
                right_edge + arrow_offset,
                count_inf / 2,
            ),
            xytext=(
                right_edge - 0.7,
                count_inf / 2,
            ),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.8,
                color="darkgrey",
            ),
        )

    if np.any(left_censored):

        count_low = weights[left_censored].sum()
        mid_y = count_low / 2

        # Use the actual censored bar edge, not necessarily
        # the global axis minimum.
        left_edge = plot_low[left_censored].min()

        ax.annotate(
            "",
            xy=(
                left_edge,
                mid_y,
            ),
            xytext=(
                left_edge + 0.7,
                mid_y,
            ),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.8,
                color="darkgrey",
            ),
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
    # The model is now evaluated across the complete
    # drug-level plotting range.
    x_values = np.linspace(
        plot_xmin,
        plot_xmax,
        600,
    )

    component_pdfs = [
        pi * norm.pdf(
            x_values,
            mu,
            sigma,
        )
        for pi, mu, sigma
        in zip(pis, mus, sigmas)
    ]

    mixture_pdf = np.sum(
        component_pdfs,
        axis=0,
    )

    if density:

        for i, comp in enumerate(
            component_pdfs,
            start=1,
        ):
            ax.plot(
                x_values,
                comp,
                lw=2,
                label=f"Component {i}",
            )

        ax.plot(
            x_values,
            mixture_pdf,
            "k--",
            lw=1.2,
            label="Model",
        )

    else:

        scale = (
            max(heights) / np.max(mixture_pdf)
            if np.max(mixture_pdf) > 0
            else 1.0
        )

        if len(component_pdfs) > 1:

            for i, comp in enumerate(
                component_pdfs,
                start=1,
            ):
                ax.plot(
                    x_values,
                    comp * scale,
                    lw=2,
                    label=f"Component {i}",
                )

            ax.plot(
                x_values,
                mixture_pdf * scale,
                "k--",
                lw=1.2,
                label="Model",
            )

        else:

            ax.plot(
                x_values,
                mixture_pdf * scale,
                "k--",
                lw=1.2,
                label="Model",
            )

    # ---------------------------------------------------
    # Axis formatting
    # ---------------------------------------------------

    if np.any(left_censored):
        ax.margins(x=0)

    # Use common drug-level range
    ax.set_xlim(
        axis_min,
        axis_max,
    )

    ax.set_ylabel(ylabel)

    xticks = ax.get_xticks()

    mic_labels = [
        f"{dilution_factor ** x:.2g}"
        for x in xticks
    ]

    ax.set_xticks(xticks)
    ax.set_xticklabels(mic_labels)

    ax.set_xlabel("MIC (mg/L)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        fontsize=7,
        frameon=False,
    )


    return ax
