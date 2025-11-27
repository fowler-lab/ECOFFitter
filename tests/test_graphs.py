import numpy as np
import matplotlib
matplotlib.use("Agg")

from unittest.mock import MagicMock, patch
from ecoff_fitter.graphs import plot_mic_distribution


@patch("ecoff_fitter.graphs.plt")
def test_plot_mic_distribution_all_branches(mock_plt):
    """Covers: single comp, mixture comp, ECOFF line, custom proportions,
    clipping, tick relabeling, and provided axis"""
    
    low = np.array([1, 1, 2], float)
    high = np.array([2, 2, 3], float)
    weights = np.array([1, 2, 3], float)

    # -------- Case 1: Single component + ECOFF + clipping --------
    fake_fig = MagicMock()
    ax1 = MagicMock()
    mock_plt.subplots.return_value = (fake_fig, ax1)

    plot_mic_distribution(
        low, high, weights,
        dilution_factor=2,
        mu1=1, sigma1=0.5,
        log2_ecoff=1.5,          # tests ECOFF line
        global_x_min=0, global_x_max=3,   # tests clipping
        ax=None
    )

    # Bars plotted
    ax1.bar.assert_called_once()
    # ECOFF vertical line used
    ax1.axvline.assert_called_once()
    # Single component curve exists
    assert any("Component 1" in str(c) for c in ax1.plot.call_args_list)

    # Tick relabeling (called at least once)
    assert ax1.set_xticklabels.called


    # -------- Case 2: Mixture model + custom proportions --------
    ax2 = MagicMock()
    mock_plt.subplots.return_value = (fake_fig, ax2)

    plot_mic_distribution(
        low, high, weights,
        dilution_factor=2,
        mu1=1, sigma1=0.4,
        mu2=3, sigma2=1.0,
        pi1=2, pi2=1,        # tests normalization
        ax=None
    )

    # Should plot y1, y2, y_mix = 3 curves minimum
    assert ax2.plot.call_count >= 3


    # -------- Case 3: Provided axis --------
    provided_ax = MagicMock()
    out_ax = plot_mic_distribution(
        low, high, weights,
        dilution_factor=2,
        mu1=0, sigma1=1,
        ax=provided_ax
    )

    assert out_ax is provided_ax
    provided_ax.bar.assert_called_once()
