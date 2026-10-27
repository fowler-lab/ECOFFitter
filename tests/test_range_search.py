import numpy as np
import pytest

from ecoff_fitter.least_squares import LeastSquaresRangeSearch
from ecoff_fitter.range_search import CountRangeSelector, _RangeSearchFitResult


def make_search(frequencies, detection_level=10):
    return LeastSquaresRangeSearch(
        pmic=np.arange(len(frequencies), dtype=float),
        frequencies=np.asarray(frequencies, dtype=float),
        detection_level=detection_level,
        selector="first-local-peak",
    )


def test_find_mode_returns_first_qualifying_local_peak():
    search = make_search([1, 15, 2, 20, 1])

    assert search._find_mode() == 1


def test_find_mode_keeps_first_local_peak_even_if_it_is_below_threshold():
    search = make_search([1, 8, 1, 1, 20, 1])

    assert search._find_mode() == 1


def test_find_mode_threshold_controls_candidate_replacement():
    search = make_search([1, 20, 2, 30, 1], detection_level=10)

    # The first confirmed peak is returned; later candidates are not reached.
    assert search._find_mode() == 1


def test_find_mode_skips_first_bin_and_finds_next_qualifying_peak():
    search = make_search([20, 1, 2, 15, 1], detection_level=40)

    with pytest.raises(ValueError, match="No eligible local MIC peak"):
        search._find_mode()


def test_find_mode_does_not_treat_last_bin_as_a_local_mode():
    search = make_search([1, 20, 1, 2, 15])

    assert search._find_mode() == 1


def test_largest_bin_range_search_fits_candidates_and_returns_best():
    search = LeastSquaresRangeSearch(
        pmic=np.arange(5, dtype=float),
        frequencies=np.array([1, 4, 20, 8, 1], dtype=float),
        selector="largest-bin",
    )

    result = search.fit()

    assert result["mode"] == 2
    assert result["best"]["start"] == 0
    assert result["best"]["end"] in range(3, 5)
    assert result["candidates"]
    assert result["best"]["sumsq"] >= 0


def test_largest_bin_first_best_stops_when_score_worsens():
    search = LeastSquaresRangeSearch(
        pmic=np.arange(6, dtype=float),
        frequencies=np.array([1, 4, 20, 8, 3, 1], dtype=float),
        selector="largest-bin",
        first_best=True,
    )

    result = search.fit()

    assert len(result["candidates"]) <= 4
    assert result["best"] in result["candidates"]


@pytest.mark.parametrize(
    "pmic,frequencies,selector",
    [
        (np.array([[0.0]]), np.array([1.0]), "largest-bin"),
        (np.array([0.0]), np.array([[1.0]]), "largest-bin"),
        (np.array([0.0]), np.array([1.0, 2.0]), "largest-bin"),
        (np.array([], dtype=float), np.array([], dtype=float), "largest-bin"),
    ],
)
def test_range_search_rejects_invalid_arrays(pmic, frequencies, selector):
    with pytest.raises(ValueError):
        LeastSquaresRangeSearch(pmic, frequencies, selector=selector)


def test_range_search_rejects_unknown_selector():
    with pytest.raises(ValueError, match="selector must be"):
        LeastSquaresRangeSearch(np.array([0.0]), np.array([1.0]), selector="other")


def test_largest_bin_mode_rejects_zero_counts():
    search = LeastSquaresRangeSearch(
        np.array([0.0, 1.0]), np.array([0.0, 0.0]), selector="largest-bin"
    )
    with pytest.raises(ValueError, match="No MIC observations"):
        search._find_mode()


@pytest.mark.parametrize(
    "selector,frequencies,expected",
    [
        ("largest-bin", [1, 5, 3], (0, 2)),
        ("first-local-peak", [1, 5, 1], (0, 2)),
    ],
)
def test_initial_range_and_parameters_cover_both_selectors(
    selector, frequencies, expected
):
    search = LeastSquaresRangeSearch(
        np.arange(3, dtype=float),
        np.asarray(frequencies, dtype=float),
        selector=selector,
    )
    mode = search._find_mode()
    search._detected_mode = mode

    assert search._initial_range(mode) == expected
    params = search._initial_parameters(*expected)
    assert params.shape == (3,)
    assert np.all(np.isfinite(params))


def test_predicted_count_paths_return_finite_values():
    params = np.array([1.0, np.log(0.8), 20.0])
    grid = np.arange(3, dtype=float)
    cumulative = LeastSquaresRangeSearch(grid, np.array([2.0, 4.0, 3.0]))
    density = LeastSquaresRangeSearch(
        grid, np.array([2.0, 4.0, 3.0]), cumulative=False,
        selector="first-local-peak",
    )
    bins = LeastSquaresRangeSearch(
        grid, np.array([2.0, 4.0, 3.0]), cumulative=False,
        selector="largest-bin",
    )

    assert np.all(np.isfinite(cumulative._predicted_counts(params, 0, 2)))
    assert np.all(np.isfinite(density._predicted_counts(params, 0, 2)))
    assert np.all(np.isfinite(bins._predicted_counts(params, 0, 2)))


def test_first_local_peak_non_cumulative_fit_and_single_count_initialization():
    search = LeastSquaresRangeSearch(
        pmic=np.array([-1.0, 0.0, 1.0]),
        frequencies=np.array([1.0, 20.0, 1.0]),
        cumulative=False,
        selector="first-local-peak",
    )
    mode = search._find_mode()
    search._detected_mode = mode
    single_observation = LeastSquaresRangeSearch(
        pmic=np.array([-1.0, 0.0, 1.0]),
        frequencies=np.array([0.0, 1.0, 0.0]),
        cumulative=False,
        selector="first-local-peak",
    )
    single_observation._detected_mode = 1
    params = single_observation._initial_parameters(1, 1)
    assert params[0] == pytest.approx(0.5)
    assert params[1] == pytest.approx(np.log(1.0))

    fit = search._fit_range(0, 2)
    assert np.isfinite(fit["sumsq"])
    assert fit["success"]


def test_count_selector_reads_overrides_from_params_mapping():
    import pandas as pd

    frame = pd.DataFrame(
        {"MIC": ["0.5", "1", "2", "4", "8"], "observations": [1, 5, 20, 8, 1]}
    )
    selected = CountRangeSelector(
        frame,
        range_search="largest-bin",
        params={"detection_level": 5, "cumulative": False},
    ).select()

    assert selected.detection_level == 5
    assert selected.cumulative is False


def test_count_selector_runs_largest_bin_and_first_local_peak():
    import pandas as pd

    frame = pd.DataFrame(
        {"MIC": ["0.5", "1", "2", "4", "8"], "observations": [1, 20, 2, 1, 0]}
    )
    local = CountRangeSelector(
        frame, range_search="first-local-peak", detection_level=2
    ).select()
    largest = CountRangeSelector(
        frame, range_search="largest-bin", detection_level=2
    ).select()

    assert len(local.selected_df) > 0
    assert len(largest.selected_df) > 0
    assert local.range_search == "first-local-peak"
    assert largest.range_search == "largest-bin"


def test_count_selector_rejects_micdat_non_doubling_and_parking_empty_range():
    import pandas as pd

    frame = pd.DataFrame({"MIC": ["1", "2"], "observations": [2, 4]})
    with pytest.raises(ValueError, match="requires dilution_factor=2"):
        CountRangeSelector(
            frame, range_search="first-local-peak", dilution_factor=3
        ).prepare_data()
    with pytest.raises(ValueError, match="excludes all MIC observations"):
        CountRangeSelector(
            frame, range_search="largest-bin", park_from_mic=0.5
        ).prepare_data()


def test_count_selector_rejects_invalid_selector_and_multiple_groups():
    import pandas as pd

    frame = pd.DataFrame({"MIC": ["1"], "observations": [2]})
    with pytest.raises(ValueError, match="range_search must be"):
        CountRangeSelector(frame, range_search="mixture")
    with pytest.raises(ValueError, match="expects one MIC distribution"):
        CountRangeSelector(
            pd.DataFrame({"MIC": ["1"], "A": [2], "B": [3]}),
            range_search="largest-bin",
        )


def test_range_result_adapter_generates_ecoff_and_reports_summary():
    import pandas as pd
    from scipy.optimize import OptimizeResult
    from ecoff_fitter.range_search import RangeSelection

    selection = RangeSelection(
        range_search="largest-bin",
        dilution_factor=2,
        boundary_support=1,
        percentile=99.0,
        selected_df=pd.DataFrame({"MIC": ["1"], "observations": [3]}),
        inclusion_range=(0, 0),
        mode_idx=0,
        selection_mu=0.0,
        selection_sigma=1.0,
        selection_n_est=3.0,
        selection_sumsq=0.2,
        selection_result=OptimizeResult(success=True),
        candidates=[],
        detection_level=10.0,
        cumulative=True,
        park_from_mic=None,
    )
    result = _RangeSearchFitResult(
        selection, fitter="least_squares", interval_fitter=None
    )

    ecoff, location = result.compute_ecoff()
    generated = result.generate()

    assert ecoff == pytest.approx(2 ** location)
    assert generated[-1] == (0, 0)
    assert result.model_summary()["fit_method"] == "cumulative least squares"
    assert result.define_intervals()[2].tolist() == [3]
    with pytest.raises(ValueError, match="percentile must be between"):
        result.compute_ecoff(100)
