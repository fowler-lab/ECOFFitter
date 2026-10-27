import numpy as np
import pytest

from ecoff_fitter.least_squares import LeastSquaresRangeSearch


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
