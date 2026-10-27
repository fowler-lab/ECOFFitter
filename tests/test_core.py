from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

import ecoff_fitter.core as core
from ecoff_fitter.range_search import RangeSelection


def _selection():
    return RangeSelection(
        range_search="largest-bin",
        dilution_factor=2,
        boundary_support=1,
        percentile=99.0,
        selected_df=pd.DataFrame({"MIC": ["1"], "observations": [4]}),
        inclusion_range=(0, 0),
        mode_idx=0,
        selection_mu=0.0,
        selection_sigma=1.0,
        selection_n_est=4.0,
        selection_sumsq=0.3,
        selection_result=OptimizeResult(success=True),
        candidates=[],
        detection_level=10.0,
        cumulative=True,
        park_from_mic=None,
    )


class FakeIntervalFitter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mus_ = np.array([0.25])
        self.sigmas_ = np.array([0.5])
        self.pis_ = np.array([1.0])
        self.converged_ = True
        self.model_ = object()
        self.loglike_ = -2.0

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return (2.5, 1.25, 0.25, 0.5)

    def fit(self, **kwargs):
        self.fit_kwargs = kwargs
        return self


@pytest.mark.parametrize(
    "fitter,range_search,distributions,match",
    [
        ("bad", "largest-bin", 1, "fitter must be"),
        ("interval_regression", "unknown", 1, "range_search must be"),
        ("least_squares", "mixture", 2, "requires a count-based"),
        ("least_squares", None, 1, "requires a count-based"),
        ("interval_regression", "mixture", 1, "at least two"),
        ("interval_regression", None, 2, "Multiple components require"),
        ("interval_regression", "largest-bin", 2, "fit one wild-type"),
    ],
)
def test_fit_fixed_rejects_incompatible_fitter_and_selector(
    fitter, range_search, distributions, match
):
    with pytest.raises(ValueError, match=match):
        core.CoordinateFit._fit_fixed(
            pd.DataFrame({"MIC": ["1"], "observations": [1]}),
            fitter=fitter,
            range_search=range_search,
            distributions=distributions,
        )


@pytest.mark.parametrize("selector", ["mixture", None])
def test_fit_fixed_direct_interval_path(monkeypatch, selector):
    fake = FakeIntervalFitter()
    monkeypatch.setattr(core, "IntervalRegressionFitter", lambda **kwargs: fake)

    model, result = core.CoordinateFit._fit_fixed(
        pd.DataFrame({"MIC": ["1"], "observations": [5]}),
        fitter="interval_regression",
        range_search=selector,
        distributions=2 if selector == "mixture" else 1,
        options={"max_iter": 3},
    )

    assert model is fake
    assert result == (2.5, 1.25, 0.25, 0.5)
    assert fake.generate_kwargs["options"] == {"max_iter": 3}


@pytest.mark.parametrize("fitter", ["least_squares", "interval_regression"])
def test_fit_fixed_count_selector_routes_to_final_fitter(monkeypatch, fitter):
    selection = _selection()
    selector = MagicMock()
    selector.return_value.select.return_value = selection
    monkeypatch.setattr(core, "CountRangeSelector", selector)
    fake = FakeIntervalFitter()
    monkeypatch.setattr(core, "IntervalRegressionFitter", lambda **kwargs: fake)

    model, result = core.CoordinateFit._fit_fixed(
        selection.selected_df,
        fitter=fitter,
        range_search="largest-bin",
        distributions=1,
    )

    assert result[0] == pytest.approx(2 ** result[1])
    assert selector.call_args.kwargs["range_search"] == "largest-bin"
    if fitter == "least_squares":
        assert model.interval_fitter_ is None
    else:
        assert model.interval_fitter_ is fake
        assert fake.fit_kwargs == {"options": None}


def test_select_best_k_accepts_well_separated_improved_model(monkeypatch):
    data = pd.DataFrame({"MIC": ["1", "2"], "observations": [5, 5]})
    first = MagicMock()
    first.generate.return_value = (1.0, 0.0)
    first.model_summary.return_value = {"bic": 100.0}
    second = MagicMock()
    second.generate.return_value = (2.0, 1.0)
    second.model_summary.return_value = {"bic": 70.0}
    second.pis_ = np.array([0.5, 0.5])
    second.mus_ = np.array([0.0, 3.0])
    second.sigmas_ = np.array([0.5, 0.5])
    factory = MagicMock(side_effect=[first, second])
    monkeypatch.setattr(core, "IntervalRegressionFitter", factory)

    k, score, model, result = core.CoordinateFit._select_best_k(
        data, max_distributions=2, min_delta=15, min_separation=1.0
    )

    assert (k, score, model, result) == (2, 70.0, second, (2.0, 1.0))


def test_fit_auto_component_count_routes_to_selection(monkeypatch):
    model = MagicMock()
    result = (2.0, 1.0)
    select = MagicMock(return_value=(2, 80.0, model, result))
    monkeypatch.setattr(core.CoordinateFit, "_select_best_k", select)

    chosen_k, chosen_model, chosen_result = core.CoordinateFit().fit(
        pd.DataFrame({"MIC": ["1"], "observations": [5]}),
        fitter="interval_regression",
        range_search="mixture",
        distributions="auto",
        max_distributions=3,
    )

    assert (chosen_k, chosen_model, chosen_result) == (2, model, result)
    select.assert_called_once()


def test_fit_auto_component_count_rejects_other_methods():
    with pytest.raises(ValueError, match="Automatic component selection requires"):
        core.CoordinateFit().fit(
            pd.DataFrame({"MIC": ["1"], "observations": [5]}),
            fitter="least_squares",
            range_search="largest-bin",
            distributions="auto",
        )
