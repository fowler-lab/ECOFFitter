from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import ecoff_fitter.workflow as workflow
import ecoff_fitter.core as core
from ecoff_fitter.utils import PreparedInput


def _fitter(score=100.0, pis=(1.0,), mus=(0.0,), sigmas=(1.0,), result=(2.0, 1.0)):
    fitter = MagicMock()
    fitter.generate.return_value = result
    fitter.model_summary.return_value = {"bic": score, "aic": score}
    fitter.pis_ = np.asarray(pis)
    fitter.mus_ = np.asarray(mus)
    fitter.sigmas_ = np.asarray(sigmas)
    return fitter


def test_run_ecoff_analysis_fixed_components_with_individual_fits(monkeypatch):
    global_df = pd.DataFrame({"MIC": ["1"], "observations": [5]})
    individual_df = pd.DataFrame({"MIC": ["1"], "observations": [2]})
    prepared = PreparedInput(global_df, {"assay": individual_df})
    monkeypatch.setattr(workflow, "prepare_input", lambda data, sheet_name=None: prepared)

    global_fitter = _fitter()
    individual_fitter = _fitter()
    coordinator = MagicMock(side_effect=[
        (2, global_fitter, global_fitter.generate.return_value),
        (2, individual_fitter, individual_fitter.generate.return_value),
    ])
    monkeypatch.setattr(workflow.CoordinateFit, "fit", coordinator)
    monkeypatch.setattr(
        workflow.GenerateReport,
        "from_fitter",
        MagicMock(side_effect=lambda fitter, result: (fitter, result)),
    )

    run = workflow.run_ecoff_analysis(
        "input",
        distributions=2,
        range_search="mixture",
        percentile=97.5,
        fit_individual=True,
        density=True,
        options={"max_iter": 10},
    )

    assert run.selected_k == 2
    assert run.global_fitter is global_fitter
    assert run.individual_fits["assay"] == (individual_fitter, individual_fitter.generate.return_value)
    assert set(run.individual_reports) == {"assay"}
    assert global_fitter.density is True
    assert individual_fitter.density is True
    assert coordinator.call_count == 2
    for call_args in coordinator.call_args_list:
        assert call_args.kwargs["fitter"] == "interval_regression"
        assert call_args.kwargs["range_search"] == "mixture"


def test_run_ecoff_analysis_skips_individual_fits(monkeypatch):
    prepared = PreparedInput(
        pd.DataFrame({"MIC": ["1"], "observations": [5]}),
        {"assay": pd.DataFrame({"MIC": ["1"], "observations": [2]})},
    )
    monkeypatch.setattr(workflow, "prepare_input", lambda data, sheet_name=None: prepared)
    fitter = _fitter()
    coordinator = MagicMock(return_value=(1, fitter, fitter.generate.return_value))
    monkeypatch.setattr(workflow.CoordinateFit, "fit", coordinator)
    monkeypatch.setattr(workflow.GenerateReport, "from_fitter", MagicMock())

    run = workflow.run_ecoff_analysis("input", fit_individual=False)

    assert run.individual_fits == {}
    assert run.individual_reports == {}
    assert coordinator.call_count == 1
    assert run.global_report is not None


def test_run_ecoff_analysis_auto_selection(monkeypatch):
    prepared = PreparedInput(
        pd.DataFrame({"MIC": ["1"], "observations": [5]}), {}
    )
    monkeypatch.setattr(workflow, "prepare_input", lambda data, sheet_name=None: prepared)
    fitter = _fitter()
    result = (2.0, 1.0)
    select = MagicMock(return_value=(3, fitter, result))
    monkeypatch.setattr(workflow.CoordinateFit, "fit", select)
    monkeypatch.setattr(workflow.GenerateReport, "from_fitter", MagicMock())

    run = workflow.run_ecoff_analysis(
        "input", distributions="auto", range_search="mixture",
        max_distributions=4, fit_individual=False
    )

    assert run.selected_k == 3
    assert run.global_fitter is fitter
    assert run.global_result == result
    select.assert_called_once()


def test_select_best_k_raises_when_no_observations():
    data = pd.DataFrame({"MIC": ["1", "2"], "observations": [0, 0]})
    with pytest.raises(ValueError, match="No MIC observations"):
        core.CoordinateFit._select_best_k(data)


def test_select_best_k_stops_when_score_improvement_is_too_small(monkeypatch):
    data = pd.DataFrame({"MIC": ["1", "2", "4"], "observations": [3, 5, 2]})
    first, second = _fitter(score=100), _fitter(score=90)
    factory = MagicMock(side_effect=[first, second])
    monkeypatch.setattr(core, "IntervalRegressionFitter", factory)

    k, score, fitter, result = core.CoordinateFit._select_best_k(
        data, max_distributions=3, min_delta=15
    )

    assert (k, score, fitter, result) == (1, 100, first, first.generate.return_value)
    assert factory.call_count == 2


@pytest.mark.parametrize(
    "pis,mus,sigmas",
    [
        ((0.01, 0.99), (0.0, 3.0), (1.0, 1.0)),
        ((0.5, 0.5), (0.0, 0.1), (1.0, 1.0)),
    ],
)
def test_select_best_k_rejects_small_or_overlapping_component(
    monkeypatch, pis, mus, sigmas
):
    data = pd.DataFrame({"MIC": ["1", "2", "4"], "observations": [3, 5, 2]})
    first = _fitter(score=100)
    second = _fitter(score=70, pis=pis, mus=mus, sigmas=sigmas)
    factory = MagicMock(side_effect=[first, second])
    monkeypatch.setattr(core, "IntervalRegressionFitter", factory)

    selected = core.CoordinateFit._select_best_k(
        data, max_distributions=2, min_delta=15, min_pi=0.05, min_separation=1.0
    )

    assert selected[0] == 1
    assert selected[2] is first

