from dataclasses import dataclass
from typing import Any, Dict, Tuple

from ecoff_fitter.core import CoordinateFit, Fitter
from ecoff_fitter.report import GenerateReport
from ecoff_fitter.utils import PreparedInput, prepare_input


@dataclass
class ECOFFRun:
    """Results and reports produced by a global and optional individual fits.

    Attributes:
        prepared (PreparedInput): Normalized global and individual input data.
        selected_k (int): Number of components selected for the global fit.
        global_fitter (Fitter): Fitted model for all observations.
        global_result (Tuple[Any, ...]): Global ECOFF and fitted parameters.
        global_report (GenerateReport): Report for the global fit.
        individual_fits (Dict[str, Tuple[Fitter, Tuple[Any, ...]]]):
            Fitter and result for each individual input column.
        individual_reports (Dict[str, GenerateReport]): Reports for
            individual input columns.
    """

    prepared: PreparedInput
    selected_k: int
    global_fitter: Fitter
    global_result: Tuple[Any, ...]
    global_report: GenerateReport
    individual_fits: Dict[str, Tuple[Fitter, Tuple[Any, ...]]]
    individual_reports: Dict[str, GenerateReport]


def run_ecoff_analysis(
    data: Any,
    *,
    fitter: str = "interval_regression",
    range_search: str | None = None,
    params: dict[str, Any] | str | None = None,
    dilution_factor: int = 2,
    distributions: int | str = 1,
    max_distributions: int = 3,
    criterion: str = "bic",
    min_delta: float = 15,
    boundary_support: int | None = 1,
    percentile: float = 99.0,
    fit_individual: bool = True,
    density: bool = False,
    options: dict[str, Any] | None = None,
    sheet_name: str | None = None,
    detection_level: float = 10.0,
    cumulative: bool = True,
    first_best: bool = False,
    park_from_mic: float | None = None,
) -> ECOFFRun:
    """Fit aggregate and optional individual data with independent method choices.

    Args:
        data (Any): MIC observations in a supported table, sequence, mapping,
            array-like value, or file format.
        fitter (str): ``"least_squares"`` or ``"interval_regression"``.
        range_search (str | None): ``"largest-bin"``,
            ``"first-local-peak"``, or ``"mixture"``. Mixture selection
            requires interval regression.
        params (dict[str, Any] | str | None): Optional fitting configuration.
        dilution_factor (int): Ratio between adjacent MIC dilution steps.
        distributions (int | str): Number of interval-regression components,
            or ``"auto"`` to select a count by score and mixture diagnostics.
        max_distributions (int): Maximum component count for automatic selection.
        criterion (str): Score used for automatic component selection.
        min_delta (float): Minimum score improvement for another component.
        boundary_support (int | None): Censoring-bound support setting.
        percentile (float): Percentile used to calculate the ECOFF.
        fit_individual (bool): Whether to fit each individual input column.
        density (bool): Whether reports plot probability density instead of counts.
        options (dict[str, Any] | None): Solver and EM options.
        sheet_name (str | None): Worksheet to read from Excel input.
        detection_level (float): Minimum eligible local-peak size.
        cumulative (bool): Whether range fitting uses cumulative counts.
        first_best (bool): Whether to stop when the range score increases.
        park_from_mic (float | None): Upper MIC threshold for active bins with
            the largest-bin selector.

    Returns:
        ECOFFRun: Global fit, optional individual fits, and their reports."""
    prepared = prepare_input(data, sheet_name=sheet_name)
    coordinator = CoordinateFit()
    selected_k, global_fitter, global_result = coordinator.fit(
        prepared.global_df,
        fitter=fitter,
        range_search=range_search,
        distributions=distributions,
        max_distributions=max_distributions,
        criterion=criterion,
        min_delta=min_delta,
        params=params,
        dilution_factor=dilution_factor,
        boundary_support=boundary_support,
        percentile=percentile,
        detection_level=detection_level,
        cumulative=cumulative,
        first_best=first_best,
        park_from_mic=park_from_mic,
        options=options,
    )
    setattr(global_fitter, "density", density)

    global_report = GenerateReport.from_fitter(global_fitter, global_result)
    individual_fits: Dict[str, Tuple[Fitter, Tuple[Any, ...]]] = {}
    individual_reports: Dict[str, GenerateReport] = {}
    if fit_individual:
        for name, subdf in prepared.individual_dfs.items():
            k, model, result = coordinator.fit(
                subdf,
                fitter=fitter,
                range_search=range_search,
                distributions=distributions if distributions == "auto" else selected_k,
                max_distributions=max_distributions,
                criterion=criterion,
                min_delta=min_delta,
                params=params,
                dilution_factor=dilution_factor,
                boundary_support=boundary_support,
                percentile=percentile,
                detection_level=detection_level,
                cumulative=cumulative,
                first_best=first_best,
                park_from_mic=park_from_mic,
                options=options,
            )
            setattr(model, "density", density)
            individual_fits[name] = (model, result)
            individual_reports[name] = GenerateReport.from_fitter(model, result)

    return ECOFFRun(
        prepared=prepared, selected_k=selected_k, global_fitter=global_fitter,
        global_result=global_result, global_report=global_report,
        individual_fits=individual_fits, individual_reports=individual_reports,
    )
