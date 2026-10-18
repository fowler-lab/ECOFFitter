from dataclasses import dataclass
from typing import Any, Dict, Tuple, cast

import numpy as np

from ecoff_fitter.core import ECOFFitter
from ecoff_fitter.ecoffinder_core import ECOFFinder
from ecoff_fitter.report import GenerateReport
from ecoff_fitter.utils import PreparedInput, prepare_input

Fitter = ECOFFitter | ECOFFinder


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
    method: str = "ecoffitter",
    detection_level: float = 10.0,
    cumulative: bool = True,
    first_best: bool = False,
    intervals: bool = False,
) -> ECOFFRun:
    """Fit ECOFF models to aggregate data and optionally each input group.

    Args:
        data (Any): MIC observations in a supported table, sequence, mapping,
            array-like value, or file format.
        params (dict[str, Any] | str | None): Optional model parameter mapping
            or configuration file path.
        dilution_factor (int): Ratio between adjacent MIC dilution steps.
        distributions (int | str): Number of components, or ``"auto"`` to
            select a count using ``criterion``.
        max_distributions (int): Maximum component count considered in auto
            selection.
        criterion (str): Model summary score used for auto selection, usually
            ``"bic"`` or ``"aic"``.
        min_delta (float): Minimum score improvement required to consider the
            next component count.
        boundary_support (int | None): Number of dilution steps used to extend
            censoring bounds, or None for unbounded tails.
        percentile (float): Percentile used to calculate the ECOFF.
        fit_individual (bool): Whether to fit each individual input column.
        density (bool): Whether reports plot probability density instead of
            counts.
        options (dict[str, Any] | None): Solver and EM options passed to fits.
        sheet_name (str | None): Worksheet to read when ``data`` is an Excel
            file.
        method (str): Fitting method, either ``"ecoffitter"`` or
            ``"ecoffinder"``.
        detection_level (float): Minimum mode size as a percentage of all
            observations when using ECOFFinder.
        cumulative (bool): Whether ECOFFinder fits cumulative frequencies.
        first_best (bool): Whether ECOFFinder stops after its score begins to
            increase from its best value.
        intervals (bool): Whether ECOFFinder refits its selected range using
            interval regression.

    Returns:
        ECOFFRun: Global fit, optional individual fits, and their reports.
    """
    prepared = prepare_input(data, sheet_name=sheet_name)

    if method not in {"ecoffitter", "ecoffinder"}:
        raise ValueError("method must be 'ecoffitter' or 'ecoffinder'.")

    if method == "ecoffinder":
        if distributions != 1:
            raise ValueError(
                "ECOFFinder fits one wild-type distribution; "
                "distributions must be 1."
            )

    def fit_one(input_df: Any) -> tuple[Fitter, Tuple[Any, ...]]:
        if method == "ecoffinder":
            fitter = ECOFFinder(
                input=input_df,
                dilution_factor=dilution_factor,
                detection_level=detection_level,
                cumulative=cumulative,
                first_best=first_best,
                intervals=intervals,
                boundary_support=boundary_support,
                params=params,
                percentile=percentile,
            )
            result = fitter.generate(options=options)
        else:
            fitter = ECOFFitter(
                input=input_df,
                params=params,
                dilution_factor=dilution_factor,
                distributions=cast(int, distributions),
                boundary_support=boundary_support,
            )
            result = fitter.generate(percentile=percentile, options=options)
        setattr(fitter, "density", density)
        return fitter, result

    if method == "ecoffinder":
        global_fitter, global_result = fit_one(prepared.global_df)
        k = 1
    elif distributions == "auto":
        k, score, global_fitter, global_result = select_best_k(
            data,
            dilution_factor=dilution_factor,
            boundary_support=boundary_support,
            max_distributions=max_distributions,
            percentile=percentile,
            criterion=criterion,
            min_delta=min_delta,
            params=params,
            options=options,
            sheet_name=sheet_name,
        )

    else:
        fixed_distributions = cast(int, distributions)
        global_fitter, global_result = fit_one(prepared.global_df)
        k = fixed_distributions

    global_report = GenerateReport.from_fitter(global_fitter, global_result)

    individual_fits: Dict[str, Tuple[Fitter, Tuple[Any, ...]]] = {}
    individual_reports: Dict[str, GenerateReport] = {}

    if fit_individual:
        for name, subdf in prepared.individual_dfs.items():
            if method == "ecoffinder":
                fitter, result = fit_one(subdf)
            elif distributions == "auto":
                _, _, fitter, result = select_best_k(
                    subdf,
                    dilution_factor=dilution_factor,
                    boundary_support=boundary_support,
                    max_distributions=max_distributions,
                    percentile=percentile,
                    criterion=criterion,
                    min_delta=min_delta,
                    params=params,
                    options=options,
                )
            else:
                fitter, result = fit_one(subdf)

            individual_fits[name] = (fitter, result)
            individual_reports[name] = GenerateReport.from_fitter(fitter, result)

    return ECOFFRun(
        prepared=prepared,
        selected_k=k,
        global_fitter=global_fitter,
        global_result=global_result,
        global_report=global_report,
        individual_fits=individual_fits,
        individual_reports=individual_reports,
    )


def select_best_k(
    data: Any,
    dilution_factor: int = 2,
    boundary_support: int | None = 1,
    percentile: float = 99,
    criterion: str = "bic",
    min_delta: float = 15,
    min_pi: float = 0.05,
    min_separation: float = 1.0,
    max_distributions: int = 2,
    params: dict[str, Any] | str | None = None,
    options: dict[str, Any] | None = None,
    sheet_name: str | None = None,
) -> Tuple[int, float, ECOFFitter, Tuple[Any, ...]]:
    """Select a component count by score and mixture diagnostics.

    Candidate models are fitted in increasing component count. Selection
    stops when the score improvement is too small, a component weight is
    below ``min_pi``, or adjacent components are insufficiently separated.

    Args:
        data (Any): MIC input accepted by :func:`prepare_input`.
        dilution_factor (int): Ratio between adjacent MIC dilution steps.
        boundary_support (int | None): Censoring-bound support setting.
        percentile (float): Percentile used to calculate each ECOFF.
        criterion (str): Key in ``ECOFFitter.model_summary()`` to minimize.
        min_delta (float): Minimum score improvement for adding a component.
        min_pi (float): Minimum accepted component mixture weight.
        min_separation (float): Minimum adjacent mean separation in pooled
            standard deviations.
        max_distributions (int): Maximum number of components to evaluate.
        params (dict[str, Any] | str | None): Optional parameter mapping or
            configuration file path.
        options (dict[str, Any] | None): Solver and EM options passed to fits.
        sheet_name (str | None): Worksheet to read from Excel input.

    Returns:
        tuple[int, float, ECOFFitter, tuple[Any, ...]]: Selected component
            count, criterion score, fitted model, and generated result.
    """

    df = prepare_input(
        data,
        sheet_name=sheet_name,
    ).global_df

    n_informative = int((df["observations"] > 0).sum())

    if n_informative == 0:
        raise ValueError("No MIC observations available for fitting.")

    max_k = min(
        max_distributions,
        n_informative,
    )

    best = None

    for k in range(1, max_k + 1):

        # Fit model
        fitter = ECOFFitter(
            input=df,
            params=params,
            dilution_factor=dilution_factor,
            distributions=k,
            boundary_support=boundary_support,
        )

        result = fitter.generate(
            percentile=percentile,
            options=options,
        )

        score = fitter.model_summary()[criterion]

        # K=1 is always the starting model
        if best is None:
            best = (k, score, fitter, result)
            continue

        # improve model fit?
        _, best_score, _, _ = best

        if best_score - score < min_delta:
            break

        # components sufficiently large?
        pis = np.asarray(fitter.pis_)

        if np.any(pis < min_pi):
            break

        # Are adjacent components sufficiently separated?
        order = np.argsort(fitter.mus_)

        mus = np.asarray(fitter.mus_)[order]
        sigmas = np.asarray(fitter.sigmas_)[order]

        pooled_sigmas = np.sqrt((sigmas[:-1] ** 2 + sigmas[1:] ** 2) / 2)

        separations = np.diff(mus) / pooled_sigmas

        if np.any(separations < min_separation):
            break

        best = (k, score, fitter, result)

    assert best is not None
    return best
