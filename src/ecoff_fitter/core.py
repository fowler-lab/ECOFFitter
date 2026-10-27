"""Coordinate WT range selection and distribution fitting."""

from typing import Any, Tuple

import numpy as np

from ecoff_fitter.intervals import IntervalRegressionFitter
from ecoff_fitter.range_search import CountRangeSelector, _RangeSearchFitResult
from ecoff_fitter.utils import prepare_input

Fitter = IntervalRegressionFitter | _RangeSearchFitResult


class CoordinateFit:
    """Route MIC data through the selected WT selector and fitter.

    Use :meth:`fit` as the single entry point. Count-based selectors return a
    range selection that is passed to the chosen fitter. Mixture selection
    uses interval regression directly.
    """

    @staticmethod
    def _fit_fixed(
        input: Any,
        *,
        fitter: str,
        range_search: str | None,
        distributions: int,
        params: dict[str, Any] | str | None = None,
        dilution_factor: int = 2,
        boundary_support: int | None = 1,
        percentile: float = 99.0,
        detection_level: float = 10.0,
        cumulative: bool = True,
        first_best: bool = False,
        park_from_mic: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Fitter, tuple[Any, ...]]:
        """Route one input through its selected range search and fitter.

        Args:
            input (Any): MIC data accepted by the input preparation helpers.
            fitter (str): ``"least_squares"`` or ``"interval_regression"``.
            range_search (str | None): ``"largest-bin"``,
                ``"first-local-peak"``, ``"mixture"``, or None for direct fitting.
            distributions (int): Number of interval-regression components.
            params (dict[str, Any] | str | None): Optional parameter source.
            dilution_factor (int): Ratio between adjacent MIC dilution steps.
            boundary_support (int | None): Censoring-bound support setting.
            percentile (float): Percentile used to calculate the ECOFF.
            detection_level (float): Minimum eligible local-peak size.
            cumulative (bool): Whether range fitting uses cumulative counts.
            first_best (bool): Whether to stop when the range score increases.
            park_from_mic (float | None): Upper MIC threshold for active bins.
            options (dict[str, Any] | None): Solver or EM options.

        Returns:
            tuple[Any, tuple[Any, ...]]: Fitted model and generated result."""
        selectors = {"largest-bin", "first-local-peak", "mixture"}
        if fitter not in {"least_squares", "interval_regression"}:
            raise ValueError("fitter must be 'least_squares' or 'interval_regression'.")
        if range_search not in selectors | {None}:
            raise ValueError("range_search must be 'largest-bin', 'first-local-peak', or 'mixture'.")
        if fitter == "least_squares" and range_search not in {"largest-bin", "first-local-peak"}:
            raise ValueError("Least-squares fitting requires a count-based range selector.")
        if range_search == "mixture" and fitter != "interval_regression":
            raise ValueError("Mixture range selection requires interval_regression.")
        if range_search is None and fitter != "interval_regression":
            raise ValueError("Direct fitting requires interval_regression.")
        if range_search == "mixture" and distributions < 2:
            raise ValueError("Mixture selection requires at least two components.")
        if range_search is None and distributions != 1:
            raise ValueError("Multiple components require range_search='mixture'.")
        if range_search in {"largest-bin", "first-local-peak"} and distributions != 1:
            raise ValueError("Count-based range selectors fit one wild-type distribution.")
        if range_search in {"mixture", None}:
            model = IntervalRegressionFitter(
                input=input, params=params, dilution_factor=dilution_factor,
                distributions=distributions, boundary_support=boundary_support,
            )
            result = model.generate(percentile=percentile, options=options)
        else:
            selection = CountRangeSelector(
                input=input,
                range_search=range_search,
                params=params,
                dilution_factor=dilution_factor,
                detection_level=detection_level,
                cumulative=cumulative,
                first_best=first_best,
                boundary_support=boundary_support,
                park_from_mic=park_from_mic,
                percentile=percentile,
            ).select()
            interval_fitter = None
            if fitter == "interval_regression":
                interval_fitter = IntervalRegressionFitter(
                    input=selection.selected_df,
                    dilution_factor=selection.dilution_factor,
                    distributions=1,
                    boundary_support=selection.boundary_support,
                )
                interval_fitter.fit(options=options)
            model = _RangeSearchFitResult(
                selection,
                fitter=fitter,
                interval_fitter=interval_fitter,
            )
            result = model.generate()
        return model, result


    @staticmethod
    def _select_best_k(
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
    ) -> Tuple[int, float, IntervalRegressionFitter, Tuple[Any, ...]]:
        """Select a component count by score and mixture diagnostics.

        Candidate models are fitted in increasing component count. Selection
        stops when the score improvement is too small, a component weight is
        below ``min_pi``, or adjacent components are insufficiently separated.

        Args:
            data (Any): MIC input accepted by :func:`prepare_input`.
            dilution_factor (int): Ratio between adjacent MIC dilution steps.
            boundary_support (int | None): Censoring-bound support setting.
            percentile (float): Percentile used to calculate each ECOFF.
            criterion (str): Key in ``IntervalRegressionFitter.model_summary()`` to minimize.
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
            tuple[int, float, IntervalRegressionFitter, tuple[Any, ...]]: Selected component
                count, criterion score, fitted model, and generated result."""

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
            fitter = IntervalRegressionFitter(
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


    def fit(
        self,
        input: Any,
        *,
        fitter: str,
        range_search: str | None,
        distributions: int | str,
        max_distributions: int = 3,
        criterion: str = "bic",
        min_delta: float = 15,
        params: dict[str, Any] | str | None = None,
        dilution_factor: int = 2,
        boundary_support: int | None = 1,
        percentile: float = 99.0,
        detection_level: float = 10.0,
        cumulative: bool = True,
        first_best: bool = False,
        park_from_mic: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, Fitter, tuple[Any, ...]]:
        """Fit one MIC table using the selected range-search and fitting methods.

        Args:
            input (Any): MIC data accepted by the input preparation helpers.
            fitter (str): ``"least_squares"`` or ``"interval_regression"``.
            range_search (str | None): Count-based selector, ``"mixture"``, or
                None for a direct single-component interval fit.
            distributions (int | str): Component count or ``"auto"`` for automatic
                mixture selection.
            max_distributions (int): Maximum components evaluated when automatic.
            criterion (str): Model-summary score minimized during automatic selection.
            min_delta (float): Minimum score improvement for another component.
            params (dict[str, Any] | str | None): Optional parameter source.
            dilution_factor (int): Ratio between adjacent MIC dilution steps.
            boundary_support (int | None): Censoring-bound support setting.
            percentile (float): Percentile used to calculate the ECOFF.
            detection_level (float): Minimum eligible local-peak size.
            cumulative (bool): Whether count-based range fitting uses cumulative counts.
            first_best (bool): Whether count-based search stops as its score increases.
            park_from_mic (float | None): Upper MIC threshold for active bins.
            options (dict[str, Any] | None): Solver or EM options.

        Returns:
            tuple[int, Any, tuple[Any, ...]]: Selected component count, fitted model,
                and generated result."""
        if distributions == "auto":
            if fitter != "interval_regression" or range_search != "mixture":
                raise ValueError(
                    "Automatic component selection requires interval_regression "
                    "with range_search='mixture'."
                )
            selected_k, _, model, result = self._select_best_k(
                input,
                dilution_factor=dilution_factor,
                boundary_support=boundary_support,
                percentile=percentile,
                criterion=criterion,
                min_delta=min_delta,
                max_distributions=max_distributions,
                params=params,
                options=options,
            )
            return selected_k, model, result

        selected_k = (
            1
            if range_search in {"largest-bin", "first-local-peak"}
            else int(distributions)
        )
        model, result = self._fit_fixed(
            input,
            fitter=fitter,
            range_search=range_search,
            distributions=selected_k,
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
        return selected_k, model, result
