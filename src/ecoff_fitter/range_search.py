"""Count-based WT range selection and fitted range results."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from ecoff_fitter.defence import (
    validate_input_source,
    validate_mic_data,
    validate_params,
    validate_params_source,
    validate_range_search_params,
)
from ecoff_fitter.intervals import IntervalRegressionFitter
from ecoff_fitter.least_squares import LeastSquaresRangeSearch
from ecoff_fitter.utils import prepare_input, read_range_search_params


@dataclass(frozen=True)
class RangeSelection:
    """Selected MIC rows and diagnostics from a count-based WT search."""

    range_search: str
    dilution_factor: int
    boundary_support: int | None
    percentile: float
    selected_df: pd.DataFrame
    inclusion_range: tuple[int, int]
    mode_idx: int
    selection_mu: float
    selection_sigma: float
    selection_n_est: float
    selection_sumsq: float
    selection_result: Any
    candidates: list[dict[str, Any]]
    detection_level: float
    cumulative: bool
    park_from_mic: float | None


class CountRangeSelector:
    """Select a wild-type MIC range with a count-based selector.

    Candidate ranges are evaluated using least squares on the log-dilution
    scale. The selected observations and diagnostics are returned separately.
    """

    def __init__(
        self,
        input: Any,
        *,
        range_search: str,
        dilution_factor: int = 2,
        detection_level: float = 10.0,
        cumulative: bool = True,
        first_best: bool = False,
        boundary_support: int | None = 1,
        park_from_mic: float | None = None,
        params: dict[str, Any] | str | None = None,
        percentile: float = 99.9,
    ) -> None:
        """Initialize count-based WT range selection with MIC frequencies.

        Args:
            input: Input accepted by :func:`prepare_input`.
            dilution_factor (int): Ratio between adjacent MIC dilution steps.
            detection_level (float): Minimum mode frequency as a percentage
                of the total count.
            cumulative (bool): Fit cumulative counts when True; otherwise fit
                per-dilution frequencies.
            first_best (bool): Stop range search after the count-difference
                criterion begins increasing from its best value.
            range_search (str): ``"largest-bin"`` or ``"first-local-peak"``.
            boundary_support (int | None): Censoring-bound setting retained
                from the shared parameter source.
            park_from_mic (float | None): MIC at which values enter the parked
                region for largest-bin selection; that MIC and higher bins
                are excluded from the range search.
            params (dict[str, Any] | str | None): Optional mapping or YAML/TXT
                file with common or range-search settings. Values in
                the parameter source override the corresponding arguments.
            percentile (float): ECOFF percentile retained from the shared
                parameter source.

        Raises:
            ValueError: If the dilution factor, detection level, or MIC table
                is invalid.

        Returns:
            None: Result of the operation."""

        validate_params_source(params)
        if range_search not in {"largest-bin", "first-local-peak"}:
            raise ValueError("range_search must be 'largest-bin' or 'first-local-peak'.")

        if params is not None:
            (
                dilution_factor,
                detection_level,
                cumulative,
                first_best,
                boundary_support,
                percentile,
                range_search,
                park_from_mic,
            ) = read_range_search_params(
                params,
                dilution_factor,
                detection_level,
                cumulative,
                first_best,
                boundary_support,
                percentile,
                range_search,
                park_from_mic,
            )

        validate_params(dilution_factor, 1, boundary_support)
        validate_range_search_params(
            detection_level, cumulative, first_best, percentile, range_search, park_from_mic
        )

        prepared = prepare_input(input)
        if prepared.has_individuals:
            raise ValueError(
                "Range selection expects one MIC distribution. Pass a single "
                "MIC/observations table or one assay column."
            )

        self.obj_df = prepared.global_df.copy()
        validate_input_source(self.obj_df)
        validate_mic_data(self.obj_df)
        self.dilution_factor = dilution_factor
        self.detection_level = detection_level
        self.cumulative = cumulative
        self.first_best = first_best
        self.range_search = range_search
        self.park_from_mic = park_from_mic
        self.boundary_support = boundary_support
        self.percentile = float(percentile)
        self.pmic_: NDArray[np.float64]
        self.observed_: NDArray[np.float64]
        self.observed_cumulative_: NDArray[np.float64]
        self.inclusion_range_: tuple[int, int]
        self.mu_: float
        self.sigma_: float
        self.n_est_: float
        self.sumsq_: float
        self.ecoff_: float
        self.z_percentile_: float
        self.converged_: bool
        self.result_: Any

    def prepare_data(self) -> "CountRangeSelector":
        """Convert the input table to sorted pMIC and frequency arrays.

        The first-local-peak selector uses a fixed 19-bin grid and rounded log2
        MIC mapping. The largest-bin selector keeps supplied MIC categories
        and their exact log-dilution coordinates.

        Returns:
            self: The instance with prepared pMIC and frequency arrays.

        Args:
            None."""

        df = self.obj_df.copy()
        mic = df["MIC"].astype(str).str.strip()

        # The workbook frequency algorithm expects numeric MIC categories.
        # Represent censored categories at their stated threshold for now.
        mic = mic.str.replace("<=", "", regex=False)
        mic = mic.str.replace(">", "", regex=False)
        mic = mic.str.split("/").str[0]
        mic_values = pd.to_numeric(mic, errors="raise").to_numpy(dtype=float)

        if np.any(mic_values <= 0):
            raise ValueError("MIC values must be greater than zero.")

        frequencies = df["observations"].to_numpy(dtype=float)

        if self.range_search == "first-local-peak":
            if self.dilution_factor != 2:
                raise ValueError(
                    "first-local-peak requires dilution_factor=2."
                )

            # fixed-grid #Main has pMIC bin centers -9.5, -8.5, ..., 8.5.
            # Its input categories are rounded to the nearest integer log2
            # MIC, and that integer k maps to the center k - 0.5.
            log2_mic = np.log2(mic_values)
            rounded_bins = np.sign(log2_mic) * np.floor(np.abs(log2_mic) + 0.5)
            bin_indices = rounded_bins.astype(int) + 9
            pmic = np.arange(-9.5, 9.0, 1.0, dtype=float)
            binned_frequencies = np.zeros(pmic.size, dtype=float)
            in_grid = (bin_indices >= 0) & (bin_indices < pmic.size)
            outside_nonzero = (~in_grid) & (frequencies > 0)
            if np.any(outside_nonzero):
                values = mic_values[outside_nonzero].tolist()
                raise ValueError(
                    "Nonzero MIC observations fall outside fixed-grid's "
                    f"19-bin grid: {values}."
                )
            np.add.at(
                binned_frequencies,
                bin_indices[in_grid],
                frequencies[in_grid],
            )
            self._fixed_grid_bin_indices_ = bin_indices
            frequencies = binned_frequencies
        else:
            pmic = np.log(mic_values) / np.log(self.dilution_factor)

        if self.park_from_mic is not None:
            pmic_cutoff = np.log(self.park_from_mic) / np.log(self.dilution_factor)
            active = pmic < pmic_cutoff
            pmic = pmic[active]
            frequencies = frequencies[active]
            if pmic.size == 0 or float(np.sum(frequencies)) <= 0:
                raise ValueError("Parking cutoff excludes all MIC observations.")
        if self.range_search == "first-local-peak":
            self.pmic_ = pmic
            self.observed_ = frequencies
        else:
            order = np.argsort(pmic)
            self.pmic_ = pmic[order]
            self.observed_ = frequencies[order]
        self.observed_cumulative_ = np.cumsum(self.observed_)
        return self

    def search_ranges(self) -> "CountRangeSelector":
        """Search candidate ranges and retain the best count-matching fit.

        Candidate iteration and least-squares scoring are handled by
        :class:`LeastSquaresRangeSearch`.

        Returns:
            self: The instance populated with the selected range and fit.

        Args:
            None."""

        range_search = LeastSquaresRangeSearch(
            pmic=self.pmic_,
            frequencies=self.observed_,
            detection_level=self.detection_level,
            cumulative=self.cumulative,
            first_best=self.first_best,
            selector=self.range_search,
        )
        search_result = range_search.fit()
        best = search_result["best"]
        candidates = search_result["candidates"]
        self.mode_idx_ = search_result["mode"]
        self.inclusion_range_ = (best["start"], best["end"])
        self.selection_mu_ = best["mu"]
        self.selection_sigma_ = best["sigma"]
        self.selection_n_est_ = best["n_est"]
        self.selection_sumsq_ = best["sumsq"]
        self.mu_ = self.selection_mu_
        self.sigma_ = self.selection_sigma_
        self.n_est_ = self.selection_n_est_
        self.sumsq_ = best["sumsq"]
        self.converged_ = best["success"]
        self.result_ = best["result"]
        self.range_fits_ = candidates
        self.mus_ = np.array([self.mu_], dtype=float)
        self.sigmas_ = np.array([self.sigma_], dtype=float)
        self.pis_ = np.array([1.0], dtype=float)
        return self

    def _selected_df(self) -> pd.DataFrame:
        """Return original MIC rows within the selected inclusive range.

        Args:
            None.

        Returns:
            pd.DataFrame: Result of the operation."""

        if not hasattr(self, "inclusion_range_"):
            raise RuntimeError("Call search_ranges() before selecting MIC rows.")

        start, end = self.inclusion_range_
        selected_pmic = self.pmic_[start : end + 1]
        mic_values = self.obj_df["MIC"].astype(str).str.strip()
        numeric_mic = mic_values.str.replace("<=", "", regex=False)
        numeric_mic = numeric_mic.str.replace(">", "", regex=False)
        numeric_mic = numeric_mic.str.split("/").str[0]
        numeric_mic = pd.to_numeric(numeric_mic, errors="raise")
        pmic = (
            np.log(numeric_mic.to_numpy(dtype=float))
            / np.log(self.dilution_factor)
        )
        if self.range_search == "first-local-peak":
            rounded_bins = np.sign(pmic) * np.floor(np.abs(pmic) + 0.5)
            bin_indices = rounded_bins.astype(int) + 9
            selected = (bin_indices >= start) & (bin_indices <= end)
        else:
            selected_min = self.dilution_factor ** selected_pmic[0]
            selected_max = self.dilution_factor ** selected_pmic[-1]
            selected = (numeric_mic >= selected_min) & (numeric_mic <= selected_max)
        selected_df = self.obj_df.loc[selected, ["MIC", "observations"]].copy()

        if selected_df.empty:
            raise ValueError("The selected wild-type range contains no MIC rows.")
        return selected_df


    def select(self) -> "RangeSelection":
        """Select and return the wild-type MIC range.

        Returns:
            RangeSelection: Selected MIC rows and range-search diagnostics.

        Args:
            None."""

        self.prepare_data()
        self.search_ranges()
        return RangeSelection(
            range_search=self.range_search,
            dilution_factor=self.dilution_factor,
            boundary_support=self.boundary_support,
            percentile=self.percentile,
            selected_df=self._selected_df(),
            inclusion_range=self.inclusion_range_,
            mode_idx=self.mode_idx_,
            selection_mu=self.selection_mu_,
            selection_sigma=self.selection_sigma_,
            selection_n_est=self.selection_n_est_,
            selection_sumsq=self.selection_sumsq_,
            selection_result=self.result_,
            candidates=self.range_fits_,
            detection_level=self.detection_level,
            cumulative=self.cumulative,
            park_from_mic=self.park_from_mic,
        )


class _RangeSearchFitResult:
    """Expose a fitted count-selected model through the reporting interface."""

    def __init__(
        self,
        selection: RangeSelection,
        *,
        fitter: str,
        interval_fitter: IntervalRegressionFitter | None,
    ) -> None:
        """Build the report-facing result for a selected MIC range.

        Args:
            selection (RangeSelection): Selected observations and diagnostics.
            fitter (str): Fitting method used for the selected observations.
            interval_fitter (IntervalRegressionFitter | None): Fitted interval
                model, or None when least squares supplied the final fit.

        Returns:
            None.
        """
        self.selection_ = selection
        self.fitter = fitter
        self.range_search = selection.range_search
        self.dilution_factor = selection.dilution_factor
        self.boundary_support = selection.boundary_support
        self.percentile = selection.percentile
        self.distributions = 1
        self.density = False
        self.inclusion_range_ = selection.inclusion_range
        self.mode_idx_ = selection.mode_idx
        self.selection_mu_ = selection.selection_mu
        self.selection_sigma_ = selection.selection_sigma
        self.selection_n_est_ = selection.selection_n_est
        self.selection_sumsq_ = selection.selection_sumsq
        self.range_fits_ = selection.candidates
        self.interval_fitter_ = interval_fitter

        if interval_fitter is None:
            self.mu_ = selection.selection_mu
            self.sigma_ = selection.selection_sigma
            self.n_est_ = selection.selection_n_est
            self.converged_ = bool(selection.selection_result.success)
            self.result_ = selection.selection_result
            self.mus_ = np.array([self.mu_], dtype=float)
            self.sigmas_ = np.array([self.sigma_], dtype=float)
            self.pis_ = np.array([1.0], dtype=float)
        else:
            self.mu_ = float(interval_fitter.mus_[0])
            self.sigma_ = float(interval_fitter.sigmas_[0])
            self.n_est_ = float(np.sum(selection.selected_df["observations"]))
            self.converged_ = bool(interval_fitter.converged_)
            self.result_ = interval_fitter.model_
            self.mus_ = interval_fitter.mus_.copy()
            self.sigmas_ = interval_fitter.sigmas_.copy()
            self.pis_ = interval_fitter.pis_.copy()
            self.loglike_ = interval_fitter.loglike_

    def define_intervals(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Return the selected observations as censoring intervals.

        Args:
            None.

        Returns:
            tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]: Result of the operation."""

        if self.interval_fitter_ is None:
            fitter = IntervalRegressionFitter(
                input=self.selection_.selected_df,
                dilution_factor=self.dilution_factor,
                distributions=1,
                boundary_support=self.boundary_support,
            )
            return fitter.define_intervals()
        return self.interval_fitter_.define_intervals()

    def compute_ecoff(self, percentile: float | None = None) -> tuple[float, float]:
        """Calculate an ECOFF and its log-dilution location.

        Args:
            percentile (float | None): Input value.

        Returns:
            tuple[float, float]: Result of the operation."""

        if percentile is None:
            percentile = self.percentile
        if not 0 < percentile < 100:
            raise ValueError("percentile must be between 0 and 100.")
        z_percentile = self.mu_ + norm.ppf(percentile / 100) * self.sigma_
        ecoff = float(self.dilution_factor**z_percentile)
        self.ecoff_ = ecoff
        self.z_percentile_ = float(z_percentile)
        return ecoff, float(z_percentile)

    def generate(self, percentile: float | None = None) -> tuple[Any, ...]:
        """Return ECOFF, fitted parameters, and the selected inclusive range.

        Args:
            percentile (float | None): Input value.

        Returns:
            tuple[Any, ...]: Result of the operation."""

        ecoff, z_percentile = self.compute_ecoff(percentile)
        result = (
            ecoff, z_percentile, self.mu_, self.sigma_, self.n_est_,
            self.inclusion_range_,
        )
        self.generated_result_ = result
        return result

    def model_summary(self) -> dict[str, Any]:
        """Return selected range, fit parameters, and optimization status.

        Args:
            None.

        Returns:
            dict[str, Any]: Result of the operation."""

        return {
            "model_family": (
                "interval-censored normal"
                if self.fitter == "interval_regression"
                else "normal cumulative least squares"
            ),
            "model_type": "selected MIC range search",
            "range_selection_method": "cumulative least squares",
            "final_fit_method": (
                "interval regression"
                if self.fitter == "interval_regression"
                else "cumulative least squares"
            ),
            "dilution_factor": self.dilution_factor,
            "detection_level": self.selection_.detection_level,
            "range_search": self.range_search,
            "park_from_mic": self.selection_.park_from_mic,
            "cumulative": self.selection_.cumulative,
            "inclusion_range": self.inclusion_range_,
            "mean_log2_mic": self.mu_,
            "sd_log2_mic": self.sigma_,
            "estimated_wt_count": self.n_est_,
            "range_selection_mean_log2_mic": self.selection_mu_,
            "range_selection_sd_log2_mic": self.selection_sigma_,
            "range_selection_estimated_count": self.selection_n_est_,
            "sum_squared_error": self.selection_sumsq_,
            "fit_method": (
                "interval regression"
                if self.fitter == "interval_regression"
                else "cumulative least squares"
            ),
            "converged": self.converged_,
        }
