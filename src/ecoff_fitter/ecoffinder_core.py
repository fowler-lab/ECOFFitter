from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from ecoff_fitter.defence import (
    validate_ecoffinder_params,
    validate_input_source,
    validate_mic_data,
    validate_params,
    validate_params_source,
)
from ecoff_fitter.range_search import ECOFFinderRangeSearch
from ecoff_fitter.utils import prepare_input, read_ecoffinder_params


class ECOFFinder:
    """Select a wild-type MIC range with the ECOFFinder workbook method.

    Candidate ranges are evaluated using cumulative least squares on the log2
    MIC scale. If ``intervals=True``, the selected range is then fitted with
    interval regression using :class:`ECOFFitter`.

    The range-search implementation is an approximation pending validation
    against reference outputs from the Excel workbook.
    """

    def __init__(
        self,
        input: Any,
        dilution_factor: int = 2,
        detection_level: float = 10.0,
        cumulative: bool = True,
        first_best: bool = False,
        intervals: bool = False,
        boundary_support: int | None = 1,
        params: dict[str, Any] | str | None = None,
        percentile: float = 99.9,
    ) -> None:
        """Initialize ECOFFinder with MIC frequencies.

        Args:
            input: Input accepted by :func:`prepare_input`.
            dilution_factor (int): Ratio between adjacent MIC dilution steps.
            detection_level (float): Minimum mode frequency as a percentage
                of the total count.
            cumulative (bool): Fit cumulative counts when True; otherwise fit
                per-dilution frequencies.
            first_best (bool): Stop range search after the count-difference
                criterion begins increasing from its best value.
            intervals (bool): If True, use the selected ECOFFinder range to
                fit an interval-regression model with ECOFFitter.
            boundary_support (int | None): Number of dilution steps used by
                the interval fit to extend censored bounds; None leaves the
                tails unbounded.
            params (dict[str, Any] | str | None): Optional mapping or YAML/TXT
                file with common or ECOFFinder-specific settings. Values in
                the parameter source override the corresponding arguments.
            percentile (float): ECOFF percentile used by ``generate()`` when
                no percentile is passed there.

        Raises:
            ValueError: If the dilution factor, detection level, or MIC table
                is invalid.
        """
        validate_params_source(params)

        if params is not None:
            (
                dilution_factor,
                distributions,
                detection_level,
                cumulative,
                first_best,
                intervals,
                boundary_support,
                percentile,
            ) = read_ecoffinder_params(
                params,
                dilution_factor,
                detection_level,
                cumulative,
                first_best,
                intervals,
                boundary_support,
                percentile,
            )
            if distributions != 1:
                raise ValueError("ECOFFinder supports one WT distribution only.")

        validate_params(dilution_factor, 1, boundary_support)
        validate_ecoffinder_params(
            detection_level,
            cumulative,
            first_best,
            intervals,
            percentile,
        )

        prepared = prepare_input(input)
        if prepared.has_individuals:
            raise ValueError(
                "ECOFFinder expects one MIC distribution. Pass a single "
                "MIC/observations table or one assay column."
            )

        self.obj_df = prepared.global_df.copy()
        validate_input_source(self.obj_df)
        validate_mic_data(self.obj_df)
        self.dilution_factor = dilution_factor
        self.detection_level = detection_level
        self.cumulative = cumulative
        self.first_best = first_best
        self.intervals = intervals
        self.boundary_support = boundary_support
        self.percentile = float(percentile)
        self.distributions = 1
        self.density = False

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
        self.interval_fitter_: Any

    def prepare_data(self) -> "ECOFFinder":
        """Convert the input table to sorted pMIC and frequency arrays.

        Exact MIC rows are placed on the log2 scale. Censored values are
        represented by their reported boundary for this workbook-style
        frequency fit; this draft does not yet reproduce the workbook's
        special handling of censored tail bins.

        Returns:
            self: The instance with prepared pMIC and frequency arrays.
        """
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

        pmic = np.log(mic_values) / np.log(self.dilution_factor)
        frequencies = df["observations"].to_numpy(dtype=float)
        order = np.argsort(pmic)

        self.pmic_ = pmic[order]
        self.observed_ = frequencies[order]
        self.observed_cumulative_ = np.cumsum(self.observed_)
        return self

    def search_ranges(self) -> "ECOFFinder":
        """Search candidate ranges and retain the best count-matching fit.

        Candidate iteration and least-squares scoring are handled by
        :class:`ECOFFinderRangeSearch`.

        Returns:
            self: The instance populated with the selected range and fit.
        """
        range_search = ECOFFinderRangeSearch(
            pmic=self.pmic_,
            frequencies=self.observed_,
            detection_level=self.detection_level,
            cumulative=self.cumulative,
            first_best=self.first_best,
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
        """Return original MIC rows within the selected inclusive range."""
        if not hasattr(self, "inclusion_range_"):
            raise RuntimeError("Call search_ranges() before selecting MIC rows.")

        start, end = self.inclusion_range_
        selected_pmic = self.pmic_[start : end + 1]
        selected_min = self.dilution_factor ** selected_pmic[0]
        selected_max = self.dilution_factor ** selected_pmic[-1]
        mic_values = self.obj_df["MIC"].astype(str).str.strip()
        numeric_mic = mic_values.str.replace("<=", "", regex=False)
        numeric_mic = numeric_mic.str.replace(">", "", regex=False)
        numeric_mic = numeric_mic.str.split("/").str[0]
        numeric_mic = pd.to_numeric(numeric_mic, errors="raise")
        selected = (numeric_mic >= selected_min) & (numeric_mic <= selected_max)
        selected_df = self.obj_df.loc[selected, ["MIC", "observations"]].copy()

        if selected_df.empty:
            raise ValueError("The selected wild-type range contains no MIC rows.")
        return selected_df

    def define_intervals(self) -> tuple[
        NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]
    ]:
        """Return selected MIC rows as log-scale censoring intervals.

        Uses ECOFFitter's existing interval conversion so report plots retain
        the input's left-, right-, and interval-censoring semantics.

        Returns:
            tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
                Lower bounds, upper bounds, and observation weights.
        """
        from ecoff_fitter.core import ECOFFitter

        fitter = ECOFFitter(
            input=self._selected_df(),
            dilution_factor=self.dilution_factor,
            distributions=1,
            boundary_support=self.boundary_support,
        )
        return fitter.define_intervals()

    def fit_intervals(
        self, options: dict[str, Any] | None = None
    ) -> "ECOFFinder":
        """Fit the selected wild-type range with ECOFFitter interval regression.

        The range is selected by the ECOFFinder least-squares search. Rows
        outside that selected range are excluded from the interval fit.

        Returns:
            self: The instance with interval-regression estimates.
        """
        from ecoff_fitter.core import ECOFFitter

        selected_df = self._selected_df()

        fitter = ECOFFitter(
            input=selected_df,
            dilution_factor=self.dilution_factor,
            distributions=1,
            boundary_support=self.boundary_support,
        )
        fitter.fit(options=options)

        self.interval_fitter_ = fitter
        self.mu_ = float(fitter.mus_[0])
        self.sigma_ = float(fitter.sigmas_[0])
        self.mus_ = fitter.mus_.copy()
        self.sigmas_ = fitter.sigmas_.copy()
        self.pis_ = fitter.pis_.copy()
        self.n_est_ = float(np.sum(selected_df["observations"]))
        self.loglike_ = fitter.loglike_
        self.converged_ = fitter.converged_
        self.result_ = fitter.model_
        return self

    def fit(self, options: dict[str, Any] | None = None) -> "ECOFFinder":
        """Prepare input, select the WT range, and fit the selected data.

        Candidate ranges are always selected with the ECOFFinder cumulative
        least-squares procedure. When ``intervals`` is True, ECOFFitter then
        refits the MIC rows in the selected range using interval likelihood.

        Returns:
            self: The fitted ECOFFinder instance.
        """
        self.prepare_data()
        self.search_ranges()
        if self.intervals:
            return self.fit_intervals(options=options)
        return self

    def compute_ecoff(self, percentile: float | None = None) -> tuple[float, float]:
        """Calculate the ECOFF at a percentile of the fitted WT distribution.

        Args:
            percentile (float): Percentile between 0 and 100.

        Returns:
            tuple[float, float]: ECOFF on the MIC scale and its log2 location.
        """
        if percentile is None:
            percentile = self.percentile
        if not 0 < percentile < 100:
            raise ValueError("percentile must be between 0 and 100.")
        if not hasattr(self, "mu_"):
            raise RuntimeError("Call fit() before compute_ecoff().")

        z_percentile = self.mu_ + norm.ppf(percentile / 100) * self.sigma_
        ecoff = float(self.dilution_factor**z_percentile)
        self.ecoff_ = ecoff
        self.z_percentile_ = float(z_percentile)
        return ecoff, float(z_percentile)

    def generate(
        self,
        percentile: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        """Fit the distribution and return ECOFF and model parameters.

        Args:
            percentile (float | None): Percentile used to calculate the ECOFF;
                defaults to the configured value.
            options (dict[str, Any] | None): Optional ECOFFitter solver
                settings; used when ``intervals=True``.

        Returns:
            tuple[Any, ...]: ECOFF, log2 ECOFF, mean, standard deviation,
                estimated WT count, and selected inclusive range.
        """
        self.fit(options=options)
        ecoff, z_percentile = self.compute_ecoff(percentile)
        result = (
            ecoff,
            z_percentile,
            self.mu_,
            self.sigma_,
            self.n_est_,
            self.inclusion_range_,
        )
        self.generated_result_: tuple[Any, ...] = result
        return result

    def make_report(self) -> Any:
        """Create a :class:`GenerateReport` for the fitted ECOFFinder model.

        If ``generate()`` has already been called, its ECOFF result is used.
        Otherwise the model is fitted if needed and a default 99.9th
        percentile report result is created.

        Returns:
            GenerateReport: Report using the existing ECOFFitter report tools.
        """
        from ecoff_fitter.report import GenerateReport

        if not hasattr(self, "mu_"):
            self.fit()
        if not hasattr(self, "generated_result_"):
            ecoff, z_percentile = self.compute_ecoff(self.percentile)
            self.generated_result_ = (
                ecoff,
                z_percentile,
                self.mu_,
                self.sigma_,
            )
        return GenerateReport.from_fitter(self, self.generated_result_)

    def model_summary(self) -> dict[str, Any]:
        """Return selected range, fitted parameters, and optimization status."""
        if not hasattr(self, "mu_"):
            raise RuntimeError("Call fit() before model_summary().")
        return {
            "model_family": (
                "interval-censored normal"
                if self.intervals
                else "normal cumulative least squares"
            ),
            "model_type": "ECOFFinder range search",
            "range_selection_method": "cumulative least squares",
            "final_fit_method": (
                "interval regression" if self.intervals else "cumulative least squares"
            ),
            "dilution_factor": self.dilution_factor,
            "detection_level": self.detection_level,
            "cumulative": self.cumulative,
            "inclusion_range": self.inclusion_range_,
            "mean_log2_mic": self.mu_,
            "sd_log2_mic": self.sigma_,
            "estimated_wt_count": self.n_est_,
            "range_selection_mean_log2_mic": self.selection_mu_,
            "range_selection_sd_log2_mic": self.selection_sigma_,
            "range_selection_estimated_count": self.selection_n_est_,
            "sum_squared_error": self.selection_sumsq_,
            "fit_method": (
                "interval regression" if self.intervals else "cumulative least squares"
            ),
            "converged": self.converged_,
        }
