import numpy as np
from typing import Any, Optional, Tuple
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm
from intreg.intreg import IntReg
from ecoff_fitter.utils import read_input, read_params
from ecoff_fitter.defence import (
    validate_input_source,
    validate_params_source,
    validate_mic_data,
    validate_params,
)
from ecoff_fitter.mixture import MixtureModel


class ECOFFitter:
    """
    Estimate epidemiological cutoff values (ECOFFs) using interval regression
    on MIC (Minimum Inhibitory Concentration) data.

    If a single distribution is specified, a single censored-normal interval
    regression model is fitted and the ECOFF is taken from the specified percentile.

    If more than one distribution is specified, a finite mixture of interval-
    censored normals is fitted and the ECOFF is taken from the lowest-mean
    (wild-type) component at the given percentile.
    """

    model_: IntReg | MixtureModel | None
    x: NDArray[np.floating]
    mus_: NDArray[np.floating]
    sigmas_: NDArray[np.floating]
    pis_: NDArray[np.floating]
    loglike_: float
    converged_: bool
    n_iter_: int | None
    ecoff_: float
    z_percentile_: float
    y_low_: NDArray[np.floating]
    y_high_: NDArray[np.floating]
    weights_: NDArray[np.floating]

    def __init__(
        self,
        input: pd.DataFrame | str,
        params: dict[str, Any] | str | None = None,
        dilution_factor: int = 2,
        distributions: int = 1,
        boundary_support: int | None = 1,
    ) -> None:
        """
        Initialize the ECOFFitter.

        Input may be a DataFrame with 'MIC' and 'observations' columns, or a
        file (txt, csv, tsv, xlsx, xls) containing the same structure. A
        single-column table or array of MIC values is also accepted.

        Parameters may be supplied directly, or via a dictionary or config
        file (yaml, yml, txt) containing dilution_factor, distributions, and
        boundary_support.

        Args:
            input: Input MIC data or file path.
            params (dict | str | None): Optional parameter source overriding
                manual arguments.
            dilution_factor (int): MIC dilution factor (default 2).
            distributions (int): Number of normal components to fit.
            boundary_support (int | None): Number of intervals defining
                censoring bounds (default 1).
        """

        validate_input_source(input)
        validate_params_source(params)

        # read input dataframe or file
        self.obj_df = read_input(input)
        # check input values
        validate_mic_data(self.obj_df)

        if params is not None:
            # overide explicit arguments with input dict/file
            dilution_factor, distributions, boundary_support, percentile = read_params(
                params, dilution_factor, distributions, boundary_support
            )
            self.percentile = percentile

        # check parameter values
        validate_params(dilution_factor, distributions, boundary_support)

        self.dilution_factor = dilution_factor
        self.distributions = distributions
        self.boundary_support = boundary_support

    def fit(self, options: dict[str, Any] | None = None) -> "ECOFFitter":
        """
        Define MIC intervals and fit either a single censored-normal model
        or a finite mixture model.

        Args:
            options (dict): Optional solver or EM settings.

        Returns:
            self: The fitted ECOFFitter instance.
        """
        # Define and log-transform intervals
        self.y_low_, self.y_high_, self.weights_ = self.define_intervals()

        if self.distributions == 1:
            return self.fit_single(options)

        else:
            # multiple gaussians
            return self.fit_mixture(options)

    def fit_single(self, options: dict[str, Any] | None = None) -> "ECOFFitter":
        """
        Fit a single-component censored normal distribution using interval
        regression.

        Args:
            options (dict | None): Optimization options for the solver.

        Returns:
            self: The fitted ECOFFitter instance with stored parameters.
        """

        # single guassian
        model = IntReg(self.y_low_, self.y_high_, weights=self.weights_)
        model.fit(method="L-BFGS-B", options=options)
        result = model.result

        self.model_ = model
        self.x = result.x
        self.mus_ = np.array([result.x[0]])
        self.sigmas_ = np.array([np.exp(result.x[1])])
        self.pis_ = np.array([1.0])
        self.loglike_ = -result.fun
        self.converged_ = result.success
        self.n_iter_ = result.nit if hasattr(result, "nit") else None

        return self

    def fit_mixture(self, options: dict[str, Any] | None = None) -> "ECOFFitter":
        """
        Fit a K-component finite mixture of censored normals using the EM
        algorithm followed by optional refinement.

        Args:
            options (dict | None): EM and refinement options.

        Returns:
            self: The fitted ECOFFitter instance with stored mixture parameters.
        """

        options = options or {}
        max_iter = options.get("max_iter", 500)
        tol = options.get("tol", 1e-6)

        # Ensure weights array
        weights = np.asarray(self.weights_, dtype=float)

        # Run EM algorithm
        model = MixtureModel(self.y_low_, self.y_high_, weights, self.distributions)
        model.fit(max_iter=max_iter, tol=tol, refine=options.get("refine", True))

        self.model_ = model
        self.x = model.x
        self.mus_ = model.mus
        self.sigmas_ = model.sigmas
        self.pis_ = model.pis
        self.loglike_ = model.loglike
        self.converged_ = model.converged
        self.n_iter_ = model.n_iter

        return self

    def define_intervals(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """
        Construct MIC interval bounds and apply left-, right-, and interval-
        censoring rules, then transform to log dilution space.

        Args:
            df (DataFrame | None): Optional MIC data to override stored input.

        Returns:
            tuple: (y_low_log, y_high_log, weights) – log-transformed interval
            bounds and observation weights.
        """

        if df is None:
            df = self.obj_df

        y_low = np.zeros(len(df))
        y_high = np.zeros(len(df))
        weights = df.observations.to_numpy()

        # Calculate tail dilution factor if not censored
        if self.boundary_support is not None:
            tail_dilution_factor = self.dilution_factor**self.boundary_support

        # Process each MIC value and define intervals
        for i, mic in enumerate(df.MIC):
            if mic.startswith("<="):  # Left-censored value
                lower_bound = float(mic[2:])
                y_low[i] = (
                    1e-6
                    if self.boundary_support is None
                    else lower_bound / tail_dilution_factor
                )
                y_high[i] = lower_bound

            elif mic.startswith(">"):  # Right-censored value
                upper_bound = float(mic[1:])
                y_low[i] = upper_bound
                y_high[i] = (
                    np.inf
                    if self.boundary_support is None
                    else upper_bound * tail_dilution_factor
                )

            else:  # Exact MIC value
                mic_value = float(mic)
                y_low[i] = mic_value / self.dilution_factor
                y_high[i] = mic_value

        # Apply log transformation to intervals
        y_low_log, y_high_log = self.log_transf_intervals(y_low, y_high)

        return y_low_log, y_high_log, weights

    def log_transf_intervals(
        self, y_low: NDArray[np.floating], y_high: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Transform interval bounds into log base–dilution_factor space.

        Args:
            y_low (array-like): Lower interval bounds.
            y_high (array-like): Upper interval bounds.

        Returns:
            tuple: Log-transformed (y_low_log, y_high_log).
        """
        log_base = np.log(self.dilution_factor)
        # Transform intervals to log space
        y_low = np.clip(y_low, 1e-12, None)
        y_high = np.clip(y_high, 1e-12, None)

        y_low_log = np.log(y_low, where=(y_low > 0)) / log_base
        y_high_log = np.log(y_high, where=(y_high > 0)) / log_base

        return y_low_log, y_high_log

    def generate(
        self, percentile: int | float = 99, options: dict[str, Any] | None = None
    ) -> Tuple[Any, ...]:
        """
        Fit the model and compute the ECOFF at a specified percentile.

        Args:
            percentile (float): Desired percentile (default 99).
            options (dict): Optional fit settings.

        Returns:
            tuple: ECOFF value, z-percentile, and fitted parameters.
        """

        if hasattr(self, "percentile"):
            percentile = self.percentile

        self.fit(options=options)  # updates self.model_, self.mus_, etc.
        results = self.compute_ecoff(percentile)
        self.ecoff_ = results[0]
        self.z_percentile_ = results[1]

        return results

    def compute_ecoff(self, percentile: float) -> Tuple[Any, ...]:
        """
        Compute the ECOFF and percentile location from the fitted model.

        Args:
            percentile (float): Percentile in the wild-type component.

        Returns:
            tuple:
              For 1 component:
                  (ecoff, z_percentile, mu, sigma)
              For K components:
                  (ecoff, z_percentile, mu1, sigma1, ..., muK, sigmaK)
        """

        assert 0 < percentile < 100, "percentile must be between 0 and 100"

        z = norm.ppf(percentile / 100)

        # SINGLE COMPONENT
        if self.distributions == 1:
            mu = self.mus_[0]
            sigma = self.sigmas_[0]
            z_percentile = mu + z * sigma
            ecoff = self.dilution_factor**z_percentile
            return ecoff, z_percentile, mu, sigma

        # MULTI-COMPONENT
        K = self.distributions

        mus = self.mus_
        sigmas = self.sigmas_

        # WT = component with lowest mean
        wt_idx = np.argmin(mus)
        mu_wt = mus[wt_idx]
        sigma_wt = sigmas[wt_idx]

        # compute ECOFF on WT
        z_percentile = mu_wt + z * sigma_wt
        ecoff = self.dilution_factor**z_percentile

        # flatten
        mus_sigmas = []
        for k in range(K):
            mus_sigmas.extend([mus[k], sigmas[k]])

        return (ecoff, z_percentile, *mus_sigmas)
