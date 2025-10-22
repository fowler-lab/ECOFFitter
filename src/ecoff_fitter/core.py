import numpy as np
from scipy.stats import norm
from intreg.intreg import IntReg
from .utils import read_input, read_params
from scipy.optimize import minimize

class ECOFFitter:

    def __init__(
            self,
            input,
            params: dict | str | None = None,
            dilution_factor: int = 2,
            distributions: int = 1,
            tail_dilutions: int | None = 1,
            ):
        
        self.obj_df = read_input(input)
        
        if params is not None:
            dilution_factor, distributions, tail_dilutions = read_params(params)
        
        self.dilution_factor = dilution_factor
        self.distributions = distributions
        self.tail_dilutions = tail_dilutions

    def fit(self, options={}):
        """
        Fit the interval regression model for wild-type samples.

        Returns:
            OptimizeResult: The result of the optimization containing fitted parameters.
        """
        # Define and log-transform intervals
        y_low, y_high, weights = self.define_intervals()

        if self.distributions == 1:

            return IntReg(y_low, y_high, weights=weights).fit(
                method="L-BFGS-B", initial_params=None, options=options
            )
        
        elif self.distributions == 2:
            return self.fit_mixture(y_low, y_high, weights, options)
        
        else:
            raise NotImplementedError("Only 1 or 2 componenent mixtures supported")
        
    def fit_mixture(self, y_low, y_high, weights, options=None):

        """Fit a 2-component finite mixture of censored normals using the EM algorithm."""

        options = options or {}
        max_iter = options.get("max_iter", 500)
        tol = options.get("tol", 1e-6)

        # Ensure weights array
        weights = np.asarray(weights, dtype=float)

        # ---- Initialization: single-component fit ----
        base = IntReg(y_low, y_high, weights=weights).fit()
        mu, log_sigma = base.x
        mu1, mu2 = mu - 0.5, mu + 0.5
        sigma1 = sigma2 = np.exp(log_sigma)
        pi1 = 0.5
        pi2 = 1 - pi1

        # ---- Run EM algorithm ----
        result = EcoffFitter._em_algorithm(
            y_low, y_high, weights,
            mu1, sigma1, mu2, sigma2,
            pi1, pi2,
            max_iter=max_iter, tol=tol
        )

        # ---- Optional refinement using minimizer ----
        def _neg_log_L(params):
            mu1, log_sigma1, mu2, log_sigma2, logit_pi = params
            sigma1, sigma2 = np.exp(log_sigma1), np.exp(log_sigma2)
            pi1 = 1 / (1 + np.exp(-logit_pi))
            pi2 = 1 - pi1

            p1 = IntReg.interval_prob(y_low, y_high, mu1, sigma1)
            p2 = IntReg.interval_prob(y_low, y_high, mu2, sigma2)

            mixture_p = pi1 * p1 + pi2 * p2
            return -np.sum(weights * np.log(mixture_p))
        
        if result.converged and options.get('refine', True):
            init = result.x
            result_ref = minimize(_neg_log_L, init, method='L-BFGS-B')
            result.x = result_ref.x
            mu1, log_sigma1, mu2, log_sigma2, logit_pi = result.x
            result.params_.update({
                "mu1": mu1, "sigma1": np.exp(log_sigma1),
                "mu2": mu2, "sigma2": np.exp(log_sigma2),
                "pi1": 1 / (1 + np.exp(-logit_pi)),
                "pi2": 1 - 1 / (1 + np.exp(-logit_pi))
            })
        
        return result
    

    @staticmethod
    def _em_algorithm(
        y_low, y_high, weights,
        mu1, sigma1, mu2, sigma2,
        pi1, pi2, max_iter=500, tol=1e-6
    ):
        """
        Core EM algorithm for a 2-component finite mixture of interval-censored normals.
        """

        p1 = IntReg.interval_prob(y_low, y_high, mu1, sigma1)
        p2 = IntReg.interval_prob(y_low, y_high, mu2, sigma2)
        prev_ll = -np.inf

        for it in range(max_iter):

            # ----- E-step -----
            r1 = pi1 * p1
            r2 = pi2 * p2
            total = np.clip(r1 + r2, 1e-300, np.inf)  # avoid division by zero
            r1 /= total
            r2 /= total

            # ----- M-step -----
            pi1 = np.sum(weights * r1) / np.sum(weights)
            pi2 = 1 - pi1

            w1 = weights * r1
            w2 = weights * r2

            res1 = IntReg(y_low, y_high, weights=w1).fit()
            res2 = IntReg(y_low, y_high, weights=w2).fit()

            mu1, log_sigma1 = res1.x
            mu2, log_sigma2 = res2.x
            sigma1, sigma2 = np.exp(log_sigma1), np.exp(log_sigma2)

            # ----- Recompute p1, p2 using new parameters -----
            p1 = IntReg.interval_prob(y_low, y_high, mu1, sigma1)
            p2 = IntReg.interval_prob(y_low, y_high, mu2, sigma2)

            # ----- Compute new log-likelihood -----
            ll = np.sum(weights * np.log(pi1 * p1 + pi2 * p2))

            # ----- Check convergence -----
            if np.abs(ll - prev_ll) < tol:
                converged = True
                break

            prev_ll = ll
        else:
            converged = False


        class Result:
            pass

        result = Result()
        result.x = np.array([
            mu1, np.log(sigma1), mu2, np.log(sigma2), np.log(pi1 / pi2)
        ])
        result.n_iter = it + 1
        result.converged = converged
        result.loglike = ll
        result.params_ = dict(
            mu1=mu1, sigma1=sigma1,
            mu2=mu2, sigma2=sigma2,
            pi1=pi1, pi2=pi2
        )

        return result    

        
    def define_intervals(self, df=None):
        """
        Define MIC intervals based on the dilution factor and censoring settings.

        Args:
            df (DataFrame): DataFrame containing MIC data. If public access, can optionally supply a df to override the wt.

        Returns:
            tuple: Log-transformed lower and upper bounds for MIC intervals.
        """

        if df is None:
            df = self.obj_df

        y_low = np.zeros(len(df))
        y_high = np.zeros(len(df))
        weights = df.observations.to_numpy()

        # Calculate tail dilution factor if not censored
        if self.tail_dilutions is not None:
            tail_dilution_factor = self.dilution_factor**self.tail_dilutions

        # Process each MIC value and define intervals
        for i, mic in enumerate(df.MIC):
            if mic.startswith("<="):  # Left-censored
                lower_bound = float(mic[2:])
                y_low[i] = 1e-6 if self.tail_dilutions is  None else lower_bound / tail_dilution_factor
                y_high[i] = lower_bound
            elif mic.startswith(">"):  # Right-censored
                upper_bound = float(mic[1:])
                y_low[i] = upper_bound
                y_high[i] = (
                    np.inf if self.tail_dilutions is None else upper_bound * tail_dilution_factor
                )
            else:  # Exact MIC value
                mic_value = float(mic)
                y_low[i] = mic_value / self.dilution_factor
                y_high[i] = mic_value

        # Apply log transformation to intervals
        y_low_log, y_high_log = self.log_transf_intervals(y_low, y_high)

        return y_low_log, y_high_log, weights
    
    def log_transf_intervals(self, y_low, y_high):
        """
        Apply log transformation to interval bounds with the specified dilution factor.

        Args:
            y_low (array-like): Lower bounds of the intervals.
            y_high (array-like): Upper bounds of the intervals.

        Returns:
            tuple: Log-transformed lower and upper bounds.
        """
        log_base = np.log(self.dilution_factor)
        # Transform intervals to log space
        y_low = np.clip(y_low, 1e-12, None)
        y_high = np.clip(y_high, 1e-12, None)

        y_low_log = np.log(y_low, where=(y_low > 0)) / log_base
        y_high_log = np.log(y_high, where=(y_high > 0)) / log_base

        return y_low_log, y_high_log

    def generate(self, percentile=99, options={}):
        """
        Calculate the ECOFF value based on the fitted model and a specified percentile.

        Args:
            percentile (float): The desired percentile (e.g., 99 for 99th percentile).

        Returns:
            tuple: ECOFF in the original scale, the specified percentile in the log-transformed scale,
                   mean (mu), standard deviation (sigma), and the model result.
        """

        assert (
            0 < percentile < 100
        ), "percentile must be a float or integer between 0 and 100"

        model = self.fit(options=options)

        if self.distributions == 1:
            # Extract model parameters
            mu, log_sigma = model.x
            sigma = np.exp(log_sigma)
            # Calulcate z-score for the given percentile
            z = norm.ppf(percentile / 100)
            # Calculate the percentile in log scale
            z_percentile = mu + z * sigma
            # Convert the percentile back to the original MIC scale
            ecoff = self.dilution_factor**z_percentile

            return ecoff, z_percentile, mu, sigma, model
        
        else:
            mu1, log_sigma1, mu2, log_sigma2, logit_pi = model.x

            sigma_1 = np.exp(log_sigma1)
            sigma_2 = np.exp(log_sigma2)

            # pick WT component’s parameters
            mu_wt = mu1 if mu1 < mu2 else mu2
            sigma_wt = sigma_1 if mu1 < mu2 else sigma_2

            # calculate ECOFF percentile
            z = norm.ppf(percentile / 100)
            z_percentile = mu_wt + z * sigma_wt
            ecoff = self.dilution_factor ** z_percentile

            # return both components for reference (same order as your original)
            return ecoff, z_percentile, mu1, sigma_1, mu2, sigma_2, model


        

        






        