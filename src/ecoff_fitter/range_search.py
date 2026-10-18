from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.stats import norm


class ECOFFinderRangeSearch:
    """Find the WT inclusion range using the ECOFFinder least-squares method.

    This class owns mode detection, candidate range iteration, least-squares
    fitting, and selection by the observed-versus-estimated count difference.
    The current candidate-range iteration is an approximation pending
    validation against reference workbook outputs.
    """

    def __init__(
        self,
        pmic: NDArray[np.float64],
        frequencies: NDArray[np.float64],
        detection_level: float = 10.0,
        cumulative: bool = True,
        first_best: bool = False,
    ) -> None:
        """Initialize the range search with sorted MIC frequencies.

        Args:
            pmic (NDArray[np.float64]): Sorted log-dilution MIC values.
            frequencies (NDArray[np.float64]): Frequency at each MIC value.
            detection_level (float): Minimum eligible peak size as a
                percentage of total observations.
            cumulative (bool): Whether the least-squares fit uses cumulative
                counts rather than per-dilution counts.
            first_best (bool): Whether to stop when the count-difference
                criterion starts increasing after its best value.

        Raises:
            ValueError: If input arrays are not one-dimensional and equal in
                length, or if no observations are present.
        """
        self.pmic = np.asarray(pmic, dtype=float)
        self.frequencies = np.asarray(frequencies, dtype=float)
        if self.pmic.ndim != 1 or self.frequencies.ndim != 1:
            raise ValueError("pMIC and frequencies must be one-dimensional.")
        if self.pmic.shape != self.frequencies.shape:
            raise ValueError("pMIC and frequencies must have the same length.")
        if self.pmic.size == 0:
            raise ValueError("At least one MIC value is required.")

        self.detection_level = detection_level
        self.cumulative = cumulative
        self.first_best = first_best
        self.cumulative_frequencies = np.cumsum(self.frequencies)

    def _find_mode(self) -> int:
        """Find the leftmost local mode meeting the detection threshold."""
        total = float(np.sum(self.frequencies))
        if total <= 0:
            raise ValueError("No MIC observations available for fitting.")

        threshold = total * self.detection_level / 100
        local_modes = []
        for i, count in enumerate(self.frequencies):
            left = self.frequencies[i - 1] if i > 0 else -np.inf
            right = (
                self.frequencies[i + 1] if i + 1 < len(self.frequencies) else -np.inf
            )
            if count >= left and count >= right and count >= threshold:
                local_modes.append(i)

        if not local_modes:
            raise ValueError("No MIC mode meets the configured detection level.")
        return local_modes[0]

    def _initial_range(self, mode: int) -> tuple[int, int]:
        """Set the initial range from the first count through above the mode."""
        nonzero = np.flatnonzero(self.frequencies > 0)
        start = int(nonzero[0])
        end = min(mode + 1, len(self.pmic) - 1)
        return start, max(start, end)

    def _initial_parameters(self, start: int, end: int) -> NDArray[np.float64]:
        """Calculate starting values for mean, log-SD, and fitted count."""
        x = self.pmic[start : end + 1]
        counts = self.frequencies[start : end + 1]
        total = max(float(np.sum(counts)), 1.0)
        weights = np.maximum(counts, 1e-12)
        mu = float(np.average(x, weights=weights))
        sigma = float(np.sqrt(np.average((x - mu) ** 2, weights=weights)))
        sigma = max(sigma, 0.1)
        return np.array([mu, np.log(sigma), total], dtype=float)

    def _predicted_counts(
        self, params: NDArray[np.float64], start: int, end: int
    ) -> NDArray[np.float64]:
        """Calculate cumulative or per-bin normal predictions."""
        mu, log_sigma, n_est = params
        sigma = float(np.exp(log_sigma))
        x = self.pmic[start : end + 1]

        if self.cumulative:
            predicted = n_est * norm.cdf(x, loc=mu, scale=sigma)
        else:
            lower = x - 0.5
            upper = x + 0.5
            predicted = n_est * (
                norm.cdf(upper, loc=mu, scale=sigma)
                - norm.cdf(lower, loc=mu, scale=sigma)
            )
        return np.asarray(predicted, dtype=float)

    def _fit_range(
        self,
        start: int,
        end: int,
        initial: NDArray[np.float64] | None = None,
    ) -> dict[str, Any]:
        """Fit the normal model to one candidate inclusion range."""
        if initial is None:
            initial = self._initial_parameters(start, end)

        observed = (
            self.cumulative_frequencies[start : end + 1]
            if self.cumulative
            else self.frequencies[start : end + 1]
        )

        def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
            return observed - self._predicted_counts(params, start, end)

        result = least_squares(
            residuals,
            initial,
            bounds=(
                np.array([-9.0, np.log(0.1), 1.0]),
                np.array([9.0, np.log(9.0), 1_000_000.0]),
            ),
        )
        mu, log_sigma, n_est = result.x
        residual = residuals(result.x)
        return {
            "start": start,
            "end": end,
            "mu": float(mu),
            "sigma": float(np.exp(log_sigma)),
            "n_est": float(n_est),
            "sumsq": float(np.sum(residual**2)),
            "observed_count": float(np.sum(self.frequencies[start : end + 1])),
            "success": bool(result.success),
            "result": result,
        }

    def fit(self) -> dict[str, Any]:
        """Iterate candidate ranges and return the best count-matching fit.

        Returns:
            dict[str, Any]: Selected fit and all candidate range fits.
        """
        mode = self._find_mode()
        start, initial_end = self._initial_range(mode)
        last_nonzero = len(self.frequencies) - 1
        zero_rows = np.flatnonzero(self.frequencies[initial_end + 1 :] == 0)
        if zero_rows.size:
            last_nonzero = initial_end + int(zero_rows[0])

        candidates = []
        params = None
        best_difference = np.inf
        for end in range(initial_end, max(initial_end, last_nonzero) + 1):
            fit = self._fit_range(start, end, params)
            params = np.array(
                [fit["mu"], np.log(fit["sigma"]), fit["n_est"]], dtype=float
            )
            difference = abs(fit["observed_count"] - fit["n_est"])
            fit["count_difference"] = difference
            candidates.append(fit)

            if self.first_best and difference > best_difference:
                break
            best_difference = min(best_difference, difference)

        best = min(candidates, key=lambda fit: fit["count_difference"])
        return {"best": best, "candidates": candidates, "mode": mode}
