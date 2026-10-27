from typing import Any, Callable, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.stats import norm

MIN_SIGMA = 0.1


class LeastSquaresRangeSearch:
    """Find a WT inclusion range with least-squares candidate fits.

    The first-local-peak rule uses a fixed MIC grid and selects the first
    eligible local peak. The largest-bin rule selects the highest-frequency
    active bin.
    """

    def __init__(
        self,
        pmic: NDArray[np.float64],
        frequencies: NDArray[np.float64],
        detection_level: float = 10.0,
        cumulative: bool = True,
        first_best: bool = False,
        selector: str = "first-local-peak",
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
            selector (str): ``"largest-bin"`` or ``"first-local-peak"``.

        Raises:
            ValueError: If input arrays are not one-dimensional and equal in
                length, or if no observations are present.

        Returns:
            None."""
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
        if not isinstance(selector, str) or selector not in {"first-local-peak", "largest-bin"}:
            raise ValueError("selector must be 'largest-bin' or 'first-local-peak'.")
        self.selector = selector
        self.cumulative_frequencies = np.cumsum(self.frequencies)

    def _find_mode(self) -> int:
        """Return a mode index using the selected workbook's detection rule.

        The first-local-peak rule scans for the first eligible local peak.
        The largest-bin rule selects the largest remaining raw-count bin, with
        the first bin winning ties.


        Returns:
            int: Zero-based selected mode index.

        Args:
            None."""
        total = float(np.sum(self.frequencies))
        if total <= 0:
            raise ValueError("No MIC observations available for fitting.")

        if self.selector == "largest-bin":
            return int(np.argmax(self.frequencies))

        maximum_row = 1
        for row in range(1, len(self.frequencies) + 1):
            count = self.frequencies[row - 1]
            maximum_count = self.frequencies[maximum_row - 1]
            if maximum_count <= count:
                if count / total * 100 > self.detection_level:
                    maximum_row = row
            else:
                if row == maximum_row + 1 and maximum_row > 1:
                    return maximum_row - 1

        raise ValueError("No eligible local MIC peak was detected.")

    def _initial_range(self, mode: int) -> tuple[int, int]:
        """Set the initial range using the selected mode's starting row.

        Args:
            mode (int): Zero-based selected mode index.

        Returns:
            tuple[int, int]: Inclusive start and end indices."""
        if self.selector == "first-local-peak":
            # Mirror VBA FirstNonZero: scan backward from the one-based Mode
            # to the nearest zero. Include that row only when it is directly
            # below the mode; otherwise start just after the zero.
            mode_row = mode + 1
            start_row = 1
            for row in range(mode_row, 0, -1):
                if self.frequencies[row - 1] == 0:
                    start_row = row if row == mode_row - 1 else row + 1
                    break
            start = start_row - 1
        else:
            nonzero = np.flatnonzero(self.frequencies > 0)
            start = int(nonzero[0])
        end = min(mode + 1, len(self.pmic) - 1)
        return start, max(start, end)

    def _initial_parameters(self, start: int, end: int) -> NDArray[np.float64]:
        """Calculate starting values for mean, log-SD, and fitted count.

        Args:
            start (int): Inclusive start index.
            end (int): Inclusive end index.

        Returns:
            NDArray[np.float64]: Initial mean, log-SD, and count."""
        x = self.pmic[start : end + 1]
        counts = self.frequencies[start : end + 1]
        if self.selector == "first-local-peak":
            # Mirror Initial!Mean: OFFSET(pMIC, Mode, 0), plus 0.5 when
            # fitting noncumulative density counts.
            mu = float(self.pmic[self._detected_mode])
            if not self.cumulative:
                mu += 0.5

            # Mirror XDev(target, From, To, 1): population SD over the
            # inclusive range, ignoring zero-frequency bins, defaulting to 1
            # when N <= 1, then clipping to SDevMin/SDevMax (0.1 and 9).
            total = float(np.sum(counts))
            if total > 1:
                weighted_mean = float(np.sum(x * counts) / total)
                variance = float(
                    np.sum((x - weighted_mean) ** 2 * counts) / total
                )
                sigma = float(np.sqrt(variance))
            else:
                sigma = 1.0
            sigma = float(np.clip(sigma, MIN_SIGMA, 9.0))
            nest = total
        else:
            total = max(float(np.sum(counts)), 1.0)
            weights = np.maximum(counts, 1e-12)
            weighted_mean = float(np.average(x, weights=weights))
            mu = weighted_mean
            sigma = float(
                np.sqrt(np.average((x - weighted_mean) ** 2, weights=weights))
            )
            sigma = max(sigma, MIN_SIGMA)
            nest = total
        return np.array([mu, np.log(sigma), nest], dtype=float)

    def _predicted_counts(
        self, params: NDArray[np.float64], start: int, end: int
    ) -> NDArray[np.float64]:
        """Calculate cumulative or per-bin normal predictions.

        Args:
            params (NDArray[np.float64]): Mean, log-SD, and fitted count.
            start (int): Inclusive start index.
            end (int): Inclusive end index.

        Returns:
            NDArray[np.float64]: Predicted counts across the range."""
        mu, log_sigma, n_est = params
        sigma = float(np.exp(log_sigma))
        x = self.pmic[start : end + 1]

        if self.cumulative:
            predicted = n_est * norm.cdf(x, loc=mu, scale=sigma)
        elif self.selector == "first-local-peak":
            # The fixed-grid method returns the density at each
            # bin center, multiplied by Nest; it does not integrate over a
            # one-dilution interval.
            predicted = n_est * norm.pdf(x, loc=mu, scale=sigma)
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
        """Fit the normal model to one candidate inclusion range.

        Args:
            start (int): Inclusive start index.
            end (int): Inclusive end index.
            initial (NDArray[np.float64] | None): Optional starting parameters.

        Returns:
            dict[str, Any]: Fit parameters, score, and optimizer result."""
        if initial is None:
            initial = self._initial_parameters(start, end)

        # The largest-bin method fits cumulative counts from its fixed
        # lower grid boundary (pMIC = -10), including zero-count rows. Keep
        # the candidate's selected lower bound unchanged for downstream use.
        fit_start = (
            int(np.searchsorted(self.pmic, -10.0, side="left"))
            if self.cumulative and self.selector == "largest-bin"
            else start
        )
        observed = (
            self.cumulative_frequencies[fit_start : end + 1]
            if self.cumulative
            else self.frequencies[fit_start : end + 1]
        )

        def residuals(params: NDArray[np.float64]) -> NDArray[np.float64]:
            """Calculate observed minus predicted counts.

            Args:
                params (NDArray[np.float64]): Mean, log-SD, and fitted count.

            Returns:
                NDArray[np.float64]: Residuals for the candidate range."""
            return observed - self._predicted_counts(params, fit_start, end)

        solver = cast(Callable[..., Any], least_squares)
        result = solver(
            residuals,
            initial,
            bounds=(
                np.array([-9.0, np.log(MIN_SIGMA), 1.0]),
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

        Args:
            None."""
        mode = self._find_mode()
        self._detected_mode = mode
        start, initial_end = self._initial_range(mode)

        if self.selector == "first-local-peak":
            # Match Analyze's range loop: start from its current To (Reset
            # gives Mode + 1), refit each candidate from Initial parameters,
            # and stop allowing candidates to replace the current best after
            # the first zero-frequency bin. Analyze(False) continues fitting
            # later bins for display, so retain those candidate diagnostics.
            best_difference = float(
                np.sum(self.frequencies[start : initial_end + 1])
            )
            best_end = initial_end
            stopped = False
            candidates = []

            for end in range(initial_end, len(self.frequencies)):
                fit = self._fit_range(start, end)
                difference = abs(fit["observed_count"] - fit["n_est"])
                fit["count_difference"] = difference
                fit["sd_at_floor"] = bool(
                    np.isclose(fit["sigma"], MIN_SIGMA, rtol=0.0, atol=1e-6)
                )
                candidates.append(fit)

                if self.frequencies[end] == 0:
                    stopped = True
                if not stopped and best_difference >= difference:
                    best_difference = difference
                    best_end = end
                elif self.first_best:
                    stopped = True

            # Analyze restores the selected range and runs Solver one more
            # time from the workbook's Initial values.
            best = self._fit_range(start, best_end)
            best["count_difference"] = abs(
                best["observed_count"] - best["n_est"]
            )
            best["sd_at_floor"] = bool(
                np.isclose(best["sigma"], MIN_SIGMA, rtol=0.0, atol=1e-6)
            )
            return {"best": best, "candidates": candidates, "mode": mode}

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
            sd_at_floor = np.isclose(
                fit["sigma"], MIN_SIGMA, rtol=0.0, atol=1e-6
            )
            fit["sd_at_floor"] = bool(sd_at_floor)
            candidates.append(fit)

            # A floor-SD fit is a degenerate candidate. Do not let it trigger
            # first_best stopping or set the score threshold for valid fits.
            if self.selector == "largest-bin" and not sd_at_floor:
                if self.first_best and difference > best_difference:
                    break
                best_difference = min(best_difference, difference)
            elif self.selector != "largest-bin":
                if self.first_best and difference > best_difference:
                    break
                best_difference = min(best_difference, difference)

        eligible = candidates
        if self.selector == "largest-bin":
            non_floor = [fit for fit in candidates if not fit["sd_at_floor"]]
            if non_floor:
                eligible = non_floor
        best = min(eligible, key=lambda fit: fit["count_difference"])
        return {"best": best, "candidates": candidates, "mode": mode}
