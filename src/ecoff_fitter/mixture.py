import numpy as np
from intreg.intreg import IntReg
from sklearn.cluster import KMeans
from scipy.optimize import minimize


class MixtureModel:
    """
    Finite mixture of interval-censored normal distributions fitted using
    Expectation–Maximization (EM) followed by optional joint refinement.

    Workflow:
        1. K-means clustering on interval midpoints for initial paramaters.
        2. EM iteratively updates responsibilities and component parameters.
        3. Optional refinement using mixture likelihood via L-BFGS-B.
    """

    def __init__(self, y_low, y_high, weights, distributions):
        """
        Initialise a K-component mixture model using K-means clustering.

        Initial parameters are generated as follows:
            - μ_k: K-means cluster centres of interval midpoints.
            - σ_k: Weighted within-cluster variance, clipped to a minimum.
            - π_k: Weighted cluster proportions.

        Args:
            y_low (array-like): Lower bounds of log2 MIC intervals.
            y_high (array-like): Upper bounds of log2 MIC intervals.
            weights (array-like): Observation weights for each interval.
            distributions (int): Number of mixture components (K).

        """
        self.K = distributions

        y_low = np.asarray(y_low, dtype=float)
        y_high = np.asarray(y_high, dtype=float)
        weights = np.asarray(weights, dtype=float)

        mid = (y_low + y_high) / 2
        mid_reshaped = mid.reshape(-1, 1)  # sklearn expects 2D

        kmeans = KMeans(
            n_clusters=self.K,
            n_init="auto",
            random_state=0,
        ).fit(mid_reshaped, sample_weight=weights)

        # Cluster centres → initial mus
        mus = kmeans.cluster_centers_.flatten()

        labels = kmeans.labels_

        # Est. within-cluster std dev in log-space
        sigmas = np.zeros(self.K)
        for k in range(self.K):
            idx = np.where(labels == k)[0]
            if len(idx) > 1:
                sigmas[k] = np.sqrt(
                    np.average((mid[idx] - mus[k]) ** 2, weights=weights[idx])
                )
            else:
                # fallback for small clusters
                sigmas[k] = 0.5

        sigmas = np.clip(sigmas, 1e-3, None)

        # ----- Mixture proportions -----
        Nk = np.array([weights[labels == k].sum() for k in range(self.K)])
        pis = Nk / Nk.sum()

        self.y_low, self.y_high, self.weights = y_low, y_high, weights
        self.mus, self.sigmas, self.pis = mus, sigmas, pis

    def fit(self, max_iter=500, tol=1e-6, refine=True):
        """
        Fit the mixture model using EM and optional refinement.

        Args:
            max_iter (int): Maximum number of EM iterations.
            tol (float): Convergence tolerance for change in log-likelihood.
            refine (bool): If True, perform joint optimisation of
                (μ, log σ, logits) after EM using L-BFGS-B.

        Returns:
            self: The fitted MixtureModel instance.
        """

        self.em(max_iter, tol)

        if refine:
            self.refine_mixture()

        return self

    def em(self, max_iter=500, tol=1e-6):
        """
        Expectation–Maximization (EM) algorithm for a K-component mixture
        of interval-censored normal distributions.

        E-step:
            Computes responsibilities r[n,k] = P(component k | interval n).

        M-step:
            Updates mixture weights and fits each component's (mu, sigma)
            via weighted interval regression.

        Args:
            max_iter (int): Maximum EM iterations.
            tol (float): Convergence tolerance for log-likelihood change.

        Returns:
            self: Fitted parameters, log-likelihood, and convergence info.
        """

        N = len(self.y_low)

        converged = False

        prev_ll = -np.inf

        p = np.zeros((N, self.K))

        for it in range(max_iter):

            # ----- E STEP -----
            # p[n,k] = P( interval_n | component k )
            for k in range(self.K):
                p[:, k] = IntReg.interval_prob(
                    self.y_low, self.y_high, self.mus[k], self.sigmas[k]
                )

            # responsibilities (N × K)
            weighted = self.pis * p
            total = np.clip(weighted.sum(axis=1, keepdims=True), 1e-300, np.inf)
            r = weighted / total

            # ----- M STEP -----
            # update mixture weights
            Nk = (self.weights[:, None] * r).sum(axis=0)
            self.pis = Nk / Nk.sum()

            # update mus and sigmas via interval regression
            new_mus, new_sigmas = np.zeros(self.K), np.zeros(self.K)

            for k in range(self.K):
                w_k = self.weights * r[:, k]
                fit_k = IntReg(self.y_low, self.y_high, weights=w_k)
                fit_k.fit()
                mu_k, log_sigma_k = fit_k.result.x
                new_mus[k] = mu_k
                new_sigmas[k] = np.exp(log_sigma_k)

            self.mus, self.sigmas = new_mus, new_sigmas

            # -----LOG-LIKELIHOOD - recompute ps using new parameters-----
            for k in range(self.K):
                p[:, k] = IntReg.interval_prob(
                    self.y_low, self.y_high, self.mus[k], self.sigmas[k]
                )

            mixture_p = (self.pis * p).sum(axis=1)
            ll = np.sum(self.weights * np.log(np.clip(mixture_p, 1e-300, np.inf)))

            # Check convergence
            if it > 0 and abs(ll - prev_ll) < tol:
                converged = True
                break

            prev_ll = ll

        else:
            converged = False

        # [mu1, logσ1, ..., muK, logσK, logits...]
        log_sigmas = np.log(self.sigmas)
        logits = np.log(self.pis[:-1] / self.pis[-1])

        x = np.concatenate([self.mus, log_sigmas, logits])

        self.x = x
        self.converged = converged
        self.n_iter = it + 1
        self.loglike = ll

        return self

    def refine_mixture(self):
        """
        Refinement step for a general K-component mixture model using L-BFGS-B.

        Optimizes all parameters jointly:
            mus, log(sigmas), and mixture logits (K−1 free parameters),
        using the full interval-censored mixture likelihood.

        Returns:
            self: Updated parameter vector and mixture component estimates.
        """

        y_low = np.asarray(self.y_low, float)
        y_high = np.asarray(self.y_high, float)
        weights = np.asarray(self.weights, float)

        def unpack_params(params):
            """Convert flat parameter vector into mus, sigmas, pis."""
            mus = params[: self.K]
            sigmas = np.exp(params[self.K : 2 * self.K])

            # mixture logits, last component has implicit logit = 0
            logits = params[2 * self.K :]
            logits_full = np.concatenate([logits, [0.0]])
            exp_logits = np.exp(logits_full - np.max(logits_full))
            pis = exp_logits / exp_logits.sum()

            return mus, sigmas, pis

        def neg_log_likelihood(params):
            mus, sigmas, pis = unpack_params(params)

            # P(interval | component k)
            # N × K matrix
            p_mat = np.column_stack(
                [
                    IntReg.interval_prob(y_low, y_high, mus[k], sigmas[k])
                    for k in range(self.K)
                ]
            )

            mix = p_mat @ pis
            mix = np.clip(mix, 1e-300, np.inf)

            return -np.sum(weights * np.log(mix))

        res = minimize(neg_log_likelihood, self.x, method="L-BFGS-B")
        self.x = res.x
        self.mus, self.sigmas, self.pis = unpack_params(self.x)

        # Convenience dict
        self.params_ = (
            {f"mu{k+1}": self.mus[k] for k in range(self.K)}
            | {f"sigma{k+1}": self.sigmas[k] for k in range(self.K)}
            | {f"pi{k+1}": self.pis[k] for k in range(self.K)}
        )

        return self
