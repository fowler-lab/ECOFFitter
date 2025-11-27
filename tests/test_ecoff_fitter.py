import pytest
import numpy as np
import pandas as pd
from ecoff_fitter import ECOFFitter
from intreg.intreg import IntReg

# ----------------- __init__.py checks ----------------- #

def test_ecoffitter_importable_from_root():
    """The package root should expose ECOFFitter in its public API."""
    import ecoff_fitter
    assert hasattr(ecoff_fitter, "ECOFFitter")
    from ecoff_fitter import ECOFFitter  # should not fail
    assert ECOFFitter is ecoff_fitter.ECOFFitter


def test_main_invokes_cli(monkeypatch):
    """The top-level main() should call the CLI's main function."""
    import ecoff_fitter
    called = {"flag": False}

    def fake_cli_main():
        called["flag"] = True

    monkeypatch.setattr("ecoff_fitter.cli.main", fake_cli_main)

    ecoff_fitter.main()
    assert called["flag"], "main() did not dispatch to cli.main"


#------------------- Data fixtures ----------------------- #


@pytest.fixture
def simple_data():
    """Small, clean MIC dataset with exact (non-censored) values."""
    return pd.DataFrame(
        {"MIC": ["0.5", "1", "2", "4", "8"], "observations": [5, 10, 20, 10, 5]}
    )


@pytest.fixture
def censored_data():
    """Dataset containing left- and right-censored MIC values."""
    return pd.DataFrame(
        {"MIC": ["<=0.5", "1", "2", ">4"], "observations": [3, 8, 12, 2]}
    )


# ------------------ Initialization & Validation ------------------ #


def test_init_with_dataframe(simple_data):
    """ECOFFitter should initialize properly from a DataFrame input."""
    fitter = ECOFFitter(simple_data, dilution_factor=2, distributions=1)
    assert fitter.dilution_factor == 2
    assert fitter.distributions == 1
    assert "MIC" in fitter.obj_df.columns
    assert "observations" in fitter.obj_df.columns


def test_init_with_invalid_distributions(simple_data):
    """ECOFFitter should reject unsupported distribution counts."""
    with pytest.raises(NotImplementedError):
        ECOFFitter(simple_data, distributions=3).fit()


# ------------------ Interval Definition ------------------ #


def test_define_intervals_uncensored(simple_data):
    """Intervals for uncensored MICs should be properly defined in log space."""
    fitter = ECOFFitter(simple_data)
    y_low, y_high, weights = fitter.define_intervals()

    # All intervals should be in ascending order
    assert np.all(y_low < y_high)
    assert np.allclose(weights, simple_data["observations"].values)

    # Check expected log2 transforms
    log2 = lambda x: np.log(x) / np.log(2)
    expected_low = log2(np.array([0.25, 0.5, 1, 2, 4]))
    expected_high = log2(np.array([0.5, 1, 2, 4, 8]))
    assert np.allclose(y_low, expected_low, atol=1e-3)
    assert np.allclose(y_high, expected_high, atol=1e-3)


def test_define_intervals_with_censoring(censored_data):
    """Left- and right-censored MICs should be handled correctly."""
    fitter = ECOFFitter(censored_data, tail_dilutions=None)
    y_low, y_high, _ = fitter.define_intervals()

    # <=0.5 → left-censored, low = 0.25
    assert y_low[0] < np.log(0.5) / np.log(2)
    # >4 → right-censored, high = inf
    assert np.isinf(y_high[-1])


def test_log_transf_intervals_direct(simple_data):
    """Direct log transform should produce expected log2-scale values."""
    fitter = ECOFFitter(simple_data)
    y_low = np.array([0.5, 1.0, 2.0])
    y_high = np.array([1.0, 2.0, 4.0])
    low_log, high_log = fitter.log_transf_intervals(y_low, y_high)
    assert np.allclose(low_log, np.log2(y_low), atol=1e-6)
    assert np.allclose(high_log, np.log2(y_high), atol=1e-6)


# ------------------ Model Fitting ------------------ #


@pytest.mark.parametrize("distributions", [1, 2])
def test_fit_model_runs(simple_data, distributions):
    """Fitting should run successfully for both single and two-component models."""
    fitter = ECOFFitter(simple_data, distributions=distributions)
    model = fitter.fit()
    assert hasattr(model, "x")
    assert np.all(np.isfinite(model.x))


def test_fit_mixture_returns_expected_structure(simple_data):
    """Mixture model fit should return both components and mixture proportions."""
    fitter = ECOFFitter(simple_data, distributions=2)
    result = fitter.fit()

    # Mixture parameters
    params = result.params_
    assert all(k in params for k in ["mu1", "sigma1", "mu2", "sigma2", "pi1", "pi2"])
    assert np.isclose(params["pi1"] + params["pi2"], 1, atol=1e-3)
    assert 0 <= params["pi1"] <= 1


# ------------------ ECOFF Generation ------------------ #


def test_generate_returns_valid_output(simple_data):
    """Generate() should produce a valid ECOFF and model output."""
    fitter = ECOFFitter(simple_data, distributions=1)
    ecoff, z_percentile, mu, sigma, model = fitter.generate(percentile=99)

    assert isinstance(ecoff, float)
    assert sigma > 0
    assert np.isclose(ecoff, fitter.dilution_factor**z_percentile, rtol=1e-5)


def test_compute_ecoff_matches_generate(simple_data):
    """compute_ecoff() and generate() should produce matching results."""
    fitter = ECOFFitter(simple_data)
    _, _, _, _, model = fitter.generate(99)
    res_direct = fitter.compute_ecoff(model, 99)
    res_from_generate = fitter.generate(99)[:-1]  # drop model
    assert np.allclose(res_direct[:2], res_from_generate[:2], atol=1e-6)


@pytest.mark.parametrize("percentile", [95, 97.5, 99])
def test_compute_ecoff_percentiles_increase(simple_data, percentile):
    """Higher percentiles should produce larger ECOFF values."""
    fitter = ECOFFitter(simple_data)
    _, _, _, _, model = fitter.generate(95)
    ecoff_low, *_ = fitter.compute_ecoff(model, 95)
    ecoff_high, *_ = fitter.compute_ecoff(model, 99)
    assert ecoff_high > ecoff_low


# ------------------ Integration-style check ------------------ #


def test_full_pipeline_mixture(simple_data):
    """Full pipeline test for a 2-component mixture."""
    fitter = ECOFFitter(simple_data, distributions=2)
    ecoff, z_percentile, mu1, sigma1, mu2, sigma2, model = fitter.generate(97.5)

    assert isinstance(ecoff, float)
    assert sigma1 > 0 and sigma2 > 0
    assert mu1 != mu2
    assert np.isfinite(z_percentile)


def simulate_intervals(mu, sigma, n=200, dilution_factor=2, tail_dilutions=1):
    """Generate synthetic censored MIC-style intervals in log2 scale."""
    y = np.random.normal(mu, sigma, size=n)
    y_low = y - np.random.uniform(0.05, 0.1, size=n)
    y_high = y + np.random.uniform(0.05, 0.1, size=n)
    weights = np.ones(n)
    return y_low, y_high, weights


def test_em_algorithm_converges_and_recovers_parameters():

    np.random.seed(1)

    # Known true parameters
    mu1_true, sigma1_true = 1.0, 0.2
    mu2_true, sigma2_true = 2.0, 0.3
    pi1_true = 0.6
    pi2_true = 1 - pi1_true

    # Generate mixed sample
    n = 300
    comp1 = np.random.normal(mu1_true, sigma1_true, int(pi1_true * n))
    comp2 = np.random.normal(mu2_true, sigma2_true, int(pi2_true * n))
    y = np.concatenate([comp1, comp2])
    y_low, y_high = y - 0.05, y + 0.05
    weights = np.ones_like(y)

    result = ECOFFitter._em_algorithm(
        y_low,
        y_high,
        weights,
        mu1=0.5,
        sigma1=0.5,
        mu2=2.5,
        sigma2=0.5,
        pi1=0.5,
        pi2=0.5,
        max_iter=200,
        tol=1e-5,
    )

    # --- Convergence checks ---
    assert result.converged, "EM algorithm did not converge"
    assert result.n_iter < 500

    # --- Parameter plausibility ---
    assert abs(result.params_["mu1"] - mu1_true) < 0.3
    assert abs(result.params_["mu2"] - mu2_true) < 0.3
    assert result.params_["pi1"] + result.params_["pi2"] == pytest.approx(1.0, rel=1e-6)

    # --- Log-likelihood finite ---
    assert np.isfinite(result.loglike)


@pytest.mark.parametrize("mu1,mu2", [(1.0, 1.0), (1.0, 5.0)])
def test_em_algorithm_handles_overlap_and_separation(mu1, mu2):
    """Test extreme overlap (mu1≈mu2) and complete separation."""
    y_low = np.array([0.5, 1, 2, 4])
    y_high = y_low + 0.1
    weights = np.ones_like(y_low)

    result = ECOFFitter._em_algorithm(
        y_low, y_high, weights, mu1, 0.2, mu2, 0.2, 0.5, 0.5, max_iter=100, tol=1e-5
    )

    assert result.converged
    assert 0 <= result.params_["pi1"] <= 1
    assert np.isfinite(result.loglike)


def test_fit_mixture_refinement_improves_likelihood(monkeypatch):
    np.random.seed(101)
    fitter = ECOFFitter(
        {"MIC": ["1", "2", "4"], "observations": [3, 5, 2]}, distributions=2
    )

    y_low, y_high, weights = fitter.define_intervals()

    # Mock EM output
    fake_em_result = ECOFFitter._em_algorithm(
        y_low, y_high, weights, 0.5, 0.3, 1.5, 0.3, 0.5, 0.5
    )
    monkeypatch.setattr(ECOFFitter, "_em_algorithm", lambda *a, **k: fake_em_result)

    result = fitter.fit_mixture(y_low, y_high, weights, options={"refine": True})

    assert result.converged
    assert "mu1" in result.params_
    assert np.isfinite(result.params_["mu1"])

