import pytest
import numpy as np
import pandas as pd

from ecoff_fitter import ECOFFitter


#  __init__.py AND PACKAGE IMPORT TESTS

def test_ecoffitter_importable_from_root():
    """Package root must expose ECOFFitter."""
    import ecoff_fitter
    assert hasattr(ecoff_fitter, "ECOFFitter")
    assert ecoff_fitter.ECOFFitter is ECOFFitter


def test_main_invokes_cli(monkeypatch):
    """ecoff_fitter.main() must dispatch to ecoff_fitter.cli.main()."""
    import ecoff_fitter

    called = {"hit": False}

    def fake_main():
        called["hit"] = True

    monkeypatch.setattr("ecoff_fitter.cli.main", fake_main)
    ecoff_fitter.main()
    assert called["hit"] is True


#  DATA FIXTURES

@pytest.fixture
def simple_data():
    return pd.DataFrame({
        "MIC": ["0.5", "1", "2", "4", "8"],
        "observations": [5, 10, 20, 10, 5],
    })


@pytest.fixture
def censored_data():
    return pd.DataFrame({
        "MIC": ["<=0.5", "1", "2", ">4"],
        "observations": [3, 8, 12, 2],
    })



#  INITIALIZATION


def test_init_with_dataframe(simple_data):
    fitter = ECOFFitter(simple_data, dilution_factor=2, distributions=1)
    assert fitter.dilution_factor == 2
    assert fitter.distributions == 1
    assert "MIC" in fitter.obj_df.columns
    assert "observations" in fitter.obj_df.columns


#  INTERVAL CONSTRUCTION

def test_define_intervals_uncensored(simple_data):
    fitter = ECOFFitter(simple_data)
    y_low, y_high, weights = fitter.define_intervals()

    assert np.all(y_low < y_high)
    assert np.all(weights == simple_data["observations"].to_numpy())

    log2 = lambda x: np.log(x) / np.log(2)

    expected_low = log2(np.array([0.25, 0.5, 1, 2, 4]))
    expected_high = log2(np.array([0.5, 1, 2, 4, 8]))

    assert np.allclose(y_low, expected_low, atol=1e-3)
    assert np.allclose(y_high, expected_high, atol=1e-3)


def test_define_intervals_with_censoring(censored_data):
    fitter = ECOFFitter(censored_data, boundary_support=None)
    y_low, y_high, _ = fitter.define_intervals()

    # left censored: <=0.5 → low = ~0
    assert y_low[0] < np.log2(0.5)

    # right-censored: >4 → high = +inf
    assert np.isinf(y_high[-1])


def test_log_transf_intervals_direct(simple_data):
    fitter = ECOFFitter(simple_data)
    y_low = np.array([0.5, 1.0, 2.0])
    y_high = np.array([1.0, 2.0, 4.0])
    low_log, high_log = fitter.log_transf_intervals(y_low, y_high)

    assert np.allclose(low_log, np.log2(y_low))
    assert np.allclose(high_log, np.log2(y_high))


# ============================================================
#  FITTING — SINGLE & MIXTURE
# ============================================================

@pytest.mark.parametrize("dists", [1, 2])
def test_fit_model_runs(simple_data, dists):
    fitter = ECOFFitter(simple_data, distributions=dists)
    result = fitter.fit()

    # After fitting, Fitter must have mus_, sigmas_, etc.
    assert hasattr(fitter, "mus_")
    assert hasattr(fitter, "sigmas_")
    assert hasattr(fitter, "pis_")
    assert np.all(np.isfinite(fitter.mus_))
    assert np.all(fitter.sigmas_ > 0)
    assert np.all(fitter.pis_ >= 0)
    assert np.isclose(np.sum(fitter.pis_), 1.0)


def test_fit_mixture_returns_structure(simple_data):
    fitter = ECOFFitter(simple_data, distributions=2)
    fitter.fit()

    assert len(fitter.mus_) == 2
    assert len(fitter.sigmas_) == 2
    assert len(fitter.pis_) == 2

    assert np.isclose(fitter.pis_.sum(), 1.0)
    assert np.all(fitter.sigmas_ > 0)


# ============================================================
#  ECOFF CALCULATION
# ============================================================

def test_generate_returns_valid_output(simple_data):
    fitter = ECOFFitter(simple_data, distributions=1)
    ecoff, z_pt, mu, sigma = fitter.generate(99)

    assert isinstance(ecoff, float)
    assert sigma > 0
    assert np.isclose(ecoff, fitter.dilution_factor ** z_pt, rtol=1e-5)


def test_compute_ecoff_matches_generate(simple_data):
    fitter = ECOFFitter(simple_data)

    ecoff_g, z_g, mu_g, sig_g = fitter.generate(99)
    ecoff_d, z_d, mu_d, sig_d = fitter.compute_ecoff(99)

    assert np.isclose(ecoff_g, ecoff_d)
    assert np.isclose(z_g, z_d)
    assert np.isclose(mu_g, mu_d)
    assert np.isclose(sig_g, sig_d)


def test_compute_ecoff_percentiles_increase(simple_data):
    fitter = ECOFFitter(simple_data)
    fitter.fit()

    ecoff95, *_ = fitter.compute_ecoff(95)
    ecoff99, *_ = fitter.compute_ecoff(99)

    assert ecoff99 > ecoff95


# ============================================================
#  FULL MIxTURE PIPELINE
# ============================================================

def test_full_pipeline_mixture(simple_data):
    fitter = ECOFFitter(simple_data, distributions=2)
    results = fitter.generate(97.5)

    ecoff = results[0]
    z = results[1]
    mu1, sigma1, mu2, sigma2 = results[2:]

    assert isinstance(ecoff, float)
    assert sigma1 > 0 and sigma2 > 0
    assert mu1 != mu2
    assert np.isfinite(z)


# ============================================================
#  EM / Mixture STRESS TESTS (via MixtureModel inside fitter)
# ============================================================



@pytest.mark.parametrize("mu1,mu2", [(1.0, 1.0), (1.0, 5.0)])
def test_em_mixture_handles_overlap_and_separation(mu1, mu2):
    y_low = np.array([0.5, 1, 2, 4])
    y_high = y_low + 0.05
    weight = np.ones_like(y_low)

    df = pd.DataFrame({
        "MIC": ["1"] * len(y_low),
        "observations": weight,
    })

    fitter = ECOFFitter(df, distributions=2)
    fitter.y_low_ = np.log2(y_low)
    fitter.y_high_ = np.log2(y_high)
    fitter.weights_ = weight

    fitter.fit_mixture({"max_iter": 50})
    assert fitter.converged_
    assert 0 <= fitter.pis_[0] <= 1
    assert np.isfinite(fitter.loglike_)


# ============================================================
#  TEST refine=True BRANCH
# ============================================================

def test_fit_mixture_refine_flag(monkeypatch, simple_data):
    """Ensure refine=True path runs without error."""
    fitter = ECOFFitter(simple_data, distributions=2)

    # monkeypatch MixtureModel.fit to simulate convergence
    class FakeModel:
        def __init__(self, y_low, y_high, w, k): pass
        def fit(self, **kwargs):
            self.mus = np.array([0.0, 1.0])
            self.sigmas = np.array([0.3, 0.3])
            self.pis = np.array([0.5, 0.5])
            self.x = np.array([0.0, np.log(0.3), 1.0, np.log(0.3)])
            self.loglike = -10.0
            self.converged = True
            self.n_iter = 10

    monkeypatch.setattr("ecoff_fitter.mixture.MixtureModel", FakeModel)

    fitter.fit({"refine": True})

    assert fitter.converged_
    assert len(fitter.mus_) == 2
    assert np.isfinite(fitter.mus_[0])

def test_model_summary_basic(simple_data):
    """
    model_summary() should return a structured, self-consistent summary
    after fitting.
    """
    fitter = ECOFFitter(simple_data, distributions=2)
    fitter.fit()

    summary = fitter.model_summary()

    # basic structure
    assert isinstance(summary, dict)

    # required keys
    required_keys = {
        "model_family",
        "model_type",
        "components",
        "wild_type_component",
        "n_observations",
        "log_likelihood",
        "converged",
        "pis",
        "aic",
        "bic",
    }
    assert required_keys.issubset(summary.keys())

    # internal consistency
    assert summary["components"] == fitter.distributions
    assert summary["n_observations"] == fitter.obj_df.observations.sum()
    assert np.isclose(np.sum(summary["pis"]), 1.0)
    assert summary["converged"] is True
    assert summary["bic"] >= summary["aic"]

