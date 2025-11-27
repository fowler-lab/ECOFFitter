"""
ECOFF Fitter — Estimate epidemiological cutoff values (ECOFFs)
using interval regression on MIC (Minimum Inhibitory Concentration) data.

Main API:
    from ecoff_fitter import ECOFFitter
    fitter = ECOFFitter(input="mic_data.csv")
    ecoff, z, mu, sigma, model = fitter.generate(percentile=99)

CLI:
    python -m ecoff_fitter --input mic_data.csv --percentile 99
"""

from importlib.metadata import version, PackageNotFoundError

# --- Public API imports ---
from .core import ECOFFitter

__all__ = ["ECOFFitter"]

# --- Optional: version handling ---
try:
    __version__ = version("ecoff_fitter")
except PackageNotFoundError:
    __version__ = "0.0.0"

# --- Optional: CLI hook for `python -m ecoff_fitter` ---
def main():
    """Entry point for running ecoff_fitter as a module (CLI)."""
    from .cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
