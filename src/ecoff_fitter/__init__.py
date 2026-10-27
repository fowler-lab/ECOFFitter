"""ECOFF analysis tools for MIC data.

Direct interval-regression fitting:

    from ecoff_fitter import IntervalRegressionFitter

    fitter = IntervalRegressionFitter(input="mic_data.csv")
    result = fitter.generate(percentile=99)

Multi-assay workflow:

    from ecoff_fitter import run_ecoff_analysis

    run = run_ecoff_analysis("mic_data.csv", fitter="interval_regression")
"""

from importlib.metadata import PackageNotFoundError, version

from .intervals import IntervalRegressionFitter
from .workflow import ECOFFRun, run_ecoff_analysis

__all__ = [
    "IntervalRegressionFitter",
    "run_ecoff_analysis",
    "ECOFFRun",
]

try:
    __version__ = version("ecoff_fitter")
except PackageNotFoundError:
    __version__ = "0.0.0"


def main() -> None:
    """Run the command-line entry point.


    Returns:
        None.

    Args:
        None."""
    from .cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
