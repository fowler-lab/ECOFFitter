"""
ECOFF Fitter — Estimate epidemiological cutoff values (ECOFFs)
using interval regression on MIC (Minimum Inhibitory Concentration) data.

Single-fit API
--------------
from ecoff_fitter import ECOFFitter

fitter = ECOFFitter(input="mic_data.csv")
result = fitter.generate(percentile=99)

Multi-fit / workflow API
------------------------
from ecoff_fitter import run_ecoff_analysis

run = run_ecoff_analysis(
    "mic_data.csv",
    fit_individual=True,
)

print(run.global_report.to_text())

CLI
---
python -m ecoff_fitter --input mic_data.csv
"""

from importlib.metadata import PackageNotFoundError, version

from .core import ECOFFitter
from .ecoffinder_core import ECOFFinder
from .workflow import ECOFFRun, run_ecoff_analysis

__all__ = [
    "ECOFFitter",
    "ECOFFinder",
    "run_ecoff_analysis",
    "ECOFFRun",
]

try:
    __version__ = version("ecoff_fitter")
except PackageNotFoundError:
    __version__ = "0.0.0"


def main() -> None:
    """Entry point for `python -m ecoff_fitter`."""
    from .cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
