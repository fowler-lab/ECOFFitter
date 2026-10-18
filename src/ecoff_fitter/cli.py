#!/usr/bin/env python3
"""
Command-line interface for ECOFFitter.

Usage example:
    python -m ecoff_fitter.cli --input data/mic_data.csv --distributions 2 --percentile 99
"""

import argparse
from typing import Any, List, Optional
from ecoff_fitter import ECOFFitter
from ecoff_fitter.report import GenerateReport, CombinedReport
from ecoff_fitter.defence import validate_output_path
from ecoff_fitter.utils import read_multi_obs_input
from unittest.mock import MagicMock, patch
from ecoff_fitter.workflow import run_ecoff_analysis


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="ecoff_fitter",
        description="Estimate epidemiological cutoff values (ECOFFs) using interval regression on MIC data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to the input MIC dataset (CSV, TSV, XLSX, or XLS) "
            "with columns 'MIC' and assay name."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("ecoffitter", "ecoffinder"),
        default="ecoffitter",
        help="Fitting method to use (default: ECOFFitter).",
    )
    parser.add_argument(
        "--params",
        help=(
            "Optional YAML or TXT parameter file. ECOFFitter accepts "
            "dilution_factor, distributions, boundary_support, and percentile; "
            "ECOFFinder also accepts detection_level, cumulative, first_best, "
            "and intervals. File values override matching options."
        ),
    )
    parser.add_argument(
        "--dilution_factor",
        type=int,
        default=2,
        help="Dilution factor for MIC steps (default 2).",
    )
    parser.add_argument(
        "--distributions",
        type=int,
        default=1,
        help="Number of normal components to fit with ECOFFitter; ECOFFinder uses one WT range.",
    )
    parser.add_argument(
        "--boundary_support",
        type=int,
        default=1,
        help="boundary support for censored data handling (None to disable).",
    )
    parser.add_argument(
        "--density",
        action="store_true",
        help="Plot MIC distribution as density rather than counts.",
    )
    parser.add_argument(
        "--detection_level",
        type=float,
        default=10.0,
        help="Minimum ECOFFinder mode size as a percentage of observations.",
    )
    parser.add_argument(
        "--cumulative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fit cumulative frequencies with ECOFFinder; use --no-cumulative "
            "for per-dilution frequencies."
        ),
    )
    parser.add_argument(
        "--first_best",
        action="store_true",
        help="Stop ECOFFinder range search after its score starts increasing.",
    )
    parser.add_argument(
        "--intervals",
        action="store_true",
        help="Refit ECOFFinder's selected range with interval regression.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile to calculate the ECOFF (0–100).",
    )
    parser.add_argument(
        "--fit_individual",
        action="store_true",
        help="Fit individual populations in addition to the aggregate fit.",
    )
    parser.add_argument(
        "--outfile",
        help="Optional path to save ECOFF results to a text or pdf file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed model information and parameters.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Run the command-line interface.

    Args:
        argv (Optional[List[str]]): Argument list to parse; defaults to the
            process command line.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        workflow_args = {
            "params": args.params,
            "dilution_factor": args.dilution_factor,
            "distributions": args.distributions,
            "boundary_support": args.boundary_support,
            "percentile": args.percentile,
            "fit_individual": args.fit_individual,
            "density": args.density,
        }
        if args.method == "ecoffinder":
            workflow_args.update(
                method=args.method,
                detection_level=args.detection_level,
                cumulative=args.cumulative,
                first_best=args.first_best,
                intervals=args.intervals,
            )
        run = run_ecoff_analysis(args.input, **workflow_args)

        text = "\n\nECOFF RESULTS\n=====================================\n\n"
        text += run.global_report.to_text("GLOBAL FIT", verbose=args.verbose)

        if run.individual_reports:
            text += "\nINDIVIDUAL FITS:\n-------------------------------------\n"
            for name, report in run.individual_reports.items():
                text += report.to_text(label=name, verbose=args.verbose)

        if args.outfile:
            validate_output_path(args.outfile)

            if not run.individual_reports:
                if args.outfile.endswith(".pdf"):
                    run.global_report.save_pdf(args.outfile)
                else:
                    run.global_report.write_out(args.outfile)
            else:
                combined = CombinedReport(
                    args.outfile,
                    run.global_report,
                    run.individual_reports,
                )
                if args.outfile.endswith(".pdf"):
                    combined.save_pdf()
                else:
                    combined.write_out()

        print(text)

    except Exception as e:
        print("Error", str(e))


if __name__ == "__main__":
    main()
