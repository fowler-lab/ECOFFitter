#!/usr/bin/env python3
"""
Command-line interface for ECOFF analysis.

Usage example:
    python -m ecoff_fitter.cli --input data/mic_data.csv --distributions 2 --percentile 99
"""

import argparse
from typing import List, Optional
from ecoff_fitter.report import GenerateReport, CombinedReport
from ecoff_fitter.defence import validate_output_path
from ecoff_fitter.workflow import run_ecoff_analysis


def _parse_optional_int(value: str) -> int | None:
    """Parse a non-negative integer or the word ``none``.

    Args:
        value (str): Text supplied for an optional integer argument.

    Returns:
        int | None: Parsed integer, or None when disabled.
    """
    if value.strip().lower() == "none":
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer or 'none'") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer or 'none'")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.


    Returns:
        argparse.ArgumentParser: Configured command-line parser.

    Args:
        None."""
    parser = argparse.ArgumentParser(
        prog="ecoff_fitter",
        description="Estimate epidemiological cutoff values (ECOFFs) from MIC data.",
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
        "--fitter",
        choices=("least_squares", "interval_regression"),
        help="Model fitting method: least squares or interval regression.",
    )
    parser.add_argument(
        "--range_search",
        choices=("largest-bin", "first-local-peak", "mixture"),
        help="Wild-type selector: largest bin, first local peak, or mixture model.",
    )
    parser.add_argument(
        "--params",
        help=(
            "Optional YAML or TXT parameter file for dilution, component count, "
            "censoring support, percentile, and range-search settings. File "
            "values override matching options."
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
        help="Number of normal components for interval regression or mixture selection.",
    )
    parser.add_argument(
        "--boundary_support",
        type=_parse_optional_int,
        default=1,
        help="Number of dilution steps used as support for censored observations.",
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
        help="Minimum eligible local-peak size as a percentage of observations.",
    )
    parser.add_argument(
        "--park_from_mic",
        type=float,
        help=(
            "Exclude this MIC and higher bins before largest-bin selection."
        ),
    )
    parser.add_argument(
        "--cumulative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fit cumulative frequencies; use --no-cumulative for per-bin frequencies."
        ),
    )
    parser.add_argument(
        "--first_best",
        action="store_true",
        help="Stop range search after its score starts increasing.",
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

    Returns:
        None."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.range_search == "mixture" and args.distributions < 2:
        parser.error("--range_search mixture requires --distributions of at least 2")

    try:
        workflow_args = {
            "params": args.params,
            "dilution_factor": args.dilution_factor,
            "distributions": args.distributions,
            "boundary_support": args.boundary_support,
            "percentile": args.percentile,
            "fit_individual": args.fit_individual,
            "density": args.density,
            "fitter": args.fitter or "interval_regression",
            "range_search": args.range_search,
            "detection_level": args.detection_level,
            "cumulative": args.cumulative,
            "first_best": args.first_best,
            "park_from_mic": args.park_from_mic,
        }
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
