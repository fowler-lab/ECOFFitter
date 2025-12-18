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
        "--params",
        help=(
            "Optional path to a parameter file (YAML, TXT, or key=value list) "
            "defining dilution_factor, distributions, and boundary_support. "
            "Overrides manual CLI options if provided."
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
        help="Number of normal components to fit.",
    )
    parser.add_argument(
        "--boundary_support",
        type=int,
        default=1,
        help="boundary support for censored data handling (None to disable).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile to calculate the ECOFF (0–100).",
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
    """Main entry point for the ECOFFitter CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:

        data_dict = read_multi_obs_input(args.input)
        df_global = data_dict['global']
        df_individual = data_dict['individual']

        global_fitter = ECOFFitter(
            input=df_global,
            params=args.params,
            distributions=args.distributions,
            boundary_support=args.boundary_support,
            dilution_factor=args.dilution_factor
        )

        global_result = global_fitter.generate(percentile=args.percentile)

        individual_results = {}
        for col, subdf in df_individual.items():

            fitter = ECOFFitter(
                input=subdf,
                params=args.params,
                dilution_factor=args.dilution_factor,
                distributions=args.distributions,
                boundary_support=args.boundary_support,
            )

            result = fitter.generate(percentile=args.percentile)
            individual_results[col] = (fitter, result)

        text = "\n\nECOFF RESULTS\n=====================================\n\n"

        global_report = GenerateReport.from_fitter(global_fitter, global_result)

        if len(individual_results) > 1:
            text += global_report.to_text("GLOBAL FIT")
            text += "\nINDIVIDUAL FITS:\n-------------------------------------\n"

        # Individual fits
        for name, (fitter, result) in individual_results.items():
            rep = GenerateReport.from_fitter(fitter, result)
            text += rep.to_text(label=name)       
        
        if args.outfile:
            validate_output_path(args.outfile)
            if len(individual_results.keys())==1:
                
                if args.outfile.endswith(".pdf"):
                    global_report.save_pdf(args.outfile)
                else:
                    global_report.write_out(args.outfile)
            elif (len(individual_results.keys()))>1:
                # Build section reports
                indiv_reports = {
                    name: GenerateReport.from_fitter(fitter, result)
                    for name, (fitter, result) in individual_results.items()
                }
                # Build combined PDF
                combined = CombinedReport(args.outfile, global_report, indiv_reports)
                if args.outfile.endswith(".pdf"):
                    combined.save_pdf()
                else:
                    combined.write_out()

        print (text)

    except Exception as e:
        print ('Error', str(e))


if __name__ == "__main__":
    main()
