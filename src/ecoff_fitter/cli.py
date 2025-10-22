import argparse
from ecoff_fitter import ECOFFitter


def parse_ecoff_generator():
    """
    Parse command-line options for the GenerateEcoff class.

    Returns:
        argparse.Namespace: Parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Generate ECOFF values for wild-type samples using interval regression."
    )
    parser.add_argument(
        "--samples",
        required=True,
        type=str,
        help="Path to the samples file containing 'UNIQUEID' and 'MIC' columns.",
    )
    parser.add_argument(
        "--mutations",
        required=True,
        type=str,
        help="Path to the mutations file containing 'UNIQUEID' and 'MUTATION' columns.",
    )
    parser.add_argument(
        "--dilution_factor",
        type=int,
        default=2,
        help="The factor for dilution scaling (default: 2 for doubling).",
    )
    parser.add_argument(
        "--censored",
        action="store_true",
        help="Flag to indicate if censored data is used (default: False).",
    )
    parser.add_argument(
        "--tail_dilutions",
        type=int,
        default=1,
        help="Number of dilutions to extend for interval tails if uncensored (default: 1).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99,
        help="The desired percentile for calculating the ECOFF (default: 99).",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="Optional path to save the ECOFF result to a file.",
    )
    return parser


def main_ecoff_generator(args):
    """
    Main function to execute ECOFF generation from the command line.
    """

    # Instantiate the GenerateEcoff class
    generator = EcoffFitter(
        samples=args.samples,
        mutations=args.mutations,
        dilution_factor=args.dilution_factor,
        censored=args.censored,
        tail_dilutions=args.tail_dilutions,
    )

    # Generate ECOFF
    ecoff, z_percentile, mu, sigma, model = generator.generate(
        percentile=args.percentile
    )

    # Display results
    print(f"ECOFF (Original Scale): {ecoff}")
    print(f"Percentile (Log Scale): {z_percentile}")
    print(f"Mean (mu): {mu}")
    print(f"Standard Deviation (sigma): {sigma}")

    # Optionally save results
    if args.outfile:
        with open(args.outfile, "w") as f:
            f.write(
                f"ECOFF: {ecoff}\n"
                f"Percentile (Log Scale): {z_percentile}\n"
                f"Mean (mu): {mu}\n"
                f"Standard Deviation (sigma): {sigma}\n"
                f"Model: {model}\n"
            )
