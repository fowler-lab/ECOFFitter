import pandas as pd
import os
import re


def validate_input_source(input):
    """
    Validate the input source for ECOFFitter.

    Args:
        input (str | pd.DataFrame | dict): Input MIC data or file path or dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the input type is invalid.
    """

    if isinstance(input, pd.DataFrame):
        if not {"MIC", "observations"}.issubset(input.columns):
            raise ValueError(
                "Input DataFrame must contain 'MIC' and 'observations' columns."
            )
    elif isinstance(input, dict):
        if not {"MIC", "observations"}.issubset(input.keys()):
            raise ValueError(
                "Input dictionary must contain 'MIC' and 'observations' keys."
            )
    elif isinstance(input, str):
        if not os.path.exists(input):
            raise FileNotFoundError(f"Input file not found: {input}")
    else:
        raise ValueError("Input must be a pandas DataFrame or a valid file path.")


def validate_params_source(params):
    """
    Pre-validate the params argument before attempting to read it.

    Args:
        params (dict | str | None): Parameter input.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
        ValueError: If params is not a dictionary, file path, or None.
    """
    if params is None:
        return

    if isinstance(params, str):
        if not os.path.exists(params):
            raise FileNotFoundError(f"Parameter file not found: {params}")
        return  # file will be parsed later

    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary, file path string, or None.")



def validate_mic_data(df):
    """
    Validate MIC and observations columns.

    Args:
        df (DataFrame): Input MIC data.

    Raises:
        ValueError: If invalid MIC formats or negative observation counts are found.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if (df["observations"] < 0).any():
        raise ValueError("Observation counts must be non-negative integers.")

    invalid_mic = df["MIC"].isna() | (df["MIC"].astype(str).str.strip() == "")
    if invalid_mic.any():
        raise ValueError("MIC column contains empty or invalid entries.")

    # MIC pattern check
    valid_pattern = re.compile(r"^(<=|>)?\d+(\.\d+)?$")
    bad_rows = df.loc[~df["MIC"].astype(str).str.match(valid_pattern)]
    if not bad_rows.empty:
        raise ValueError(f"Invalid MIC format found in rows: {bad_rows.index.tolist()}")


def validate_params(dilution_factor, distributions, tail_dilutions):
    """
    Validate ECOFFitter configuration values.

    Args:
        dilution_factor (int): The MIC dilution step.
        distributions (int): Number of components in the mixture.
        tail_dilutions (int | None): Number of tail dilutions to include.

    Raises:
        ValueError: If any parameter is outside acceptable range.
    """

    if not isinstance(dilution_factor, int) or dilution_factor <= 1:
        raise ValueError("dilution_factor must be an integer > 1.")
    
    if distributions not in [1, 2]:
        raise NotImplementedError("Only 1 or 2-component mixtures are supported.")
    
    if tail_dilutions is not None and (
        not isinstance(tail_dilutions, int) or tail_dilutions < 0
    ):
        raise ValueError("tail_dilutions must be a non-negative integer or None.")



def validate_output_path(path: str) -> bool:
    """
    Checks if the given path is safe and writable, and that the file extension is .txt or .pdf.

    Returns True if valid, otherwise raises ValueError.
    """
    # Check extension
    allowed_exts = ('.txt', '.pdf')
    if not path.lower().endswith(allowed_exts):
        raise ValueError(f"File must end with {allowed_exts}, got '{path}'")

    directory = os.path.dirname(path) or '.'

    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    if not os.access(directory, os.W_OK):
        raise PermissionError(f"No write permission in directory: {directory}")

    return True

