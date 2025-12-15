import pandas as pd
from typing import Any
from pandas import DataFrame
import os
import re


def validate_input_source(input: str | DataFrame | dict[str, Any]) -> None:
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


def validate_params_source(
    params: dict[str, Any] | str | list[Any] | tuple[Any, ...] | DataFrame | Any | None,
) -> None:
    """
    Pre-validate the params argument before attempting to read it.

    Args:
        params (dict | str | list | array | DataFrame | None): Parameter input.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
        ValueError: If params is not a supported type.
    """

    if params is None:
        return

    if isinstance(params, str):
        if not os.path.exists(params):
            raise FileNotFoundError(f"Parameter file not found: {params}")
        return  # file will be read later

    if isinstance(params, dict):
        return

    if isinstance(params, pd.DataFrame):
        return

    if isinstance(params, (list, tuple)):
        return

    if hasattr(params, "__array__"):
        return

    raise ValueError(
        "params must be a dictionary, DataFrame, list, array, file path string, or None."
    )


def validate_mic_data(df: DataFrame) -> None:
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


def validate_params(
    dilution_factor: int, distributions: int, boundary_support: int | None
) -> None:
    """
    Validate ECOFFitter configuration values.

    Args:
        dilution_factor (int): The MIC dilution step.
        distributions (int): Number of components in the mixture.
        boundary_support (int | None): Number of boundary support to include.

    Raises:
        ValueError: If any parameter is outside acceptable range.
    """

    if not isinstance(dilution_factor, int) or dilution_factor <= 1:
        raise ValueError("dilution_factor must be an integer > 1.")

    if not isinstance(distributions, int):
        raise NotImplementedError(
            "The number of mixture components must be an integer."
        )

    if boundary_support is not None and (
        not isinstance(boundary_support, int) or boundary_support < 0
    ):
        raise ValueError("boundary_support must be a non-negative integer or None.")


def validate_output_path(path: str) -> bool:
    """
    Checks if the given path is safe and writable, and that the file extension is .txt or .pdf.

    Returns True if valid, otherwise raises ValueError.
    """
    # Check extension
    allowed_exts = (".txt", ".pdf")
    if not path.lower().endswith(allowed_exts):
        raise ValueError(f"File must end with {allowed_exts}, got '{path}'")

    directory = os.path.dirname(path) or "."

    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    if not os.access(directory, os.W_OK):
        raise PermissionError(f"No write permission in directory: {directory}")

    return True
