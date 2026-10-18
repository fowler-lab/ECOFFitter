from typing import Any, Dict, Tuple, Optional, cast
import pandas as pd
from pandas import DataFrame
import yaml
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedInput:
    """Normalized aggregate and per-column MIC observation tables.

    Attributes:
        global_df (DataFrame): MIC and summed observation counts.
        individual_dfs (Dict[str, DataFrame]): MIC and counts for each input
            group, keyed by its column name.
    """
    global_df: DataFrame
    individual_dfs: Dict[str, DataFrame]

    @property
    def has_individuals(self) -> bool:
        """Whether input preparation produced any individual group tables."""
        return len(self.individual_dfs) > 0



def read_input(
    data: DataFrame | list[Any] | tuple[Any, ...] | dict[str, Any] | str,
    sheet_name: Optional[str] = None,
) -> DataFrame:
    """
    Strict single-fit reader for ECOFFitter.

    Accepts raw MIC values or a canonical ``MIC`` and ``observations`` table.

    Args:
        data (DataFrame | list[Any] | tuple[Any, ...] | dict[str, Any] | str):
            Input observations or supported file path.
        sheet_name (Optional[str]): Worksheet name for Excel input.

    Returns:
        DataFrame: Canonical table with ``MIC`` and ``observations`` columns.

    Raises:
        ValueError: If the normalized input contains individual group columns.
    """
    prepared = prepare_input(data, sheet_name=sheet_name)

    if prepared.has_individuals:
        raise ValueError(
            "ECOFFitter expects a single MIC + observations table. "
            "For multi-column input, use prepare_input() or run_ecoff_analysis()."
        )

    return prepared.global_df

def prepare_input(
    data: DataFrame | list[Any] | tuple[Any, ...] | dict[str, Any] | str,
    sheet_name: Optional[str] = None,
) -> PreparedInput:
    """
    Normalize input for both single-fit and multi-fit workflows.

    Input rules:
    - 1 column only: treat as raw MIC values and collapse to MIC + observations.
    - MIC + observations: canonical single-fit input, no individual fits.
    - MIC + one or more additional columns: build a global fit plus one
      individual fit per additional column.

    Args:
        data (DataFrame | list[Any] | tuple[Any, ...] | dict[str, Any] | str):
            Raw MIC observations or supported file path.
        sheet_name (Optional[str]): Worksheet name for Excel input.

    Returns:
        PreparedInput: Aggregate table and any per-column tables.
    """
    df = _load_raw_frame(data, sheet_name=sheet_name)

    if df.shape[1] == 1:
        col = df.columns[0]
        mic_series = df[col].astype(str).str.strip()
        global_df = mic_series.to_frame(name="MIC").groupby("MIC").size().reset_index(name="observations")
        return PreparedInput(global_df=global_df, individual_dfs={})

    if "MIC" not in df.columns:
        raise ValueError("Input must contain a 'MIC' column.")

    obs_cols = [c for c in df.columns if c != "MIC"]
    if len(obs_cols) == 0:
        raise ValueError("Input must contain at least one additional column besides 'MIC'.")

    df = df.copy()
    df["MIC"] = df["MIC"].astype(str).str.strip()

    for col in obs_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # Canonical single-fit input: MIC + observations only
    if len(obs_cols) == 1 and obs_cols[0] == "observations":
        global_df = df[["MIC", "observations"]].copy()
        return PreparedInput(global_df=global_df, individual_dfs={})

    # Multi-column workflow:
    # one global fit (sum across all observation columns),
    # and one individual fit per column.
    individual_dfs: Dict[str, DataFrame] = {}
    for col in obs_cols:
        subdf = df[["MIC", col]].copy()
        subdf.rename(columns={col: "observations"}, inplace=True)
        individual_dfs[col] = subdf

    global_df = df[["MIC"]].copy()
    global_df["observations"] = df[obs_cols].sum(axis=1).astype(float)

    return PreparedInput(global_df=global_df, individual_dfs=individual_dfs)

def _load_raw_frame(
    data: DataFrame | list[Any] | tuple[Any, ...] | dict[str, Any] | str,
    sheet_name: Optional[str] = None,
) -> DataFrame:
    """Load input data into a DataFrame and normalize its column labels.

    Args:
        data: DataFrame, sequence, mapping, array-like object, or file path.
        sheet_name: Worksheet name when loading an Excel file.

    Returns:
        DataFrame: Loaded table with whitespace removed from column names.

    Raises:
        ValueError: If the input type or file extension is unsupported.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, (list, tuple)):
        df = pd.DataFrame({"MIC": data})
    elif hasattr(data, "__array__"):
        df = pd.DataFrame({"MIC": list(data)})
    elif isinstance(data, dict):
        df = pd.DataFrame.from_dict(data)
    elif isinstance(data, str):
        ext = os.path.splitext(data)[-1].lower()
        if ext == ".csv":
            df = pd.read_csv(data)
        elif ext in [".tsv", ".txt"]:
            df = pd.read_csv(data, sep=r"\s+")
        elif ext in [".xlsx", ".xls"]:
            val = pd.read_excel(data, sheet_name=sheet_name)
            df = next(iter(val.values())) if isinstance(val, dict) else val
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    else:
        raise ValueError("Input must be DataFrame, list, array, dict, or file path.")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_params(
    params: str | dict[str, Any],
    dflt_dilution: int,
    dflt_dists: int,
    dflt_tails: Optional[int],
) -> Tuple[int, int, Optional[int], float]:
    """
    Read ECOFF model parameters from a YAML, TXT, or dictionary source.

    Args:
        params (str | dict): File path to a YAML or text parameter file,
            or a dictionary containing configuration values.
        dflt_dilution (int): Default dilution factor.
        dflt_dists (int): Default number of distributions.
        dflt_tails (int | None): Default number of boundary support.

    Returns:
        tuple[int, int, Optional[int], float]: Dilution factor, component
            count, boundary support, and percentile.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file type is unsupported.
        AssertionError: If ``params`` is not a dictionary after path handling.
    """

    if isinstance(params, str):
        if not os.path.exists(params):
            raise FileNotFoundError(f"Parameter file not found: {params}")

        ext = os.path.splitext(params)[-1].lower()

        if ext in [".yaml", ".yml"]:
            with open(params, "r") as f:
                params = yaml.safe_load(f) or {}

        elif ext == ".txt":

            parsed: Dict[str, Any] = {}
            with open(params, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    key, val = [x.strip() for x in line.split("=", 1)]
                    if key == "dilution_factor":
                        parsed[key] = int(val)
                    elif key == "boundary_support":
                        if val.lower() == "none":
                            parsed[key] = None
                        else:
                            parsed[key] = int(val)
                    elif key == "distributions":
                        parsed[key] = int(val)
                    elif key == "percentile":
                        parsed[key] = float(val)
                    else:
                        parsed[key] = val
            params = parsed
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    else:
        assert isinstance(
            params, dict
        ), "params must either be a file path or a dictionary"

    # --- Apply defaults for any missing keys ---
    dilution_factor = params.get("dilution_factor", dflt_dilution)
    distributions = params.get("distributions", dflt_dists)
    boundary_support = params.get("boundary_support", dflt_tails)
    percentile = params.get("percentile", 99)

    return dilution_factor, distributions, boundary_support, percentile


def read_multi_obs_input(
    data: DataFrame | list[Any] | tuple[Any, ...] | dict[str, Any] | Any | str,
    sheet_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Return aggregate and individual tables prepared from the input.

    Args:
        data: Input observations accepted by :func:`prepare_input`.
        sheet_name: Worksheet name when loading an Excel file.

    Returns:
        Dict[str, Any]: Mapping with ``"global"`` and ``"individual"``
            entries.
    """
    prepared = prepare_input(data, sheet_name=sheet_name)
    return {"global": prepared.global_df, "individual": prepared.individual_dfs}
