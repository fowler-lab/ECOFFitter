import pandas as pd
import yaml
import os


def read_input(data, sheet_name=None):
    """
    Read MIC input data from a DataFrame, array-like, dict, or file
    and validate required columns. If given a single-column input,
    automatically aggregate it into MIC + observations format.

    Returns:
        DataFrame with columns:
            - MIC (str)
            - observations (int)
    """

    # Load into a DataFrame ----
    if isinstance(data, pd.DataFrame):
        df = data.copy()

    elif isinstance(data, (list, tuple)):
        df = pd.DataFrame({"MIC": data})

    elif hasattr(data, "__array__"):  # numpy arrays
        df = pd.DataFrame({"MIC": list(data)})

    elif isinstance(data, dict):
        df = pd.DataFrame.from_dict(data)

    elif isinstance(data, str):
        ext = os.path.splitext(data)[-1].lower()

        if ext in [".csv"]:
            df = pd.read_csv(data)
        elif ext in [".tsv", ".txt"]:
            df = pd.read_csv(data, sep=r"\s+")
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(data, sheet_name=sheet_name)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    else:
        raise ValueError("Input must be DataFrame, list, array, dict, or file path.")

    df.columns = [str(c).strip() for c in df.columns]

    # Handle single-column input automatically
    if df.shape[1] == 1:
        col = df.columns[0]
        df["MIC"] = df[col].astype(str).str.strip()
        
        df = (
            df.groupby("MIC")
            .size()
            .reset_index(name="observations")
        )

    expected = ["MIC", "observations"]
    missing = [c for c in expected if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"If using single-column input, it must be one column only."
        )

    df["MIC"] = df["MIC"].astype(str).str.strip()
    df["observations"] = (
        pd.to_numeric(df["observations"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df = df.dropna(subset=["MIC"]).reset_index(drop=True)
    return df

def read_params(params, dflt_dilution, dflt_dists, dflt_tails):
    """
    Read ECOFF model parameters from a file or dictionary, falling back to provided defaults.

    Args:
        params (str | dict): File path to a YAML or text parameter file,
            or a dictionary containing configuration values.
        dflt_dilution (int): Default dilution factor.
        dflt_dists (int): Default number of distributions.
        dflt_tails (int | None): Default number of tail dilutions.

    Returns:
        tuple: (dilution_factor, distributions, tail_dilutions)

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file type is unsupported.
        AssertionError: If input is not a valid file path or dictionary.
    """

    if isinstance(params, str):
        if not os.path.exists(params):
            raise FileNotFoundError(f"Parameter file not found: {params}")

        ext = os.path.splitext(params)[-1].lower()

        if ext in [".yaml", ".yml"]:
            with open(params, "r") as f:
                params = yaml.safe_load(f) or {}

        elif ext == ".txt":
            parsed = {}
            with open(params, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    key, val = [x.strip() for x in line.split("=", 1)]
                    if key == "dilution_factor":
                        parsed[key] = int(val)
                    elif key == "tail_dilutions":
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
    tail_dilutions = params.get("tail_dilutions", dflt_tails)
    percentile = params.get("percentile", None)

    return dilution_factor, distributions, tail_dilutions, percentile


def read_multi_obs_input(data, sheet_name=None):
    """
    Read MIC input but allow multiple observation columns.
    Returns a dict:
        {
            "global": df_global,             # aggregated observations
            "individual": {col: df_col}      # one df per obs column
        }

    Each returned df has:
        MIC (str)
        observations (int)
    """

    # --- Load input similarly to original read_input() ---
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
            df = pd.read_excel(data, sheet_name=sheet_name)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    else:
        raise ValueError("Input must be DataFrame, list, array, dict, or file path.")

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # If only 1 column, fallback to old logic
    if df.shape[1] == 1:
        col = df.columns[0]
        df["MIC"] = df[col].astype(str).str.strip()
        df_single = (
            df.groupby("MIC")
              .size()
              .reset_index(name="observations")
        )
        return {
            "global": df_single,
            "individual": {"observations": df_single.copy()}
        }

    # Require MIC column
    if "MIC" not in df.columns:
        raise ValueError("Input must contain a 'MIC' column.")

    df["MIC"] = df["MIC"].astype(str).str.strip()

    # Identify observation columns
    obs_cols = [c for c in df.columns if c != "MIC"]

    # Validate numeric obs columns
    for col in obs_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Build individual fitter DataFrames
    individual = {}
    for col in obs_cols:
        subdf = df[["MIC", col]].copy()
        subdf.rename(columns={col: "observations"}, inplace=True)
        individual[col] = subdf

    # Build global aggregated DataFrame
    df_global = df[["MIC"]].copy()
    df_global["observations"] = df[obs_cols].sum(axis=1).astype(int)

    return {
        "global": df_global,
        "individual": individual
    }

