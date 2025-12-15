import pytest
import pandas as pd
import os
import yaml
from ecoff_fitter.utils import read_input, read_params



def test_read_input_strict_numeric(tmp_path):
    # A CSV with messy spacing + numeric edge cases
    csv_path = tmp_path / "mic.csv"
    csv_path.write_text(
        "MIC,observations\n"
        " 1 , 3.0 \n"
        ">2, 4.7\n"
        "3.5, 0\n"
    )

    df = read_input(str(csv_path))

    # Check column types
    assert df["MIC"].dtype == object  # stringified, not numeric
    assert df["observations"].dtype == int  # forced to int

    # Exact expected values
    assert df["MIC"].tolist() == ["1", ">2", "3.5"]
    assert df["observations"].tolist() == [3, 4, 0]


def test_read_input_single_column_conversion():
    # A single-column MIC list containing duplicates
    data = pd.DataFrame({
        "MIC": ["0.06", "0.06", "0.12", "1", "1", "1"]
    })

    df = read_input(data)

    # Should create exactly these columns
    assert set(df.columns) == {"MIC", "observations"}

    # MIC values should aggregate correctly
    expected = pd.DataFrame({
        "MIC": ["0.06", "0.12", "1"],
        "observations": [2, 1, 3],
    })


def test_read_params_yaml_strict(tmp_path):
    defaults = (2, 1, None)

    yaml_path = tmp_path / "params.yaml"
    yaml_path.write_text(
        "dilution_factor: 8\n"
        "distributions: 2\n"
        "boundary_support: 5\n"
        "percentile: 95\n"
    )

    dilution, dists, tails, percentile = read_params(str(yaml_path), *defaults)

    # Strict numerical expectations
    assert dilution == 8
    assert isinstance(dilution, int)
    assert dists == 2
    assert tails == 5
    assert percentile == 95


def test_read_params_txt(tmp_path):
    defaults = (10, 1, None)

    txt_path = tmp_path / "params.txt"
    txt_path.write_text(
        "dilution_factor = 4\n"
        "distributions = 2\n"
        "boundary_support = None\n"
        "percentile = 99\n"
        "extra_value = 123\n"
    )

    dilution, dists, tails, percentile = read_params(str(txt_path), *defaults)

    # Strict numeric matches
    assert dilution == 4
    assert isinstance(dilution, int)
    assert dists == 2
    assert isinstance(dists, int)
    assert tails is None
    assert percentile == 99

    # Extra params should be ingested but ignored for output keys



def test_read_params_dict():
    defaults = (2, 1, None)

    params = {
        "dilution_factor": 16,
        "distributions": "2",      # test string that should remain a string (no coercion!)
        "boundary_support": None,
    }

    dilution, dists, tails, percentile = read_params(params, *defaults)

    # Only integer keys are parsed as provided, not coerced
    assert dilution == 16
    assert isinstance(dilution, int)

    # Should NOT coerce "2" to int — read_params only casts file values, not dict values
    assert dists == "2"
    assert isinstance(dists, str)

    # Defaults respected
    assert tails is None

    #check default percentile = 99 from parser
    assert percentile == 99


def test_read_params_txt_invalid_format(tmp_path):
    defaults = (2, 1, None)

    txt_path = tmp_path / "params.txt"
    txt_path.write_text("badline_without_equals\n")

    # Should raise because it tries to split on "=" and fails
    with pytest.raises(ValueError):
        read_params(str(txt_path), *defaults)
