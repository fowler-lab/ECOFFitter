import pytest
import pandas as pd
import os
from ecoff_fitter.defence import (
    validate_input_source,
    validate_params_source,
    validate_mic_data,
    validate_params,
    validate_output_path,
)


def test_validate_input_source():
    # valid DataFrame
    df_ok = pd.DataFrame({"MIC": [1], "observations": [2]})
    validate_input_source(df_ok)

    # bad DataFrame
    df_bad = pd.DataFrame({"MIC": [1]})
    with pytest.raises(ValueError):
        validate_input_source(df_bad)

    # valid dict
    validate_input_source({"MIC": [1], "observations": [2]})

    # bad dict
    with pytest.raises(ValueError):
        validate_input_source({"MIC": [1]})

    # missing file
    with pytest.raises(FileNotFoundError):
        validate_input_source("no_such_file.txt")

    # invalid type
    with pytest.raises(ValueError):
        validate_input_source(123)


def test_validate_params_source(tmp_path):
    validate_params_source(None)  # OK

    # missing file path
    with pytest.raises(FileNotFoundError):
        validate_params_source("missing.json")

    # valid dict
    validate_params_source({"a": 1})

    # invalid type
    with pytest.raises(ValueError):
        validate_params_source(123)



def test_validate_mic_data():
    # valid
    df_ok = pd.DataFrame({"MIC": ["1", "2.0", ">8"], "observations": [1, 2, 3]})
    validate_mic_data(df_ok)

    # empty
    with pytest.raises(ValueError):
        validate_mic_data(pd.DataFrame({"MIC": [], "observations": []}))

    # negative obs
    with pytest.raises(ValueError):
        validate_mic_data(pd.DataFrame({"MIC": ["1"], "observations": [-1]}))

    # empty/invalid MIC
    with pytest.raises(ValueError):
        validate_mic_data(pd.DataFrame({"MIC": [""], "observations": [1]}))

    # bad format
    with pytest.raises(ValueError):
        validate_mic_data(pd.DataFrame({"MIC": ["bad"], "observations": [1]}))


def test_validate_params():
    # valid settings
    validate_params(2, 1, 0)
    validate_params(2, 2, None)

    # invalid dilution_factor
    with pytest.raises(ValueError):
        validate_params(1, 1, 0)

    # invalid distributions
    with pytest.raises(NotImplementedError):
        validate_params(2, 3, 0)

    # invalid tail_dilutions
    with pytest.raises(ValueError):
        validate_params(2, 1, -1)


def test_validate_output_path(tmp_path, monkeypatch):
    # OK
    path_ok = tmp_path / "output.txt"
    assert validate_output_path(str(path_ok))

    # bad extension
    with pytest.raises(ValueError):
        validate_output_path(str(tmp_path / "bad.exe"))

    # missing directory
    with pytest.raises(ValueError):
        validate_output_path("no_such_dir/file.txt")

    # no write permission
    locked = tmp_path / "locked"
    locked.mkdir()
    monkeypatch.setattr(os, "access", lambda *args: False)
    with pytest.raises(PermissionError):
        validate_output_path(str(locked / "file.txt"))
