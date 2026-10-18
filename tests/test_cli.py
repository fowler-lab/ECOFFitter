import pytest
from unittest.mock import MagicMock
import ecoff_fitter.cli as cli


@pytest.fixture
def mock_run(monkeypatch):
    """Return a mocked workflow result for CLI tests."""
    run = MagicMock()
    run.global_report.to_text.return_value = "GLOBAL FIT\n"
    run.individual_reports = {}
    monkeypatch.setattr(cli, "run_ecoff_analysis", MagicMock(return_value=run))
    return run


@pytest.fixture
def mock_validate(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(cli, "validate_output_path", mock)
    return mock


def test_parser_accepts_basic_args():
    parser = cli.build_parser()
    args = parser.parse_args(["--input", "data.csv", "--percentile", "95"])
    assert args.input == "data.csv"
    assert args.percentile == 95


def test_parser_requires_input():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

def test_main_runs_minimal(mock_run, capsys):
    cli.main(["--input", "fake.csv"])
    # Output contains header
    out = capsys.readouterr().out
    assert "ECOFF RESULTS" in out
    cli.run_ecoff_analysis.assert_called_once()


def test_main_outfile_txt(mock_run, mock_validate):
    cli.main(["--input", "fake.csv", "--outfile", "results.txt"])
    mock_validate.assert_called_once_with("results.txt")
    mock_run.global_report.write_out.assert_called_once_with("results.txt")


def test_main_outfile_pdf(mock_run, mock_validate):
    cli.main(["--input", "fake.csv", "--outfile", "report.pdf"])
    mock_validate.assert_called_once_with("report.pdf")
    mock_run.global_report.save_pdf.assert_called_once_with("report.pdf")


def test_main_runs_with_multiple_individuals(monkeypatch, mock_validate):
    """
    Ensure the CLI runs when multiple individual datasets are present
    and that CombinedReport is invoked.
    """

    run = MagicMock()
    run.individual_reports = {"A": MagicMock(), "B": MagicMock()}
    monkeypatch.setattr(cli, "run_ecoff_analysis", MagicMock(return_value=run))

    # Mock CombinedReport so we can detect when it's used
    combined_instance = MagicMock()
    combined_cls = MagicMock(return_value=combined_instance)
    monkeypatch.setattr(cli, "CombinedReport", combined_cls)

    cli.main(["--input", "fake.csv", "--outfile", "combined.pdf", "--fit_individual"])

    mock_validate.assert_called_once_with("combined.pdf")

    combined_cls.assert_called_once()
    combined_instance.save_pdf.assert_called_once()


def test_main_default_skips_individual_fits(monkeypatch, mock_validate):
    run = MagicMock()
    run.individual_reports = {}
    monkeypatch.setattr(cli, "run_ecoff_analysis", MagicMock(return_value=run))

    combined_cls = MagicMock()
    monkeypatch.setattr(cli, "CombinedReport", combined_cls)

    cli.main(["--input", "fake.csv", "--outfile", "combined.pdf"])

    combined_cls.assert_not_called()
    run.global_report.save_pdf.assert_called_once_with("combined.pdf")
