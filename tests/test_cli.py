import pytest
from unittest.mock import MagicMock
import ecoff_fitter.cli as cli


@pytest.fixture
def mock_loader(monkeypatch):
    """
    Pretend that the input contains only a single global dataset
    and no individual datasets.
    """
    monkeypatch.setattr(
        cli,
        "read_multi_obs_input",
        lambda path: {"global": MagicMock(), "individual": {}},
    )


@pytest.fixture
def mock_fitter(monkeypatch):
    """Mock ECOFFitter so it always returns a simple fixed result."""
    mock_instance = MagicMock()
    mock_instance.generate.return_value = (4.0, 2.0, 1.0, 0.5)

    monkeypatch.setattr(cli, "ECOFFitter", MagicMock(return_value=mock_instance))
    return mock_instance


@pytest.fixture
def mock_report(monkeypatch):
    """Mock GenerateReport.from_fitter."""
    report_instance = MagicMock()
    report_cls = MagicMock()
    report_cls.from_fitter.return_value = report_instance

    monkeypatch.setattr(cli, "GenerateReport", report_cls)
    return report_instance


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

def test_main_runs_minimal(mock_loader, mock_fitter, mock_report, capsys):
    cli.main(["--input", "fake.csv"])
    # Output contains header
    out = capsys.readouterr().out
    assert "ECOFF RESULTS" in out
    # ECOFFitter used
    assert cli.ECOFFitter.called
    # global fitter generate called
    mock_fitter.generate.assert_called_once()


def test_main_outfile_txt(mock_loader, mock_fitter, mock_report, mock_validate):
    cli.main(["--input", "fake.csv", "--outfile", "results.txt"])
    mock_validate.assert_called_once_with("results.txt")


def test_main_outfile_pdf(mock_loader, mock_fitter, mock_report, mock_validate):
    cli.main(["--input", "fake.csv", "--outfile", "report.pdf"])
    mock_validate.assert_called_once_with("report.pdf")


def test_main_runs_with_multiple_individuals(monkeypatch, mock_fitter, mock_report, mock_validate):
    """
    Ensure the CLI runs when multiple individual datasets are present
    and that CombinedReport is invoked.
    """

    # Make two fake individual datasets
    fake_global = MagicMock()
    fake_indiv1 = MagicMock()
    fake_indiv2 = MagicMock()

    def fake_loader(path):
        return {
            "global": fake_global,
            "individual": {
                "A": fake_indiv1,
                "B": fake_indiv2,
            },
        }

    monkeypatch.setattr(cli, "read_multi_obs_input", fake_loader)

    # Mock CombinedReport so we can detect when it's used
    combined_instance = MagicMock()
    combined_cls = MagicMock(return_value=combined_instance)
    monkeypatch.setattr(cli, "CombinedReport", combined_cls)

    cli.main(["--input", "fake.csv", "--outfile", "combined.pdf"])

    mock_validate.assert_called_once_with("combined.pdf")

    combined_cls.assert_called_once()
    combined_instance.save_pdf.assert_called_once()
