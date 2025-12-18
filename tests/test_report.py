import io
import sys
import pytest
from unittest.mock import MagicMock, patch, call, ANY

import ecoff_fitter.cli as cli
import ecoff_fitter.report as report


@pytest.fixture
def mock_loader(monkeypatch):
    """
    Ensures CLI never touches the filesystem and always receives
    one global dataset and zero individual datasets (simplest case).
    """
    monkeypatch.setattr(
        cli,
        "read_multi_obs_input",
        lambda _: {"global": MagicMock(), "individual": {}},
    )


@pytest.fixture
def mock_fitter(monkeypatch):
    mock = MagicMock(name="ECOFFitterMock")
    mock.generate.return_value = (4.0, 2.0, 1.0, 0.5)
    monkeypatch.setattr(cli, "ECOFFitter", MagicMock(return_value=mock))
    return mock


@pytest.fixture
def mock_report(monkeypatch):
    report_instance = MagicMock(name="ReportInstance")
    report_cls = MagicMock()
    report_cls.from_fitter.return_value = report_instance
    monkeypatch.setattr(cli, "GenerateReport", report_cls)
    return report_instance


@pytest.fixture
def mock_validate(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(cli, "validate_output_path", mock)
    return mock


# Basic parser tests

def test_build_parser_help(capsys):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    output = (capsys.readouterr().err or "").lower()
    assert "usage" in output
    assert "--input" in output


def test_parser_accepts_basic_args():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["--input", "data.csv", "--distributions", "2", "--percentile", "97.5"]
    )
    assert args.input == "data.csv"
    assert args.distributions == 2
    assert args.percentile == 97.5


# CLI integration tests

def test_main_runs_with_minimal_args(mock_loader, mock_fitter, mock_report):
    cli.main(["--input", "fake.csv"])

    # ECOFFitter MUST be called at least once
    assert cli.ECOFFitter.called

    # global fitter generate() must run once
    mock_fitter.generate.assert_called_once_with(percentile=99.0)

    # one GenerateReport created
    cli.GenerateReport.from_fitter.assert_called()



def test_main_invalid_percentile_prints_error(mock_loader, monkeypatch, capsys):
    mock_fitter = MagicMock()
    mock_fitter.generate.side_effect = AssertionError("percentile must be between 0 and 100")

    monkeypatch.setattr(cli, "ECOFFitter", MagicMock(return_value=mock_fitter))

    cli.main(["--input", "data.csv", "--percentile", "200"])

    out = capsys.readouterr().out.lower()
    assert "error" in out
    assert "percentile" in out


# GenerateReport tests

def test_generate_report_from_fitter_single_distribution():
    fitter = MagicMock()
    fitter.distributions = 1
    fitter.dilution_factor = 2
    fitter.mus_ = [1.0]
    fitter.sigmas_ = [0.5]
    fitter.define_intervals.return_value = ("low", "high", "weights")
    fitter.compute_ecoff.side_effect = [(10,), (8,), (6,)]

    r = report.GenerateReport.from_fitter(fitter, (4.0, None))

    assert r.ecoff == 4.0
    assert r.mus == [1.0]
    assert r.sigmas == [0.5]
    assert r.z == (10, 8, 6)


def test_generate_report_from_fitter_two_distributions():
    fitter = MagicMock()
    fitter.distributions = 2
    fitter.dilution_factor = 2
    fitter.mus_ = [1.0, 2.0]
    fitter.sigmas_ = [0.2, 0.5]
    fitter.define_intervals.return_value = ("low", "high", "weights")
    fitter.compute_ecoff.side_effect = [(10,), (8,), (6,)]

    r = report.GenerateReport.from_fitter(fitter, (4.0, None))

    assert r.mus == [1.0, 2.0]
    assert r.sigmas == [0.2, 0.5]
    assert r.z == (10, 8, 6)


# write_out
def test_generate_report_write_out(tmp_path):
    path = tmp_path / "out.txt"

    fitter = MagicMock()
    fitter.distributions = 2
    fitter.dilution_factor = 2
    fitter.mus_ = [1, 2]
    fitter.sigmas_ = [0.2, 0.4]
    fitter.define_intervals.return_value = ("a", "b", "c")

    r = cli.GenerateReport(fitter=fitter, ecoff=4.0, z=(10, 8, 6))

    r.write_out(str(path))
    text = path.read_text()

    assert "ECOFF: 4.0000" in text
    assert "log scale:" in text

    # Rounded 4-decimal sigma outputs
    assert "1.1487" in text    # 2**0.2
    assert "1.3195" in text    # 2**0.4


# save_pdf tests

@patch("ecoff_fitter.report.plot_mic_distribution")
@patch("ecoff_fitter.report.PdfPages")
@patch("ecoff_fitter.report.plt")
def test_generate_report_save_pdf(mock_plt, mock_pdf, mock_plot):
    fake_fig = MagicMock()
    fake_ax1 = MagicMock()
    fake_ax2 = MagicMock()

    mock_plt.subplots.return_value = (fake_fig, (fake_ax1, fake_ax2))
    mock_pdf.return_value.__enter__.return_value = MagicMock()

    fitter = MagicMock()
    fitter.distributions = 1
    fitter.dilution_factor = 2
    fitter.mus_ = [1]
    fitter.sigmas_ = [0.2]
    fitter.define_intervals.return_value = ("low", "high", "weights")

    r = cli.GenerateReport(fitter=fitter, ecoff=4.0, z=(10, 8, 6))

    r.save_pdf("out.pdf")

    mock_pdf.assert_called_once_with("out.pdf")
    fake_ax2.axis.assert_called_once_with("off")
    mock_plt.close.assert_called_once_with(fake_fig)


# to_text tests
def test_generate_report_to_text_single_dist():
    fitter = MagicMock()
    fitter.distributions = 1
    fitter.dilution_factor = 2
    fitter.mus_ = [1.0]
    fitter.sigmas_ = [0.5]

    r = cli.GenerateReport(fitter=fitter, ecoff=4.0, z=(10, 8, 6))
    text = r.to_text()

    assert "ECOFF: 4.0000" in text
    assert "log scale: 2.0000" in text
    assert "μ = 2.0000" in text
    assert "σ (folds) = 1.4142" in text


def test_generate_report_to_text_two_dist_verbose():
    fitter = MagicMock()
    fitter.distributions = 2
    fitter.dilution_factor = 2
    fitter.mus_ = [1, 2]
    fitter.sigmas_ = [0.2, 0.5]
    fitter.model_ = "FAKE_MODEL_DETAILS"

    r = cli.GenerateReport(fitter=fitter, ecoff=4.0, z=(10, 8, 6))
    text = r.to_text(verbose=True)

    assert "Component 1" in text
    assert "Component 2" in text
    assert "--- Model details ---" in text
    assert "FAKE_MODEL_DETAILS" in text
