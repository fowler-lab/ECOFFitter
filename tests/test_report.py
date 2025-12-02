import io
import sys
import pytest
from unittest.mock import MagicMock, patch, call

import ecoff_fitter.cli as cli
import ecoff_fitter.report as report


# have called a combination of cli.GenerateREport and report.GenerateReport for depth

@pytest.fixture
def mock_fitter(monkeypatch):
    """Mock ECOFFitter to avoid real fitting."""
    mock = MagicMock(name="ECOFFitterMock")

    # New result format: (ecoff, z, mu, sigma)
    mock.generate.return_value = (4.0, 2.0, 1.0, 0.5)

    monkeypatch.setattr(cli, "ECOFFitter", MagicMock(return_value=mock))
    return mock


@pytest.fixture
def mock_report(monkeypatch):
    """Mock GenerateReport to avoid plotting and file writing."""
    report_instance = MagicMock(name="ReportInstance")
    report_cls = MagicMock()
    report_cls.from_fitter.return_value = report_instance
    monkeypatch.setattr(cli, "GenerateReport", report_cls)
    return report_instance


@pytest.fixture
def mock_validate(monkeypatch):
    """Mock validate_output_path to skip filesystem checks."""
    mock = MagicMock()
    monkeypatch.setattr(cli, "validate_output_path", mock)
    return mock


# -----------------------------
# Basic parser and help tests
# -----------------------------

def test_build_parser_help(capsys):
    """Parser should show usage and required arguments."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    captured = capsys.readouterr()
    output = captured.err or captured.out
    assert "usage:" in output.lower()
    assert "--input" in output


def test_parser_accepts_basic_args():
    """Parser should correctly interpret key CLI arguments."""
    parser = cli.build_parser()
    args = parser.parse_args(
        ["--input", "data.csv", "--distributions", "2", "--percentile", "97.5"]
    )
    assert args.input == "data.csv"
    assert args.distributions == 2
    assert args.percentile == 97.5


# -----------------------------
# Integration-like CLI tests
# -----------------------------

def test_main_runs_with_minimal_args(mock_fitter, mock_report, monkeypatch):
    """Main should call ECOFFitter and GenerateReport correctly."""
    argv = ["--input", "fake.csv"]

    cli.main(argv)

    cli.ECOFFitter.assert_called_once_with(
        input="fake.csv",
        params=None,
        dilution_factor=2,
        distributions=1,
        tail_dilutions=1,
    )

    mock_fitter.generate.assert_called_once_with(percentile=99.0)
    cli.GenerateReport.from_fitter.assert_called_once()


def test_main_verbose_flag_triggers_print(mock_fitter, mock_report, capsys):
    """Verbose flag should pass through to print_stats(verbose=True)."""
    argv = ["--input", "data.csv", "--verbose"]
    cli.main(argv)
    mock_report.print_stats.assert_called_once_with(True)


def test_main_outfile_txt(mock_fitter, mock_report, mock_validate):
    """If outfile ends with .txt, report.write_out() should be used."""
    argv = ["--input", "file.csv", "--outfile", "result.txt"]
    cli.main(argv)
    mock_validate.assert_called_once_with("result.txt")
    mock_report.write_out.assert_called_once_with("result.txt")
    mock_report.save_pdf.assert_not_called()


def test_main_outfile_pdf(mock_fitter, mock_report, mock_validate):
    """If outfile ends with .pdf, report.save_pdf() should be used."""
    argv = ["--input", "file.csv", "--outfile", "report.pdf"]
    cli.main(argv)
    mock_validate.assert_called_once_with("report.pdf")
    mock_report.save_pdf.assert_called_once_with("report.pdf")
    mock_report.write_out.assert_not_called()


def test_main_invalid_percentile_raises(monkeypatch):
    """Percentile outside 0–100 should raise AssertionError."""
    mock_fitter = MagicMock()
    mock_fitter.generate.side_effect = AssertionError(
        "percentile must be between 0 and 100"
    )
    monkeypatch.setattr(cli, "ECOFFitter", MagicMock(return_value=mock_fitter))

    argv = ["--input", "data.csv", "--percentile", "200"]
    with pytest.raises(AssertionError):
        cli.main(argv)


# -----------------------------
# GenerateReport.from_fitter
# -----------------------------

def test_generate_report_from_fitter_single_distribution():
    """from_fitter should correctly build a report for 1-distribution models."""
    fitter = MagicMock()
    fitter.distributions = 1
    fitter.dilution_factor = 2

    fitter.mus_ = [1.0]
    fitter.sigmas_ = [0.5]

    fitter.define_intervals.return_value = ("low", "high", "weights")

    fitter.compute_ecoff.side_effect = [
        (10.0,),   # 99
        (8.0,),    # 97.5
        (6.0,),    # 95
    ]

    # result tuple is ignored for mus/sigmas in the new API
    result = (4.0, "ignored", 1.0, 0.5)

    r = report.GenerateReport.from_fitter(fitter, result)

    assert r.ecoff == 4.0
    assert r.mus == [1.0]
    assert r.sigmas == [0.5]

    assert r.z == (10.0, 8.0, 6.0)
    assert r.intervals == ("low", "high", "weights")



def test_generate_report_from_fitter_two_distributions():
    """from_fitter should correctly build a report for 2-distribution models."""
    fitter = MagicMock()
    fitter.distributions = 2
    fitter.dilution_factor = 2

    fitter.mus_ = [1.0, 2.0]
    fitter.sigmas_ = [0.2, 0.5]

    fitter.define_intervals.return_value = ("low", "high", "weights")

    fitter.compute_ecoff.side_effect = [
        (10.0,),  # 99
        (8.0,),   # 97.5
        (6.0,),   # 95
    ]

    # result tuple is ignored for mus/sigmas in new API
    result = (4.0, "ignored", 1.0, 0.2, 2.0, 0.5)

    r = report.GenerateReport.from_fitter(fitter, result)

    assert r.ecoff == 4.0
    assert r.mus == [1.0, 2.0]
    assert r.sigmas == [0.2, 0.5]

    assert r.z == (10.0, 8.0, 6.0)
    assert r.intervals == ("low", "high", "weights")



# -----------------------------
# Write-out tests
# -----------------------------
def test_generate_report_write_out(tmp_path):
    """write_out should write correct text output for a 2-distribution report."""
    path = tmp_path / "out.txt"

    # ---- Mock a minimal fitter ----
    fitter = MagicMock()
    fitter.distributions = 2
    fitter.dilution_factor = 2
    fitter.mus_ = [1.0, 2.0]
    fitter.sigmas_ = [0.2, 0.4]
    fitter.define_intervals.return_value = ("a", "b", "c")

    # ---- Build report using new API ----
    r = cli.GenerateReport(
        fitter=fitter,
        ecoff=4.0,
        z=(10.0, 8.0, 6.0),
    )

    r.write_out(str(path))
    text = path.read_text()

    # Check essential content
    assert "ECOFF: 4.00" in text
    assert "99th percentile: 10.00" in text
    assert "97.5th percentile: 8.00" in text
    assert "95th percentile: 6.00" in text

    # Component values must still appear exactly
    assert f"{2**1.0}" in text
    assert f"{2**0.2}" in text
    assert f"{2**2.0}" in text
    assert f"{2**0.4}" in text

@patch("ecoff_fitter.report.plot_mic_distribution")
@patch("ecoff_fitter.report.PdfPages")
@patch("ecoff_fitter.report.plt")
def test_generate_report_save_pdf(mock_plt, mock_pdf, mock_plot):
    """save_pdf should produce a PDF using PdfPages."""
    fake_fig = MagicMock(name="Figure")
    fake_ax1 = MagicMock(name="PlotAxis")
    fake_ax2 = MagicMock(name="TextAxis")

    mock_plt.subplots.return_value = (fake_fig, (fake_ax1, fake_ax2))
    mock_pdf.return_value.__enter__.return_value = MagicMock()

    # ---- mock fitter ----
    fitter = MagicMock()
    fitter.distributions = 1
    fitter.dilution_factor = 2
    fitter.mus_ = [1.0]
    fitter.sigmas_ = [0.2]
    fitter.define_intervals.return_value = ("low", "high", "weights")

    r = cli.GenerateReport(
        fitter=fitter,
        ecoff=4.0,
        z=(10.0, 8.0, 6.0),
    )

    r.save_pdf("out.pdf")

    mock_pdf.assert_called_once_with("out.pdf")
    mock_plot.assert_called_once()
    fake_ax1.legend.assert_called_once()
    fake_ax2.axis.assert_called_once_with("off")
    mock_plt.close.assert_called_once_with(fake_fig)

def test_generate_report_print_stats_single_dist(capsys):
    # ---- Mock fitter ----
    fitter = MagicMock()
    fitter.distributions = 1
    fitter.dilution_factor = 2
    fitter.mus_ = [1.0]
    fitter.sigmas_ = [0.5]
    fitter.define_intervals.return_value = ("a", "b", "c")

    r = cli.GenerateReport(
        fitter=fitter,
        ecoff=4.0,
        z=(10, 8, 6),
    )

    r.print_stats(verbose=False)
    out = capsys.readouterr().out

    assert "ECOFF (original scale): 4" in out
    assert "μ: 2.00" in out     # 2**1.0
    assert "σ (folds): 1.41" in out  # 2**0.5
    assert "Model details" not in out

def test_generate_report_print_stats_two_dist_verbose(capsys):
    # ---- Mock fitter ----
    fitter = MagicMock()
    fitter.distributions = 2
    fitter.dilution_factor = 2
    fitter.mus_ = [1.0, 2.0]
    fitter.sigmas_ = [0.2, 0.5]
    fitter.define_intervals.return_value = ("a", "b", "c")
    fitter.model_ = "FAKE_MODEL_DETAILS"

    r = cli.GenerateReport(
        fitter=fitter,
        ecoff=4.0,
        z=(10, 8, 6),
    )

    r.print_stats(verbose=True)
    out = capsys.readouterr().out

    # WT component (lowest mean)
    assert "μ1: 2.0000" in out
    assert "σ1 (folds): 1.1487" in out   # 2**0.2

    # Resistant component
    assert "μ2: 4.0000" in out
    assert "σ2 (folds): 1.4142" in out   # 2**0.5


    assert "Model details" in out
    assert "FAKE_MODEL_DETAILS" in out
