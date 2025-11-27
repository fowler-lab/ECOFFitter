import io
import sys
import pytest
from unittest.mock import MagicMock, patch, call

import ecoff_fitter.cli as cli
import ecoff_fitter.report as report

#have called a combination of cli.GenerateREport and report.GenerateReport for depth

@pytest.fixture
def mock_fitter(monkeypatch):
    """Mock ECOFFitter to avoid real fitting."""
    mock = MagicMock(name="ECOFFitterMock")
    mock.generate.return_value = ("ecoff", "z", "mu", "sigma", "model")
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
    output = captured.err or captured.out  # argparse prints help to stderr
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


def test_generate_report_from_fitter_single_distribution():
    """from_fitter should correctly build a report for 1-distribution models."""
    fitter = MagicMock()
    fitter.distributions = 1
    fitter.dilution_factor = 2

    fitter.define_intervals.return_value = ("low", "high", "weights")

    fitter.compute_ecoff.side_effect = [
        (10.0,),   # 99
        (8.0,),    # 97.5
        (6.0,),    # 95
    ]

    # ecoff, _, mu, sigma, model
    result = (4.0, "ignored", 1.0, 0.5, "MODEL_OBJ")

    r = report.GenerateReport.from_fitter(fitter, result)

    assert r.ecoff == 4.0
    assert r.mu == 1.0
    assert r.sigma == 0.5
    assert r.mu1 is None    # ensures correct branch
    assert r.mu2 is None

    assert r.z == (10.0, 8.0, 6.0)
    assert r.intervals == ("low", "high", "weights")

    calls = [
        call("MODEL_OBJ", percentile=99),
        call("MODEL_OBJ", percentile=97.5),
        call("MODEL_OBJ", percentile=95),
    ]
    fitter.compute_ecoff.assert_has_calls(calls)


def test_generate_report_from_fitter_two_distributions():
    """from_fitter should correctly build a report for 2-distribution models."""
    # ---- Mock fitter with 2 distributions ----
    fitter = MagicMock()
    fitter.distributions = 2
    fitter.dilution_factor = 2

    # define_intervals should return something iterable
    fitter.define_intervals.return_value = ("low", "high", "weights")

    # compute_ecoff returns a tuple; we only use index 0
    fitter.compute_ecoff.side_effect = [
        (10.0,),  # 99th percentile
        (8.0,),  # 97.5th percentile
        (6.0,),  # 95th percentile
    ]

    # ---- Mock result tuple for 2-distribution branch ----
    # ecoff, _, mu1, sigma1, mu2, sigma2, model
    result = (4.0, "ignored", 1.0, 0.2, 2.0, 0.5, "MODEL_OBJ")

    report = cli.GenerateReport.from_fitter(fitter, result)

    assert report.ecoff == 4.0
    assert report.distributions == 2
    assert report.dilution_factor == 2
    assert report.mu1 == 1.0
    assert report.sigma1 == 0.2
    assert report.mu2 == 2.0
    assert report.sigma2 == 0.5
    assert report.model == "MODEL_OBJ"
    assert report.z == (10.0, 8.0, 6.0)

    assert report.intervals == ("low", "high", "weights")

    calls = [call("MODEL_OBJ", percentile=p) for p in (99, 97.5, 95)]
    fitter.compute_ecoff.assert_has_calls(calls, any_order=False)


def test_generate_report_write_out(tmp_path):
    """write_out should write correct text output for a 2-distribution report."""
    path = tmp_path / "out.txt"

    report = cli.GenerateReport(
        ecoff=4.0,
        z=(10.0, 8.0, 6.0),
        distributions=2,
        dilution_factor=2,
        mu1=1.0,
        sigma1=0.2,
        mu2=2.0,
        sigma2=0.4,
        model="MODEL",
        intervals=("a", "b", "c"),
    )

    report.write_out(str(path))

    text = path.read_text()

    # Check essential content
    assert "ECOFF: 4.00" in text
    assert "99th percentile: 10.00" in text
    assert "97.5th percentile: 8.00" in text
    assert "95th percentile: 6.00" in text

    # These appear unsafely (no rounding) but should still exist
    assert f"μ₁: {2**1.0}" in text
    assert f"σ₁: {2**0.2}" in text
    assert f"μ₂: {2**2.0}" in text
    assert f"σ₂: {2**0.4}" in text


@patch("ecoff_fitter.report.plot_mic_distribution")
@patch("ecoff_fitter.report.PdfPages")
@patch("ecoff_fitter.report.plt")
def test_generate_report_save_pdf(mock_plt, mock_pdf, mock_plot):
    """save_pdf should produce a PDF using PdfPages without touching real filesystem."""
    fake_fig = MagicMock(name="Figure")
    fake_ax1 = MagicMock(name="PlotAxis")
    fake_ax2 = MagicMock(name="TextAxis")

    # Mock plt.subplots return
    mock_plt.subplots.return_value = (fake_fig, (fake_ax1, fake_ax2))

    # Mock PdfPages context manager
    mock_pdf.return_value.__enter__.return_value = MagicMock()

    r = report.GenerateReport(
        ecoff=4.0,
        z=(10.0, 8.0, 6.0),
        distributions=1,
        dilution_factor=2,
        mu=1.0,
        sigma=0.2,
        model="MODEL",
        intervals=("low", "high", "weights"),
    )

    r.save_pdf("out.pdf")

    # PdfPages should be opened
    mock_pdf.assert_called_once_with("out.pdf")

    # plot_mic_distribution should be called exactly once
    mock_plot.assert_called_once()

    # Axes methods should be triggered
    fake_ax1.legend.assert_called_once()
    fake_ax2.axis.assert_called_once_with("off")

    # Figure must be closed
    mock_plt.close.assert_called_once_with(fake_fig)


def test_generate_report_print_stats_single_dist(capsys):
    r = cli.GenerateReport(
        ecoff=4.0,
        z=(10, 8, 6),
        distributions=1,
        dilution_factor=2,
        mu=1.0,
        sigma=0.5,
        intervals=("a", "b", "c"),
        model=None,
    )

    r.print_stats(verbose=False)
    out = capsys.readouterr().out

    assert "ECOFF (original scale): 4" in out
    assert "μ (mean): 2.00" in out  # 2**1.0
    assert "σ (std dev): 1.41" in out  # 2**0.5
    assert "Model details" not in out


def test_generate_report_print_stats_two_dist_verbose(capsys):
    r = cli.GenerateReport(
        ecoff=4.0,
        z=(10, 8, 6),
        distributions=2,
        dilution_factor=2,
        mu1=1.0,
        sigma1=0.2,
        mu2=2.0,
        sigma2=0.5,
        intervals=("a", "b", "c"),
        model="FAKE_MODEL_DETAILS",
    )

    r.print_stats(verbose=True)
    out = capsys.readouterr().out

    # Component stats
    assert "WT component" in out
    assert "μ=2.00" in out  # 2**1.0
    assert "σ=1.15" in out  # 2**0.2
    assert "Resistant component" in out
    assert "μ=4.00" in out  # 2**2.0
    assert "σ=1.41" in out  # 2**0.5

    # Verbose model printing
    assert "Model details" in out
    assert "FAKE_MODEL_DETAILS" in out
