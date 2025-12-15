from dataclasses import dataclass
from typing import Any, Tuple, Optional, Dict, cast
from matplotlib.figure import Figure
from numpy.typing import NDArray
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from ecoff_fitter.graphs import plot_mic_distribution
import matplotlib.pyplot as plt


@dataclass
class GenerateReport:
    """
    Container for ECOFF fitting results and reporting utilities.

    Now stores a reference to the ECOFFitter instance (Option A),
    avoiding duplication of distributions, mus, sigmas, intervals, etc.
    """

    fitter: Any
    ecoff: float
    z: Tuple[float, float, float]  # Percentile-based ECOFFs (99, 97.5, 95)

    @classmethod
    def from_fitter(cls, fitter: Any, result: Tuple[Any, ...]) -> "GenerateReport":
        """
        Construct a GenerateReport from an ECOFFitter and generate() output.

        result = (ecoff, z_percentile, mu, sigma)   # single
        result = (ecoff, z_percentile, mu1, sigma1, ..., muK, sigmaK) # mixture
        """

        # Percentiles derived from fitter
        z0 = fitter.compute_ecoff(percentile=99)[0]
        z1 = fitter.compute_ecoff(percentile=97.5)[0]
        z2 = fitter.compute_ecoff(percentile=95)[0]

        ecoff = result[0]  # first element always ECOFF

        return cls(
            fitter=fitter,
            ecoff=ecoff,
            z=(z0, z1, z2),
        )

    @property
    def distributions(self) -> int:
        return cast(int, self.fitter.distributions)

    @property
    def dilution_factor(self) -> float:
        return cast(float, self.fitter.dilution_factor)

    @property
    def mus(self) -> NDArray[np.floating]:
        return cast(NDArray[np.floating], self.fitter.mus_)

    @property
    def sigmas(self) -> NDArray[np.floating]:
        return cast(NDArray[np.floating], self.fitter.sigmas_)

    @property
    def pis(self) -> Optional[NDArray[np.floating]]:
        return getattr(self.fitter, "pis_", None)

    @property
    def model(self) -> Any:
        return getattr(self.fitter, "model_", None)


    @property
    def intervals(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        return cast(
            tuple[
                NDArray[np.floating],
                NDArray[np.floating],
                NDArray[np.floating]
            ],
            self.fitter.define_intervals(),
        )

    def print_stats(self, verbose: bool = False) -> None:
        print(f"\nECOFF (original scale): {self.ecoff:.2}")

        if self.distributions == 1:
            mu = self.mus[0]
            sigma = self.sigmas[0]
            print(f"μ: {self.dilution_factor**mu:.2f}")
            print(f"σ (folds): {self.dilution_factor**sigma:.2f}")
        else:
            print("\nComponent means and sigmas (original scale):")
            for i, (mu, sigma) in enumerate(zip(self.mus, self.sigmas), start=1):
                print(
                    f"  μ{i}: {self.dilution_factor**mu:.4f}, "
                    f"σ{i} (folds): {self.dilution_factor**sigma:.4f}"
                )

        if verbose and self.model is not None:
            print("\n--- Model details ---")
            print(self.model)

    def write_out(self, path: str) -> None:
        z0, z1, z2 = self.z

        with open(path, "w") as f:
            f.write(f"ECOFF: {self.ecoff:.2f}\n")
            f.write(f"99th percentile: {z0:.2f}\n")
            f.write(f"97.5th percentile: {z1:.2f}\n")
            f.write(f"95th percentile: {z2:.2f}\n")

            if self.distributions == 1:
                mu = self.mus[0]
                sigma = self.sigmas[0]
                f.write(
                    f"μ: {self.dilution_factor**mu}, "
                    f"σ (folds): {self.dilution_factor**sigma}\n"
                )
            else:
                for i, (mu, sigma) in enumerate(zip(self.mus, self.sigmas), start=1):
                    f.write(
                        f"μ{i}: {self.dilution_factor**mu}, "
                        f"σ{i} (folds): {self.dilution_factor**sigma}\n"
                    )

        print(f"\nResults saved to: {path}")

    def save_pdf(self, outfile: str) -> None:
        with PdfPages(outfile) as pdf:
            fig = self._make_pdf()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"PDF report saved to: {outfile}")

    def _make_pdf(self, title: Optional[str] = None) -> Figure:
        fig, (ax_plot, ax_text) = plt.subplots(
            nrows=1, ncols=2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]}
        )

        low_log, high_log, weights = self.intervals

        plot_mic_distribution(
            low_log=low_log,
            high_log=high_log,
            pis=self.pis,
            weights=weights,
            dilution_factor=self.dilution_factor,
            mus=self.mus,
            sigmas=self.sigmas,
            log2_ecoff=np.log2(self.ecoff) if self.ecoff else None,
            ax=ax_plot,
        )
        ax_plot.legend(fontsize=7, frameon=False)

        if title:
            ax_plot.set_title(title)

        # Right-hand text -------------
        z0, z1, z2 = self.z
        ax_text.axis("off")

        lines = [
            f"ECOFF: {self.ecoff:.2f}",
            f"99th percentile: {z0:.2f}",
            f"97.5th percentile: {z1:.2f}",
            f"95th percentile: {z2:.2f}",
        ]

        if self.distributions == 1:
            mu = self.mus[0]
            sigma = self.sigmas[0]
            lines += [
                f"μ: {self.dilution_factor**mu:.2f}",
                f"σ: {self.dilution_factor**sigma:.2f}",
            ]
        else:
            for i, (mu, sigma) in enumerate(zip(self.mus, self.sigmas), start=1):
                lines.append(
                    f"μ{i}: {self.dilution_factor**mu:.4f}, "
                    f"σ{i} (folds): {self.dilution_factor**sigma:.4f}"
                )

        ax_text.text(
            0.05,
            0.9,
            "\n".join(lines),
            fontsize=11,
            va="top",
            family="monospace",
        )

        fig.tight_layout(rect=(0, 0, 1, 0.95))

        return fig


class CombinedReport:
    def __init__(
        self,
        outfile: str,
        global_report: GenerateReport,
        individual_reports: Dict[str, GenerateReport],
    ) -> None:
        """
        outfile: PDF filename
        global_report: GenerateReport instance
        individual_reports: dict {column_name: GenerateReport}
        """
        self.outfile = outfile
        self.global_report = global_report
        self.individual_reports = individual_reports

    def save_pdf(self) -> None:
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(self.outfile) as pdf:

            # ----- GLOBAL PAGE -----
            fig = self.global_report._make_pdf(title="GLOBAL FIT")
            pdf.savefig(fig)
            plt.close(fig)

            # ----- INDIVIDUAL PAGES -----
            for name, report in self.individual_reports.items():
                fig = report._make_pdf(title=f"INDIVIDUAL FIT: {name}")
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Combined PDF saved to {self.outfile}")
