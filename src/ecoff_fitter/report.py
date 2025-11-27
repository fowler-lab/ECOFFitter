from dataclasses import dataclass
from typing import Any
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from ecoff_fitter.graphs import plot_mic_distribution
import matplotlib.pyplot as plt


@dataclass
class GenerateReport:
    """
    Container for ECOFF fitting results and reporting utilities.

    This class stores fitted model outputs, summary statistics,
    and provides methods to print, write, and visualize ECOFF results.
    """

    ecoff: float
    z: tuple
    distributions: int
    dilution_factor: int
    mu: float | None = None
    sigma: float | None = None
    mu1: float | None = None
    sigma1: float | None = None
    mu2: float | None = None
    sigma2: float | None = None
    model: Any = None
    intervals: tuple = None

    @classmethod
    def from_fitter(cls, fitter, result):
        """
        Factory method to construct a GenerateReport object from an ECOFFitter instance
        and a fitted model result.

        Args:
            fitter (ECOFFitter): The ECOFFitter object used to fit the data.
            result (tuple): The tuple returned from `ECOFFitter.generate()`,
                containing the ECOFF, z-score percentile(s), parameter estimates,
                and fitted model object.

        Returns:
            GenerateReport: A populated report object with fitted parameters,
            percentile calculations, and interval data for plotting.
        """
        intervals = fitter.define_intervals()

        if fitter.distributions == 1:
            ecoff, _, mu, sigma, model = result

            z0 = fitter.compute_ecoff(model, percentile=99)[0]
            z1 = fitter.compute_ecoff(model, percentile=97.5)[0]
            z2 = fitter.compute_ecoff(model, percentile=95)[0]

            return cls(
                ecoff,
                (z0, z1, z2),
                fitter.distributions,
                fitter.dilution_factor,
                mu=mu,
                sigma=sigma,
                model=model,
                intervals=intervals,
            )
        elif fitter.distributions == 2:
            ecoff, _, mu1, sigma1, mu2, sigma2, model = result

            z0 = fitter.compute_ecoff(model, percentile=99)[0]
            z1 = fitter.compute_ecoff(model, percentile=97.5)[0]
            z2 = fitter.compute_ecoff(model, percentile=95)[0]

            return cls(
                ecoff,
                (z0, z1, z2),
                fitter.distributions,
                fitter.dilution_factor,
                mu1=mu1,
                sigma1=sigma1,
                mu2=mu2,
                sigma2=sigma2,
                model=model,
                intervals=intervals,
            )
        else:
            raise ValueError(
                f"Unsupported number of distributions: {fitter.distributions}"
            )

    def print_stats(self, verbose=False):
        """
        Print key ECOFF statistics and fitted parameter estimates to console.

        Args:
            verbose (bool): If True, also prints the full model details object.

        Displays:
            ECOFF value, mean(s), and standard deviation(s) on the original MIC scale.
        """
        print(f"\nECOFF (original scale): {self.ecoff:.2}")
        if self.distributions == 1:
            print(f"μ (mean): {self.dilution_factor**self.mu:.2f}")
            print(f"σ (std dev): {self.dilution_factor**self.sigma:.2f}")
        else:
            print(
                f"WT component: μ={self.dilution_factor**self.mu1:.2f}, σ={self.dilution_factor**self.sigma1:.2f}"
            )
            print(
                f"Resistant component: μ={self.dilution_factor**self.mu2:.4f}, σ={self.dilution_factor**self.sigma2:.2f}"
            )

        if verbose and self.model is not None:
            print("\n--- Model details ---")
            print(self.model)

    def write_out(self, path: str):
        """
        Write ECOFF summary statistics and fitted parameter values to a text file.

        Args:
            path (str): File path to save the output text report.

        Output:
            Creates a plain-text file listing ECOFF, percentile values,
            and fitted Gaussian component parameters on the original MIC scale.
        """
        z0, z1, z2 = self.z

        with open(path, "w") as f:
            f.write(f"ECOFF: {self.ecoff:.2f}\n")
            f.write(f"99th percentile: {z0:.2f}\n")
            f.write(f"97.5th percentile: {z1:.2f}\n")
            f.write(f"95th percentile: {z2:.2f}\n")

            if self.distributions == 1:
                f.write(
                    f"μ: {self.dilution_factor**self.mu}, σ: {self.dilution_factor**self.sigma}\n"
                )
            else:
                f.write(
                    f"μ₁: {self.dilution_factor**self.mu1}, σ₁: {self.dilution_factor**self.sigma1}\n"
                    f"μ₂: {self.dilution_factor**self.mu2}, σ₂: {self.dilution_factor**self.sigma2}\n"
                )
        print(f"\nResults saved to: {path}")

    def save_pdf(self, outfile: str):
        """
        Generate a formatted PDF report summarizing the ECOFF analysis.

        The PDF includes:
            - MIC histogram with fitted normal or mixture curves.
            - ECOFF threshold marked on the plot.
            - Summary statistics and parameter estimates.

        Args:
            outfile (str): Path to save the generated PDF report.

        Output:
            A single-page PDF combining the MIC distribution plot and textual summary.
        """
        with PdfPages(outfile) as pdf:
            fig, (ax_plot, ax_text) = plt.subplots(
                nrows=1, ncols=2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]}
            )

            # --- Left: Distribution plot ---
            plot_mic_distribution(
                low_log=self.intervals[0],
                high_log=self.intervals[1],
                weights=self.intervals[2],
                dilution_factor=self.dilution_factor,
                mu1=self.mu if self.distributions == 1 else self.mu1,
                sigma1=self.sigma if self.distributions == 1 else self.sigma1,
                mu2=None if self.distributions == 1 else self.mu2,
                sigma2=None if self.distributions == 1 else self.sigma2,
                log2_ecoff=np.log2(self.ecoff) if self.ecoff else None,
                ax=ax_plot,
            )
            ax_plot.legend(fontsize=7, frameon=False)

            z0, z1, z2 = self.z

            # --- Right: Text block ---
            ax_text.axis("off")
            lines = [
                f"ECOFF: {self.ecoff:.2f}",
                f"99th percentile: {z0:.2f}",
                f"97.5th percentile: {z1:.2f}",
                f"95th percentile: {z2:.2f}",
            ]
            if self.distributions == 1:
                lines += [
                    f"μ (mean): {self.dilution_factor**self.mu:.2f}",
                    f"σ (std dev): {self.dilution_factor**self.sigma:.2f}",
                ]
            else:
                lines += [
                    f"μ₁: {self.dilution_factor**self.mu1:.2f}, σ₁: {self.dilution_factor**self.sigma1:.2f}",
                    f"μ₂: {self.dilution_factor**self.mu2:.2f}, σ₂: {self.dilution_factor**self.sigma2:.2f}",
                ]
            ax_text.text(
                0.05,
                0.9,
                "\n".join(lines),
                fontsize=11,
                va="top",
                family="monospace",
            )

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

        print(f"PDF report saved to: {outfile}")
