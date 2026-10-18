import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ecoff_fitter.workflow import run_ecoff_analysis
from ecoff_fitter.report import CombinedReport
from ecoff_fitter.defence import validate_output_path
from ecoff_fitter.graphs import plot_mic_distribution

class ECOFFGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ECOFFitter")

        # INPUT FIELDS
        input_frame = tk.Frame(root)
        input_frame.pack(pady=5, anchor="w")

        tk.Label(input_frame, text="MIC data file:").pack(side="left")
        self.input_entry = tk.Entry(input_frame, width=50)
        self.input_entry.pack(side="left", padx=5)
        tk.Button(input_frame, text="Browse", command=self.select_input).pack(side="left")

        params_frame = tk.Frame(root)
        params_frame.pack(pady=5, anchor="w")

        tk.Label(params_frame, text="Params file (optional):").pack(side="left")
        self.params_entry = tk.Entry(params_frame, width=50)
        self.params_entry.pack(side="left", padx=5)
        tk.Button(params_frame, text="Browse", command=self.select_params).pack(side="left")

        options_frame = tk.Frame(root)
        options_frame.pack(pady=5, anchor="w")

        tk.Label(options_frame, text="Populations:").pack(side="left")
        self.dist_entry = tk.Entry(options_frame, width=5)
        self.dist_entry.insert(0, "1")
        self.dist_entry.pack(side="left", padx=10)

        tk.Label(options_frame, text="Dilution factor:").pack(side="left")
        self.dil_entry = tk.Entry(options_frame, width=5)
        self.dil_entry.insert(0, "2")
        self.dil_entry.pack(side="left", padx=10)

        tk.Label(options_frame, text="Truncation support:").pack(side="left")
        self.tails_entry = tk.Entry(options_frame, width=5)
        self.tails_entry.insert(0, "1")
        self.tails_entry.pack(side="left", padx=10)

        tk.Label(options_frame, text="Percentile:").pack(side="left")
        self.percentile_entry = tk.Entry(options_frame, width=5)
        self.percentile_entry.insert(0, "99")
        self.percentile_entry.pack(side="left", padx=10)

        self.verbose_var = tk.BooleanVar(value=False)
        verbose_frame = tk.Frame(root)
        verbose_frame.pack(pady=5, anchor="w")
        tk.Checkbutton(
            verbose_frame,
            text="model diagnostics",
            variable=self.verbose_var,
        ).pack(side="left")

        self.density_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            verbose_frame,
            text="plot density",
            variable=self.density_var,
        ).pack(side="left", padx=10)


        self.fit_individual_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            verbose_frame,
            text="fit individual populations",
            variable=self.fit_individual_var,
        ).pack(side="left", padx=10)

        # OUTPUT FILE
        output_frame = tk.Frame(root)
        output_frame.pack(pady=5, anchor="w")

        tk.Label(output_frame, text="Output file (optional):").pack(side="left")
        self.output_entry = tk.Entry(output_frame, width=50)
        self.output_entry.pack(side="left", padx=5)
        tk.Button(output_frame, text="Browse", command=self.select_output).pack(side="left")

        # RUN BUTTON
        tk.Button(root, text="Fit", command=self.run_ecoff,
                  bg="#4CAF50", fg="white").pack(pady=10)

        # RESULTS TEXT WINDOW
        self.output_text = tk.Text(root, height=15, width=80, state="disabled")
        self.output_text.pack(padx=10, pady=10)

        # SCROLLABLE PLOT AREA
        container = tk.Frame(root)
        container.pack(fill="both", expand=True)

        self.plot_canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=self.plot_canvas.yview)
        self.plot_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.plot_canvas.pack(side="left", fill="both", expand=True)

        self.plot_frame = tk.Frame(self.plot_canvas)
        self.plot_canvas.create_window((0, 0), window=self.plot_frame, anchor="nw")

        def update_scroll_region(event):
            self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))

        self.plot_frame.bind("<Configure>", update_scroll_region)

        self.canvas = None  # reference for cleanup

    # FILE DIALOG HELPERS
    def select_input(self):
        file = filedialog.askopenfilename(title="Select MIC Data File")
        if file:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, file)

    def select_params(self):
        file = filedialog.askopenfilename(title="Select Params File")
        if file:
            self.params_entry.delete(0, tk.END)
            self.params_entry.insert(0, file)

    def select_output(self):
        file = filedialog.asksaveasfilename(
            title="Save Output",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if file:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, file)

    # Fit
    def run_ecoff(self):
        input_file = self.input_entry.get()
        params_file = self.params_entry.get() or None
        distributions = int(self.dist_entry.get())
        dilution_factor = int(self.dil_entry.get())
        tails = int(self.tails_entry.get()) if self.tails_entry.get().strip() else None
        percentile = float(self.percentile_entry.get())
        outfile = self.output_entry.get()
        verbose = self.verbose_var.get()
        density = self.density_var.get()

        if not input_file:
            messagebox.showerror("Error", "You must select an input MIC data file.")
            return

        try:
            run = run_ecoff_analysis(
                input_file,
                params=params_file,
                distributions=distributions,
                dilution_factor=dilution_factor,
                boundary_support=tails,
                percentile=percentile,
                fit_individual=self.fit_individual_var.get(),
                density=density,
            )

            text = "ECOFF RESULTS\n=====================================\n\n"
            text += run.global_report.to_text("GLOBAL FIT", verbose=verbose)

            if run.individual_reports:
                text += "\nINDIVIDUAL FITS:\n-------------------------------------\n"
                for name, report in run.individual_reports.items():
                    text += report.to_text(label=name, verbose=verbose)

            if outfile:
                validate_output_path(outfile)

                if not run.individual_reports:
                    if outfile.endswith(".pdf"):
                        run.global_report.save_pdf(outfile)
                    else:
                        run.global_report.write_out(outfile)
                else:
                    combined = CombinedReport(outfile, run.global_report, run.individual_reports)
                    if outfile.endswith(".pdf"):
                        combined.save_pdf()
                    else:
                        combined.write_out()

            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            # Global plot
            fig_global = plt.Figure(figsize=(6, 3), dpi=100)
            axg = fig_global.add_subplot(111)

            gl_low, gl_high, gl_w = run.global_fitter.define_intervals()
            plot_mic_distribution(
                low_log=gl_low,
                high_log=gl_high,
                weights=run.global_fitter.weights_,
                dilution_factor=dilution_factor,
                mus=run.global_fitter.mus_,
                sigmas=run.global_fitter.sigmas_,
                pis=run.global_fitter.pis_,
                log2_ecoff=run.global_result[1],
                density=density,
                ax=axg,
            )
            axg.set_title("Aggregate")
            fig_global.tight_layout()
            widget = FigureCanvasTkAgg(fig_global, master=self.plot_frame).get_tk_widget()
            widget.pack(fill="both", expand=True, pady=10)

            # Individual plots
            if self.fit_individual_var.get():
                for col, (fitter, result) in run.individual_fits.items():
                    fig_i = plt.Figure(figsize=(6, 3), dpi=100)
                    axi = fig_i.add_subplot(111)

                    low, high, w = fitter.define_intervals()
                    plot_mic_distribution(
                        low_log=low,
                        high_log=high,
                        weights=fitter.weights_,
                        dilution_factor=dilution_factor,
                        mus=fitter.mus_,
                        sigmas=fitter.sigmas_,
                        pis=fitter.pis_,
                        log2_ecoff=result[1],
                        density=density,
                        ax=axi,
                    )

                    axi.set_title(f"{col}")
                    fig_i.tight_layout()
                    widget = FigureCanvasTkAgg(fig_i, master=self.plot_frame).get_tk_widget()
                    widget.pack(fill="both", expand=True, pady=10)

            self.write_output(text)

        except Exception as e:
            messagebox.showerror("Error", str(e))


    def write_output(self, msg):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, msg)
        self.output_text.config(state="disabled")


# MAIN
if __name__ == "__main__":
    root = tk.Tk()
    app = ECOFFGUI(root)
    root.mainloop()
