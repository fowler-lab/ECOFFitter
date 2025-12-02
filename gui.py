import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from ecoff_fitter import ECOFFitter
from ecoff_fitter.report import GenerateReport
from ecoff_fitter.defence import validate_output_path
from ecoff_fitter.graphs import plot_mic_distribution
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class ECOFFGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ECOFFitter")

        # Input file selection
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

        # Model options
        options_frame = tk.Frame(root)
        options_frame.pack(pady=5, anchor="w")

        tk.Label(options_frame, text="Distributions:").pack(side="left")
        self.dist_entry = tk.Entry(options_frame, width=5)
        self.dist_entry.insert(0, "1")
        self.dist_entry.pack(side="left", padx=10)

        tk.Label(options_frame, text="Dilution factor:").pack(side="left")
        self.dil_entry = tk.Entry(options_frame, width=5)
        self.dil_entry.insert(0, "2")
        self.dil_entry.pack(side="left", padx=10)

        tk.Label(options_frame, text="Series tails:").pack(side="left")
        self.tails_entry = tk.Entry(options_frame, width=5)
        self.tails_entry.insert(0, "1")
        self.tails_entry.pack(side="left", padx=10)

        tk.Label(options_frame, text="Percentile:").pack(side="left")
        self.percentile_entry = tk.Entry(options_frame, width=5)
        self.percentile_entry.insert(0, "99")
        self.percentile_entry.pack(side="left", padx=10)

        # Output file selection 
        output_frame = tk.Frame(root)
        output_frame.pack(pady=5, anchor="w")

        tk.Label(output_frame, text="Output file (optional):").pack(side="left")
        self.output_entry = tk.Entry(output_frame, width=50)
        self.output_entry.pack(side="left", padx=5)
        tk.Button(output_frame, text="Browse", command=self.select_output).pack(side="left")

        #  Run button 
        tk.Button(root, text="Run ECOFF", command=self.run_ecoff, bg="#4CAF50", fg="white").pack(pady=10)

        #  Results terminal 
        self.output_text = tk.Text(root, height=15, width=80, state="disabled")
        self.output_text.pack(padx=10, pady=10)

        # Plot area
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(pady=10)
        self.canvas = None

    # File selection methods
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

    def run_ecoff(self):
        input_file = self.input_entry.get()
        params_file = self.params_entry.get() or None
        distributions = int(self.dist_entry.get())
        dilution_factor = int(self.dil_entry.get())
        tails = int(self.tails_entry.get())
        percentile = float(self.percentile_entry.get())
        outfile = self.output_entry.get()

        if not input_file:
            messagebox.showerror("Error", "You must select an input MIC data file.")
            return

        try:
            fitter = ECOFFitter(
                input=input_file,
                params=params_file,
                distributions=distributions,
                tail_dilutions=tails,
                dilution_factor=dilution_factor,

            )
            result = fitter.generate(percentile=percentile)

            # Format result text
            text = f"ECOFF results:\n\n"
            text += f"ECOFF: {result[0]:.4f}\n"
            text += f"z-value: {result[1]:.4f}\n"

            for i in range(fitter.distributions):
                text += f"\nComponent {i+1}:"
                text += f"\n  mu = {dilution_factor**fitter.mus_[i]:.4f}"
                text += f"\n  sigma (folds)= {dilution_factor**fitter.sigmas_[i]:.4f}\n"

            # Write output only if user selected a file
            if outfile:
                validate_output_path(outfile)
                report = GenerateReport.from_fitter(fitter, result)

                if outfile.endswith(".pdf"):
                    report.save_pdf(outfile)
                else:
                    report.write_out(outfile)

                text += f"\n\nSaved to: {outfile}"

            if self.canvas:
                self.canvas.get_tk_widget().destroy()

            # Create new figure
            fig = plt.Figure(figsize=(6, 3.5), dpi=100)
            ax = fig.add_subplot(111)

            low_log, high_log, weights = fitter.define_intervals()

            # Call user’s plot function using our axis
            plot_mic_distribution(
                low_log=low_log,
                high_log=high_log,
                weights=fitter.weights_,
                dilution_factor=dilution_factor,
                mus=fitter.mus_,
                sigmas=fitter.sigmas_,
                pis=fitter.pis_,
                log2_ecoff=result[1],
                ax=ax
            )

            # Embed figure in Tkinter
            fig.subplots_adjust(bottom=0.20)
            self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack()

            self.write_output(text)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # --- Write to GUI console ---
    def write_output(self, msg):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, msg)
        self.output_text.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = ECOFFGUI(root)
    root.mainloop()
