[![codecov](https://codecov.io/gh/DylanAdlard/ECOFFitter/graph/badge.svg?token=0Q0YSOZYKW)](https://codecov.io/gh/DylanAdlard/ECOFFitter)

# ECOFFitter

**ECOFFitter** is a Python tool for fitting interval regression models
to censored Minimum Inhibitory Concentration (MIC) data.\
It provides a command-line interface (CLI) to process input MIC
distributions, estimate epidemiological cutoffs (ECOFFs), and generate
model outputs and diagnostic plots.

The package supports: - interval-censored, left-censored,
right-censored, and uncensored MIC data - mixed datasets with variable
censoring - automated output files (PDF reports, text outputs, plots) -
configuration via parameter files

Demo input files are provided in `demo_files/` to illustrate basic use.

---

## 📦 Installation

### Install from PyPI

```bash
pip install ecoffitter
```

### Install from source

Assuming in project directory:

```bash
pip install -e .
```

---

## 🛠 Creating the Environment

### Conda environment (env.yml)

```bash
conda env create -f env.yml
conda activate ECOFFitter
```

### Pip environment (requirements.txt)

```bash
python -m venv ecoff-env
source ecoff-env/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Command-Line Usage

Once installed, you can call the CLI.

Example using demo files:

```bash
ecoffitter   -input demo_files/input.txt   -params demo_files/params.txt   -output demo_files/output.txt   --pdf demo_files/output.pdf
```

usage:

```bash
ecoff_fitter [-h] --input INPUT [--params PARAMS]
[--dilution_factor DILUTION_FACTOR]
[--distributions {1,2}] [--tail_dilutions TAIL_DILUTIONS]
[--percentile PERCENTILE] [--outfile OUTFILE] [--verbose]
```

---

## ⚙️ CLI Parameters

### CLI Options

| Option              | Argument          | Description                                                                                                                                                      | Default |
| ------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `-h`, `--help`      | —                 | Show help message and exit.                                                                                                                                      | —       |
| `--input`           | `INPUT`           | Path to the input MIC dataset (CSV, TSV, XLSX, or XLS). Must contain columns `MIC` and `observations`.                                                           | `None`  |
| `--params`          | `PARAMS`          | Optional parameter file (`YAML`, `TXT`, or `key=value` list). Defines `dilution_factor`, `distributions`, and `tail_dilutions`. Overrides CLI flags if provided. | `None`  |
| `--dilution_factor` | `DILUTION_FACTOR` | Dilution factor                                                                                                                                                  | `2`     |
| `--distributions`   | `{1,2}`           | Number of Gaussian mixture components to fit.                                                                                                                    | `1`     |
| `--tail_dilutions`  | `TAIL_DILUTIONS`  | Extra upper-dilution steps for censored MICs (set to `None` to disable).                                                                                         | `1`     |
| `--percentile`      | `PERCENTILE`      | Percentile used to compute the ECOFF (0–100).                                                                                                                    | `99.0`  |
| `--outfile`         | `OUTFILE`         | Path to save ECOFF results (`.txt` or `.pdf`).                                                                                                                   | `None`  |
| `--verbose`         | —                 | Print detailed model information and parameters.                                                                                                                 | `False` |

---

## 📄 License

MIT License.
