[![codecov](https://codecov.io/gh/DylanAdlard/ECOFFitter/graph/badge.svg?token=0Q0YSOZYKW)](https://codecov.io/gh/DylanAdlard/ECOFFitter)

# ECOFFitter

**ECOFFitter** is a Python tool for fitting interval regression models
to censored Minimum Inhibitory Concentration (MIC) data.\
It provides a command-line interface (CLI) to process input MIC
distributions, estimate epidemiological cutoffs (ECOFFs), and generate
model outputs and diagnostic plots.

The package supports:

- interval-censored, left-censored,
  right-censored, and uncensored MIC data
- mixed datasets with variable
  censoring
- automated output files (PDF reports, text outputs, plots)
- configuration via parameter files

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

## 📥 Input

### 1. MIC Data Input File

The MIC dataset describes the distribution of Minimum Inhibitory Concentration (MIC) values and the number of isolates observed at each dilution.

It must be a **tabular file** in one of these formats:

- `.csv`
- `.tsv`
- `.txt`
- `.xlsx` / `.xls`

#### Required columns

| Column         | Type            | Description                                                                                                 |
| -------------- | --------------- | ----------------------------------------------------------------------------------------------------------- |
| `MIC`          | float or string | MIC dilution (e.g., `0.125`, `0.25`, `1`, `2`, …). Can include censored values such as `<=0.125` or `>=64`. |
| `observations` | integer         | Number of isolates observed at that MIC dilution.                                                           |

#### Example (from `demo_files/input.txt`)

```text
MIC     observations
0.125   12
0.25    34
0.5     51
1       63
2       4
4       20
8       7
>=16    4
```

- censored balues (e.g >=16) are automatically detected and handled

### 2. Parameter File (`--params`)

This is an **optional** configuration file (YAML, TXT, or `key=value` list) that overrides CLI arguments.

Typical fields include:

| Key               | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `dilution_factor` | Fold-change between MIC dilutions (e.g., `2`).                  |
| `distributions`   | Number of Gaussian mixture components (`1` or `2`).             |
| `tail_dilutions`  | How many dilution steps to extend upper tail for censored data. |
| `percentile`      | Percentile used to compute the ECOFF (e.g., `99`).              |

#### Example (`demo_files/params.txt`)

```text
dilution_factor=2
distributions=1
tail_dilutions=1
percentile=99
```

or YAML:

```text
dilution_factor: 2
distributions: 2
tail_dilutions: 1
percentile: 99
```

## 📤 Output

The tool produces one or more of the following.

---

### 1. Text Output

If `--outfile` ends in `.txt`, the tool writes:

- ECOFF estimate
- fitted mean & variance (per component)
- mixture weights (if 2 distributions)
- percentiles
- likelihood values

### 2. PDF Report

If `--outfile` ends in `.pdf`, the tool writes:

- histogram of observed MICs
- fitted distribution curve(s)
- ECOFF location marker
- table of model parameters
- censoring diagnostics

## 🚀 Command-Line Usage

Once installed, you can call the CLI.

Example using demo files:

```bash
ecoffitter   --input demo_files/input.txt   --params demo_files/params.txt   --outfile demo_files/output.txt
```

Instead of using a parameter file, you can also specify parameters directly.

Usage:

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
