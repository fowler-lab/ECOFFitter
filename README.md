[![codecov](https://codecov.io/gh/fowler-lab/ECOFFitter/graph/badge.svg?token=VEkosXEIAQ)](https://codecov.io/gh/fowler-lab/ECOFFitter)

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

## 🛠 Installation

### Install from source

Assuming in project directory (after git cloning):

```bash
pip install -e .
```

---

## 📥 Input

### 1. MIC Data Input File

The MIC dataset describes the distribution of Minimum Inhibitory Concentration (MIC) values and the number of isolates observed at each dilution.

Alternatively, an array or single-column file of observed MICs can be supplied (ie 1 row per sample).

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

or

A single column/array of observed MIC values.

#### Example (from `demo_files/censored.txt`)

```text
MIC     observations
0.06    1925
0.12    1165
0.25    341
0.5     69
1       27
2       27
4       13
8       29
<=0.03  1096
>8      2196
```

- censored balues (e.g >=16) are automatically detected and handled

### 2. Parameter File (`--params`)

This is an **optional** configuration file (YAML, TXT, or `key=value` list) that overrides CLI arguments.

Typical fields include:

| Key                | Description                                                     |
| ------------------ | --------------------------------------------------------------- |
| `dilution_factor`  | Fold-change between MIC dilutions (e.g., `2`).                  |
| `distributions`    | Number of Gaussian mixture components.                          |
| `boundary_support` | How many dilution steps to extend upper tail for censored data. |
| `percentile`       | Percentile used to compute the ECOFF (e.g., `99`).              |

#### Example (`demo_files/params.txt`)

```text
dilution_factor=2
distributions=1
boundary_support=1
percentile=99
```

or YAML:

```text
dilution_factor: 2
distributions: 2
boundary_support: 1
percentile: 99
```

## 📤 Output

The tool produces one or more of the following:

---

### 1. Text Output

If `--outfile` ends in `.txt`, the tool writes:

- ECOFF estimate
- fitted mean & variance (per component)
- mixture weights (if multiple distributions)

### 2. PDF Report

If `--outfile` ends in `.pdf`, the tool writes:

- histogram of observed MICs
- fitted distribution curve(s)
- ECOFF location marker
- fitted mean & variance (per component)
- mixture weights (if multiple distributions)

## 🚀 Command-Line Usage

Once installed, you can call the CLI.

Example using demo files:

```bash
ecoff-fitter   --input demo_files/censored.txt   --params demo_files/params.txt   --outfile demo_files/output.txt
```

Instead of using a parameter file, you can also specify parameters directly.

Usage:

```bash
ecoff-fitter [-h] --input INPUT [--params PARAMS]
[--dilution_factor DILUTION_FACTOR]
[--distributions DISTRIBUTIONS] [--boundary_support BOUNDARY_SUPPORT]
[--percentile PERCENTILE] [--outfile OUTFILE] [--verbose]
```

---

## ⚙️ CLI Parameters

### CLI Options

| Option               | Argument           | Description                                                                                                                                                        | Default |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `-h`, `--help`       | —                  | Show help message and exit.                                                                                                                                        | —       |
| `--input`            | `INPUT`            | Path to the input MIC dataset (CSV, TSV, XLSX, or XLS). Must contain columns `MIC` and `observations`.                                                             | `None`  |
| `--params`           | `PARAMS`           | Optional parameter file (`YAML`, `TXT`, or `key=value` list). Defines `dilution_factor`, `distributions`, and `boundary_support`. Overrides CLI flags if provided. | `None`  |
| `--dilution_factor`  | `DILUTION_FACTOR`  | Dilution factor                                                                                                                                                    | `2`     |
| `--distributions`    | `DISTRIBUTIONS`    | Number of Gaussian mixture components to fit.                                                                                                                      | `1`     |
| `--boundary_support` | `boundary_support` | Extra upper-dilution steps for censored MICs (set to `None` to disable).                                                                                           | `1`     |
| `--percentile`       | `PERCENTILE`       | Percentile used to compute the ECOFF (0–100).                                                                                                                      | `99.0`  |
| `--outfile`          | `OUTFILE`          | Path to save ECOFF results (`.txt` or `.pdf`).                                                                                                                     | `None`  |
| `--verbose`          | —                  | Print detailed model information and parameters.                                                                                                                   | `False` |

---

## 📄 License

MIT License.
