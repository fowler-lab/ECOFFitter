[![codecov](https://codecov.io/gh/fowler-lab/ECOFFitter/graph/badge.svg?token=VEkosXEIAQ)](https://codecov.io/gh/fowler-lab/ECOFFitter)

# ECOFFitter

**ECOFFitter** fits interval-censored normal models to Minimum Inhibitory
Concentration (MIC) observations and estimates epidemiological cutoff values
(ECOFFs). It supports a single distribution, a finite mixture, and automatic
selection of the number of mixture components. Text and PDF reports can include
aggregate results and optional per-assay results.

Demo input files are in `demo_files/`.

## Installation

Install from a local checkout:

```bash
pip install -e .
```

## Input data

Input can be a CSV, TSV, TXT, XLSX, or XLS file, or a DataFrame, array-like
sequence, or mapping when using the Python API. MIC values may be exact (for
example `0.25`), left-censored (such as `<=0.125`), or right-censored (such as
`>64`). Ratio values such as `1/2` are accepted; the numerator is used as the
MIC value.

A single-column input contains one MIC per row. Duplicate MIC values are
aggregated into counts. A canonical single-fit table has `MIC` and
`observations` columns:

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

For aggregate and per-assay analyses, use a table with a `MIC` column and one
or more assay columns. Counts across assay columns are summed for the aggregate
fit. Each assay is fitted separately when individual fits are enabled.

Supported input extensions are `.csv`, `.tsv`, `.txt`, `.xlsx`, and `.xls`.
For Excel files, the first worksheet is used unless a sheet is selected through
the Python API.

## Configuration

The optional `--params` file can be YAML or TXT (`key=value`). Values in this
file override the corresponding manually supplied parameters. Supported keys
are `dilution_factor`, `distributions`, `boundary_support`, and `percentile`.

TXT example (`demo_files/params.txt`):

```text
dilution_factor=2
distributions=1
boundary_support=1
percentile=99
```

YAML example:

```yaml
dilution_factor: 2
distributions: 2
boundary_support: 1
percentile: 99
```

Set `boundary_support` to `None` in a parameter file to use unbounded tails for
censored observations.

## Python API

Fit one dataset directly:

```python
from ecoff_fitter import ECOFFitter

fitter = ECOFFitter(input="demo_files/censored.txt", distributions=1)
result = fitter.generate(percentile=99)
print(result)
```

Run the aggregate and optional per-assay workflow:

```python
from ecoff_fitter import run_ecoff_analysis

run = run_ecoff_analysis(
    "demo_files/multi_assay.txt",
    distributions=2,
    fit_individual=True,
    percentile=99,
)

print(run.global_report.to_text())
for assay, report in run.individual_reports.items():
    print(report.to_text(label=assay))
```

Use the ECOFFinder workbook-style range search through the same workflow by
setting `method="ecoffinder"`. The selected wild-type range is found with
least squares; `intervals=True` then refits that range using ECOFFitter's
interval regression:

```python
run = run_ecoff_analysis(
    "demo_files/multi_assay.txt",
    method="ecoffinder",
    detection_level=10,
    intervals=True,
    fit_individual=True,
)

print(run.global_report.to_text(verbose=True))
```

ECOFFinder fits one wild-type distribution and does not support
`distributions="auto"`. It accepts parameter mappings or YAML/TXT files with
`dilution_factor`, `boundary_support`, `percentile`, `detection_level`,
`cumulative`, `first_best`, and `intervals`. File values override matching
arguments. A `distributions` entry is accepted only when set to `1`.

Set `distributions="auto"` in `run_ecoff_analysis` to select a component
count by the AIC or BIC criterion. `max_distributions` sets the largest count
to consider; `criterion`, `min_delta`, `min_pi`, and `min_separation` control
the selection checks. The selected aggregate count is available as
`run.selected_k`.

## Command-line usage

The CLI accepts a dataset with a `MIC` column and either `observations` or one
or more assay columns:

```bash
ecoff-fitter \
  --input demo_files/multi_assay.txt \
  --method ecoffinder \
  --intervals \
  --percentile 99 \
  --fit_individual \
  --outfile demo_files/output.pdf
```

Run `ecoff-fitter --help` for the complete option list. The same CLI is
available as `python -m ecoff_fitter`.

| Option | Default | Description |
| --- | --- | --- |
| `--input INPUT` | Required | Input MIC dataset. |
| `--method METHOD` | `ecoffitter` | Fitting method: `ecoffitter` or `ecoffinder`. |
| `--params PARAMS` | None | YAML or TXT parameter file; supported keys depend on the selected method, and file values override matching options. |
| `--dilution_factor N` | `2` | Ratio between adjacent MIC dilution steps. |
| `--distributions K` | `1` | Number of normal components to fit. The CLI accepts an integer; automatic selection is available in the Python workflow API. |
| `--boundary_support N` | `1` | Number of dilution steps used to extend censoring bounds. Use `None` in a parameter file for unbounded tails. |
| `--percentile P` | `99.0` | Percentile used to calculate the ECOFF; must be between 0 and 100. |
| `--fit_individual` | Off | Fit each assay column as well as the aggregate observations. |
| `--density` | Off | Plot probability density instead of counts. |
| `--detection_level P` | `10.0` | Minimum eligible ECOFFinder mode size as a percentage of observations. |
| `--cumulative` / `--no-cumulative` | On | Use cumulative counts in ECOFFinder range fitting, or per-dilution frequencies. |
| `--first_best` | Off | Stop ECOFFinder range search after its score begins increasing from its best value. |
| `--intervals` | Off | Refit ECOFFinder's selected range using interval regression. |
| `--outfile PATH` | None | Write a text report (`.txt`) or PDF report (`.pdf`). |
| `--verbose` | Off | Include model details in printed text output. |

## Output

The results are printed to the terminal. With `--outfile`, the CLI also writes a
text or PDF report. PDF reports show observed MIC intervals, fitted component
curves, and the ECOFF marker. When per-assay fits are enabled, the report
contains the aggregate fit and individual assay fits.

## License

MIT License.
