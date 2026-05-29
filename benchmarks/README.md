# ReHLine Mini Benchmark

This folder is a lightweight GridSearchCV benchmark harness for quick ReHLine
version checks. It focuses on wall-clock time for `GridSearchCV.fit`, not on
paper-grade solver comparisons. Regression rows report the best MSE found by
GridSearchCV; classification rows report the best accuracy.

## One-line usage

Edit `benchmarks/mini_config.json`, then run:

```bash
python -m benchmarks.mini_gridsearch
```

This writes a Markdown result file named `rehline-<version>.md` under
`benchmarks/results/`, for example:

```text
benchmarks/results/rehline-0.1.3.dev33-g8af87e306.md
```

The config controls tasks, datasets, hyperparameter grids, X preprocessing, CV
folds, repeats, and the output directory:

```json
{
  "task_datasets": {
    "ridge_quantile": [
      "california_housing",
      "make_regression_100k",
      "make_friedman1_5k_100"
    ],
    "ridge_quantile_monotonic": [
      "california_housing",
      "make_regression_100k",
      "make_friedman1_5k_100"
    ],
    "elasticnet_quantile": [
      "california_housing",
      "make_regression_100k",
      "make_friedman1_5k_100"
    ],
    "elasticnet_quantile_monotonic": [
      "california_housing",
      "make_regression_100k",
      "make_friedman1_5k_100"
    ],
    "ridge_svm": [
      "digits_low_high",
      "make_classification_100k",
      "openml_bioresponse"
    ],
    "elasticnet_svm": [
      "digits_low_high",
      "make_classification_100k",
      "openml_bioresponse"
    ]
  },
  "cv": 2,
  "repeats": 1,
  "n_jobs": null,
  "max_iter": 5000000,
  "tol": 0.0001,
  "C_grid": [0.1, 1.0, 10.0],
  "l1_ratio_grid": [0.5],
  "quantile_grid": [0.25],
  "preprocess_X": "standard",
  "output_dir": "benchmarks/results"
}
```

`task_datasets` locks each task to its own dataset list, so regression tasks do
not accidentally run on all regression datasets and classification tasks do not
accidentally run on all classification datasets. `preprocess_X` supports
`"standard"` (default), `"minmax"`, and `"none"`.

Use a different config:

```bash
python -m benchmarks.mini_gridsearch --config path/to/config.json
```

Run the larger dataset suite:

```bash
python -m benchmarks.mini_gridsearch --config benchmarks/large_config.json
```

`benchmarks/large_config.json` keeps the same task-specific schema but focuses
on heavier datasets such as `openml_buzz_twitter`, `openml_guillermo`,
`make_regression_300k`, `make_classification_300k`, and `covtype_binary_100k`.
Its output goes to `benchmarks/results/large/`.

## Python usage

```python
from benchmarks import run_default_benchmark

results = run_default_benchmark()
print(results)
```

Markdown table output:

```python
from benchmarks import run_default_benchmark

print(run_default_benchmark(as_markdown=True))
```

## Select tasks and datasets

```python
from benchmarks import available_datasets, available_tasks, run_gridsearch_benchmark

tasks = available_tasks()
datasets = available_datasets()

results = run_gridsearch_benchmark(
    tasks=[tasks["ridge_quantile"], tasks["elasticnet_svm"]],
    datasets=[datasets["diabetes"], datasets["breast_cancer"]],
    cv=3,
    repeats=3,
)
print(results)
```

Built-in tasks:

- `ridge_quantile`
- `ridge_quantile_monotonic`
- `ridge_quantile_eps`
- `ridge_mae`
- `ridge_huber`
- `ridge_svr`
- `elasticnet_quantile`
- `elasticnet_quantile_monotonic`
- `elasticnet_quantile_eps`
- `elasticnet_mae`
- `elasticnet_huber`
- `elasticnet_svr`
- `ridge_svm`
- `ridge_smooth_svm`
- `ridge_squared_svm`
- `elasticnet_svm`
- `elasticnet_smooth_svm`
- `elasticnet_squared_svm`

The extra tasks mirror the sklearn-compatible examples under
`doc/source/examples`: `MAE.ipynb`, `Huber.ipynb`, `SVR.ipynb`,
`QR_eps.ipynb`, `CustomQR.ipynb`, `MonotonicSVM.ipynb`,
`Smooth_SVM.ipynb`, `Squared_SVM.ipynb`, `GridSearchCV_reg_losses.ipynb`,
and `GridSearchCV_SVM_losses.ipynb`. `ridge_mse` and `elasticnet_mse` are
intentionally excluded from this mini suite because these cases dominated the
runtime in local tests.
Examples such as `CQR.ipynb`, `Path_solution.ipynb`, `Warm_start.ipynb`,
`RankRegression.ipynb`, and `NMF.ipynb` are better handled by separate
benchmark runners because they are not plain sklearn `GridSearchCV` tasks over
one estimator/loss pair.

Built-in datasets:

- `toy_regression`
- `make_regression_10k`
- `make_regression_100k`
- `make_regression_300k`
- `california_housing`
- `diabetes`
- `friedman1`
- `make_friedman1_5k_100`
- `openml_buzz_twitter`
- `sparse_uncorrelated`
- `linnerud_weight`
- `toy_classification`
- `make_classification_100k`
- `make_classification_300k`
- `openml_guillermo`
- `openml_bioresponse`
- `covtype_binary`
- `covtype_binary_50k`
- `covtype_binary_100k`
- `covtype_binary_full`
- `breast_cancer`
- `iris_binary`
- `wine_binary`
- `digits_0_1`
- `digits_low_high`

The default mini benchmark mixes a small loader dataset, medium regression data,
generated 100k-scale dense data, and one compact OpenML classification dataset.
`fetch_covtype` variants and larger OpenML datasets are available for stress
testing, but they are not part of the default mini config because they can
dominate total runtime. `fetch_*` datasets may
download once to the sklearn cache; `openml_*` datasets may download once to
the OpenML cache; `load_*` and `make_*` datasets do not download data.
Multi-class sklearn datasets are exposed as binary variants for the SVM tasks.

Default dataset mix:

| dataset | sklearn source | task | scale | notes |
| --- | --- | --- | --- | --- |
| `california_housing` | `fetch_california_housing` | regression | medium, 20,640 x 8 | downloads once to sklearn cache |
| `make_regression_100k` | `make_regression` | regression | large, 100,000 x 20 | generated locally |
| `make_friedman1_5k_100` | `make_friedman1` | regression | medium/high-dimensional, 5,000 x 100 | generated locally; suitable for default mini benchmark |
| `digits_low_high` | `load_digits` | classification | small, 1,797 x 64 | digits `0-4` vs `5-9`, no download |
| `make_classification_100k` | `make_classification` | classification | large, 100,000 x 20 | generated locally |
| `openml_bioresponse` | `fetch_openml(data_id=4134)` | classification | 3,751 x 1,776 | downloads once to OpenML cache |

Optional stress datasets:

| dataset | sklearn source | task | scale | notes |
| --- | --- | --- | --- | --- |
| `make_regression_300k` | `make_regression` | regression | large, 300,000 x 20 | generated locally |
| `openml_buzz_twitter` | `fetch_openml(data_id=4549)` | regression | 583,250 x 77 | target is `Annotation`; downloads once to OpenML cache |
| `make_classification_300k` | `make_classification` | classification | large, 300,000 x 20 | generated locally |
| `openml_guillermo` | `fetch_openml(data_id=41159)` | classification | 20,000 x 4,296 | high-dimensional; has an ARFF fallback cache for known OpenML md5 mismatch |
| `covtype_binary_50k` | `fetch_covtype` | classification | 50,000 x 54 | fixed subsample after filtering classes 1/2 |
| `covtype_binary_100k` | `fetch_covtype` | classification | 100,000 x 54 | fixed subsample after filtering classes 1/2 |
| `covtype_binary_full` | `fetch_covtype` | classification | 495k x 54 after filtering classes 1/2 | stress-only; can take hours |
| `covtype_binary` | `fetch_covtype` | classification | alias for full binary covtype | kept for compatibility |

## Mini Config Hyperparameter Grids

| task | grid | candidates |
| --- | --- | --- |
| `ridge_quantile` | `C=[0.1, 1, 10]`, `qt=[0.25]` | 3 |
| `ridge_quantile_monotonic` | `C=[0.1, 1, 10]`, `qt=[0.25]`, `constraint=[monotonic increasing]` | 3 |
| `elasticnet_quantile` | `C=[0.1, 1, 10]`, `l1_ratio=[0.5]`, `qt=[0.25]` | 3 |
| `elasticnet_quantile_monotonic` | `C=[0.1, 1, 10]`, `l1_ratio=[0.5]`, `qt=[0.25]`, `constraint=[monotonic increasing]` | 3 |
| `ridge_svm` | `C=[0.1, 1, 10]` | 3 |
| `elasticnet_svm` | `C=[0.1, 1, 10]`, `l1_ratio=[0.5]` | 3 |

Override grids from Python:

```python
from benchmarks import run_default_benchmark

print(
    run_default_benchmark(
        C_grid=[0.1, 1, 10],
        l1_ratio_grid=[0.2, 0.5, 0.8],
        quantile_grid=[0.25, 0.5, 0.75],
        preprocess_X="standard",
        as_markdown=True,
    )
)
```

## Command line

```bash
python -m benchmarks.mini_gridsearch --task ridge_quantile --dataset diabetes --cv 3
```

CLI flags override values from the config file.

Override grids from CLI by repeating flags:

```bash
python -m benchmarks.mini_gridsearch \
  --task elasticnet_quantile \
  --dataset diabetes \
  --C 0.1 --C 1 --C 10 \
  --l1-ratio 0.2 --l1-ratio 0.5 --l1-ratio 0.8 \
  --quantile 0.25 --quantile 0.5 --quantile 0.75 \
  --preprocess-X standard
```

The CLI writes Markdown by default. To choose a path:

```bash
python -m benchmarks.mini_gridsearch --task ridge_quantile --dataset diabetes --output results.md
python -m benchmarks.mini_gridsearch --task ridge_quantile --dataset diabetes --output results.csv
```

## Custom datasets

```python
from benchmarks import DatasetSpec, available_tasks, run_gridsearch_benchmark

def my_data():
    return X, y

results = run_gridsearch_benchmark(
    tasks=[available_tasks()["ridge_quantile"]],
    datasets=[DatasetSpec("my_dataset", "regression", my_data)],
)
```

Use `problem_type="regression"` for quantile-regression tasks and
`problem_type="classification"` for SVM tasks.
