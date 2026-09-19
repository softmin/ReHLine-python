# ReHLine Benchmarks

This directory contains maintained correctness suites and repeatable ReHLine
release-regression runners. The correctness suites validate supported losses,
constraints and estimator behavior; they do not establish performance rankings.
Release timing comparisons require matching optimization problems and full
objective validation. The general GridSearchCV harness also offers unverified
timing runs, which must not be used as evidence of equivalent solutions.

Exploratory solver comparisons, tolerance probes, one-off review scripts and
discussion reports belong in the Git-ignored `local/` directory. Passing numerical
checks alone does not promote an experiment into this maintained suite. Before
adding a solver comparison, define a reproducible case matrix, match the complete
objective (including regularization, intercept handling and sample weights) and
constraints, and specify common accuracy gates and failure reporting. Timings
for different optimization problems must not be presented as solver speedups.

Generated reports are local or CI artifacts and are ignored by Git. See
[results guidance](results/README.md) before publishing a result.

## Independent correctness mini benchmark

Generate many small problems and compare ReHLine with CVXPY:

```bash
pip install -e ".[benchmark]"
python -m benchmarks.correctness --cases 1200
```

The default suite uses 5–40 observations and 1–8 coefficients, with a fixed seed
(`20260912`). Each of 1,200 problems is checked with shrinking off, shrinking on,
and a warm refit: 3,600 objective comparisons. No dataset downloads are needed.
It covers MSE, MAE, quantile, epsilon-insensitive quantile, Huber, SVR, hinge,
smooth hinge, squared hinge, and arbitrary ReLU, ReHU and mixed PLQ losses.
Cases combine weighted Ridge/ElasticNet, zero sample weights, zero L1 weights,
finite/zero/infinite ReHU thresholds, nonnegative/monotonic/box/general linear
constraints and equalities. Designs include correlated columns, rank deficiency,
zero rows, offsets and synthetic intercept columns.

CVXPY models are built from the mathematical losses without ReHLine's loss
conversion functions. Both solutions are independently evaluated against the
**full original objective**:

```text
C * sum(sample_weight * loss) + (1 - l1_ratio)/2 * ||beta||²
                            + l1_ratio * sum(omega * abs(beta))
```

The gate compares objective values with `rtol=1e-8`, `atol=1e-9`; checks reported
objectives and dual values; requires ReHLine convergence; and checks both
solutions' constraint violations against `1e-8`. The native ElasticNet objective
is converted back to the original normalization before comparison. Coefficient
differences are diagnostic and do not replace the objective checks.

CVXPY uses CLARABEL with strict solver tolerances. If its reference fails status,
feasibility or independent objective checks, OSQP is tried and both attempts are
recorded. Only `optimal` is accepted; `optimal_inaccurate` never passes as a
reference. Smooth hinge uses an equivalent QP epigraph so the OSQP fallback
also works with CVXPY 1.6. If neither reference is reliable, the benchmark fails. These are
numerical checks within explicit tolerances, not exact arithmetic proofs.

The process exits nonzero on any failure. Results go to
`benchmarks/results/correctness/report.json`; failing inputs are saved as NPZ
files next to it. Reports include seeds, versions, native binary hashes,
reference attempts and all objective comparisons. Run more cases or replay one:

```bash
python -m benchmarks.correctness --cases 4800 --seed 20260913 --output /tmp/stress.json
python -m benchmarks.correctness --case 71 --seed 20260912
python -m benchmarks.correctness --replay /tmp/stress-cases/case-71.npz
```

CI runs the correctness suites with current dependencies and with CVXPY 1.6.0
(NumPy 1.26.4 / SciPy 1.13.1, Python 3.12). Separate Linux, macOS and Windows
jobs build wheels, then run the full tests and 256 cases from each of the six
independent suites outside the checkout. Reports and failed cases are retained
as artifacts. CVXPY is an optional benchmark dependency, not a runtime dependency.

### Equivalent constraint representations

```bash
python -m benchmarks.constraint_scaling --cases 600
python -m benchmarks.constraint_scaling --case 71 --seed 20260918
```

This additional suite uses at most 20 samples and 5 coefficients. Each problem
is solved in four equivalent constraint representations: original, multiplied
by `1e-12`, multiplied by `1e12`, and mixed per-row scales from `1e-200` to
`1e200`. Both shrinking modes and a warm refit give 7,200 comparisons for
600 problems. CVXPY receives the well-scaled original constraints; objective
and feasibility checks also use that original problem, avoiding a reference
that silently loses the same tiny rows. The complete weighted Ridge/ElasticNet
objective, reported objective, dual bound and convergence status must pass.

The release workflow also downloads the actual cibuildwheel artifacts on
Linux, macOS and Windows and selects the compatible CPython 3.12 wheel. It
installs that exact binary, runs all six suites plus pytest outside the checkout,
and records the wheel SHA-256 and validation status. PyPI upload depends on these
jobs. Other wheel/Python combinations retain their cibuildwheel pytest checks;
the independent CVXPY gate does not claim to cover every Python ABI.

## Release timing with objective validation

For a repeatable GridSearchCV comparison with automatic objective checks, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m benchmarks.release --label before --output /tmp/before.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m benchmarks.release --label after --output /tmp/after.json \
  --baseline /tmp/before.json
```

This uses `benchmarks/release_objective_config.json`: ten task types, 24
task/dataset combinations, three folds, three repeats and one worker, with
`tol=1e-9`. It includes full Iris, Wine and Digits with explicit Ridge/ElasticNet
OvR and OvO tasks. It records 720 timed estimator fits (4,380 native subproblems),
then independently audits another 240 estimator fits (1,460 native subproblems)
covering every parameter candidate, fold and final refit. The extra
audit is outside the timer, and every timed objective/dual value must agree with
its audited counterpart. Timing includes only lightweight diagnostic recording.
Run each version sequentially; use `--package /path/containing/rehline` to select
a different package, or use `tools/run_release_benchmark.py` from the desired
installed version's Python environment. Reports identify the imported package
and native binary hash. With `--baseline`, objective mismatch exits nonzero.

Existing audit reports can also be compared directly:

```bash
python -m benchmarks.objectives \
  /tmp/objective-before.json \
  /tmp/objective-after.json
```

The audit hashes each optimization problem, independently recomputes its PLQ
objective and dual value, and checks primal and dual feasibility. Monotonic
constraints receive a feasible coefficient adjustment for a valid primal upper
bound. The comparison fails on missing fits, changed problems, objective mismatch
or insufficient optimality accuracy (`rtol=1e-8`, `atol=1e-10`). Each audit runs
one repeat over all candidates and folds; timing runs retain three repeats.
This release audit supports unconstrained and adjacent monotonic constraints;
other constraints fail explicitly. The independent CVXPY suite above covers
general linear constraints as well.

Each native subproblem is identified by candidate/fold (or final refit) and
class or class pair. The audit independently derives the expected subproblems
from labels retained after sample/class weighting; it does not infer completeness
from the recorded solver calls. Binary fits count once, OvR fits count `K` times,
and OvO fits count `K * (K - 1) / 2` times for `K > 2`. Counts may vary across
folds and candidates. Missing, duplicate or unexpected identities fail validation,
and threaded completion order does not affect matching. Reports include
`n_classes`, `n_estimator_fits`, `n_objective_fits` and the complete fit manifest.
Both native and original objectives/dual bounds are recorded; ElasticNet values
are multiplied by `1 - l1_ratio` to recover the original normalization.

The same checks are available through `run_gridsearch_benchmark` with
`verify_objective=True, n_jobs=1`, or the mini runner's `--verify-objective` flag:

```bash
python -m benchmarks.mini_gridsearch --config benchmarks/release_objective_config.json \
  --verify-objective --output /tmp/checked.md
```

Verified mini runs add objective status and maximum relative bound gap to the
Markdown report and write all audited problems to a JSON sidecar (`checked.json`).
Configuration keys `verify_objective`, `objective_rtol` and `objective_atol`
control the same behavior. Verification requires serial outer GridSearchCV
execution (`n_jobs=1`). Inner multiclass `model__n_jobs` may use threads; the
recorder explicitly selects joblib's threading backend. Because recording wraps
solver/estimator methods temporarily, it must not run concurrently with other
model fits in the same process. All wrappers are restored on success or failure.

## One-line usage

The general GridSearchCV harness reports elapsed time, MSE and accuracy.
Objective verification is opt-in here; use the release runner above for mandatory
validation or the correctness runner for independent CVXPY references.

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
- `ridge_svm_ovr`, `ridge_svm_ovo`
- `elasticnet_svm_ovr`, `elasticnet_svm_ovo`

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
- `iris`, `wine`, `digits` (all original classes)
- `multiclass_4`, `multiclass_10`, `multiclass_30` (600 samples, 20 features)
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
Multiclass sklearn datasets are available with all their original classes and
as separately named binary variants. The release configuration includes both
OvR and OvO tasks on the full datasets.

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

### Public-estimator correctness

Run `python -m benchmarks.estimator_correctness --cases 1200` with the optional
`.[benchmark]` dependencies installed. This adds independently modeled CQR
problems, MF convex blocks checked against CVXPY, full weighted MF histories and
final factor feasibility, and raw ReHLine cloning checked against reference
objectives. Constrained MF fits also run three equivalent row scalings (small,
large and mixed), comparing full weighted objectives, iteration counts, warnings
and convergence status using normalized feasibility units. Joint MF is nonconvex; CVXPY is an oracle for fixed-factor blocks,
not for the global factorization optimum. CQR checks include both slope and
intercept ridge penalties. Any reference failure, objective mismatch, loss of
constraints or inner nonconvergence fails the command. Use `--case INDEX` and
`--seed SEED` to reproduce a reported case.

## Implicit CQR correctness and memory

Run `python -m benchmarks.cqr_correctness --cases 600` to compare CQR's implicit
training design with the original dense joint problem and independently
formulated CVXPY references. The suite generates 1–7 quantiles, unsorted and
repeated levels, weights with zeros, rank-deficient/correlated designs and
synthetic intercept columns. Both shrinking modes and cold/warm fits are
checked: six full-objective comparisons per case.

All native solves use `tol=1e-10`; objective and dual comparisons use
`rtol=1e-8, atol=1e-9`. CVXPY references must have a feasible-dual gap below
`1e-10 + 1e-9 * abs(objective)`. Reference fallback never depends on the ReHLine
answer. A failed case exits nonzero; reproduce it with `--case INDEX --seed SEED`.
The report defaults to `results/correctness/cqr.json`.

Each cold/warm CQR fit also exports an inference snapshot and compares its
predictions, quantile labels and independently recomputed full objective with
the fitted model and CVXPY. Reports record these as `inference_comparisons`.
The API correctness suite performs the same objective/constraint checks for
sklearn regressor and binary/OvR/OvO classifier exports, including both score
formats. Snapshot comparisons are reported separately from solver comparisons.

Training stores the original `(n, d)` features plus loss/dual arrays of size
`O(n*q)`. It avoids the former `(n*q, d+q)` design allocation, while retaining
joint optimization and both slope and intercept penalties. Release objective
audits stream the equivalent dense design when hashing inputs, so comparisons
against older dense implementations remain valid without allocating that matrix.
The test suite also checks warm starts with changed shapes and measures Python
preprocessing allocations at `n=10000, d=40, q=20`.

## Constraint and multiclass API correctness

Run `python -m benchmarks.api_correctness --cases 1200` for small independent
CVXPY checks of constructor/combined constraints and binary, OvR and OvO fits.
It varies losses, sample/class weights, intercept scaling and ElasticNet penalties.
Regression cases cover float64, float32, uint8 and int8 targets, including the
signed minimum, across MSE, MAE, quantile, Huber, SVR and epsilon-quantile losses.
CVXPY receives the original numeric values converted independently to floating
point. Each row reports `target_dtype`; the dtype/loss expansion changes case
contents relative to earlier benchmark revisions.
Classification cases cycle through an explicit Cartesian product of 2–5 classes,
four constraint modes, intercept on/off, Ridge/ElasticNet and OvR/OvO: 128 cells.
The first 256 mixed regression/classification cases cover every cell; later
cycles add balanced, nonuniform and zero class weights. Balanced reference
weights use `w_i * sum(w) / (K * sum(w[y == y_i]))`, calculated independently
on original retained classes before OvR/OvO decomposition. Reports record actual
retained-class coverage, not just requested class counts. The case generator
changed with this matrix, so historical case numbers require their original
benchmark revision to reproduce exactly.
Every binary problem is checked against the original full objective and requested
constraints, including cold, warm and threaded fits. Identical cold starts must
match exactly across thread counts. Warm prediction differences at numerical
decision ties are reported separately; the objective/feasibility gates remain strict.
Use `--seed` and `--case` to reproduce an individual case. The default report is
`results/correctness/api.json`; CVXPY is only a benchmark dependency.

The API and fairness suites also exercise ``decision_function_shape`` on each
cold, warm and threaded classifier fit. Binary and OvO models switch between
both formats without refitting: predictions, coefficients, objective values and
iteration counts must remain exactly unchanged. Raw multiclass scores must use
SVC's first-class-positive pair convention; binary scores stay second-class
positive. Per-case ``score_format_checks`` counts these format round trips.
Original weighted objectives and constraints are checked against CVXPY after
the round trip, including asymmetric constraints. Unit tests additionally reject
incorrect score signs and missing pair columns.

## Fairness covariance correctness

Run `python -m benchmarks.fairness_correctness --cases 600` to check centered
fairness constraints on raw and sklearn Ridge/ElasticNet estimators, including
OvR/OvO, zero loss/class weights, intercepts and constant sensitive columns.
The `covariance_case` field distinguishes constant decimal sensitive columns,
constant other columns, zero bounds, shifted data and ordinary data. Constant
columns include zero-tolerance checks, where spurious roundoff constraints can
change the optimum. These cases supersede earlier seed/case contents.
The oracle computes covariance independently from pairwise differences between
observations, then solves each original weighted objective with CVXPY. Cold,
warm and threaded solutions must satisfy that independent constraint and match
the full objective. Inputs are read-only to check that centering does not alter
the model's training features. Use `--seed` and `--case` to replay a case.
