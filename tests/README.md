# Test policy

Benchmark runners and independent reference helpers live in the separate
[ReHLine-benchmarking](https://github.com/softmin/ReHLine-benchmarking) repository.
Install the sibling checkout into the test environment before running tests:

```sh
python -m pip install -e ".[test,benchmark]"
python -m pip install --no-deps -e ../ReHLine-benchmarking
```

The benchmark package is a test dependency, not a ReHLine runtime dependency.
CI checks it out explicitly; `REHLINE_BENCHMARKING_REF` (repository variable)
can select a matching commit/tag instead of `main`. Publish the benchmark
migration before enabling the corresponding Python-repository CI changes.
Performance-runner tests now live in ReHLine-benchmarking/tests. Numerical solver
acceptance and release/wheel gates remain here and continue to use their shared
independent references. CI report artifacts go to `test-results/`.

For a wheel-test environment without an installed benchmark package, use
`python tools/test_installed.py --benchmark-source ../ReHLine-benchmarking`.
This copies only benchmark helpers/configs into the temporary test directory;
it does not add the Python source checkout to the import path.

Routine PR, source and installed-wheel CI runs:

```sh
python -m pytest tests -m "not numerical_stress"
```

Numerical acceptance uses independently evaluated objectives, with
`abs(f - reference) / max(1, abs(reference)) <= 1e-8`, and normalized constraint
violation at most `1e-8`. Coefficient agreement and a full KKT certificate are
not additional general accuracy requirements. Tests specifically covering
convergence diagnostics, API warnings, coordinate descent, weighting semantics,
input validation, serialization and predictions remain strict. Convergence
warnings are still errors unless an individual numerical test explicitly
captures them and verifies its alternative objective certificate.

The sklearn estimator-contract checks receive up to five million sweeps to
finish their shifted-data fits. Their external elementwise output-equivalence
checks run in the stress suite; ordinary independent objective/weighting tests
remain required. The marker is assigned by check name, across all estimators,
not by matching a previously failing node or catching an assertion failure.

Required CVXPY checks cover all 12 loss families and all 6 constraint geometries
using normal-scale designs, `tol=1e-8` and a one-million-sweep budget. They check
the full weighted objective, normalized feasibility and validity of the dual
lower bound; the absolute KKT residual remains a separately recorded diagnostic.
Source and exact installed-wheel reference jobs run 256 such problems (768 fits):

```sh
python -m benchmarks.correctness.core --profile routine --cases 256 \
  --max-samples 20 --max-dim 5 --tol 1e-8 --max-iter 1000000 \
  --output test-results/correctness/routine.json
```

Small API, CQR, fairness and constraint-scaling reference regressions also remain
in routine pytest. The full randomized matrices use the stress marker.

Coordinate-order regressions cover both cyclic/random orders with shrinking
enabled/disabled, legacy defaults and native positional calls, reproducible
seeds, warm refits, full dual descent, mixed-loss references, compact/dense CQR
and parameter forwarding through estimators/path helpers. Timing comparisons
run separately with `python -m benchmarks.diagnostics.coordinate_order`; they do not gate CI.

Shrinking regressions also verify that relaxed `eps_shrink` recovery happens
before final accuracy without declaring an unfinished prefix converged. They
independently reconstruct primal/dual objectives, cover empty and zero-curvature
groups, and check strict final certificates across loss scales and warm refits.
The recovery threshold changes the working-set schedule; it does not relax the
solver's final `tol` or introduce a timing requirement in CI.

Final solver convergence requires the relative primal-dual gap and normalized
feasibility at `tol`; absolute KKT residuals are still reported. Analytic ReHU
regressions cover early termination with a small independently reconstructed
gap but a larger KKT residual, including implicit CQR, constraints and both
coordinate orders/shrinking settings. Final budget diagnostics use the same rule.
Recovery-event regressions verify termination with shrunken coordinates using
independently reconstructed full objectives. An infeasible-constraint case with
a small numerical gap must keep iterating, including under extreme positive row
scalings. Failed probes preserve the incremental state before restoration.
Optional recovery probes require two full scans' worth of coordinate visits
since the previous certificate check. Regressions reconstruct visited work from
active sets, including coordinates removed during the sweep, and verify that
deferring a probe does not defer restoration. The full-set and final diagnostic
certificates remain independent of this scheduling budget.

## Numerical stress

The separate `Numerical stress (diagnostic)` workflow runs weekly and on manual
dispatch. It is not called by PR or release workflows. Its accuracy report uses
six deterministic release-review problems: two constrained boxes, two MSE fits
with large feature offsets and a synthetic intercept, and two shifted SVM fits.
Both update modes are measured for boxes and MSE, giving ten cold fits. References
include the full weighted loss and penalties, including the synthetic intercept.
Box references are analytic optimal corners; MSE references enumerate primal
sign patterns; SVM references solve an independent primal epigraph QP with SciPy.

```sh
python -m benchmarks.diagnostics.numerical_stress --max-iter 1000000 --output test-results/numerical-stress/report.json
python -m benchmarks.diagnostics.numerical_stress --max-iter 20000000 --output /tmp/stress.json
python -m pytest tests -m numerical_stress --no-cov -o junit_family=legacy --junitxml=stress.xml
```

The report records objective error, normalized feasibility, KKT residual,
iterations, elapsed time, the actual solver convergence flag, versions and native
binary hash. JSON and Markdown are saved under
`test-results/numerical-stress/` (CI); CI uploads them and displays the summary.
`not_met` means the requested accuracy was not achieved within that budget. It
is never relabeled as converged. The report command exits nonzero for invalid
outputs, false convergence diagnostics, or a worsening dual trace. Add
`--require-accuracy` to also fail the command on an accuracy miss.

The workflow's separate strict pytest job preserves the original fixed-budget
accuracy assertions and sklearn output comparisons. It can remain red when
these targets are missed; the XML artifact retains failures. It has no
`continue-on-error`, blanket warning suppression, or expected-failure override.
The dispatch budget controls the report; strict pytest retains its recorded
test budgets. Plain `pytest` still includes both suites.

The six complete random scans (correctness, public estimators, API/multiclass,
CQR, constraint scaling and fairness) run in separate stress jobs against current
CVXPY and CVXPY 1.6.0. Their original strict thresholds and failures are retained,
with JSON reports and any saved replay cases uploaded even on failure.

The exact release-artifact gate still requires the installed binary to pass
routine objective/reference checks and pytest. Solver defaults and the
`polish_primal` correction algorithm are unchanged.
