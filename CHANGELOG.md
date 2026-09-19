# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Scale-invariant constraint correctness benchmark, checking four equivalent
  row representations against the original CVXPY problem. The release workflow
  verifies actual cibuildwheel artifacts on CPython 3.12 on each platform before
  publishing, with artifact checksums and mandatory objective comparisons.
- `scaled_constraint_violation_` alongside original-unit constraint diagnostics.

- Explicit `to_inference()` exports for CQR and all four sklearn estimators,
  retaining independent predictions, fitted metadata and final diagnostics
  without loss/dual caches. CQR paths accept `compact=True` for these snapshots
  while keeping their return tuple and array shapes. Snapshots cannot refit.
- Independent implicit-CQR correctness suite comparing the joint weighted
  objective and dual bound with both dense ReHLine and CVXPY, including warm
  starts, repeated quantiles and rank-deficient designs.
- Correctness CI for CVXPY 1.6.0 and current dependencies, plus objective checks
  against installed wheels on Linux, macOS and Windows.
- Multiclass release benchmarks for Ridge/ElasticNet OvR and OvO on Iris, Wine
  and Digits, with per-candidate/fold/class-pair objective auditing. Audit counts
  derive independently from retained labels, including zero class weights and
  folds with missing classes. Original ElasticNet objectives are recorded too.
- Explicit 128-cell classifier correctness matrix over class count, constraints,
  intercept, penalty and training strategy, with coverage recorded in reports.
- Classifier `decision_function_shape='ovr'|'ovo'` selects class-aligned scores
  (default) or raw OvO pair margins independently of the training strategy.
  Raw multiclass margins follow SVC's first-class-positive convention; binary
  margins remain second-class-positive. Format changes require no refit and
  preserve predictions and objectives.
- Public-estimator correctness benchmark for CQR, MF convex blocks and full
  weighted MF histories, plus raw-estimator clone checks against CVXPY.
- MF exposes its weighted training objective, outer iteration count, convergence
  status and final constraint violation.

- Independent CVXPY correctness benchmark covering 1,200 small weighted and
  constrained convex problems, with full objective comparisons, shrinking/warm
  start checks, saved failure inputs and a CI job.
- Optional objective verification in GridSearchCV benchmarks, outside timing,
  and automatic baseline objective comparison in the release benchmark runner.

### Fixed

- Named losses convert numeric targets to float64 before arithmetic, preventing
  unsigned negation and signed integer overflow from silently changing the
  objective. Regression tests and CVXPY API benchmarks cover integer targets.
- Fairness covariance subtracts a reference row before centering, keeping
  constant decimal columns exactly zero even with `tol_sen=0`. Genuine tiny
  nonzero constraints remain active; training features are unchanged.
- Sparse `make_mf_dataset` pair sampling uses memory proportional to requested
  interactions, with uniform sampling without replacement. Explicit zero counts
  produce empty data; dimensions, counts and density are validated. Sparse
  datasets can differ from previous releases for the same seed, while repeated
  calls with the same arguments remain reproducible.

- Classifier `class_weight="balanced"` now balances sample-weight mass, matching
  integer row replication for fixed constraints and LinearSVC >= 1.7. The formula
  is independent of the installed sklearn version and applies globally before
  binary/OvR/OvO decomposition. Extreme representable weights are handled safely.
- MF outer stopping and final constraint warnings use row-normalized feasibility,
  consistently with block solves. `constraint_violation_` retains original units;
  MF also exposes `scaled_constraint_violation_`. Equivalent row scalings are
  checked in the public-estimator correctness benchmark.

- Constraint row scaling no longer causes premature successful convergence.
  Native solves normalize rows safely, preserve original-unit duals, and require
  an objective-gap check as well as KKT/feasibility checks. Warm refits convert
  cached multipliers when row scales change; tiny nonzero rows are not rejected
  as zero because of squared-norm underflow.
- MF emits `ConvergenceWarning` when its outer sweep budget is exhausted even
  if all inner block solves converged. Short-budget benchmark/tests explicitly
  check the warning and final status instead of suppressing convergence errors.

- Numerical overflow/nonfinite solver computations raise `OverflowError`
  instead of reporting successful convergence with a zero gap and infinite or
  NaN objectives. Existing fitted state survives a failed refit.
- CQR copies fitted quantile labels, preventing later changes to a constructor
  array from silently relabeling existing prediction columns.
- Failed fits preserve the last successful fitted state for all public
  estimators, including raw ReHLine, ERM, CQR and MF. This covers multiclass
  worker failures and convergence warnings promoted to errors. MF predictions
  and objective evaluation retain the configuration of the successful fit.
- Binary zero margins select the first sorted class, consistently with native
  OvO and sklearn linear classifiers wrapped in `OneVsOneClassifier`.
- Named-loss APIs reject nonempty `U/V/S/T/Tau` instead of silently ignoring
  them. Manual loss matrices remain supported by `ReHLine` and `ReHLine_solver`.
- CVXPY reference problems with no positive-weight rows use the constrained
  penalty alone, including with CVXPY 1.6, which rejects empty matrix expressions.
- Smooth-hinge CVXPY references use an equivalent quadratic-program epigraph,
  allowing the OSQP fallback on CVXPY 1.6 without relaxing accuracy requirements.
- CQR references supply full-shape multiplication constants for CVXPY 1.6.0's
  SCIPY canonicalization backend, avoiding its implicit-broadcast shape errors.
- Built-in fairness constraints use centered empirical covariance, with temporary
  statistics that leave training features unchanged. Document reference rows,
  loss-weight semantics, intercept handling and the limits of score-based fairness.
- ERM and sklearn estimators enforce constructor `A/b` constraints. When supplied
  with `constraint`, both constraint lists are enforced, with a notice per fit.
- Multiclass warm starts reuse compatible per-class/pair dual states.
- OvO decision scores align with classes for sklearn calibration and scoring.
- MF uses sample weights consistently in its objective history and stopping check;
  `obj(X, y, sample_weight=...)` evaluates explicitly weighted data.
- MF cold-start and zero-weight blocks solve the constrained minimum-norm problem.
  Integer-valued numeric IDs are normalized before indexing.
- Raw ReHLine cloning and `set_params` preserve loss/constraint parameters, including
  legacy private-array assignment, while ERM clones omit generated training state.
- Stabilize near-converged constrained primal recovery under cancellation, with
  full stationarity/KKT, feasibility and objective-gap checks at the original tolerance.

- Zero-weight quadratic and squared-hinge terms no longer produce NaN cutpoints.
- All sklearn estimators apply `intercept_scaling` consistently in training and prediction.
- Built-in constraints apply to feature coefficients, excluding the synthetic intercept.
  Custom constraints accept `d` columns for features or `d + 1` columns to explicitly
  constrain the actual intercept when `fit_intercept=True`.
- Validate solver shapes, finite values, sample weights and parameter ranges before
  native computation, including parameters changed with `set_params`.
- Regularization paths report `C * loss + 0.5 * ||beta||^2`; CQR paths retain
  independent model snapshots for each regularization value.
- Cold starts avoid large initial coefficients under extreme class weights.
- Low-level estimators reset incompatible cached duals when refitting with changed
  sample counts, losses, constraints, quantiles or L1 penalty dimensions.
- CQR accepts scalar sample weights consistently with other estimators.
- Classifiers remove zero effective-weight samples before multiclass decomposition,
  including zero weights supplied through `class_weight`.
- Release the GIL during native solves so independent threaded fits can run concurrently.
- Accelerate slow coordinate descent on correlated data with bounded joint dual
  updates, retaining the same objective, feasibility bounds and KKT tolerance.

### Changed

- CQR training evaluates its joint design implicitly from the original features,
  avoiding the `(n * q, d + q)` dense matrix. Shared slopes, quantile intercepts,
  both ridge penalties and convergence requirements are unchanged.
- Default OvO inference aggregates bounded blocks of pair margins, reducing
  temporary memory while preserving score conventions and vote/tie rules.
- OvO creates training subsets inside bounded worker tasks instead of retaining
  all pairwise feature copies. Raw-score users should select
  `decision_function_shape='ovo'` and account for the sign reversal from the
  former second-class-positive multiclass output. Stored coefficients retain
  their training orientation. The proposed `pairwise_decision_function` method
  was replaced before release by this parameter.
- CQR prediction broadcasts the shared linear prediction instead of building a
  dense matrix per quantile; the optimized objective and intercept penalty are unchanged.
- Bundle Eigen 5.0.1 headers and license files in source distributions for offline
  compilation. Git builds automatically download missing headers and verify the
  pinned archive and file checksums; downloaded files are ignored by Git. The
  optional `tools/prepare_eigen.py` command supports offline preparation from a
  local archive. Source-build checks deny Python network access.

- Require scikit-learn 1.6 or newer. The sklearn constructors retain parameters
  unchanged; `loss=None` selects median regression at fit time, and
  `multi_class=None` selects OvR. Parameter validation now occurs in `fit`.
- Convergence requires a full projected-gradient/KKT check. Difficult problems
  may require more iterations than before; unconverged native solves emit a warning.
- `n_iter_` counts completed sweeps (starting at one). Final `objective_`,
  `dual_objective_`, `dual_gap_`, `constraint_violation_`, `kkt_residual_` and
  `converged_` are available independently of verbosity.
- Test built wheels outside the source checkout and rebuild/test sdists before
  publishing. Release publication depends on the cross-platform test workflow.
- Unexpected convergence warnings fail the test suite; tests of nonconvergence
  explicitly assert the warning.

### Compatibility notes

- Binary predictions at exactly zero margin now select `classes_[0]` instead
  of `classes_[1]`. Nonzero margins, multiclass voting and score signs are
  unchanged. No numerical epsilon is used to widen the tie.
- The sklearn regressors expose predictions through `predict`; migrate calls to
  their former `decision_function` alias to `predict`. Classifiers and low-level
  ERM estimators retain `decision_function`. All sklearn checks run without
  expected-failure exceptions.
- Historical `dual_obj_` traces retain the minimized negative-dual convention.
  The new scalar `dual_objective_` is the maximized dual lower bound.
- `dual_gap_` is infinite when constraint violation exceeds `tol`. Within
  feasibility tolerance it reports the non-negative difference between the unconstrained
  objective (loss and penalties) and the dual bound; it is not an exact certificate
  for a strictly infeasible iterate. Use `kkt_residual_` and
  `constraint_violation_` alongside it.
- With intercept scaling `s`, the regularized synthetic coefficient is
  `intercept_ / s`; intercept regularization is retained.

## [0.1.1] - 2025-10-07

### Added

- Support for monotonic constraints (both increasing and decreasing) in solvers
- Monotonic constraint documentation and tutorials

### Fixed

- Various bug fixes and improvements

## [0.1.0] - 2025-06-10

### Added

- Full scikit-learn compatibility with `plq_Ridge_Classifier` and `plq_Ridge_Regressor`
- Integration with scikit-learn `Pipeline`
- Support for `GridSearchCV` and standard evaluation metrics

### Changed

- Improved documentation and tutorials

## [0.0.7] - 2025-06-10

### Added

- Multi-class classification support
- ElasticNet penalty support for `plqERM`
- ElasticNet penalty support for ReHLine solver

### Fixed

- Various bug fixes and improvements

## [0.0.6] - 2025-01-14

### Added

- Matrix factorization support with `plqMF_Ridge` class
- `make_mf_dataset` function for matrix factorization datasets

### Changed

- Updated dependencies

## [0.0.5] - 2024-10-31

### Added

- Additional features and improvements

### Changed

- Documentation improvements and code formatting

## [0.0.4] - 2024-09-03

### Added

- Additional features and improvements

### Changed

- Dependency updates

## [0.0.3] - 2024-04-24

### Added

- Core solver improvements and bug fixes

## [0.0.1] - 2023-10-18

### Added

- Initial release of ReHLine Python package
- ReHLine solver for regularized composite ReLU-ReHU loss minimization
- Support for convex piecewise linear-quadratic loss functions
- Linear equality and inequality constraints support
- Ridge regression and classification estimators

[Unreleased]: https://github.com/softmin/ReHLine-python/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/softmin/ReHLine-python/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/softmin/ReHLine-python/compare/v0.0.7...v0.1.0
[0.0.7]: https://github.com/softmin/ReHLine-python/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/softmin/ReHLine-python/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/softmin/ReHLine-python/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/softmin/ReHLine-python/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/softmin/ReHLine-python/compare/v0.0.1...v0.0.3
[0.0.1]: https://github.com/softmin/ReHLine-python/tree/v0.0.1
