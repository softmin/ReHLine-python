"""Check implicit CQR against the dense joint problem and independent CVXPY.

Run python -m benchmarks.cqr_correctness --cases 600. No timing claims are made
by this small-problem correctness suite; all fits use the same strict tolerance.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from rehline import CQR_Ridge, ReHLine_solver


def make_case(index, seed=20260918):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    n, d, q = int(rng.integers(3, 31)), int(rng.integers(1, 7)), 1 + index % 7
    X, y = rng.normal(size=(n, d)), rng.normal(size=n)
    geometry = (index // 7) % 5
    if geometry == 1:
        X[:, -1] = 1
    elif geometry == 2:
        X[::2] = 0
    elif geometry == 3:
        X[:, -1] = X[:, 0]
    elif geometry == 4:
        X = X[:, :1] + X * 0.01
    levels = rng.permutation(np.linspace(0.1, 0.9, q))
    if index % 4 == 0:
        levels[:] = 0.5  # Repeated/unsorted levels must retain output order.
    weights = rng.uniform(0.1, 2, n)
    weights[::3] = 0
    if index % 3 == 0:
        weights[:] = 0.7
    return dict(X=X, y=y, quantiles=levels, weight=weights, C=float(10 ** rng.uniform(-2, 0)))


def objective(case, beta, intercept):
    residual = case["y"][:, None] - (case["X"] @ beta)[:, None] - intercept[None, :]
    loss = np.maximum(case["quantiles"] * residual, (case["quantiles"] - 1) * residual)
    return float(case["C"] * np.sum(case["weight"][:, None] * loss) + 0.5 * (beta @ beta + intercept @ intercept))


def solve_reference(case):
    import cvxpy as cp

    X, y, levels = case["X"], case["y"], case["quantiles"]
    n, d, q = len(X), X.shape[1], len(levels)
    beta, alpha = cp.Variable(d), cp.Variable(q)
    residual = y[:, None] - cp.reshape(X @ beta, (n, 1), order="C") - cp.reshape(alpha, (1, q), order="C")
    loss = cp.Variable((n, q))
    # CVXPY 1.6.0's SCIPY backend requires full-shape multiplication constants.
    level_grid = np.broadcast_to(levels, (n, q))
    weight_grid = np.broadcast_to(case["weight"][:, None], (n, q))
    constraints = [loss >= cp.multiply(level_grid, residual), loss >= cp.multiply(level_grid - 1, residual)]
    problem = cp.Problem(
        cp.Minimize(
            case["C"] * cp.sum(cp.multiply(weight_grid, loss)) + 0.5 * (cp.sum_squares(beta) + cp.sum_squares(alpha))
        ),
        constraints,
    )
    attempts = []
    for solver in ("CLARABEL", "OSQP"):
        options = (
            dict(tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11, max_iter=500)
            if solver == "CLARABEL"
            else dict(eps_abs=1e-11, eps_rel=1e-11, max_iter=200000, polishing=True)
        )
        try:
            with warnings.catch_warnings(record=True):
                problem.solve(solver=solver, canon_backend="SCIPY", **options)
            if problem.status != cp.OPTIMAL or beta.value is None or alpha.value is None:
                raise ValueError(f"CVXPY status {problem.status}")
            value = objective(case, beta.value, alpha.value)
            if not np.isfinite(value) or not np.isclose(value, problem.value, rtol=1e-8, atol=1e-9):
                raise ValueError("Independent objective disagrees with CVXPY")
            # Construct a feasible dual independently of ReHLine. Clipping
            # the first epigraph multiplier and setting the second to C*w-u
            # enforces positivity and epigraph stationarity.
            bound = case["C"] * case["weight"][:, None]
            positive = np.clip(constraints[0].dual_value, 0, bound)
            residual_weight = positive + (levels - 1) * bound
            slope_dual = X.T @ residual_weight.sum(axis=1)
            intercept_dual = residual_weight.sum(axis=0)
            dual = float(
                y @ residual_weight.sum(axis=1) - 0.5 * (slope_dual @ slope_dual + intercept_dual @ intercept_dual)
            )
            if not np.isfinite(dual) or abs(value - dual) > 1e-10 + 1e-9 * abs(value):
                raise ValueError("CVXPY reference has an uncertified objective gap")
            return value, solver
        except (ValueError, cp.error.SolverError) as error:
            attempts.append(f"{solver}: {error}")
    raise ValueError(f"No reliable independent reference: {attempts}")


def dense_problem(case):
    """Build the explicit joint QR loss directly from its mathematical formula."""
    X, y, levels, weight = case["X"], case["y"], case["quantiles"], case["weight"]
    n, d, q = len(X), X.shape[1], len(levels)
    design = np.zeros((n * q, d + q))
    for i in range(q):
        design[i * n : (i + 1) * n, :d] = X
        design[i * n : (i + 1) * n, d + i] = 1
    level = np.repeat(levels, n)
    U = np.vstack((-level, 1 - level)) * (case["C"] * np.tile(weight, q))
    return dict(X=design, U=U, V=-U * np.tile(y, q))


def check_case(index, seed=20260918):
    case = make_case(index, seed)
    reference, solver = solve_reference(case)
    differences = []
    for shrink in (0, 1):
        dense = ReHLine_solver(**dense_problem(case), max_iter=100000, tol=1e-10, shrink=shrink, verbose=0)
        if not dense.converged:
            raise AssertionError("Dense reference did not converge")
        d = case["X"].shape[1]
        dense_value = objective(case, dense.beta[:d], dense.beta[d:])
        for value in (dense_value, dense.objective, dense.dual_objective):
            if not np.isclose(value, reference, rtol=1e-8, atol=1e-9):
                raise AssertionError("Dense full objective/dual disagrees with CVXPY")
        differences.append(abs(dense_value - reference))
        model = CQR_Ridge(case["quantiles"], C=case["C"], max_iter=100000, tol=1e-10, shrink=shrink, warm_start=True)
        for _ in range(2):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(case["X"], case["y"], sample_weight=case["weight"])
            if not model.converged_ or model.kkt_residual_ > model.tol:
                raise AssertionError("Implicit CQR failed the requested KKT tolerance")
            actual = objective(case, model.coef_, model.intercept_)
            for value in (actual, model.objective_, model.dual_objective_):
                if not np.isfinite(value) or not np.isclose(value, reference, rtol=1e-8, atol=1e-9):
                    raise AssertionError(f"Implicit CQR full objective/dual {value} differs from {reference}")
            np.testing.assert_allclose(actual, dense_value, rtol=1e-8, atol=1e-9)
            np.testing.assert_array_equal(model.quantiles_, case["quantiles"])
            snapshot = model.to_inference()
            np.testing.assert_array_equal(snapshot.predict(case["X"]), model.predict(case["X"]))
            np.testing.assert_array_equal(snapshot.quantiles_, case["quantiles"])
            for value in (
                objective(case, snapshot.coef_, snapshot.intercept_),
                snapshot.objective_,
                snapshot.dual_objective_,
            ):
                if not np.isfinite(value) or not np.isclose(value, reference, rtol=1e-8, atol=1e-9):
                    raise AssertionError("Compact CQR objective disagrees with CVXPY")
            differences.append(abs(actual - reference))
    return dict(
        index=index,
        status="passed",
        comparisons=len(differences),
        inference_comparisons=4,
        solver=solver,
        max_objective_difference=max(differences),
    )


def run_suite(cases=600, seed=20260918, case_index=None):
    import cvxpy as cp

    if cases < 1:
        raise ValueError("cases must be positive")
    rows, start = [], time.perf_counter()
    for index in range(cases) if case_index is None else [case_index]:
        try:
            rows.append(check_case(index, seed))
        except Exception as error:
            rows.append(dict(index=index, status="failed", error=repr(error)))
    return dict(
        seed=seed,
        cvxpy_version=cp.__version__,
        numpy_version=np.__version__,
        tol=1e-10,
        objective_rtol=1e-8,
        objective_atol=1e-9,
        cases=len(rows),
        passed=sum(r["status"] == "passed" for r in rows),
        comparisons=sum(r.get("comparisons", 0) for r in rows),
        inference_comparisons=sum(r.get("inference_comparisons", 0) for r in rows),
        elapsed_sec=time.perf_counter() - start,
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--case", type=int)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/correctness/cqr.json"))
    args = parser.parse_args()
    report = run_suite(args.cases, args.seed, args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
    failures = [r for r in report["rows"] if r["status"] != "passed"]
    if failures:
        print(json.dumps(failures[:10], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
