"""Independent small-problem checks for CQR, MF blocks/fit, and raw cloning.

Run python -m benchmarks.estimator_correctness --cases 1200. CVXPY is only
required by this benchmark, never by the estimators' training paths.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from benchmarks.correctness import make_case, objective, solve_reference

LOSSES = ({"name": "MSE"}, {"name": "MAE"}, {"name": "QR", "qt": 0.3}, {"name": "huber", "tau": 0.7})
FAMILIES = ("mse", "mae", "quantile", "huber")


def agree(value, reference):
    if not np.isfinite(value) or not np.isfinite(reference) or not np.isclose(value, reference, rtol=1e-8, atol=1e-9):
        raise AssertionError(f"Objective mismatch: {value} versus {reference}")


def geometry(rng, d, mode):
    if mode == 0:
        return np.empty((0, d)), np.empty(0)
    lower = rng.uniform(-0.5, 0.5, d)
    if mode == 1:
        lower = np.zeros(d)
    if mode < 3:
        return np.vstack((np.eye(d), -np.eye(d))), np.r_[-lower, lower + rng.uniform(0.5, 1.5, d)]
    A = rng.normal(size=(d + 2, d))
    b = -A @ lower + rng.uniform(0.1, 1.0, len(A))
    return A, b


def direct_loss(residual, family):
    if family == 0:
        return residual**2
    if family == 1:
        return abs(residual)
    if family == 2:
        return np.maximum(0.3 * residual, -0.7 * residual)
    z = abs(residual)
    return np.where(z <= 0.7, 0.5 * z**2, 0.7 * (z - 0.35))


def check_cqr(rng, index, seed):
    import cvxpy as cp

    from rehline import CQR_Ridge

    n, d, q = int(rng.integers(5, 20)), int(rng.integers(1, 5)), int(rng.integers(1, 5))
    X = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    levels = rng.permutation(np.linspace(0.1, 0.9, q)) if q > 1 else np.array([0.5])
    w = rng.uniform(0.1, 2, n)
    w[::4] = 0
    C = float(10 ** rng.uniform(-2, 0))
    slope, intercept = cp.Variable(d), cp.Variable(q)
    residual = y[:, None] - cp.reshape(X @ slope, (n, 1), order="C") - cp.reshape(intercept, (1, q), order="C")
    # Explicit shapes also work with the minimum supported CVXPY 1.6.0.
    level_grid = np.broadcast_to(levels, (n, q))
    weight_grid = np.broadcast_to(w[:, None], (n, q))
    loss = cp.maximum(cp.multiply(level_grid, residual), cp.multiply(level_grid - 1, residual))
    problem = cp.Problem(
        cp.Minimize(
            C * cp.sum(cp.multiply(weight_grid, loss)) + 0.5 * (cp.sum_squares(slope) + cp.sum_squares(intercept))
        )
    )
    problem.solve(canon_backend="SCIPY", solver="CLARABEL", tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11)
    if problem.status != "optimal":
        raise AssertionError(f"Unreliable CQR reference: {problem.status}")

    def value(beta, alpha):
        residual = y[:, None] - (X @ beta)[:, None] - alpha[None, :]
        return float(
            C * np.sum(w[:, None] * np.maximum(levels * residual, (levels - 1) * residual))
            + 0.5 * (beta @ beta + alpha @ alpha)
        )

    reference = value(slope.value, intercept.value)
    agree(reference, problem.value)
    differences = []
    for shrink in (0, 1):
        model = CQR_Ridge(levels, C=C, shrink=shrink, warm_start=True, tol=1e-9, max_iter=100000)
        for _ in range(2 if shrink else 1):
            model.fit(X, y, sample_weight=w)
            if not model.converged_:
                raise AssertionError("CQR did not converge")
            actual = value(model.coef_, model.intercept_)
            agree(actual, reference)
            agree(actual, model.objective_)
            agree(reference, model.dual_objective_)
            np.testing.assert_allclose(
                model.predict(X), (X @ model.coef_)[:, None] + model.intercept_[None, :], atol=1e-12
            )
            differences.append(abs(actual - reference))
    return {"kind": "cqr", "comparisons": 3, "max_objective_difference": max(differences)}


def check_mf_block(rng, index, seed):
    from rehline import plqMF_Ridge

    family = (index // 4) % 4
    d, n = int(rng.integers(1, 5)), int(rng.integers(1, 18))
    design = rng.normal(size=(n, d))
    if index % 8:
        design[:, 0] = 1.0
    target, bias = rng.normal(size=n), rng.normal(size=n)
    weight = rng.uniform(0.1, 2.0, n)
    weight[::3] = 0
    no_observations = (index // 4) % 5 == 0
    no_weight = (index // 4) % 5 == 1
    if no_observations:
        design, target, bias, weight = design[:0], target[:0], bias[:0], weight[:0]
    elif no_weight:
        weight[:] = 0
    A, b = geometry(rng, d, (index // 16) % 4)
    C = float(10 ** rng.uniform(-2, 1))
    constraints = [{"name": "custom", "A": A, "b": b}] if len(b) else None
    model = plqMF_Ridge(n_users=1, n_items=1, loss=LOSSES[family], tol=1e-9, max_iter=100000)
    z, converged = model._solve_block(design, target, weight, bias, constraints, C, {})
    if not converged:
        raise AssertionError("MF block did not converge")
    if len(b) and np.min(A @ z + b) < -1.01e-9:
        raise AssertionError("MF block violates a constraint")
    case = make_case(index, seed=seed)
    case.update(
        family=FAMILIES[family],
        X=design if len(design) else np.zeros((1, d)),
        y=target - bias if len(design) else np.zeros(1),
        weight=weight if len(design) else np.zeros(1),
        C=C,
        l1_ratio=0.0,
        omega=np.ones(d),
        A=A,
        b=b,
        qt=0.3,
        tau=0.7,
    )
    for name in ("U", "V", "S", "T", "Tau"):
        case[name] = np.empty((0, len(case["X"])))
    reference = solve_reference(case)
    actual = float(C * (weight @ direct_loss(target - design @ z - bias, family)) + 0.5 * (z @ z))
    agree(actual, objective(case, z))
    agree(actual, reference["objective"])
    return {"kind": "mf_block", "comparisons": 1, "max_objective_difference": abs(actual - reference["objective"])}


def check_mf_fit(rng, index, seed):
    from rehline import plqMF_Ridge

    family = (index // 4) % 4
    X = np.array([(u, i) for u in range(3) for i in range(4)])
    y, w = rng.normal(size=len(X)), rng.uniform(0.1, 2, len(X))
    w[X[:, 0] == 0] = 0
    rank, biased = 1 + (index // 16) % 2, bool((index // 8) % 2)
    d = rank + biased
    lower = rng.uniform(0.1, 0.3, d)
    constraints = [{"name": "custom", "A": np.eye(d), "b": -lower}] if (index // 4) % 2 else None
    model = plqMF_Ridge(
        n_users=4,
        n_items=5,
        rank=rank,
        biased=biased,
        loss=LOSSES[family],
        C=0.1,
        random_state=seed + index,
        constraint_user=constraints,
        constraint_item=constraints,
        tol=1e-9,
        max_iter=100000,
        max_iter_CD=8,
        tol_CD=1e-10,
    )

    def validate(fitted):
        # A short outer budget validates the attained objective, not a global optimum.
        with warnings.catch_warnings(record=True) as caught:
            warnings.filterwarnings("always", message="MF outer iterations failed", category=ConvergenceWarning)
            fitted.fit(X, y, sample_weight=w)
        if len(caught) != int(not fitted.converged_) or any(
            item.category is not ConvergenceWarning or "max_iter_CD" not in str(item.message) for item in caught
        ):
            raise AssertionError("MF outer convergence status and warnings disagree")
        pred = np.sum(fitted.P[X[:, 0]] * fitted.Q[X[:, 1]], axis=1)
        penalty = fitted.rho / 4 * np.square(fitted.P).sum() + (1 - fitted.rho) / 5 * np.square(fitted.Q).sum()
        if biased:
            pred += fitted.bu[X[:, 0]] + fitted.bi[X[:, 1]]
            penalty += fitted.rho / 4 * np.square(fitted.bu).sum() + (1 - fitted.rho) / 5 * np.square(fitted.bi).sum()
        actual = float(0.1 * (w @ direct_loss(y - pred, family)) + penalty)
        agree(actual, fitted.objective_)
        agree(actual, fitted.history[fitted.n_iter_, 1])
        if not fitted.inner_converged_:
            raise AssertionError("An MF inner solve did not converge")
        if fitted.scaled_constraint_violation_ > 1.01e-9:
            raise AssertionError("Final MF constraints are violated")
        if constraints:
            for factors, bias in ((fitted.P, fitted.bu), (fitted.Q, fitted.bi)):
                parameters = np.column_stack((bias, factors)) if biased else factors
                if np.min(parameters - lower) < -1.01e-9:
                    raise AssertionError("Independent factor feasibility check failed")
        # The random initial point need not be feasible. Fixed constraints remain
        # unchanged after one full sweep, so the objective should descend.
        history = fitted.history[1 : fitted.n_iter_ + 1, 1]
        if len(history) > 1 and np.max(np.diff(history)) > 1e-8 * max(1.0, np.max(abs(history))):
            raise AssertionError("Weighted MF objective increased after a feasible sweep")
        return actual

    reference = validate(model)
    differences = [abs(reference - model.objective_)]
    if constraints:
        for exponents in (np.full(d, -40), np.full(d, 40), np.arange(d) * 40 - 40):
            scale = np.exp2(exponents)
            equivalent = [{"name": "custom", "A": np.diag(scale), "b": -scale * lower}]
            scaled = clone(model).set_params(constraint_user=equivalent, constraint_item=equivalent)
            actual = validate(scaled)
            agree(actual, reference)
            if scaled.converged_ != model.converged_ or scaled.n_iter_ != model.n_iter_:
                raise AssertionError("Equivalent MF constraints changed outer convergence")
            differences.append(abs(actual - reference))
    return {"kind": "mf_fit", "comparisons": len(differences), "max_objective_difference": max(differences)}


def check_clone(rng, index, seed):
    from sklearn.base import clone

    from rehline import ReHLine

    d, n = int(rng.integers(1, 5)), int(rng.integers(5, 20))
    X, y = rng.normal(size=(n, d)), rng.normal(size=n)
    A, b = geometry(rng, d, (index // 4) % 4)
    C = float(rng.uniform(0.01, 0.2))
    model = ReHLine(
        S=np.vstack((np.full(n, np.sqrt(2)), np.full(n, -np.sqrt(2)))),
        T=np.vstack((-np.sqrt(2) * y, np.sqrt(2) * y)),
        Tau=np.full((2, n), np.inf),
        A=A,
        b=b,
        C=C,
        tol=1e-9,
        max_iter=100000,
    )
    copied = clone(model)
    model.fit(X)
    copied.fit(X)
    case = make_case(index, seed=seed)
    case.update(family="mse", X=X, y=y, weight=np.ones(n), C=C, l1_ratio=0.0, omega=np.ones(d), A=A, b=b)
    for name in ("U", "V", "S", "T", "Tau"):
        case[name] = np.empty((0, len(case["X"])))
    reference = solve_reference(case)
    differences = []
    for fitted in (model, copied):
        if not fitted.converged_:
            raise AssertionError("Raw estimator did not converge")
        actual = objective(case, fitted.coef_)
        agree(actual, fitted.objective_)
        agree(actual, reference["objective"])
        agree(reference["objective"], fitted.dual_objective_)
        if len(b):
            row_scale = np.max(abs(A), axis=1)
            row_scale[row_scale == 0] = 1
            violation = max(0.0, -np.min((A @ fitted.coef_ + b) / row_scale))
            if violation > 1.01 * fitted.tol:
                raise AssertionError("Raw clone violates the scaled constraint tolerance")
            agree(violation, fitted.scaled_constraint_violation_)
        differences.append(abs(actual - reference["objective"]))
    return {"kind": "clone", "comparisons": 2, "max_objective_difference": max(differences)}


CHECKS = (check_cqr, check_mf_block, check_mf_fit, check_clone)


def run_suite(cases=1200, seed=20260913, case_index=None):
    if cases < 1:
        raise ValueError("cases must be positive")
    rows = []
    start = time.perf_counter()
    for index in range(cases) if case_index is None else [case_index]:
        try:
            result = CHECKS[index % 4](np.random.default_rng(np.random.SeedSequence([seed, index])), index, seed)
            rows.append({"index": index, "status": "passed", **result})
        except Exception as error:
            rows.append({"index": index, "status": "failed", "kind": CHECKS[index % 4].__name__, "error": repr(error)})
    return {
        "seed": seed,
        "cases": len(rows),
        "passed": sum(row["status"] == "passed" for row in rows),
        "comparisons": sum(row.get("comparisons", 0) for row in rows),
        "elapsed_sec": time.perf_counter() - start,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--case", type=int)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/correctness/estimators.json"))
    args = parser.parse_args()
    report = run_suite(args.cases, args.seed, args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))
    failures = [row for row in report["rows"] if row["status"] == "failed"]
    if failures:
        print(json.dumps(failures[:10], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
