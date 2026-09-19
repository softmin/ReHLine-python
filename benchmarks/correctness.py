"""Many small, reproducible convex problems checked against CVXPY.

Run ``python -m benchmarks.correctness --cases 1200``. This checks numerical
correctness, not comparative performance. No downloaded datasets are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import warnings
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np

FAMILIES = (
    "mse",
    "mae",
    "quantile",
    "quantile_eps",
    "huber",
    "svr",
    "hinge",
    "smooth_hinge",
    "squared_hinge",
    "relu",
    "rehu",
    "mixed",
)
GEOMETRIES = ("none", "nonnegative", "monotonic", "box", "linear", "equality")
DESIGNS = ("normal", "correlated", "rank_deficient", "zero_rows", "offset", "intercept")
ARRAYS = ("X", "y", "weight", "omega", "A", "b", "U", "V", "S", "T", "Tau")


def make_case(index, *, seed=20260912, max_samples=40, max_dim=8):
    """Generate one case independently of execution order or total case count."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    family = FAMILIES[index % len(FAMILIES)]
    geometry = GEOMETRIES[(index // len(FAMILIES)) % len(GEOMETRIES)]
    n = int(rng.integers(5, max_samples + 1))
    d = int(rng.integers(1, max_dim + 1))
    X = rng.normal(size=(n, d))
    design = DESIGNS[int(rng.integers(len(DESIGNS)))]
    if design == "correlated":
        X = X[:, :1] + 0.01 * X
    elif design == "rank_deficient":
        X[:, -1] = X[:, 0] if d > 1 else 0
    elif design == "zero_rows":
        X[::3] = 0
    elif design == "offset":
        X += 10
    elif design == "intercept":
        X[:, -1] = rng.choice([0.2, 1.0, 5.0])
    y = X @ rng.normal(size=d) + rng.normal(size=n)
    if family in ("hinge", "smooth_hinge", "squared_hinge"):
        y = np.where(y > np.median(y), 1.0, -1.0)
    weight = rng.uniform(0.1, 2, n)
    weight[::4] = 0
    C = float(10 ** rng.uniform(-2, 1))
    l1_ratio = float(rng.choice([0.0, 0.2, 0.7]))
    omega = rng.uniform(0.1, 2, d)
    omega[::3] = 0  # Include unpenalized L1 coordinates.
    A, b = np.empty((0, d)), np.empty(0)
    anchor = rng.normal(size=d)
    if geometry == "nonnegative":
        A, b = np.eye(d), np.zeros(d)
    elif geometry == "monotonic":
        A, b = np.diff(np.eye(d), axis=0), np.zeros(d - 1)
    elif geometry == "box":
        A = np.vstack([np.eye(d), -np.eye(d)])
        b = np.r_[-anchor, anchor + rng.uniform(0.2, 2, d)]
    elif geometry == "linear":
        A = rng.normal(size=(max(1, d), d))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        b = -A @ anchor + rng.uniform(0.2, 1, len(A))
    elif geometry == "equality":
        a = rng.normal(size=(1, d))
        a /= np.linalg.norm(a)
        A = np.vstack([a, -a])
        b = -A @ anchor
    L = int(rng.integers(1, 4)) if family in ("relu", "mixed") else 0
    H = int(rng.integers(1, 3)) if family in ("rehu", "mixed") else 0
    U, V = rng.normal(size=(L, n)), rng.normal(size=(L, n))
    S, T = rng.normal(size=(H, n)), rng.normal(size=(H, n))
    Tau = rng.uniform(0.1, 2, size=(H, n))
    Tau[:, ::4] = np.inf
    Tau[:, 1::5] = 0
    return {
        "index": index,
        "seed": seed,
        "family": family,
        "geometry": geometry,
        "design": design,
        "C": C,
        "l1_ratio": l1_ratio,
        "qt": float(rng.choice([0.1, 0.5, 0.9])),
        "epsilon": float(rng.uniform(0.05, 0.5)),
        "tau": float(rng.uniform(0.1, 2)),
        "X": X,
        "y": y,
        "weight": weight,
        "omega": omega,
        "A": A,
        "b": b,
        "U": U,
        "V": V,
        "S": S,
        "T": T,
        "Tau": Tau,
    }


def _rehu(z, tau):
    z = np.maximum(z, 0)
    quadratic = z <= tau
    value = np.empty_like(z)
    value[quadratic] = 0.5 * z[quadratic] ** 2
    value[~quadratic] = tau[~quadratic] * (z[~quadratic] - 0.5 * tau[~quadratic])
    return value


def objective(case, beta):
    """Evaluate the original, unnormalized objective directly from raw data."""
    c, f = case, case["family"]
    pred = c["X"] @ beta
    r = c["y"] - pred
    positive = np.maximum
    if f == "mse":
        loss = r**2
    elif f == "mae":
        loss = abs(r)
    elif f in ("quantile", "quantile_eps"):
        loss = positive(c["qt"] * r, (c["qt"] - 1) * r)
        if f == "quantile_eps":
            loss = positive(loss - c["epsilon"], 0)
    elif f == "huber":
        loss = _rehu(abs(r), np.full_like(r, c["tau"]))
    elif f == "svr":
        loss = positive(abs(r) - c["epsilon"], 0)
    elif f in ("hinge", "smooth_hinge", "squared_hinge"):
        margin = positive(1 - c["y"] * pred, 0)
        loss = margin if f == "hinge" else margin**2
        if f == "smooth_hinge":
            loss = _rehu(margin, np.ones_like(margin))
    else:
        loss = positive(c["U"] * pred + c["V"], 0).sum(axis=0)
        loss += _rehu(c["S"] * pred + c["T"], c["Tau"]).sum(axis=0)
    penalty = 0.5 * (1 - c["l1_ratio"]) * np.dot(beta, beta)
    penalty += c["l1_ratio"] * np.dot(c["omega"], abs(beta))
    return float(c["C"] * np.dot(c["weight"], loss) + penalty)


def reference_problem(case):
    """Build CVXPY from the mathematical losses, without ReHLine converters."""
    import cvxpy as cp

    c, f = case.copy(), case["family"]
    # A zero-weight observation contributes no term to the reference problem.
    # Removing its slack variables also avoids unconstrained epigraph directions.
    active = c["weight"] > 0
    for name in ("X", "y", "weight"):
        c[name] = c[name][active]
    for name in ("U", "V", "S", "T", "Tau"):
        c[name] = c[name][:, active]
    beta = cp.Variable(c["X"].shape[1])
    penalty = 0.5 * (1 - c["l1_ratio"]) * cp.sum_squares(beta)
    penalty += c["l1_ratio"] * cp.sum(cp.multiply(c["omega"], cp.abs(beta)))
    constraints = [c["A"] @ beta + c["b"] >= 0] if c["A"].size else []
    if not active.any():
        # With no positive-weight observations only the constrained penalty
        # remains. CVXPY 1.6 does not accept zero-row matrix expressions.
        return cp.Problem(cp.Minimize(penalty), constraints), beta
    pred = c["X"] @ beta
    r = c["y"] - pred
    if f == "mse":
        loss = cp.square(r)
    elif f == "mae":
        loss = cp.abs(r)
    elif f in ("quantile", "quantile_eps"):
        loss = cp.maximum(c["qt"] * r, (c["qt"] - 1) * r)
        if f == "quantile_eps":
            loss = cp.pos(loss - c["epsilon"])
    elif f == "huber":
        loss = 0.5 * cp.huber(r, c["tau"])
    elif f == "svr":
        loss = cp.pos(cp.abs(r) - c["epsilon"])
    elif f == "smooth_hinge":
        # .5*huber(max(1-y*pred, 0), 1) has this QP epigraph. The direct
        # composition is not recognized as a QP by CVXPY 1.6, preventing an
        # OSQP fallback when the conic reference is insufficiently accurate.
        quadratic = cp.Variable(len(c["y"]), nonneg=True)
        linear = cp.Variable(len(c["y"]), nonneg=True)
        constraints.append(quadratic + linear >= 1 - cp.multiply(c["y"], pred))
        loss = 0.5 * cp.square(quadratic) + linear
    elif f in ("hinge", "squared_hinge"):
        margin = cp.pos(1 - cp.multiply(c["y"], pred))
        loss = margin if f == "hinge" else cp.square(margin)
    else:
        weighted_loss = cp.Constant(0)
        for u, v in zip(c["U"], c["V"]):
            weighted_loss += cp.sum(cp.multiply(c["weight"], cp.pos(cp.multiply(u, pred) + v)))
        # ReHU(z,tau) = min_{a,b >= 0, a+b >= z} .5*a^2 + tau*b.
        # Infinite tau fixes b=0. Zero tau contributes zero loss.
        for s, t, tau in zip(c["S"], c["T"], c["Tau"]):
            z = cp.multiply(s, pred) + t
            finite = np.isfinite(tau) & (tau > 0)
            if finite.any():
                a = cp.Variable(int(finite.sum()), nonneg=True)
                b = cp.Variable(int(finite.sum()), nonneg=True)
                constraints.append(a + b >= z[finite])
                weighted_loss += cp.sum(
                    cp.multiply(c["weight"][finite], 0.5 * cp.square(a) + cp.multiply(tau[finite], b))
                )
            infinite = np.isinf(tau)
            if infinite.any():
                weighted_loss += 0.5 * cp.sum(cp.multiply(c["weight"][infinite], cp.square(cp.pos(z[infinite]))))
    if f not in ("relu", "rehu", "mixed"):
        weighted_loss = cp.sum(cp.multiply(c["weight"], loss))
    value = c["C"] * weighted_loss + penalty
    return cp.Problem(cp.Minimize(value), constraints), beta


def constraint_violation(case, beta):
    return float(np.maximum(-(case["A"] @ beta + case["b"]), 0).max(initial=0))


def solve_reference(case, *, solver="CLARABEL", rtol=1e-8, atol=1e-9, feasibility_tol=1e-8):
    import cvxpy as cp

    problem, beta = reference_problem(case)
    attempts = []
    fallback = "OSQP" if solver == "CLARABEL" else "CLARABEL"
    # Selection depends only on reference accuracy, never on ReHLine's answer.
    for name in (solver, fallback):
        attempt = {"solver": name}
        attempts.append(attempt)
        options = {"tol_gap_abs": 1e-11, "tol_gap_rel": 1e-11, "tol_feas": 1e-11, "max_iter": 500}
        if name == "OSQP":
            options = {"eps_abs": 1e-11, "eps_rel": 1e-11, "max_iter": 200000, "polishing": True}
        try:
            with warnings.catch_warnings(record=True) as captured:
                problem.solve(solver=name, warm_start=False, **options)
            attempt.update(status=problem.status, warnings=[str(w.message) for w in captured])
            if problem.status != cp.OPTIMAL or beta.value is None:
                raise ValueError(f"CVXPY reference status: {problem.status}")
            value = objective(case, beta.value)
            if not math.isfinite(value) or not math.isclose(value, problem.value, rel_tol=rtol, abs_tol=atol):
                raise ValueError("CVXPY objective disagrees with independent evaluation")
            violation = constraint_violation(case, beta.value)
            if violation > feasibility_tol:
                raise ValueError(f"CVXPY reference constraint violation: {violation}")
            return {
                "objective": value,
                "reported_objective": float(problem.value),
                "beta": beta.value.copy(),
                "constraint_violation": violation,
                "status": problem.status,
                "solver": name,
                "attempts": attempts,
            }
        except (ValueError, cp.error.SolverError) as error:
            attempt["error"] = str(error)
    raise ValueError(f"No accurate CVXPY reference: {attempts}")


def _native_problem(case):
    c = case
    w = c["C"] * c["weight"] / (1 - c["l1_ratio"])
    root = np.sqrt(w)
    tau = np.zeros_like(c["Tau"])
    np.multiply(c["Tau"], root, out=tau, where=root > 0)
    return {
        "X": c["X"],
        "U": c["U"] * w,
        "V": c["V"] * w,
        "S": c["S"] * root,
        "T": c["T"] * root,
        "Tau": tau,
        "A": c["A"],
        "b": c["b"],
        "rho": c["l1_ratio"] * c["omega"] / (1 - c["l1_ratio"]),
    }


def solve_rehline(case, *, tol=1e-10, max_iter=100000):
    """Yield both shrinking modes and a compatible warm refit."""
    from rehline import ReHLine_solver, plqERM_ElasticNet, plqERM_Ridge

    c, f = case, case["family"]
    losses = {
        "mse": {"name": "MSE"},
        "mae": {"name": "MAE"},
        "quantile": {"name": "QR", "qt": c["qt"]},
        "quantile_eps": {"name": "check_eps", "qt": c["qt"], "epsilon": c["epsilon"]},
        "huber": {"name": "huber", "tau": c["tau"]},
        "svr": {"name": "svr", "epsilon": c["epsilon"]},
        "hinge": {"name": "svm"},
        "smooth_hinge": {"name": "sSVM"},
        "squared_hinge": {"name": "squared SVM"},
    }
    for shrink in (0, 1):
        if f in losses:
            options = dict(C=c["C"], loss=losses[f], shrink=shrink, tol=tol, max_iter=max_iter, warm_start=True)
            if c["A"].size:
                options["constraint"] = [{"name": "custom", "A": c["A"], "b": c["b"]}]
            if c["l1_ratio"]:
                model = plqERM_ElasticNet(l1_ratio=c["l1_ratio"], omega=c["omega"], **options)
            else:
                model = plqERM_Ridge(**options)
            for warm in range(2 if shrink else 1):
                model.fit(c["X"], c["y"], sample_weight=c["weight"])
                yield {
                    "shrink": shrink,
                    "warm": bool(warm),
                    "beta": model.coef_.copy(),
                    "objective": model.objective_ * (1 - c["l1_ratio"]),
                    "dual_objective": model.dual_objective_ * (1 - c["l1_ratio"]),
                    "converged": bool(model.converged_),
                    "n_iter": int(model.n_iter_),
                    "kkt_residual": float(model.kkt_residual_),
                }
        else:
            initial = {}
            for warm in range(2 if shrink else 1):
                r = ReHLine_solver(
                    **_native_problem(c), **initial, tol=tol, max_iter=max_iter, shrink=shrink, verbose=0
                )
                yield {
                    "shrink": shrink,
                    "warm": bool(warm),
                    "beta": r.beta.copy(),
                    "objective": r.objective * (1 - c["l1_ratio"]),
                    "dual_objective": r.dual_objective * (1 - c["l1_ratio"]),
                    "converged": bool(r.converged),
                    "n_iter": int(r.niter),
                    "kkt_residual": float(r.kkt_residual),
                }
                initial = {name: getattr(r, name).copy() for name in ("Lambda", "Gamma", "xi", "mu")}


def check_solution(case, reference, solution, *, rtol=1e-8, atol=1e-9, feasibility_tol=1e-8):
    """Check the objective first; coefficients are an additional diagnostic."""
    beta = solution["beta"]
    value, expected = objective(case, beta), reference["objective"]
    violation = constraint_violation(case, beta)
    errors = []
    if not solution["converged"]:
        errors.append("ReHLine did not converge")
    for name, target in (("CVXPY", expected), ("reported", solution["objective"])):
        if not math.isfinite(value) or not math.isclose(value, target, rel_tol=rtol, abs_tol=atol):
            errors.append(f"Objective differs from {name}: {value} vs {target}")
    if violation > feasibility_tol:
        errors.append(f"Constraint violation: {violation}")
    # Compare the native dual lower bound with the independently solved primal.
    # Both primal points are only feasible to feasibility_tol; this is a numeric
    # check, not a claim of exact arithmetic certification for arbitrary A/b.
    if not math.isclose(solution["dual_objective"], expected, rel_tol=rtol, abs_tol=atol):
        errors.append("Dual objective differs from CVXPY optimum")
    result = {key: value for key, value in solution.items() if key != "beta"}
    result.update(
        independent_objective=value,
        objective_abs_difference=abs(value - expected),
        objective_relative_difference=abs(value - expected) / max(abs(expected), 1e-300),
        objective_scaled_difference=abs(value - expected) / max(1.0, abs(expected)),
        objective_tolerance_fraction=abs(value - expected) / max(atol, rtol * max(abs(value), abs(expected))),
        coefficient_max_difference=float(np.max(abs(beta - reference["beta"]))),
        constraint_violation=violation,
        errors=errors,
        status="failed" if errors else "passed",
    )
    return result


def save_case(case, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in case.items() if key not in ARRAYS}
    np.savez_compressed(path, **{key: case[key] for key in ARRAYS}, metadata=json.dumps(metadata))


def load_case(path):
    with np.load(path, allow_pickle=False) as saved:
        return {**json.loads(str(saved["metadata"])), **{key: saved[key] for key in ARRAYS}}


def run_suite(
    *,
    cases=1200,
    seed=20260912,
    max_samples=40,
    max_dim=8,
    solver="CLARABEL",
    rtol=1e-8,
    atol=1e-9,
    feasibility_tol=1e-8,
    tol=1e-10,
    max_iter=100000,
    output=None,
    case_index=None,
    replay=None,
    progress=True,
):
    import cvxpy as cp

    import rehline
    from benchmarks.objectives import _check_tolerances
    from rehline import _internal

    _check_tolerances(rtol, atol)
    if cases < 1 or max_samples < 5 or max_dim < 1 or seed < 0 or (case_index is not None and case_index < 0):
        raise ValueError("Require cases >= 1, max_samples >= 5, max_dim >= 1 and nonnegative seed/case index")
    if solver not in ("CLARABEL", "OSQP") or solver not in cp.installed_solvers():
        raise ValueError(f"Reference solver unavailable or unsupported: {solver}")
    if not math.isfinite(feasibility_tol) or feasibility_tol <= 0:
        raise ValueError("feasibility_tol must be finite and positive")
    output = Path(output) if output is not None else None
    config = dict(
        cases=cases,
        seed=seed,
        max_samples=max_samples,
        max_dim=max_dim,
        solver=solver,
        rtol=rtol,
        atol=atol,
        feasibility_tol=feasibility_tol,
        tol=tol,
        max_iter=max_iter,
        case_index=case_index,
        replay=str(replay) if replay else None,
    )
    report = {
        "status": "incomplete",
        "config": config,
        "rehline_version": rehline.__version__,
        "package": rehline.__file__,
        "native_sha256": hashlib.sha256(Path(_internal.__file__).read_bytes()).hexdigest(),
        "cvxpy_version": cp.__version__,
        "numpy_version": np.__version__,
        "reference_versions": {name: version(name) for name in ("clarabel", "osqp")},
        "rows": [],
    }
    started = time.perf_counter()

    def write_report():
        rows = report["rows"]
        comparisons = [fit for row in rows for fit in row.get("fits", [])]
        report["summary"] = {
            "problems": len(rows),
            "comparisons": len(comparisons),
            "passed": sum(row["status"] == "passed" for row in rows),
            "failed": sum(row["status"] != "passed" for row in rows),
            "families": dict(Counter(row["family"] for row in rows)),
            "geometries": dict(Counter(row["geometry"] for row in rows)),
            "designs": dict(Counter(row["design"] for row in rows)),
            "reference_solvers": dict(Counter(row["reference"]["solver"] for row in rows if "reference" in row)),
            "reference_fallbacks": sum(len(row["reference"]["attempts"]) > 1 for row in rows if "reference" in row),
            "max_objective_tolerance_fraction": max(
                (fit["objective_tolerance_fraction"] for fit in comparisons), default=0
            ),
            "max_objective_scaled_difference": max(
                (fit["objective_scaled_difference"] for fit in comparisons), default=0
            ),
            "max_objective_relative_difference": max(
                (fit["objective_relative_difference"] for fit in comparisons), default=0
            ),
            "max_objective_abs_difference": max((fit["objective_abs_difference"] for fit in comparisons), default=0),
            "elapsed_sec": time.perf_counter() - started,
        }
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(_finite_json(report), indent=2, allow_nan=False) + "\n")

    write_report()
    indices = [case_index] if case_index is not None else range(cases)
    generated = (make_case(i, seed=seed, max_samples=max_samples, max_dim=max_dim) for i in indices)
    for case in [load_case(replay)] if replay else generated:
        row = {key: value for key, value in case.items() if key not in ARRAYS}
        row.update(n_samples=case["X"].shape[0], n_features=case["X"].shape[1], fits=[], status="reference_failed")
        try:
            reference = solve_reference(case, solver=solver, rtol=rtol, atol=atol, feasibility_tol=feasibility_tol)
            row["reference"] = {key: value for key, value in reference.items() if key != "beta"}
            row["status"] = "failed"
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                for solution in solve_rehline(case, tol=tol, max_iter=max_iter):
                    row["fits"].append(
                        check_solution(case, reference, solution, rtol=rtol, atol=atol, feasibility_tol=feasibility_tol)
                    )
            row["warnings"] = [str(w.message) for w in captured]
            if len(row["fits"]) == 3 and all(fit["status"] == "passed" for fit in row["fits"]):
                row["status"] = "passed"
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        if row["status"] != "passed" and output:
            failure = output.parent / (output.stem + "-cases") / f"case-{case['index']}.npz"
            save_case(case, failure)
            row["replay"] = str(failure)
        report["rows"].append(row)
        if len(report["rows"]) % 25 == 0 or row["status"] != "passed":
            write_report()
            if progress:
                summary = report["summary"]
                print(
                    f"{summary['problems']} problems: {summary['passed']} passed, {summary['failed']} failed "
                    f"({summary['elapsed_sec']:.1f}s)",
                    flush=True,
                )
    report["status"] = "passed" if all(row["status"] == "passed" for row in report["rows"]) else "failed"
    write_report()
    return report


def _finite_json(value):
    """Keep failure reports writable even when a failed solver returns NaN/Inf."""
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--max-dim", type=int, default=8)
    parser.add_argument("--solver", choices=["CLARABEL", "OSQP"], default="CLARABEL")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--feasibility-tol", type=float, default=1e-8)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=100000)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/correctness/report.json"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", dest="case_index", type=int, help="Run one generated case by index")
    selection.add_argument("--replay", type=Path, help="Replay an exact saved failure NPZ")
    args = parser.parse_args(argv)
    try:
        report = run_suite(**vars(args))
    except ImportError as error:
        raise SystemExit('Install the optional dependencies first: pip install -e ".[benchmark]"') from error
    print(json.dumps(_finite_json({"status": report["status"], **report["summary"]}), indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
