"""Report fixed-budget accuracy separately from solver correctness failures.

The six release-review problems use analytic box/quadratic references and an
independent primal epigraph QP for hinge loss. No reference uses ReHLine updates.
Budget misses remain explicit in the report; invalid outputs fail the command.
"""

import argparse
import hashlib
import json
import platform
import time
from itertools import product
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, minimize

from rehline import ReHLine_solver, _internal
from tests._helpers.core import make_case
from tests._helpers.objectives import objective_value

ACCURACY = 1e-8


def mse_problem(X, y, ratio, *, C=1.0, weight=None, omega=None, A=None, b=None):
    """Construct the complete weighted MSE problem in native normalization."""
    n, d = X.shape
    weight = np.ones(n) if weight is None else weight
    omega = np.ones(d) if omega is None else omega
    root = np.sqrt((C / (1 - ratio)) * weight)
    S = np.array([[np.sqrt(2)] * n, [-np.sqrt(2)] * n]) * root
    T = np.array([-np.sqrt(2) * y, np.sqrt(2) * y]) * root
    tau = np.where(root > 0, np.inf, 0.0)
    return dict(
        X=X,
        U=np.empty((0, 0)),
        V=np.empty((0, 0)),
        S=S,
        T=T,
        Tau=np.broadcast_to(tau, S.shape).copy(),
        A=np.empty((0, d)) if A is None else A,
        b=np.empty(0) if b is None else b,
        rho=(ratio / (1 - ratio)) * omega if ratio else None,
    )


def quadratic_reference(p):
    X, s, t = p["X"], p["S"][0], p["T"][0]
    d = X.shape[1]
    Q = np.eye(d) + X.T @ (s[:, None] ** 2 * X)
    target = -X.T @ (s * t)
    rho = np.zeros(d) if p["rho"] is None else p["rho"]
    for pattern in product((-1, 0, 1), repeat=d):
        signs = np.array(pattern)
        active = signs != 0
        beta = np.zeros(d)
        beta[active] = np.linalg.solve(Q[np.ix_(active, active)], target[active] - rho[active] * signs[active])
        if np.any(beta[active] * signs[active] < 0):
            continue
        if np.all(abs((Q @ beta - target)[~active]) <= rho[~active] + 1e-9):
            return beta
    raise ValueError("No valid quadratic reference sign pattern")


def hinge_reference(p):
    n, d = p["X"].shape
    rho = np.zeros(d) if p["rho"] is None else p["rho"]
    # Variables are beta, hinge epigraph values, and absolute-beta epigraphs.
    M = np.block(
        [
            [p["U"][0, :, None] * p["X"], -np.eye(n), np.zeros((n, d))],
            [np.eye(d), np.zeros((d, n)), -np.eye(d)],
            [-np.eye(d), np.zeros((d, n)), -np.eye(d)],
        ]
    )
    upper = np.r_[-p["V"][0], np.zeros(2 * d)]
    solution = minimize(
        lambda v: 0.5 * (v[:d] @ v[:d]) + v[d : d + n].sum() + rho @ v[d + n :],
        np.r_[np.zeros(d), np.maximum(p["V"][0], 0), np.zeros(d)],
        jac=lambda v: np.r_[v[:d], np.ones(n), rho],
        method="SLSQP",
        bounds=Bounds(np.r_[np.full(d, -np.inf), np.zeros(n + d)], np.inf),
        constraints=LinearConstraint(M, -np.inf, upper),
        options={"ftol": 1e-11, "maxiter": 3000},
    )
    if not solution.success or np.max(M @ solution.x - upper) > 1e-8:
        raise ValueError(f"Invalid hinge reference: {solution.message}")
    return solution.x[:d]


def cases():
    for index in (1332, 2844):
        c = make_case(index, seed=20260913)
        p = mse_problem(
            c["X"], c["y"], c["l1_ratio"], C=c["C"], weight=c["weight"], omega=c["omega"], A=c["A"], b=c["b"]
        )
        d = c["X"].shape[1]
        lower, upper = -c["b"][:d], c["b"][d:]
        vertices = (np.where(mask, upper, lower) for mask in product((False, True), repeat=d))
        beta = min(vertices, key=lambda v: objective_value(p, v))
        gradient = beta + c["X"].T @ (p["S"][0] * (p["S"][0] * (c["X"] @ beta) + p["T"][0]))
        gradient += p["rho"] * np.sign(beta)
        if np.min(np.where(beta == lower, 1, -1) * gradient) <= 100:
            raise ValueError("Box reference is not a certified optimal corner")
        yield f"box{index}", p, beta, "analytic corner", (0, 1)

    rng = np.random.default_rng(0)
    X = np.column_stack((100 + rng.normal(size=(80, 2)), np.ones(80)))
    y = rng.normal(size=80)
    for ratio in (0.0, 0.5):
        p = mse_problem(X, y, ratio)
        yield f"correlated_mse_{ratio}", p, quadratic_reference(p), "quadratic sign patterns", (0, 1)

    rng = np.random.RandomState(42)
    X = np.column_stack((rng.normal(loc=100, size=(100, 2)), np.ones(100)))
    y = 2 * rng.randint(0, 2, size=100) - 1
    for ratio in (0.0, 0.5):
        p = dict(
            X=X,
            U=-y[None, :] / (1 - ratio),
            V=np.ones((1, 100)) / (1 - ratio),
            S=np.empty((0, 0)),
            T=np.empty((0, 0)),
            Tau=np.empty((0, 0)),
            A=np.empty((0, 3)),
            b=np.empty(0),
            rho=np.full(3, ratio / (1 - ratio)) if ratio else None,
        )
        yield f"shifted_svm_{ratio}", p, hinge_reference(p), "primal epigraph QP (SLSQP)", (1,)


def measure(p, result, reference, *, tol, max_iter):
    """Fail on invalid diagnostics; return accuracy without requiring convergence."""
    for field in ("beta", "xi", "Lambda", "Gamma", "mu", "objective", "dual_objective", "kkt_residual"):
        if not np.isfinite(getattr(result, field)).all():
            raise ValueError(f"Nonfinite solver output: {field}")
    if not (0 < result.niter <= max_iter):
        raise ValueError("Invalid iteration count")
    if np.any(result.xi < 0) or np.any((result.Lambda < 0) | (result.Lambda > 1)):
        raise ValueError("Infeasible constraint/ReLU dual variables")
    if result.Gamma.size and (np.any(result.Gamma < 0) or np.any(result.Gamma > p["Tau"])):
        raise ValueError("Infeasible ReHU dual variables")
    if p["rho"] is not None and np.any((result.mu < 0) | (result.mu > p["rho"])):
        raise ValueError("Infeasible L1 dual variables")
    actual = objective_value(p, result.beta)
    if not np.isclose(actual, result.objective, rtol=1e-12, atol=1e-8):
        raise ValueError("Reported objective disagrees with independent evaluation")
    violation = 0.0
    if p["A"].size:
        scale = np.max(abs(p["A"]), axis=1)
        scale[scale == 0] = 1
        violation = float(max(0, -np.min((p["A"] @ result.beta + p["b"]) / scale)))
    if not np.isclose(violation, result.scaled_constraint_violation, rtol=1e-10, atol=1e-12):
        raise ValueError("Reported feasibility disagrees with independent evaluation")
    gap = abs(actual - result.dual_objective) / max(1, abs(actual), abs(result.dual_objective))
    if result.converged and (violation > tol or gap > tol):
        raise ValueError("Solver claims convergence without its reported certificate")
    trace = np.asarray(result.dual_objfns)
    if not np.isfinite(trace).all() or np.any(np.diff(trace) > 1e-12 * np.maximum(1, abs(trace[:-1]))):
        raise ValueError("Negative dual objective increased during coordinate descent")
    error = float(abs(actual - reference) / max(1, abs(reference)))
    return dict(
        status="met" if error <= ACCURACY and violation <= ACCURACY else "not_met",
        n_iter=int(result.niter),
        converged=bool(result.converged),
        objective=actual,
        reference_objective=reference,
        objective_error=error,
        feasibility=violation,
        kkt_residual=float(result.kkt_residual),
        relative_dual_gap=float(gap),
    )


def run(*, max_iter=1_000_000):
    records = []
    for name, problem, beta, reference_method, modes in cases():
        reference = objective_value(problem, beta)
        for shrink in modes:
            record = dict(case=name, shrink=shrink, max_iter=max_iter, tol=ACCURACY, reference_method=reference_method)
            start = time.perf_counter()
            try:
                result = ReHLine_solver(
                    **problem,
                    max_iter=max_iter,
                    tol=ACCURACY,
                    shrink=shrink,
                    verbose=1,
                    trace_freq=max(1, max_iter // 100),
                )
                record.update(measure(problem, result, reference, tol=ACCURACY, max_iter=max_iter))
            except (ValueError, OverflowError, RuntimeError) as error:
                record.update(status="error", error=str(error))
            record["seconds"] = time.perf_counter() - start
            records.append(record)
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        native_sha256=hashlib.sha256(Path(_internal.__file__).read_bytes()).hexdigest(),
        objective_tolerance=ACCURACY,
        feasibility_tolerance=ACCURACY,
        counts={status: sum(r["status"] == status for r in records) for status in ("met", "not_met", "error")},
        results=records,
    )


def markdown(report):
    counts = report["counts"]
    lines = [
        "# Numerical stress report",
        "",
        f"Accuracy met: **{counts['met']}**; budget target not met: **{counts['not_met']}**; errors: **{counts['error']}**.",
        "",
        "Objective error = abs(f - reference) / max(1, abs(reference)); objective and feasibility targets are 1e-8.",
        "Budget misses are reported separately from algorithm or diagnostic errors. Solver convergence is unchanged.",
        "",
        "| Case | shrink | Budget | Iterations | Objective error | Feasibility | KKT | Converged | Target |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in report["results"]:
        if r["status"] == "error":
            lines.append(f"| {r['case']} | {r['shrink']} | {r['max_iter']} | — | — | — | — | — | ERROR |")
        else:
            lines.append(
                f"| {r['case']} | {r['shrink']} | {r['max_iter']} | {r['n_iter']} | "
                f"{r['objective_error']:.3e} | {r['feasibility']:.3e} | {r['kkt_residual']:.3e} | "
                f"{r['converged']} | {r['status']} |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=Path("test-results/numerical-stress/report.json"))
    parser.add_argument("--require-accuracy", action="store_true", help="Also exit nonzero for budget misses")
    args = parser.parse_args(argv)
    if args.max_iter < 1:
        parser.error("--max-iter must be positive")
    report = run(max_iter=args.max_iter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["counts"]))
    return int(report["counts"]["error"] > 0 or (args.require_accuracy and report["counts"]["not_met"] > 0))


if __name__ == "__main__":
    raise SystemExit(main())
