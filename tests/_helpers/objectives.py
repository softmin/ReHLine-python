"""Independent full-objective, dual-bound and feasibility checks for tests."""

import hashlib
import math

import numpy as np


def _check_tolerances(rtol, atol):
    if not all(math.isfinite(value) and value >= 0 for value in (rtol, atol)) or rtol + atol == 0:
        raise ValueError("Objective tolerances must be finite, nonnegative, and not both zero")


def objective_value(problem, beta):
    """Evaluate all weighted ReLU/ReHU losses and L2/L1 penalties independently."""
    X = problem["X"]
    count = problem.get("_quantile_count", 0)
    scores = ((X @ beta[: X.shape[1]])[None, :] + beta[X.shape[1] :, None] if count else X @ beta).reshape(-1)
    value = 0.5 * np.dot(beta, beta)
    if problem["U"].size:
        value += np.maximum(problem["U"] * scores + problem["V"], 0).sum()
    if problem["S"].size:
        z = np.maximum(problem["S"] * scores + problem["T"], 0)
        tau = problem["Tau"]
        quadratic = z <= tau
        value += 0.5 * np.square(z[quadratic]).sum()
        value += (tau[~quadratic] * (z[~quadratic] - 0.5 * tau[~quadratic])).sum()
    if problem.get("rho") is not None and np.size(problem["rho"]):
        value += np.dot(problem["rho"], abs(beta))
    return float(value)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _require_close(actual, expected, name):
    _require(np.allclose(actual, expected, rtol=1e-12, atol=1e-9), f"Independent {name} disagrees with solver")


def _solver_diagnostics(result):
    return {
        "converged": bool(result.converged),
        "n_iter": int(result.niter),
        "kkt_residual": float(result.kkt_residual),
        "objective": float(result.objective),
        "dual_lower_bound": float(result.dual_objective),
        "constraint_violation": float(result.constraint_violation),
    }


def audit_solver_result(problem, result):
    """Recompute a solution's objective, dual feasibility and optimality bounds.

    Unconstrained and increasing/decreasing adjacent monotonic constraints are
    supported. Unknown constraints fail explicitly instead of claiming a valid
    primal bound for an approximately feasible coefficient vector.
    """
    p, r = problem, result
    beta = r.beta
    independent = objective_value(p, beta)
    _require_close(independent, r.objective, "objective")
    for name in ("beta", "xi", "Lambda", "Gamma", "mu"):
        _require(np.isfinite(getattr(r, name)).all(), f"Nonfinite solver variable: {name}")
    _require((r.xi >= 0).all(), "Infeasible constraint dual variables")
    _require(((r.Lambda >= 0) & (r.Lambda <= 1)).all(), "Infeasible ReLU dual variables")
    _require((r.Gamma >= 0).all(), "Infeasible ReHU dual variables")
    if p["S"].size:
        _require((r.Gamma <= p["Tau"]).all(), "Infeasible ReHU dual variables")
    reconstructed = np.zeros_like(beta)
    negative_dual = 0.0

    def transpose_product(weights):
        if not p.get("_quantile_count", 0):
            return p["X"].T @ weights
        by_quantile = weights.reshape(p["_quantile_count"], len(p["X"]))
        return np.r_[p["X"].T @ by_quantile.sum(axis=0), by_quantile.sum(axis=1)]

    if p["A"].size:
        reconstructed += p["A"].T @ r.xi
        negative_dual += np.dot(r.xi, p["b"])
    if p["U"].size:
        reconstructed -= transpose_product((p["U"] * r.Lambda).sum(axis=0))
        negative_dual -= (r.Lambda * p["V"]).sum()
    if p["S"].size:
        reconstructed -= transpose_product((p["S"] * r.Gamma).sum(axis=0))
        negative_dual += 0.5 * np.square(r.Gamma).sum() - (r.Gamma * p["T"]).sum()
    if p.get("rho") is not None and np.size(p["rho"]):
        _require(((r.mu >= 0) & (r.mu <= p["rho"])).all(), "Infeasible L1 dual variables")
        reconstructed += 2 * r.mu - p["rho"]
    _require_close(beta, reconstructed, "coefficient reconstruction")
    negative_dual += 0.5 * np.dot(reconstructed, reconstructed)
    _require_close(-negative_dual, r.dual_objective, "dual objective")
    feasible = beta.copy()
    A, b = p["A"], p["b"]
    if A.size:
        # The benchmark estimators can penalize a trailing synthetic intercept.
        supported = False
        for n_features in (len(beta), len(beta) - 1):
            expected = np.diff(np.eye(len(beta))[:n_features], axis=0)
            for sign in (1, -1):
                if np.array_equal(A, sign * expected) and (b == 0).all():
                    feasible[:n_features] = sign * np.maximum.accumulate(sign * feasible[:n_features])
                    supported = True
                    break
            if supported:
                break
        _require(supported, "Objective audit supports only unconstrained or adjacent monotonic constraints")
        _require((A @ feasible + b >= 0).all(), "Primal upper bound is not feasible")
    upper = objective_value(p, feasible)
    digest = hashlib.sha256()
    for name in ("X", "U", "V", "S", "T", "Tau", "A", "b", "rho"):
        value = p.get(name)
        value = np.empty(0) if value is None else np.asarray(value, dtype=np.float64)
        digest.update(name.encode())
        if name == "X" and p.get("_quantile_count", 0):
            # Hash the mathematical expanded design in bounded chunks. This
            # matches historical dense reports without rebuilding n*q*d data.
            n, d = value.shape
            count = p["_quantile_count"]
            digest.update(str((n * count, d + count)).encode())
            for q in range(count):
                for start in range(0, n, 256):
                    features = value[start : start + 256]
                    rows = np.zeros((len(features), d + count))
                    rows[:, :d] = features
                    rows[:, d + q] = 1
                    digest.update(rows.tobytes())
            continue
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return {
        **_solver_diagnostics(r),
        "problem_sha256": digest.hexdigest(),
        "objective": independent,
        "feasible_upper_bound": upper,
        "dual_lower_bound": float(-negative_dual),
        "dual_feasible": True,
        "certified_gap": max(0.0, upper + negative_dual),
    }
