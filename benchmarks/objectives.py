"""Independent PLQ objective checks shared by the benchmark runners.

Objectives use the native normalization: the L2 term is 0.5 * ||beta||^2.
Thus ElasticNet values include division by (1 - l1_ratio), and the synthetic
intercept coefficient is included in the penalty, just as in the solver.
"""

import argparse
import hashlib
import json
import math
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path
from threading import Lock

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from benchmarks.fit_records import FitRecords, expected_keys, keyed_records, record_key, track_fits

_recorder_lock = Lock()


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


def validate_fit(fit, *, rtol=1e-8, atol=1e-10):
    """Require a finite, converged solution with a sufficiently small bound gap."""
    _check_tolerances(rtol, atol)
    values = [fit[name] for name in ("objective", "feasible_upper_bound", "dual_lower_bound")]
    _require(
        fit["converged"] and fit["dual_feasible"] and all(math.isfinite(value) for value in values),
        "Invalid or unconverged fit",
    )
    upper, lower = fit["feasible_upper_bound"], fit["dual_lower_bound"]
    _require(abs(upper - lower) <= atol + rtol * abs(upper), "Optimality gap exceeds tolerance")
    if "objective_scale" in fit:
        scale = fit["objective_scale"]
        _require(math.isfinite(scale) and 0 < scale <= 1, "Invalid original objective normalization")
        for field in ("objective", "dual_lower_bound", "feasible_upper_bound"):
            _require(
                math.isclose(fit["original_" + field], fit[field] * scale, rel_tol=1e-12, abs_tol=1e-12),
                "Original objective normalization disagrees with native objective",
            )
    return max(0.0, upper - lower) / max(abs(upper), 1e-300)


@contextmanager
def record_solver_fits(*, audit=False, rtol=1e-8, atol=1e-10, cv=None, n_candidates=None):
    """Collect fits with serial outer CV and optional threaded multiclass solves.

    This temporarily wraps a module-level function, so it must not be used by
    concurrent benchmark runs or alongside other fits in this process.
    """
    import rehline._class as models

    _check_tolerances(rtol, atol)
    if not _recorder_lock.acquire(blocking=False):
        raise ValueError("Concurrent or nested objective recorders are not supported")
    original = models.ReHLine_solver
    fits = FitRecords()
    context = None

    def recorded(*args, **problem):
        _require(not args, "Objective recording requires keyword solver inputs")
        result = original(**problem)
        fit = audit_solver_result(problem, result) if audit else _solver_diagnostics(result)
        if audit:
            validate_fit(fit, rtol=rtol, atol=atol)
        if context is not None:
            info = getattr(context, "value", None)
            _require(info is not None, "Native solve has no estimator/subproblem identity")
            fit["fit_key"] = record_key(info["fit_id"], info["kind"], info["classes"])
            fit["objective_scale"] = float(info["objective_scale"])
            for field in ("objective", "dual_lower_bound", "feasible_upper_bound"):
                if field in fit:
                    fit["original_" + field] = fit[field] * fit["objective_scale"]
        fits.append(fit)
        return result

    models.ReHLine_solver = recorded
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            tracker = (
                track_fits(fits, cv=cv, n_candidates=n_candidates, audit=audit) if cv is not None else nullcontext()
            )
            with tracker as context:
                yield fits
    finally:
        models.ReHLine_solver = original
        _recorder_lock.release()


def validate_timed_fits(timed, audited, *, repeats, expected_per_repeat, rtol=1e-8, atol=1e-10):
    """Check every timed fit against the independently audited repeat."""
    _check_tolerances(rtol, atol)
    _require(len(audited) == expected_per_repeat, "Missing candidate or fold fits in objective audit")
    _require(len(timed) == repeats * expected_per_repeat, "Missing candidate or fold fits in timing run")
    keys = [fit["fit_key"] for fit in audited] if audited and "fit_key" in audited[0] else None
    references = keyed_records(audited, keys) if keys is not None else None
    if references is not None:
        for repeat in range(repeats):
            keyed_records(timed[repeat * expected_per_repeat : (repeat + 1) * expected_per_repeat], keys)
    for index, fit in enumerate(timed):
        reference = references[fit["fit_key"]] if references is not None else audited[index % expected_per_repeat]
        _require(fit["converged"], f"Unconverged timing fit {index}")
        for field in ("objective", "dual_lower_bound"):
            _require(
                math.isclose(fit[field], reference[field], rel_tol=rtol, abs_tol=atol),
                f"Timing fit {index} {field} differs from objective audit",
            )
        if "objective_scale" in reference:
            _require(fit.get("objective_scale") == reference["objective_scale"], "Timing fit objective scale differs")
            for field in ("original_objective", "original_dual_lower_bound"):
                _require(
                    math.isclose(fit[field], reference[field], rel_tol=rtol, abs_tol=atol),
                    f"Timing fit {index} {field} differs from objective audit",
                )


def compare(before, after, *, rtol=1e-8, atol=1e-10):
    """Compare complete benchmark reports, including every candidate and fold."""
    _check_tolerances(rtol, atol)
    # Output locations do not change the optimization problem.
    configs = [
        {key: value for key, value in report["config"].items() if key != "output_dir"} for report in (before, after)
    ]
    if configs[0] != configs[1]:
        raise ValueError("Audit configurations differ")
    if not before["rows"] or len(before["rows"]) != len(after["rows"]):
        raise ValueError("Audits must contain the same nonempty set of tasks")
    expected = [(task, dataset) for task, datasets in before["config"]["task_datasets"].items() for dataset in datasets]
    for audit in (before, after):
        if [(row["task"], row["dataset"]) for row in audit["rows"]] != expected:
            raise ValueError("Audit is incomplete for the configured tasks")
    summary = {
        "problems_checked": 0,
        "rtol": rtol,
        "atol": atol,
        "max_objective_abs_difference": 0.0,
        "max_objective_relative_difference": 0.0,
        "max_relative_certified_gap_before": 0.0,
        "max_relative_certified_gap_after": 0.0,
    }
    for left, right in zip(before["rows"], after["rows"]):
        key = (left["task"], left["dataset"])
        old_fits = left.get("objective_fits", left.get("fits", []))
        new_fits = right.get("objective_fits", right.get("fits", []))
        if key != (right["task"], right["dataset"]) or not old_fits or len(old_fits) != len(new_fits):
            raise ValueError(f"Different training problems: {key}")
        for row, fits in ((left, old_fits), (right, new_fits)):
            if "objective_manifest" in row:
                keys = expected_keys(row["objective_manifest"], row["n_candidates"], row["cv"])
                keyed_records(fits, keys)
                expected_fits = len(keys)
            else:
                expected_fits = row.get("objective_repeats", row["repeats"]) * (row["n_candidates"] * row["cv"] + 1)
            if len(fits) != expected_fits:
                raise ValueError(f"Missing candidate or fold fits: {key}")
            if "fits" in row and "objective_fits" in row:
                validate_timed_fits(
                    row["fits"], fits, repeats=row["repeats"], expected_per_repeat=expected_fits, rtol=rtol, atol=atol
                )
        if "objective_manifest" in left and "objective_manifest" in right:
            _require(left["objective_manifest"] == right["objective_manifest"], "Estimator fit manifests differ")
            keys = expected_keys(left["objective_manifest"], left["n_candidates"], left["cv"])
            old_by_key, new_by_key = keyed_records(old_fits, keys), keyed_records(new_fits, keys)
            old_fits, new_fits = [old_by_key[k] for k in keys], [new_by_key[k] for k in keys]
        for old, new in zip(old_fits, new_fits):
            if old["problem_sha256"] != new["problem_sha256"]:
                raise ValueError(f"Objective inputs differ: {key}")
            for label, fit in (("before", old), ("after", new)):
                try:
                    relative_gap = validate_fit(fit, rtol=rtol, atol=atol)
                except ValueError as error:
                    raise ValueError(f"{error}: {key}/{label}") from error
                field = f"max_relative_certified_gap_{label}"
                summary[field] = max(summary[field], relative_gap)
            for field in ("objective", "feasible_upper_bound"):
                if not math.isclose(old[field], new[field], rel_tol=rtol, abs_tol=atol):
                    raise ValueError(f"Objective mismatch: {key}, {old[field]} vs {new[field]}")
            if "objective_scale" in old and "objective_scale" in new:
                _require(old["objective_scale"] == new["objective_scale"], "Original objective scales differ")
                _require(
                    math.isclose(old["original_objective"], new["original_objective"], rel_tol=rtol, abs_tol=atol),
                    "Original objective mismatch",
                )
            difference = abs(old["objective"] - new["objective"])
            summary["problems_checked"] += 1
            summary["max_objective_abs_difference"] = max(summary["max_objective_abs_difference"], difference)
            summary["max_objective_relative_difference"] = max(
                summary["max_objective_relative_difference"], difference / max(abs(old["objective"]), 1e-300)
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
