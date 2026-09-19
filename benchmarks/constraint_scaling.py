"""Equivalent row representations checked against a well-scaled CVXPY problem.

CVXPY receives the original constraints, never their extreme representation.
This prevents two solvers from agreeing only because both lost a tiny row.
Run python -m benchmarks.constraint_scaling --cases 600; replay with --case.
"""

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from benchmarks.correctness import check_solution, make_case, solve_reference, solve_rehline


def check_case(index, seed=20260918):
    case = make_case(index, seed=seed, max_samples=20, max_dim=5)
    if not case["A"].size:
        d = case["X"].shape[1]
        case.update(A=np.vstack((np.eye(d), -np.eye(d))), b=np.r_[np.full(d, -0.2), np.ones(d)])
    reference = solve_reference(case)
    count = len(case["b"])
    variants = (
        ("original", np.ones(count)),
        ("tiny", np.full(count, 1e-12)),
        ("large", np.full(count, 1e12)),
        ("mixed", np.resize([1e-200, 1e200, 1e-10, 1e10], count)),
    )
    fits = []
    for name, scales in variants:
        scaled = dict(case, A=case["A"] * scales[:, None], b=case["b"] * scales)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            for solution in solve_rehline(scaled):
                # Check feasibility in the original geometry and compare the
                # complete objective, including ElasticNet's normalization.
                result = check_solution(case, reference, solution)
                if result["status"] != "passed":
                    raise AssertionError(f"{name}: {result['errors']}")
                fits.append(dict(representation=name, **result))
    return dict(
        index=index,
        family=case["family"],
        geometry=case["geometry"],
        reference_objective=reference["objective"],
        reference_solver=reference["solver"],
        fits=fits,
    )


def run_suite(cases=600, seed=20260918, case_index=None):
    import cvxpy

    import rehline
    from rehline import _internal

    if cases < 1 or seed < 0 or (case_index is not None and case_index < 0):
        raise ValueError("Require positive cases and nonnegative seed/case index")
    start, rows = time.perf_counter(), []
    for index in range(cases) if case_index is None else [case_index]:
        try:
            rows.append(dict(status="passed", **check_case(index, seed)))
        except Exception as error:
            rows.append(dict(index=index, status="failed", error=repr(error)))
    return dict(
        package=rehline.__file__,
        native_sha256=hashlib.sha256(Path(_internal.__file__).read_bytes()).hexdigest(),
        cvxpy_version=cvxpy.__version__,
        seed=seed,
        cases=len(rows),
        passed=sum(r["status"] == "passed" for r in rows),
        comparisons=sum(len(r.get("fits", [])) for r in rows),
        elapsed_sec=time.perf_counter() - start,
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--case", type=int)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/correctness/constraint-scaling.json"))
    args = parser.parse_args()
    report = run_suite(args.cases, args.seed, args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
    failed = [row for row in report["rows"] if row["status"] != "passed"]
    if failed:
        print(json.dumps(failed[:10], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
