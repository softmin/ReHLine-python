"""Validate fair constraints using independent pairwise covariance and CVXPY."""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.base import clone

from benchmarks.api_correctness import case_data, check_result, check_score_formats, fit, summarize
from rehline import (
    plq_ElasticNet_Classifier,
    plq_ElasticNet_Regressor,
    plq_Ridge_Classifier,
    plq_Ridge_Regressor,
    plqERM_ElasticNet,
    plqERM_Ridge,
)

from .correctness import solve_reference

ESTIMATORS = (
    plqERM_Ridge,
    plqERM_ElasticNet,
    plq_Ridge_Regressor,
    plq_ElasticNet_Regressor,
    plq_Ridge_Classifier,
    plq_ElasticNet_Classifier,
)
LOSSES = (
    ("mse", {"name": "MSE"}),
    ("mae", {"name": "MAE"}),
    ("quantile", {"name": "QR", "qt": 0.3}),
    ("huber", {"name": "huber", "tau": 0.7}),
)
CLASS_LOSSES = (
    ("hinge", {"name": "svm"}),
    ("smooth_hinge", {"name": "sSVM"}),
    ("squared_hinge", {"name": "squared SVM"}),
)


def covariance_rows(X, indices):
    """Cov(s,x) = sum_ij (s_i-s_j)(x_i-x_j) / (2*m*m).

    This small-problem oracle never uses the implementation's mean-centering
    routine, generated constraint matrices or loss converters.
    """
    differences = X[:, None, :] - X[None, :, :]
    return np.einsum("ijq,ijd->qd", differences[:, :, indices], differences) / (2 * len(X) ** 2)


def check_case(index, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    variant = index // len(ESTIMATORS)
    estimator = ESTIMATORS[index % len(ESTIMATORS)]
    classification = index % 6 >= 4
    wrapper = index % 6 >= 2
    intercept = bool(wrapper and variant % 2)
    scale = (0.3, 1.0, 3.0)[(variant // 2) % 3]
    ratio = (0.2, 0.7)[(variant // 3) % 2] if index % 2 else 0.0
    classes = 3 + (variant // 4) % 3 if classification else 0
    n = classes * 5 if classification else int(rng.integers(5, 25))
    d = int(rng.integers(2, 5))
    X = rng.normal(size=(n, d)) + rng.uniform(-4, 4, d)
    if variant % 3 == 0:
        X[:, 0] = rng.integers(0, 2, n)
    if variant % 5 == 0:
        X[:, -1] = 1.0
    indices = [0, d - 1] if variant % 2 else [0]
    bounds = rng.uniform(0.01, 0.15, len(indices))
    covariance_case = ("constant_sensitive", "constant_other", "zero_bounds", "shifted", "ordinary")[variant % 5]
    if covariance_case == "constant_sensitive":
        X[:, 0] = 0.1
        bounds[0] = 0.0
    elif covariance_case == "constant_other":
        X[:, -1] = -0.3
        if d - 1 in indices:
            bounds[-1] = 0.0
    elif covariance_case == "zero_bounds":
        bounds[:] = 0.0
    elif covariance_case == "shifted":
        X += 8.0
    weight = rng.uniform(0.1, 2, n)
    if classification:
        y = np.tile(np.arange(classes), 5)
        y = y[rng.permutation(n)]
        for c in range(classes):
            weight[np.flatnonzero(y == c)[0]] = 0.0
        family, loss = CLASS_LOSSES[(variant // 2) % 3]
    else:
        y = rng.normal(size=n)
        weight[::5] = 0.0
        family, loss = LOSSES[(variant // 2) % 4]
    C, omega = float(10 ** rng.uniform(-2, 0)), rng.uniform(0.2, 2, d)
    options = dict(
        loss=loss,
        C=C,
        tol=1e-10,
        max_iter=100000,
        warm_start=True,
        constraint=[{"name": "fair", "sen_idx": indices, "tol_sen": bounds}],
    )
    if index % 2:
        options.update(l1_ratio=ratio, omega=omega)
    if wrapper:
        options.update(fit_intercept=intercept, intercept_scaling=scale)
    effective = weight.copy()
    strategy = "none"
    if classification:
        strategy = ("ovr", "ovo")[(variant // 2) % 2]
        class_weight = {c: float(rng.uniform(0.2, 2)) for c in range(classes)}
        if variant % 4 == 0:
            class_weight[classes - 1] = 0.0
        options.update(multi_class=strategy, class_weight=class_weight, n_jobs=1)
        effective *= np.array([class_weight[c] for c in y])
    active = effective > 0 if wrapper else np.ones(n, dtype=bool)
    unique = np.unique(y[active]) if classification else None
    if classification:
        keys = list(combinations(unique, 2)) if len(unique) == 2 or strategy == "ovo" else [(c,) for c in unique]
    else:
        keys = [None]
    problems, references = [], []
    for key in keys:
        rows = active & np.isin(y, key) if key is not None and len(key) == 2 else active
        covariance = covariance_rows(X[rows], indices)
        if intercept:
            covariance = np.column_stack((covariance, np.zeros(len(indices))))
        design = np.column_stack((X[rows], np.full(rows.sum(), scale))) if intercept else X[rows]
        target = np.where(y[rows] == key[-1], 1.0, -1.0) if classification else y[rows]
        problem = case_data(
            design,
            target,
            effective[rows],
            np.vstack((-covariance, covariance)),
            np.tile(bounds, 2),
            C,
            ratio,
            np.r_[omega, 1.0] if intercept else omega,
            family,
        )
        problems.append(problem)
        references.append(solve_reference(problem))
    model = estimator(**options)
    original_X = X.copy()
    X.setflags(write=False)
    results = []
    score_format_checks = 0
    for stage in range(3):
        current = clone(model).set_params(warm_start=False) if stage == 2 else model
        if classification and stage == 2:
            current.set_params(n_jobs=2)
        fitted = fit(current, X, y, weight)
        if classification:
            score_format_checks += check_score_formats(fitted, X)
        binaries = fitted._models_ if classification and len(unique) > 2 else [fitted._model_ if wrapper else fitted]
        if len(binaries) != len(problems):
            raise AssertionError("Wrong number of binary fairness problems")
        for problem, binary, reference in zip(problems, binaries, references):
            results.append(check_result(problem, binary, reference))
    np.testing.assert_array_equal(X, original_X)
    return summarize(
        results,
        estimator=estimator.__name__,
        covariance_case=covariance_case,
        strategy=strategy,
        classes=int(len(unique)) if classification else 0,
        family=family,
        references=len(references),
        intercept=intercept,
        sensitive_columns=len(indices),
        score_format_checks=score_format_checks,
    )


def run_suite(cases=600, seed=20260916, case_index=None):
    if cases < 1:
        raise ValueError("cases must be positive")
    rows, start = [], time.perf_counter()
    for index in range(cases) if case_index is None else [case_index]:
        try:
            rows.append(dict(index=index, status="passed", **check_case(index, seed)))
        except Exception as error:
            rows.append(dict(index=index, status="failed", error=repr(error)))
    return dict(
        seed=seed,
        cases=len(rows),
        passed=sum(r["status"] == "passed" for r in rows),
        comparisons=sum(r.get("comparisons", 0) for r in rows),
        references=sum(r.get("references", 0) for r in rows),
        score_format_checks=sum(r.get("score_format_checks", 0) for r in rows),
        elapsed_sec=time.perf_counter() - start,
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--case", type=int)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/correctness/fairness.json"))
    args = parser.parse_args()
    report = run_suite(args.cases, args.seed, args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
    failures = [r for r in report["rows"] if r["status"] == "failed"]
    if failures:
        print(json.dumps(failures[:10], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
