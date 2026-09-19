"""Small public-API problems checked against independent CVXPY formulations.

Exercises explicit/combined constraints and binary, OvR and OvO classifiers,
including sample/class weights, intercept scaling, warm starts and threads.
Run ``python -m benchmarks.api_correctness --cases 1200``; replay with --case.
"""

import argparse
import json
import time
import warnings
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from benchmarks.correctness import constraint_violation, objective, solve_reference
from benchmarks.estimator_correctness import agree
from rehline import (
    plq_ElasticNet_Classifier,
    plq_ElasticNet_Regressor,
    plq_Ridge_Classifier,
    plq_Ridge_Regressor,
    plqERM_ElasticNet,
    plqERM_Ridge,
)

REGRESSORS = (plqERM_Ridge, plqERM_ElasticNet, plq_Ridge_Regressor, plq_ElasticNet_Regressor)
REGRESSION_LOSSES = (
    ("mse", {"name": "MSE"}),
    ("mae", {"name": "MAE"}),
    ("quantile", {"name": "QR", "qt": 0.3}),
    ("huber", {"name": "huber", "tau": 0.7}),
    ("svr", {"name": "SVR", "epsilon": 0}),
    ("quantile_eps", {"name": "QR_eps", "qt": 0.3, "epsilon": 1}),
)
CLASSIFICATION_LOSSES = (
    ("hinge", {"name": "svm"}),
    ("smooth_hinge", {"name": "sSVM"}),
    ("squared_hinge", {"name": "squared SVM"}),
)

# Exhaust each training/constraint/penalty/intercept cell before repeating it.
# Independent RNG draws vary data, loss and scale within these explicit cells.
CLASSIFICATION_MATRIX = tuple(product(range(2, 6), range(4), (False, True), ("ridge", "elasticnet"), ("ovr", "ovo")))


def constraint_inputs(rng, d, intercept, scale, mode, include_intercept):
    width = d + int(intercept and include_intercept)
    lower = rng.uniform(-0.6, 0.2, width)
    upper = lower + rng.uniform(0.4, 1.4, width)
    A, b = np.vstack((np.eye(width), -np.eye(width))), np.r_[-lower, upper]
    if mode == 0:
        return {}, np.empty((0, d + intercept)), np.empty(0)
    if mode == 1:
        kwargs = {"A": A, "b": b}
    elif mode == 2:
        kwargs = {"constraint": [{"name": "custom", "A": A, "b": b}]}
    else:
        kwargs = {"A": A[:width], "b": b[:width], "constraint": [{"name": "custom", "A": A[width:], "b": b[width:]}]}
    # Form the reference constraints in synthetic-coefficient coordinates without
    # using ReHLine's constraint builder or its generated training matrices.
    reference_A = A.copy()
    if intercept:
        if width == d:
            reference_A = np.column_stack((reference_A, np.zeros(len(A))))
        else:
            reference_A[:, -1] *= scale
    return kwargs, reference_A, b


def case_data(X, y, weight, A, b, C, ratio, omega, family):
    return dict(
        X=X,
        y=y,
        weight=weight,
        A=A,
        b=b,
        C=C,
        l1_ratio=ratio,
        omega=omega,
        family=family,
        qt=0.3,
        tau=0.7,
        epsilon=1 if family == "quantile_eps" else 0,
        **{name: np.empty((0, len(y))) for name in ("U", "V", "S", "T", "Tau")},
    )


def check_result(case, model, reference):
    if not model.converged_ or model.kkt_residual_ > model.tol:
        raise AssertionError("Native solver did not meet its original tolerance")
    beta = model.coef_
    actual = objective(case, beta)
    agree(actual, reference["objective"])
    reported = model.objective_ * (1 - case["l1_ratio"])
    dual = model.dual_objective_ * (1 - case["l1_ratio"])
    agree(actual, reported)
    agree(actual, dual)
    violation = constraint_violation(case, beta)
    if violation > 1.01 * model.tol:
        raise AssertionError(f"Requested constraint violated: {violation}")
    return dict(
        objective_error=abs(actual - reference["objective"]),
        reported_error=abs(actual - reported),
        dual_gap=abs(actual - dual),
        violation=violation,
    )


def fit(model, X, y, weight):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Both A/b and constraint were supplied", category=UserWarning)
        warnings.simplefilter("error", ConvergenceWarning)
        return model.fit(X, y, sample_weight=weight)


def summarize(rows, **metadata):
    return {**metadata, "comparisons": len(rows), **{f"max_{name}": max(row[name] for row in rows) for name in rows[0]}}


def check_score_formats(model, X):
    """Audit public score formats without fitting or changing training state."""
    classes = len(model.classes_)
    original_shape = model.decision_function_shape
    fields = ("coef_", "intercept_", "objective_", "dual_objective_", "n_iter_")
    before = {name: np.array(getattr(model, name), copy=True) for name in fields}
    labels = model.predict(X)
    try:
        model.set_params(decision_function_shape="ovr")
        scores = model.decision_function(X)
        expected_shape = (len(X),) if classes == 2 else (len(X), classes)
        if scores.shape != expected_shape:
            raise AssertionError("Decision scores do not align with classes")
        if classes > 2:
            np.testing.assert_array_equal(labels, model.classes_[scores.argmax(axis=1)])
        if classes == 2 or model.multi_class_ == "ovo":
            raw = model.set_params(decision_function_shape="ovo").decision_function(X)
            margins = X @ before["coef_"].T + before["intercept_"]
            np.testing.assert_array_equal(raw, margins if classes == 2 else -margins)
            np.testing.assert_array_equal(model.predict(X), labels)
            for name, value in before.items():
                np.testing.assert_array_equal(getattr(model, name), value)
            restored = model.set_params(decision_function_shape="ovr").decision_function(X)
            np.testing.assert_array_equal(restored, scores)
            return 1
        return 0
    finally:
        model.set_params(decision_function_shape=original_shape)


def check_inference(model, X, problems, references, intercept_scale):
    """Recompute each compact model's original objective against CVXPY."""
    snapshot = model.to_inference()
    np.testing.assert_array_equal(snapshot.predict(X), model.predict(X))
    if hasattr(model, "classes_"):
        np.testing.assert_array_equal(snapshot.decision_function(X), model.decision_function(X))
        check_score_formats(snapshot, X)
    for name in ("objective_", "dual_objective_", "converged_", "constraint_violation_", "kkt_residual_"):
        np.testing.assert_array_equal(getattr(snapshot, name), getattr(model, name))
    coefs = np.atleast_2d(snapshot.coef_)
    intercepts = np.atleast_1d(snapshot.intercept_)
    for i, (case, reference) in enumerate(zip(problems, references)):
        beta = np.r_[coefs[i], intercepts[i] / intercept_scale] if model.fit_intercept else coefs[i]
        actual = objective(case, beta)
        agree(actual, reference["objective"])
        for name in ("objective_", "dual_objective_"):
            agree(actual, np.atleast_1d(getattr(snapshot, name))[i] * (1 - case["l1_ratio"]))
        if constraint_violation(case, beta) > 1.01 * model.tol:
            raise AssertionError("Compact model violates the original constraints")
    return len(problems)


def check_regression(rng, index):
    variant = index // 2
    estimator = REGRESSORS[variant % 4]
    wrapper = variant % 4 >= 2
    intercept = bool(wrapper and (variant // 4) % 2)
    scale = (0.3, 1.0, 3.0)[(variant // 8) % 3]
    family, loss = REGRESSION_LOSSES[(variant // 3) % len(REGRESSION_LOSSES)]
    n, d = int(rng.integers(5, 20)), int(rng.integers(1, 5))
    X, y, weight = rng.normal(size=(n, d)), rng.normal(size=n), rng.uniform(0.1, 2, n)
    target_dtype = ("float64", "uint8", "int8", "float32")[(variant // 4) % 4]
    if target_dtype == "uint8":
        y = np.clip(np.rint(3 * y + 4), 0, 12).astype(target_dtype)
    elif target_dtype == "int8":
        y = np.clip(np.rint(3 * y), -12, 12).astype(target_dtype)
        if (variant // 16) % 5 == 0:
            y[1] = -128
    else:
        y = y.astype(target_dtype)
    weight[::5] = 0
    design = np.column_stack((X, np.full(n, scale))) if intercept else X
    mode = (variant // 2) % 4
    constraints, A, b = constraint_inputs(rng, d, intercept, scale, mode, bool((variant // 5) % 2))
    C = float(10 ** rng.uniform(-2, 0))
    ratio = (0.0, 0.2, 0.7)[(variant // 4) % 3] if variant % 2 else 0.0
    omega = rng.uniform(0.2, 2, d)
    options = dict(loss=loss, C=C, max_iter=100000, tol=1e-10, warm_start=True, shrink=variant % 2, **constraints)
    if variant % 2:
        options.update(l1_ratio=ratio, omega=omega if ratio else None)
    if wrapper:
        options.update(fit_intercept=intercept, intercept_scaling=scale)
    # The reference uses the original numeric values in floating arithmetic,
    # independently of the estimator's loss conversion.
    problem = case_data(
        design, y.astype(float), weight, A, b, C, ratio, np.r_[omega, 1.0] if intercept else omega, family
    )
    reference = solve_reference(problem)
    model = estimator(**options)
    rows = []
    inference_comparisons = 0
    for stage in range(3):
        fitted = fit(clone(model) if stage == 1 else model, X, y, weight)
        rows.append(check_result(problem, fitted._model_ if wrapper else fitted, reference))
        if wrapper:
            inference_comparisons += check_inference(fitted, X, [problem], [reference], scale)
    return summarize(
        rows,
        kind="regression",
        target_dtype=target_dtype,
        estimator=estimator.__name__,
        family=family,
        constraint_mode=mode,
        references=1,
        inference_comparisons=inference_comparisons,
    )


def check_multiclass(rng, index):
    variant = index // 2
    classes, mode, intercept, penalty, strategy = CLASSIFICATION_MATRIX[variant % len(CLASSIFICATION_MATRIX)]
    labels = np.array([f"class-{17 + 3 * k}" for k in rng.permutation(classes)])
    y = labels[np.tile(np.arange(classes), int(rng.integers(4, 8)))]
    y = y[rng.permutation(len(y))]
    d = int(rng.integers(1, 5))
    X, weight = rng.normal(size=(len(y), d)), rng.uniform(0.1, 2, len(y))
    # Keep at least two positive observations of each class for every task.
    for c in labels:
        weight[np.flatnonzero(y == c)[0]] = 0
    cw_mode = (variant // len(CLASSIFICATION_MATRIX)) % 4
    class_weight = None
    effective = weight.copy()
    if cw_mode == 1:
        class_weight = "balanced"
        for c in labels:
            effective[y == c] *= weight.sum() / (classes * weight[y == c].sum())
    elif cw_mode >= 2:
        class_weight = {c: float(rng.uniform(0.1, 2)) for c in labels}
        if cw_mode == 3 and classes > 2:
            class_weight[labels[-1]] = 0.0
        effective *= np.array([class_weight[c] for c in y])
    scale = float(rng.choice([0.3, 1.0, 3.0]))
    family, loss = CLASSIFICATION_LOSSES[int(rng.integers(len(CLASSIFICATION_LOSSES)))]
    constraints, A, b = constraint_inputs(rng, d, intercept, scale, mode, bool(rng.integers(2)))
    ratio = float(rng.choice([0.2, 0.7])) if penalty == "elasticnet" else 0.0
    C, omega = float(10 ** rng.uniform(-2, 0)), rng.uniform(0.2, 2, d)
    options = dict(
        loss=loss,
        C=C,
        max_iter=100000,
        tol=1e-10,
        warm_start=True,
        fit_intercept=intercept,
        intercept_scaling=scale,
        multi_class=strategy,
        class_weight=class_weight,
        **constraints,
    )
    estimator = plq_ElasticNet_Classifier if ratio else plq_Ridge_Classifier
    if ratio:
        options.update(l1_ratio=ratio, omega=omega)
    active = effective > 0
    unique = np.unique(y[active])
    keys = list(combinations(unique, 2)) if len(unique) == 2 or strategy == "ovo" else [(c,) for c in unique]
    problems, references = [], []
    for key in keys:
        mask = active & np.isin(y, key) if len(key) == 2 else active
        design = np.column_stack((X[mask], np.full(mask.sum(), scale))) if intercept else X[mask]
        problem = case_data(
            design,
            np.where(y[mask] == key[-1], 1.0, -1.0),
            effective[mask],
            A,
            b,
            C,
            ratio,
            np.r_[omega, 1.0] if intercept else omega,
            family,
        )
        problems.append(problem)
        references.append(solve_reference(problem))
    model, rows, cold_iterations, warm_iterations = estimator(**options, n_jobs=1), [], None, None
    cold = None
    boundary_label_changes = 0
    max_warm_margin_difference = 0.0
    score_format_checks = 0
    inference_comparisons = 0
    for stage in range(3):
        fitted = fit(clone(model).set_params(warm_start=False, n_jobs=2) if stage == 2 else model, X, y, weight)
        score_format_checks += check_score_formats(fitted, X)
        inference_comparisons += check_inference(fitted, X, problems, references, scale)
        binaries = [fitted._model_] if len(unique) == 2 else fitted._models_
        if len(binaries) != len(problems):
            raise AssertionError("Wrong number of binary problems")
        for problem, binary, reference in zip(problems, binaries, references):
            rows.append(check_result(problem, binary, reference))
        if stage == 0:
            cold_iterations = np.asarray(fitted.n_iter_).tolist()
            cold = dict(
                coef=fitted.coef_.copy(),
                intercept=np.array(fitted.intercept_),
                objective=np.asarray(fitted.objective_).copy(),
                labels=fitted.predict(X),
                margins=fitted._decision_function(X),
                scores=fitted.decision_function(X),
            )
        elif stage == 1:
            warm_iterations = np.asarray(fitted.n_iter_).tolist()
            margins = fitted._decision_function(X)
            np.testing.assert_allclose(margins, cold["margins"], rtol=1e-8, atol=1e-8)
            max_warm_margin_difference = float(np.max(abs(margins - cold["margins"])))
            # Discrete labels are only numerically determined away from zero
            # pair margins and tied class scores. Report boundary changes rather
            # than changing prediction semantics or relaxing objective checks.
            margin_eps = 1e-8 * (1 + abs(cold["margins"]))
            if len(unique) == 2:
                stable = abs(cold["margins"]) > margin_eps
            else:
                sorted_scores = np.sort(cold["scores"], axis=1)
                score_eps = margin_eps.max(axis=1)
                if strategy == "ovo":
                    score_eps *= len(unique) - 1
                stable = sorted_scores[:, -1] - sorted_scores[:, -2] > 2 * score_eps
                if strategy == "ovo":
                    stable &= np.all(abs(cold["margins"]) > margin_eps, axis=1)
            changed = fitted.predict(X) != cold["labels"]
            if np.any(changed & stable):
                raise AssertionError("Warm start changes a label away from a decision boundary")
            boundary_label_changes = int(np.sum(changed))
        else:
            np.testing.assert_allclose(fitted.objective_, model.objective_, rtol=1e-8, atol=1e-9)
            # Identical cold starts must agree exactly across thread counts,
            # including coefficients, raw margins and labels at decision ties.
            np.testing.assert_array_equal(fitted.coef_, cold["coef"])
            np.testing.assert_array_equal(fitted.intercept_, cold["intercept"])
            np.testing.assert_array_equal(fitted.objective_, cold["objective"])
            np.testing.assert_array_equal(fitted.predict(X), cold["labels"])
        if len(unique) > 2:
            scores = fitted.decision_function(X)
            if scores.shape != (len(X), len(unique)):
                raise AssertionError("Decision scores do not align with classes")
            np.testing.assert_array_equal(fitted.predict(X), unique[scores.argmax(axis=1)])
    return summarize(
        rows,
        kind="classification",
        classes=int(len(unique)),
        strategy=strategy,
        family=family,
        constraint_mode=mode,
        requested_classes=classes,
        intercept=intercept,
        penalty=penalty,
        class_weight_mode=cw_mode,
        references=len(references),
        cold_iterations=cold_iterations,
        warm_iterations=warm_iterations,
        boundary_label_changes=boundary_label_changes,
        max_warm_margin_difference=max_warm_margin_difference,
        score_format_checks=score_format_checks,
        inference_comparisons=inference_comparisons,
    )


def run_suite(cases=1200, seed=20260914, case_index=None):
    if cases < 1:
        raise ValueError("cases must be positive")
    rows, start = [], time.perf_counter()
    for index in range(cases) if case_index is None else [case_index]:
        try:
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            result = (check_multiclass if index % 2 else check_regression)(rng, index)
            rows.append(dict(index=index, status="passed", **result))
        except Exception as error:
            rows.append(dict(index=index, status="failed", error=repr(error)))
    fields = ("classes", "constraint_mode", "intercept", "penalty", "strategy")
    coverage = Counter(tuple(row[field] for field in fields) for row in rows if row.get("kind") == "classification")
    return dict(
        seed=seed,
        cases=len(rows),
        passed=sum(row["status"] == "passed" for row in rows),
        comparisons=sum(row.get("comparisons", 0) for row in rows),
        references=sum(row.get("references", 0) for row in rows),
        score_format_checks=sum(row.get("score_format_checks", 0) for row in rows),
        inference_comparisons=sum(row.get("inference_comparisons", 0) for row in rows),
        classification_matrix_size=len(CLASSIFICATION_MATRIX),
        classification_cells_covered=len(coverage),
        classification_coverage=[dict(zip(fields, key), cases=count) for key, count in sorted(coverage.items())],
        elapsed_sec=time.perf_counter() - start,
        rows=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--case", type=int)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/correctness/api.json"))
    args = parser.parse_args()
    report = run_suite(args.cases, args.seed, args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("rows", "classification_coverage")}))
    failures = [row for row in report["rows"] if row["status"] == "failed"]
    if failures:
        print(json.dumps(failures[:10], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
