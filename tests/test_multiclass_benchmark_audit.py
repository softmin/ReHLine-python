"""Audit estimator fits and every retained class/class-pair by identity."""

import copy
import json
import time

import numpy as np
import pytest

from benchmarks import BenchmarkTask, DatasetSpec, available_datasets, available_tasks, run_gridsearch_benchmark
from benchmarks.fit_records import expected_keys, keyed_records
from benchmarks.objectives import compare, record_solver_fits
from rehline import plq_ElasticNet_Classifier, plq_Ridge_Classifier


def report(strategy="ovo", jobs=2, variable_classes=True):
    X = np.random.default_rng(42).normal(size=(60, 3))
    y = np.tile(["a", "b", "c", "d"], 15)
    weights = [None, {"d": 0}] if variable_classes else [None]
    task = BenchmarkTask(
        "multiclass",
        "classification",
        plq_ElasticNet_Classifier(
            loss={"name": "svm"}, l1_ratio=0.2, multi_class=strategy, n_jobs=jobs, tol=1e-10, max_iter=100000
        ),
        {"model__C": [0.01], "model__class_weight": weights},
        "accuracy",
    )
    rows = run_gridsearch_benchmark(
        [task],
        [DatasetSpec("tiny", "classification", lambda: (X, y))],
        cv=2,
        repeats=2,
        n_jobs=1,
        verify_objective=True,
        return_dataframe=False,
    )
    return {"config": {"task_datasets": {"multiclass": ["tiny"]}}, "rows": rows}


@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
@pytest.mark.parametrize("jobs", [1, 2])
def test_variable_class_counts_have_complete_subproblem_audits(strategy, jobs):
    result = report(strategy, jobs)
    row = result["rows"][0]
    assert row["n_classes"] == 4
    assert row["n_estimator_fits"] == 5
    assert {len(entry["classes"]) for entry in row["objective_manifest"]} == {3, 4}
    expected = expected_keys(row["objective_manifest"], row["n_candidates"], row["cv"])
    assert len(row["objective_fits"]) == len(expected)
    assert row["n_objective_fits"] == len(expected)
    assert len(row["fits"]) == 2 * len(expected)
    assert compare(result, result)["problems_checked"] == len(expected)
    for fit in row["objective_fits"]:
        assert fit["original_objective"] == fit["objective"] * 0.8
    json.dumps(result, allow_nan=False)


def test_thread_completion_order_does_not_change_correspondence(monkeypatch):
    from rehline._sklearn_mixin import _SklearnReHLine

    reference = report(jobs=1, variable_classes=False)
    original = _SklearnReHLine._fit_multiclass_task

    def delayed(model, X, y, weight, key, rows, previous):
        if key == ("a", "b"):
            time.sleep(0.01)
        return original(model, X, y, weight, key, rows, previous)

    monkeypatch.setattr(_SklearnReHLine, "_fit_multiclass_task", delayed)
    threaded = report(jobs=2, variable_classes=False)
    left = reference["rows"][0]["objective_fits"]
    right = threaded["rows"][0]["objective_fits"]
    assert [fit["fit_key"] for fit in left] != [fit["fit_key"] for fit in right]
    assert compare(reference, threaded)["problems_checked"] == 18
    # Audit records may also be persisted in any order.
    threaded["rows"][0]["objective_fits"].reverse()
    assert compare(reference, threaded)["problems_checked"] == 18


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "wrong_pair", "objective", "normalization", "manifest", "timing"]
)
def test_multiclass_audit_rejects_incomplete_or_mismatched_pairs(corruption):
    original = report(variable_classes=False)
    changed = copy.deepcopy(original)
    row = changed["rows"][0]
    fits = row["objective_fits"]
    if corruption == "missing":
        fits.pop()
    elif corruption == "duplicate":
        fits[-1] = fits[0]
    elif corruption == "wrong_pair":
        fits[0]["fit_key"] = '[0,"ovo",["a","unknown"]]'
    elif corruption == "objective":
        fits[0]["objective"] += 1
    elif corruption == "normalization":
        fits[0]["original_objective"] += 1
    elif corruption == "manifest":
        row["objective_manifest"][0]["classes"].pop()
    else:
        row["fits"][0]["fit_key"] = row["fits"][1]["fit_key"]
    with pytest.raises(ValueError):
        compare(original, changed)


def test_missing_native_pair_detected_at_recording(monkeypatch):
    from rehline._sklearn_mixin import _SklearnReHLine

    original = _SklearnReHLine._fit_multiclass_task
    seen = []

    def omitted(model, X, y, weight, key, rows, previous):
        if seen:
            return seen[0]  # Return a different pair's model without solving this pair.
        result = original(model, X, y, weight, key, rows, previous)
        seen.append(result)
        return result

    monkeypatch.setattr(_SklearnReHLine, "_fit_multiclass_task", omitted)
    with pytest.raises(ValueError, match="binary subproblem"):
        report(jobs=1, variable_classes=False)


def test_failed_audit_restores_all_wrapped_methods():
    import rehline._class as native
    from rehline._sklearn_mixin import _SklearnReHLine

    before = (
        _SklearnReHLine.fit,
        _SklearnReHLine._fit_model,
        _SklearnReHLine._fit_multiclass_task,
        native.ReHLine_solver,
    )
    with pytest.raises(RuntimeError, match="failure"):
        with record_solver_fits(cv=2, n_candidates=1):
            raise RuntimeError("failure")
    after = (
        _SklearnReHLine.fit,
        _SklearnReHLine._fit_model,
        _SklearnReHLine._fit_multiclass_task,
        native.ReHLine_solver,
    )
    assert before == after
    assert compare(report(), report())["problems_checked"] > 0


def test_rare_class_changes_native_counts_between_folds():
    X = np.random.default_rng(9).normal(size=(21, 2))
    y = np.r_[np.tile([0, 1], 10), 2]
    task = BenchmarkTask(
        "rare",
        "classification",
        plq_Ridge_Classifier(loss={"name": "svm"}, multi_class="ovo", tol=1e-10, max_iter=100000),
        {"model__C": [0.01]},
        "accuracy",
    )
    with pytest.warns(UserWarning, match="least populated class"):
        row = run_gridsearch_benchmark(
            [task],
            [DatasetSpec("rare", "classification", lambda: (X, y))],
            cv=2,
            n_jobs=1,
            verify_objective=True,
            return_dataframe=False,
        )[0]
    assert row["n_estimator_fits"] == 3
    assert sorted(len(e["classes"]) for e in row["objective_manifest"]) == [2, 3, 3]
    assert row["n_objective_fits"] == 7


def test_new_multiclass_catalog_and_tasks():
    datasets, tasks = available_datasets(), available_tasks()
    for name, count in [
        ("iris", 3),
        ("wine", 3),
        ("digits", 10),
        ("multiclass_4", 4),
        ("multiclass_10", 10),
        ("multiclass_30", 30),
    ]:
        _, y = datasets[name].factory()
        assert len(np.unique(y)) == count
    for penalty in ("ridge", "elasticnet"):
        for strategy in ("ovr", "ovo"):
            assert tasks[f"{penalty}_svm_{strategy}"].estimator.multi_class == strategy


def test_legacy_binary_report_comparison_is_preserved():
    # Simulate a historic binary report without manifests or native identities.
    task = BenchmarkTask(
        "binary",
        "classification",
        plq_Ridge_Classifier(loss={"name": "svm"}, tol=1e-10),
        {"model__C": [0.01]},
        "accuracy",
    )
    X = np.arange(20.0).reshape(10, 2)
    y = np.tile([0, 1], 5)
    rows = run_gridsearch_benchmark(
        [task],
        [DatasetSpec("tiny", "classification", lambda: (X, y))],
        cv=2,
        n_jobs=1,
        verify_objective=True,
        return_dataframe=False,
    )
    current = {"config": {"task_datasets": {"binary": ["tiny"]}}, "rows": rows}
    legacy = copy.deepcopy(current)
    legacy["rows"][0].pop("objective_manifest")
    for name in ("fits", "objective_fits"):
        for fit in legacy["rows"][0][name]:
            for key in list(fit):
                if key in ("fit_key", "objective_scale") or key.startswith("original_"):
                    fit.pop(key)
    assert compare(legacy, current)["problems_checked"] == 3
