import copy
import json

import numpy as np
import pytest

from benchmarks import BenchmarkTask, DatasetSpec, benchmark_results_to_markdown, run_gridsearch_benchmark
from benchmarks.objectives import audit_solver_result, compare, record_solver_fits, validate_timed_fits
from rehline import ReHLine_solver, plq_Ridge_Regressor


@pytest.fixture
def audit_report():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(24, 3))
    y = X @ np.array([1.0, -2.0, 0.5])
    task = BenchmarkTask(
        "regression",
        "regression",
        plq_Ridge_Regressor(loss={"name": "MSE"}, tol=1e-10),
        {"model__C": [0.1, 1.0]},
        "neg_mean_squared_error",
    )
    rows = run_gridsearch_benchmark(
        [task],
        [DatasetSpec("tiny", "regression", lambda: (X, y))],
        cv=2,
        repeats=2,
        n_jobs=1,
        verify_objective=True,
        return_dataframe=False,
    )
    return {"config": {"task_datasets": {"regression": ["tiny"]}}, "rows": rows}


def test_benchmark_audits_all_candidates_and_timing_repeats(audit_report):
    row = audit_report["rows"][0]
    assert row["n_objective_fits"] == 5
    assert len(row["fits"]) == 10
    assert compare(audit_report, audit_report)["problems_checked"] == 5
    assert "| objective_validation | passed |" in benchmark_results_to_markdown([row])
    assert "| n_objective_fits | 5 |" in benchmark_results_to_markdown([row])
    json.dumps(audit_report, allow_nan=False)


@pytest.mark.parametrize("corruption", ["objective", "problem", "gap", "dual", "convergence", "fold", "row", "timing"])
def test_objective_gate_rejects_corrupt_results(audit_report, corruption):
    changed = copy.deepcopy(audit_report)
    row = changed["rows"][0]
    fit = row["objective_fits"][0]
    if corruption == "objective":
        fit.update(objective=10.0, feasible_upper_bound=10.0, dual_lower_bound=10.0)
    elif corruption == "problem":
        fit["problem_sha256"] = "different inputs"
    elif corruption == "gap":
        fit["dual_lower_bound"] -= 1
    elif corruption == "dual":
        fit["dual_feasible"] = False
    elif corruption == "convergence":
        fit["converged"] = False
    elif corruption == "fold":
        row["objective_fits"].pop()
    elif corruption == "row":
        changed["rows"].clear()
    else:
        row["fits"][7]["objective"] += 1
        with pytest.raises(ValueError, match="Timing fit"):
            validate_timed_fits(row["fits"], row["objective_fits"], repeats=2, expected_per_repeat=5)
        return
    with pytest.raises(ValueError):
        compare(audit_report, changed)


def test_audit_restores_solver_after_failure():
    import rehline._class as models

    original = models.ReHLine_solver
    with pytest.raises(RuntimeError, match="test failure"), record_solver_fits(audit=True):
        raise RuntimeError("test failure")
    assert models.ReHLine_solver is original


def test_objective_verification_rejects_parallel_execution():
    with pytest.raises(ValueError, match="n_jobs=1"):
        run_gridsearch_benchmark(tasks=[], datasets=[], verify_objective=True, n_jobs=2)


def test_independent_objective_catches_wrong_native_value():
    from types import SimpleNamespace

    problem = {
        "X": np.ones((3, 1)),
        "U": np.ones((1, 3)),
        "V": np.ones((1, 3)),
        "S": np.empty((0, 3)),
        "T": np.empty((0, 3)),
        "Tau": np.empty((0, 3)),
        "A": np.empty((0, 1)),
        "b": np.empty(0),
    }
    result = ReHLine_solver(**problem, verbose=0, tol=1e-10)
    bad = SimpleNamespace(beta=result.beta, objective=result.objective + 0.1)
    with pytest.raises(ValueError, match="Independent objective"):
        audit_solver_result(problem, bad)


@pytest.mark.parametrize("rtol,atol", [(float("nan"), 1e-9), (float("inf"), 0), (-1, 0), (0, 0)])
def test_invalid_objective_tolerances_fail(rtol, atol):
    with pytest.raises(ValueError, match="tolerances"):
        compare({}, {}, rtol=rtol, atol=atol)


def test_objective_audit_is_outside_the_timer(monkeypatch):
    from benchmarks import mini_gridsearch, objectives

    clock = [0.0]
    original = objectives.audit_solver_result

    def expensive_audit(*args):
        clock[0] += 100
        return original(*args)

    monkeypatch.setattr(mini_gridsearch.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(objectives, "audit_solver_result", expensive_audit)
    task = BenchmarkTask(
        "tiny", "regression", plq_Ridge_Regressor(tol=1e-10), {"model__C": [0.1]}, "neg_mean_squared_error"
    )
    data = DatasetSpec("tiny", "regression", lambda: (np.ones((10, 1)), np.arange(10)))
    row = run_gridsearch_benchmark([task], [data], cv=2, n_jobs=1, verify_objective=True, return_dataframe=False)[0]
    assert row["elapsed_sec_mean"] == 0
    assert row["objective_audit_sec"] == 300


def test_configured_objective_run_writes_json_sidecar(tmp_path):
    from benchmarks import run_configured_benchmark

    config = {
        "task_datasets": {"ridge_svm": ["breast_cancer"]},
        "cv": 2,
        "n_jobs": 1,
        "C_grid": [0.1],
        "tol": 1e-9,
        "verify_objective": True,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = run_configured_benchmark(path, output=tmp_path / "checked.md")
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["rows"][0]["objective_validation"] == "passed"
    assert compare(report, report)["problems_checked"] == 3
