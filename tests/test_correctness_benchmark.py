import json

import numpy as np
import pytest

from benchmarks import correctness


def test_case_generation_and_replay_are_exact(tmp_path):
    case = correctness.make_case(71)
    path = tmp_path / "failure.npz"
    correctness.save_case(case, path)
    restored = correctness.load_case(path)
    repeated = correctness.make_case(71)
    for name in correctness.ARRAYS:
        np.testing.assert_array_equal(case[name], restored[name])
        np.testing.assert_array_equal(case[name], repeated[name])
    assert restored["family"] == "mixed"
    assert restored["geometry"] == "equality"


def test_cvxpy_reference_matches_closed_form_weighted_ridge():
    pytest.importorskip("cvxpy")
    case = correctness.make_case(0)
    case["l1_ratio"] = 0
    X, w, y = case["X"], case["weight"], case["y"]
    expected = np.linalg.solve(X.T @ (w[:, None] * X) + np.eye(X.shape[1]) / (2 * case["C"]), X.T @ (w * y))
    reference = correctness.solve_reference(case)
    np.testing.assert_allclose(reference["beta"], expected, atol=1e-7)
    assert reference["objective"] == pytest.approx(correctness.objective(case, expected), abs=1e-9)


def test_objective_check_rejects_wrong_solution():
    case = correctness.make_case(0)
    beta = np.zeros(case["X"].shape[1])
    value = correctness.objective(case, beta)
    reference = {"beta": beta, "objective": value}
    result = {"beta": beta + 10, "objective": value, "dual_objective": value, "converged": True}
    checked = correctness.check_solution(case, reference, result)
    assert checked["status"] == "failed"
    assert any("CVXPY" in message for message in checked["errors"])


@pytest.mark.parametrize("family", correctness.FAMILIES)
@pytest.mark.parametrize("constrained", [False, True])
def test_zero_weight_reference_matches_analytic_penalty_minimum(family, constrained):
    pytest.importorskip("cvxpy")
    case = correctness.make_case(0)
    case["family"] = family
    case["weight"][:] = 0
    d = case["X"].shape[1]
    lower = np.linspace(0.1, 0.5, d)
    case["A"] = np.eye(d) if constrained else np.empty((0, d))
    case["b"] = -lower if constrained else np.empty(0)
    expected = lower if constrained else np.zeros(d)
    reference = correctness.solve_reference(case)
    np.testing.assert_allclose(reference["beta"], expected, atol=1e-8)
    assert reference["objective"] == pytest.approx(correctness.objective(case, expected), abs=1e-9)


def test_all_small_problem_families_against_cvxpy(tmp_path):
    pytest.importorskip("cvxpy")
    report = correctness.run_suite(cases=72, output=tmp_path / "report.json", progress=False)
    assert report["status"] == "passed", [row for row in report["rows"] if row["status"] != "passed"]
    assert report["summary"]["comparisons"] == 216
    assert set(report["summary"]["families"]) == set(correctness.FAMILIES)
    assert set(report["summary"]["geometries"]) == set(correctness.GEOMETRIES)
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "passed"


def test_reference_failure_is_saved_and_fails_cli(tmp_path, monkeypatch):
    pytest.importorskip("cvxpy")

    def unavailable(*args, **kwargs):
        raise ValueError("unreliable reference")

    monkeypatch.setattr(correctness, "solve_reference", unavailable)
    output = tmp_path / "failed.json"
    assert correctness.main(["--cases", "1", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["summary"]["passed"] == 0
    assert report["rows"][0]["status"] == "reference_failed"
    assert correctness.load_case(report["rows"][0]["replay"])["index"] == 0


def test_nan_solution_still_writes_a_failed_report(tmp_path, monkeypatch):
    pytest.importorskip("cvxpy")

    def corrupt(case, **kwargs):
        yield {
            "beta": np.full(case["X"].shape[1], np.nan),
            "objective": np.nan,
            "dual_objective": np.nan,
            "converged": False,
        }

    monkeypatch.setattr(correctness, "solve_rehline", corrupt)
    output = tmp_path / "nan.json"
    report = correctness.run_suite(cases=1, output=output, progress=False)
    assert report["status"] == "failed"
    assert json.loads(output.read_text())["rows"][0]["fits"][0]["independent_objective"] is None


@pytest.mark.parametrize("index, seed", [(0, 20260912), (415, 20260917)])
def test_inaccurate_reference_uses_second_solver(monkeypatch, index, seed):
    cp = pytest.importorskip("cvxpy")
    original = cp.Problem.solve
    calls = []

    def inaccurate_first(problem, **kwargs):
        calls.append(kwargs["solver"])
        if len(calls) == 1:
            problem._status = cp.OPTIMAL_INACCURATE
            return None
        return original(problem, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", inaccurate_first)
    case = correctness.make_case(index, seed=seed)
    problem, _ = correctness.reference_problem(case)
    assert problem.is_qp()
    reference = correctness.solve_reference(case)
    assert calls == ["CLARABEL", "OSQP"]
    assert reference["status"] == "optimal"
    assert reference["attempts"][0]["status"] == "optimal_inaccurate"
    assert reference["attempts"][0]["error"]
