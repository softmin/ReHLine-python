"""The fairness oracle uses independent pair differences and full objectives."""

import numpy as np
import pytest

from benchmarks.fairness_correctness import covariance_rows, run_suite


def test_pairwise_covariance_oracle():
    X = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(covariance_rows(X, [0]), [[0.25, -0.25]])
    np.testing.assert_array_equal(covariance_rows(X + [100, -10], [0]), [[0.25, -0.25]])


def test_small_fairness_benchmark():
    pytest.importorskip("cvxpy")
    report = run_suite(cases=30)
    assert report["passed"] == 30, report["rows"]
    assert report["comparisons"] >= 90
    assert {row["covariance_case"] for row in report["rows"]} == {
        "constant_sensitive",
        "constant_other",
        "zero_bounds",
        "shifted",
        "ordinary",
    }


def test_objective_oracle_rejects_spurious_constant_covariance(monkeypatch):
    pytest.importorskip("cvxpy")
    from rehline import _class

    original = _class._make_constraint_rehline_param

    def unsafe_covariance(constraint, X, y=None):
        A, b = original(constraint, X, y)
        for spec in constraint:
            if spec["name"] == "fair":
                centered = X - X.mean(axis=0)
                covariance = centered[:, spec["sen_idx"]].T @ centered / len(X)
                A = np.repeat(covariance, 2, axis=0)
                A[::2] *= -1
        return A, b

    correct = run_suite(case_index=0)
    assert correct["passed"] == 1, correct
    monkeypatch.setattr(_class, "_make_constraint_rehline_param", unsafe_covariance)
    broken = run_suite(case_index=0)
    assert broken["passed"] == 0, broken
    assert "Objective mismatch" in broken["rows"][0]["error"]


def test_readme_low_level_example_runs():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "README.md"
    if not path.exists():
        pytest.skip("Source README is absent from the isolated wheel test directory")
    example = path.read_text().split("### Low-Level API for Custom Problems", 1)[1]
    code = example.split("```python", 1)[1].split("```", 1)[0]
    namespace = {}
    exec(compile(code, str(path), "exec"), namespace)
    model = namespace["clf"]
    assert model.converged_
    assert model.constraint_violation_ <= model.tol
