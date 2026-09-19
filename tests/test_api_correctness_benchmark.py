"""Ensure the public-API audit checks independent objective and feasibility."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import api_correctness as benchmark


def test_small_public_api_suite():
    pytest.importorskip("cvxpy")
    result = benchmark.run_suite(cases=16)
    assert result["passed"] == 16, result["rows"]
    assert result["comparisons"] >= 48
    assert sum(row.get("score_format_checks", 0) for row in result["rows"]) > 0


def test_warm_boundary_differences_are_reported():
    pytest.importorskip("cvxpy")
    result = benchmark.run_suite(case_index=27)
    assert result["passed"] == 1, result["rows"]
    assert result["rows"][0]["max_warm_margin_difference"] < 1e-8
    assert "boundary_label_changes" in result["rows"][0]


@pytest.mark.parametrize("error", ["objective", "constraint", "convergence"])
def test_audit_rejects_false_solver_diagnostics(error):
    case = benchmark.case_data(
        np.ones((2, 1)), np.zeros(2), np.ones(2), np.ones((1, 1)), np.array([-1.0]), 1.0, 0.0, np.ones(1), "mse"
    )
    model = SimpleNamespace(
        coef_=np.zeros(1), objective_=0.0, dual_objective_=0.0, converged_=True, kkt_residual_=0.0, tol=1e-10
    )
    if error == "objective":
        model.objective_ = 1.0
    elif error == "convergence":
        model.converged_ = False
    with pytest.raises(AssertionError):
        benchmark.check_result(case, model, {"objective": 0.0})


@pytest.mark.parametrize("error", ["sign", "missing_pair"])
def test_audit_rejects_incorrect_pair_scores(error, monkeypatch):
    pytest.importorskip("cvxpy")
    from rehline._sklearn_mixin import _ReHLineClassifier

    original = _ReHLineClassifier.decision_function

    def incorrect(model, X):
        scores = original(model, X)
        if model.decision_function_shape == "ovo" and len(model.classes_) > 2:
            return -scores if error == "sign" else scores[:, :-1]
        return scores

    monkeypatch.setattr(_ReHLineClassifier, "decision_function", incorrect)
    result = benchmark.run_suite(case_index=131)
    assert result["passed"] == 0
    assert "AssertionError" in result["rows"][0]["error"]


def test_classification_matrix_covers_all_combinations():
    pytest.importorskip("cvxpy")
    result = benchmark.run_suite(cases=2 * len(benchmark.CLASSIFICATION_MATRIX))
    assert result["passed"] == result["cases"], [row for row in result["rows"] if row["status"] == "failed"]
    assert result["classification_cells_covered"] == result["classification_matrix_size"] == 128
    fields = ("classes", "constraint_mode", "intercept", "penalty", "strategy")
    assert {tuple(row[field] for field in fields) for row in result["classification_coverage"]} == set(
        benchmark.CLASSIFICATION_MATRIX
    )
    regression = [row for row in result["rows"] if row.get("kind") == "regression"]
    assert {row["target_dtype"] for row in regression} == {"float64", "float32", "uint8", "int8"}
    assert {row["family"] for row in regression} == {family for family, _ in benchmark.REGRESSION_LOSSES}


def test_objective_oracle_rejects_integer_loss_arithmetic(monkeypatch):
    pytest.importorskip("cvxpy")
    from rehline import _base

    original = _base.numeric_array

    def unsafe_targets(value, name, **kwargs):
        if name == "y":
            return np.asarray(value)  # Reproduce arithmetic before float conversion.
        return original(value, name, **kwargs)

    correct = benchmark.run_suite(case_index=8)
    assert correct["passed"] == 1, correct
    assert correct["rows"][0]["target_dtype"] == "uint8"
    monkeypatch.setattr(_base, "numeric_array", unsafe_targets)
    broken = benchmark.run_suite(case_index=8)
    assert broken["passed"] == 0, broken
    assert "Objective mismatch" in broken["rows"][0]["error"]


@pytest.mark.parametrize("classes,strategy", [(2, "ovr"), (3, "ovr"), (3, "ovo")])
def test_balanced_reference_rejects_the_old_unweighted_class_frequencies(monkeypatch, classes, strategy):
    pytest.importorskip("cvxpy")
    from rehline import _sklearn_mixin

    def old_weights(labels, weights, count):
        return weights * len(labels) / (count * np.bincount(labels)[labels])

    cell = benchmark.CLASSIFICATION_MATRIX.index((classes, 0, True, "ridge", strategy))
    index = 2 * (len(benchmark.CLASSIFICATION_MATRIX) + cell) + 1
    correct = benchmark.run_suite(case_index=index)
    assert correct["passed"] == 1, correct
    monkeypatch.setattr(_sklearn_mixin, "balanced_sample_weights", old_weights)
    result = benchmark.run_suite(case_index=index)
    assert result["passed"] == 0
    assert "Objective mismatch" in result["rows"][0]["error"]
