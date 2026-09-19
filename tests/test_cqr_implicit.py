"""Validate the compact CQR representation at the solver and estimator boundary."""

import copy
import tracemalloc

import numpy as np
import pytest

from benchmarks.cqr_correctness import check_case, dense_problem, make_case, run_suite
from benchmarks.objectives import audit_solver_result
from rehline import CQR_Ridge, ReHLine_solver


def test_cqr_against_dense_and_cvxpy():
    pytest.importorskip("cvxpy")
    report = run_suite(cases=70)
    assert report["passed"] == 70, [r for r in report["rows"] if r["status"] != "passed"]
    assert report["comparisons"] == 420


@pytest.mark.parametrize("index", [97, 468])
def test_cqr_tight_objective_gate_for_small_scale_and_warm_refits(index):
    pytest.importorskip("cvxpy")
    assert check_case(index)["comparisons"] == 6


def test_cqr_reference_gate_rejects_wrong_solution(monkeypatch):
    pytest.importorskip("cvxpy")
    import benchmarks.cqr_correctness as benchmark

    original = benchmark.CQR_Ridge.fit

    def wrong(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        result.coef_ += 1  # Leave diagnostics untouched to check independent evaluation.
        return result

    monkeypatch.setattr(benchmark.CQR_Ridge, "fit", wrong)
    report = run_suite(cases=1)
    assert report["passed"] == 0
    assert "objective" in report["rows"][0]["error"]


@pytest.mark.parametrize("shrink", [0, 1])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
def test_cqr_audit_matches_dense_problem_hash_and_full_bounds(shrink, layout, monkeypatch):
    import rehline._class as module

    case = make_case(19)
    X = case["X"]
    if layout == "fortran":
        X = np.asfortranarray(X)
    elif layout == "strided":
        X = np.repeat(X, 2, axis=1)[:, ::2]
    X.setflags(write=False)
    calls = []
    original = module.ReHLine_solver

    def record(**kwargs):
        result = original(**kwargs)
        calls.append((kwargs, result))
        return result

    monkeypatch.setattr(module, "ReHLine_solver", record)
    CQR_Ridge(case["quantiles"], C=case["C"], shrink=shrink, tol=1e-9, max_iter=100000).fit(
        X, case["y"], sample_weight=case["weight"]
    )
    problem, result = calls[0]
    dense = dict(problem, X=dense_problem(case)["X"])
    dense.pop("_quantile_count")
    old = audit_solver_result(dense, original(**dense))
    new = audit_solver_result(problem, result)
    assert old["problem_sha256"] == new["problem_sha256"]
    for field in ("objective", "dual_lower_bound", "feasible_upper_bound"):
        assert new[field] == pytest.approx(old[field], rel=1e-8, abs=1e-9)
    np.testing.assert_array_equal(X, case["X"])


@pytest.mark.parametrize("change", ["samples", "features", "quantiles", "same_dual_shape", "weights", "C"])
def test_cqr_warm_refit_with_changed_problem_matches_cold(change):
    rng = np.random.default_rng(39)
    X, y = rng.normal(size=(24, 3)), rng.normal(size=24)
    model = CQR_Ridge([0.2, 0.8], C=0.1, warm_start=True, tol=1e-9, max_iter=100000).fit(X, y)
    weight = np.ones(len(y))
    if change == "samples":
        X, y, weight = X[:12], y[:12], weight[:12]
    elif change == "features":
        X = np.column_stack((X, X[:, 0]))
    elif change == "quantiles":
        model.set_params(quantiles=[0.9, 0.1, 0.5, 0.5])
    elif change == "same_dual_shape":
        X, y, weight = X[:12], y[:12], weight[:12]
        model.set_params(quantiles=[0.9, 0.1, 0.5, 0.5])
    elif change == "weights":
        weight[::2] = 0
    else:
        model.set_params(C=0.7)
    model.fit(X, y, sample_weight=weight)
    cold = CQR_Ridge(**dict(model.get_params(), warm_start=False)).fit(X, y, sample_weight=weight)
    assert model.objective_ == pytest.approx(cold.objective_, rel=1e-8, abs=1e-9)
    assert model.dual_objective_ == pytest.approx(cold.dual_objective_, rel=1e-8, abs=1e-9)


def test_cqr_preprocessing_never_expands_the_feature_matrix(monkeypatch):
    import rehline._class as module

    X, y = np.zeros((10000, 40)), np.ones(10000)
    model = CQR_Ridge(np.linspace(0.05, 0.95, 20))
    before = copy.deepcopy(vars(model))

    class Stop(Exception):
        pass

    def inspect(**kw):
        assert kw["X"].shape == X.shape
        assert np.shares_memory(kw["X"], X)
        assert kw["_quantile_count"] == 20
        assert kw["U"].shape == (2, 200000)
        raise Stop()

    monkeypatch.setattr(module, "ReHLine_solver", inspect)
    tracemalloc.start()
    try:
        with pytest.raises(Stop):
            model.fit(X, y)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 30_000_000  # The old expanded X alone would require 96 MB.
    assert vars(model).keys() == before.keys()
    assert model.coef_ is None


@pytest.mark.parametrize("count", [-1, 1.5, True, 2**31, np.iinfo(np.int64).max])
def test_invalid_implicit_dimensions_are_rejected_before_native_code(count):
    with pytest.raises(ValueError, match="composite quantile"):
        ReHLine_solver(X=np.ones((3, 2)), U=None, V=None, _quantile_count=count)
