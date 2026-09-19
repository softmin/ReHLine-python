"""Equivalent constraints must preserve the optimum and the dual certificate."""

import numpy as np
import pytest

from rehline import ReHLine_solver, plqERM_Ridge
from rehline._internal import rehline_cqr_internal, rehline_internal, rehline_result


@pytest.mark.parametrize("shrink", [0, 1, 2])
@pytest.mark.parametrize("scales", [[1, 1], [1e-10, 1e-10], [1e-200, 1e200], [1e200, 1e-200]])
def test_equivalent_constraints_preserve_analytic_optimum_and_duals(scales, shrink):
    # min ||beta||^2/2, beta[0]>=1, beta[0]/2+beta[1]>=2.
    # Unique optimum [1, 1.5], original multipliers [0.25, 1.5].
    A = np.array([[1.0, 0], [0.5, 1.0]])
    b = np.array([-1.0, -2.0])
    scales = np.array(scales)
    problem = dict(
        X=np.zeros((1, 2)),
        U=None,
        V=None,
        A=A * scales[:, None],
        b=b * scales,
        tol=1e-10,
        max_iter=100000,
        shrink=shrink,
        verbose=0,
    )
    result = ReHLine_solver(**problem)
    for _ in range(2):
        assert result.converged
        np.testing.assert_allclose(result.beta, [1, 1.5], atol=1e-9, rtol=0)
        np.testing.assert_allclose(result.xi * scales, [0.25, 1.5], atol=1e-9, rtol=0)
        np.testing.assert_allclose(problem["A"].T @ result.xi, result.beta, atol=1e-9, rtol=0)
        assert result.objective == pytest.approx(1.625, abs=1e-9)
        assert result.dual_objective == pytest.approx(1.625, abs=1e-9)
        assert result.dual_gap <= 1.625e-10
        assert result.scaled_constraint_violation <= problem["tol"]
        assert result.kkt_residual <= problem["tol"]
        result = ReHLine_solver(**problem, xi=result.xi)


def test_public_solver_does_not_certify_a_large_objective_gap():
    model = plqERM_Ridge(
        loss={"name": "MSE"},
        A=1e-10 * np.array([[1.0, 0], [0.5, 1.0]]),
        b=1e-10 * np.array([-1.0, -2.0]),
        tol=1e-9,
        max_iter=100000,
    )
    model.fit(np.zeros((2, 2)), np.zeros(2))
    assert model.converged_
    assert model.objective_ == pytest.approx(1.625, abs=1e-8)
    assert model.dual_gap_ <= model.tol * max(1, abs(model.objective_), abs(model.dual_objective_))


@pytest.mark.parametrize("shrink", [0, 1])
def test_small_projected_residual_does_not_hide_complementarity_gap(shrink):
    # Almost opposite rows give large duals. The projected residual is tiny
    # even though the feasible primal is far from the analytic optimum 0.5.
    r = ReHLine_solver(
        X=np.zeros((1, 2)),
        U=None,
        V=None,
        A=np.array([[1.0, 0.0], [-1.0, 1e-8]]),
        b=np.array([0.0, -1e-8]),
        xi=np.full(2, 1.1e8),
        max_iter=1,
        tol=1e-8,
        shrink=shrink,
        verbose=0,
    )
    assert r.kkt_residual <= 1e-8
    assert r.scaled_constraint_violation <= 1e-8
    assert r.dual_gap > 0.1
    assert not r.converged


@pytest.mark.parametrize("quantiles", [0, 2])
@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_direct_native_entry_and_implicit_design_scale_constraints(quantiles, scale):
    d = 1 + quantiles
    A = np.eye(d) * scale
    b = -np.arange(1.0, d + 1) * scale
    empty = np.empty((0, max(1, quantiles)))
    result = rehline_result()
    native = rehline_cqr_internal if quantiles else rehline_internal
    extra = (quantiles,) if quantiles else ()
    for _ in range(2):
        native(
            result,
            np.zeros((1, 1)),
            A,
            b,
            np.empty(0),
            empty,
            empty,
            empty,
            empty,
            empty,
            *extra,
            10000,
            1e-10,
            1,
            0,
            100,
        )
        expected = np.arange(1.0, d + 1)
        np.testing.assert_allclose(result.beta, expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(result.xi * scale, expected, atol=1e-12, rtol=0)
        assert result.objective == pytest.approx(expected @ expected / 2, abs=1e-12)
        assert result.converged


def test_zero_rows_and_unrepresentable_scaling_are_distinguished():
    base = dict(X=np.zeros((1, 1)), U=None, V=None, verbose=0)
    for scale in (0, 1e-200, 1e200):
        r = ReHLine_solver(**base, A=np.array([[scale]]), b=np.array([0.0]))
        assert r.converged and r.objective == 0
    with pytest.raises(ValueError, match="zero constraint"):
        ReHLine_solver(**base, A=np.array([[0.0]]), b=np.array([-1.0]))
    with pytest.raises(OverflowError, match="floating-point range"):
        ReHLine_solver(**base, A=np.array([[1e-320]]), b=np.array([-1.0]))
    # The primal is representable but its original-unit multiplier is not.
    with pytest.raises(OverflowError, match="floating-point range"):
        ReHLine_solver(**base, A=np.array([[1e-320]]), b=np.array([-1e-320]))


def test_raw_and_scaled_violation_have_explicit_units():
    # Impossible pair, beta>=1 and beta<=0. Retain the original-unit residual.
    A = np.array([[1e8], [-1e-8]])
    b = np.array([-1e8, 0.0])
    r = ReHLine_solver(X=np.zeros((1, 1)), U=None, V=None, A=A, b=b, max_iter=1, verbose=0)
    assert not r.converged and np.isinf(r.dual_gap)
    assert r.constraint_violation == max(0, -np.min(A @ r.beta + b))
    assert r.scaled_constraint_violation == pytest.approx(max(0, -np.min((A @ r.beta + b) / [1e8, 1e-8])))


@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
@pytest.mark.parametrize("elasticnet", [False, True])
@pytest.mark.parametrize("intercept", [False, True])
def test_multiclass_scaled_constraints_preserve_full_subproblem_objectives(strategy, elasticnet, intercept):
    from itertools import combinations

    from sklearn.base import clone

    from rehline import plq_ElasticNet_Classifier, plq_Ridge_Classifier

    rng = np.random.default_rng(48)
    X, y = rng.normal(size=(24, 3)), np.arange(24) % 3
    weight = np.linspace(0.1, 1.2, 24)
    d = 3 + int(intercept)
    A = np.vstack([np.eye(d), -np.eye(d)])
    b = np.r_[np.full(d, -0.1), np.full(d, 0.5)]
    cls = plq_ElasticNet_Classifier if elasticnet else plq_Ridge_Classifier
    options = dict(
        loss={"name": "svm"},
        A=A,
        b=b,
        multi_class=strategy,
        C=0.1,
        fit_intercept=intercept,
        intercept_scaling=2.0,
        warm_start=True,
        tol=1e-10,
        max_iter=100000,
        class_weight={0: 0.5, 1: 1.0, 2: 2.0},
    )
    if elasticnet:
        options["l1_ratio"] = 0.3
    baseline = cls(**options).fit(X, y, sample_weight=weight)
    keys = list(combinations(range(3), 2)) if strategy == "ovo" else list(range(3))

    def independent_objectives(candidate):
        values = []
        for i, key in enumerate(keys):
            rows = np.isin(y, key) if strategy == "ovo" else np.ones(len(y), dtype=bool)
            target = np.where(y[rows] == (key[1] if strategy == "ovo" else key), 1, -1)
            weights = weight[rows] * np.array([options["class_weight"][label] for label in y[rows]])
            score = X[rows] @ candidate.coef_[i] + candidate.intercept_[i]
            beta = np.r_[candidate.coef_[i], candidate.intercept_[i] / 2] if intercept else candidate.coef_[i]
            ratio = 0.3 if elasticnet else 0.0
            value = 0.1 * (weights @ np.maximum(1 - target * score, 0))
            value += (1 - ratio) * (beta @ beta) / 2 + ratio * abs(beta).sum()
            values.append(value)
        return np.array(values)

    baseline_values = independent_objectives(baseline)
    model = clone(baseline)
    for scales in (np.resize([1e-200, 1e200], len(b)), np.resize([1e100, 1e-100], len(b))):
        model.set_params(A=A * scales[:, None], b=b * scales).fit(X, y, sample_weight=weight)
        for _ in range(2):
            assert model.converged_.all()
            np.testing.assert_allclose(model.objective_, baseline.objective_, rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(model.dual_objective_, baseline.dual_objective_, rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(model.coef_, baseline.coef_, rtol=0, atol=1e-8)
            actual = independent_objectives(model)
            np.testing.assert_allclose(actual, baseline_values, rtol=1e-8, atol=1e-9)
            np.testing.assert_allclose(actual, model.objective_ * (0.7 if elasticnet else 1.0), rtol=1e-8, atol=1e-9)
            snapshot = model.to_inference()
            np.testing.assert_array_equal(snapshot.predict(X), model.predict(X))
            np.testing.assert_array_equal(snapshot.scaled_constraint_violation_, model.scaled_constraint_violation_)
            model.fit(X, y, sample_weight=weight)


@pytest.mark.parametrize("shrink", [0, 1])
def test_row_permutation_duplicates_and_direct_dual_transfer(shrink):
    A, b = np.array([[1.0, 0.0], [0.5, 1.0]]), np.array([-1.0, -2.0])
    options = dict(X=np.zeros((1, 2)), U=None, V=None, shrink=shrink, tol=1e-10, max_iter=100000, verbose=0)
    original = ReHLine_solver(**options, A=A, b=b)
    rows, scales = np.array([1, 0, 1]), np.array([1e-200, 1e200, 1e100])
    # Split the duplicate row's multiplier, then express it in the new units.
    xi = original.xi[rows] * np.array([0.5, 1.0, 0.5]) / scales
    equivalent = ReHLine_solver(**options, A=A[rows] * scales[:, None], b=b[rows] * scales, xi=xi)
    assert equivalent.converged
    assert equivalent.objective == pytest.approx(1.625, abs=1e-9)
    assert equivalent.dual_objective == pytest.approx(1.625, abs=1e-9)


def test_scaled_constraint_benchmark_and_failure_gate(monkeypatch):
    pytest.importorskip("cvxpy")
    import benchmarks.constraint_scaling as module

    report = module.run_suite(cases=12)
    assert report["passed"] == 12, report
    assert report["comparisons"] == 144
    original = module.solve_rehline

    def corrupt(*args, **kwargs):
        for result in original(*args, **kwargs):
            result["objective"] += 1
            yield result

    monkeypatch.setattr(module, "solve_rehline", corrupt)
    assert module.run_suite(cases=1)["passed"] == 0


@pytest.mark.parametrize("warm_start", [False, True])
def test_failed_scaled_refit_preserves_duals_scale_metadata_and_predictions(warm_start):
    import pickle

    model = plqERM_Ridge(
        loss={"name": "MSE"},
        A=np.array([[1e-200]]),
        b=np.array([-1e-200]),
        warm_start=warm_start,
        tol=1e-10,
        max_iter=100000,
    )
    X, y = np.zeros((2, 1)), np.zeros(2)
    model.fit(X, y)
    model.set_params(A=np.array([[1e-320]]), b=np.array([-1.0]))
    before = pickle.dumps(vars(model))
    with pytest.raises(OverflowError, match="floating-point range"):
        model.fit(X, y)
    assert pickle.dumps(vars(model)) == before
    model.set_params(A=np.array([[1e200]]), b=np.array([-1e200])).fit(X, y)
    assert model.converged_ and model.objective_ == pytest.approx(0.5, abs=1e-12)
    assert model._xi[0] == pytest.approx(1e-200, rel=1e-10, abs=0)
