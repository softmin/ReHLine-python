"""Constant covariance is exactly zero; small genuine covariance stays active."""

import numpy as np
import pytest
from sklearn.base import clone

from rehline import (
    _make_constraint_rehline_param,
    plq_ElasticNet_Classifier,
    plq_ElasticNet_Regressor,
    plq_Ridge_Classifier,
    plq_Ridge_Regressor,
    plqERM_ElasticNet,
    plqERM_Ridge,
    plqMF_Ridge,
)


def covariance_rows(X, indices):
    diff = X[:, None, :] - X[None, :, :]
    return np.einsum("ijq,ijd->qd", diff[:, :, indices], diff) / (2 * len(X) ** 2)


@pytest.mark.parametrize("constant", [0.1, -0.3, 1e12 + 0.1])
@pytest.mark.parametrize("n", [3, 6, 100])
def test_constant_sensitive_and_other_columns_have_exact_zero_covariance(constant, n):
    X = np.column_stack((np.full(n, constant), np.arange(n) * 0.125, np.full(n, -0.7)))
    X.setflags(write=False)
    constraints = [{"name": "fair", "sen_idx": [0, 1, 2], "tol_sen": [0.0, 0.0, 0.0]}]
    A, b = _make_constraint_rehline_param(constraints, X)
    reference = covariance_rows(X, [0, 1, 2])
    np.testing.assert_array_equal(A[1::2], reference)
    np.testing.assert_array_equal(A[0::2], -reference)
    np.testing.assert_array_equal(A[:, [0, 2]], 0)
    np.testing.assert_array_equal(b, 0)


@pytest.mark.parametrize("scale", [1.0, 1e-12])
def test_nonzero_tiny_covariance_is_not_dropped(scale, assert_objective_close):
    X = np.array([[0.0], [1.0], [2.0]]) * scale
    A, _ = _make_constraint_rehline_param([{"name": "fair", "sen_idx": [0], "tol_sen": 0}], X)
    assert A[1, 0] > 0
    np.testing.assert_allclose(A[1], covariance_rows(X, [0])[0], atol=0, rtol=1e-14)
    # In the original geometry, zero covariance forces beta = 0 even if its
    # generated coefficient is tiny. Explicitly normalize only the reference.
    m = plqERM_Ridge(
        loss={"name": "MSE"}, tol=1e-8, max_iter=100000, constraint=[{"name": "fair", "sen_idx": [0], "tol_sen": 0}]
    ).fit(X, [1.0, 2.0, 3.0])
    assert m.converged_ and m.scaled_constraint_violation_ <= 1e-8
    assert_objective_close(np.square(np.array([1.0, 2.0, 3.0]) - X @ m.coef_).sum() + 0.5 * (m.coef_ @ m.coef_), 14.0)


def test_nearly_constant_and_large_offset_covariance_matches_pairwise_reference():
    values = np.array([0.1, np.nextafter(0.1, 1.0), 0.1, np.nextafter(0.1, 0.0)])
    X = np.column_stack((values, [1e12, 1e12 + 0.25, 1e12 + 0.5, 1e12 + 0.75]))
    A, _ = _make_constraint_rehline_param([{"name": "fair", "sen_idx": [0, 1], "tol_sen": [0, 0]}], X)
    expected = covariance_rows(X, [0, 1])
    assert expected[0, 0] > 0
    np.testing.assert_allclose(A[1::2], expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("estimator", [plqERM_Ridge, plqERM_ElasticNet, plq_Ridge_Regressor, plq_ElasticNet_Regressor])
@pytest.mark.parametrize("shrink", [0, 1])
def test_constant_fairness_regression_has_unconstrained_optimum(estimator, shrink, assert_objective_close):
    X, y = np.full((100, 1), 0.1), np.ones(100)
    options = dict(loss={"name": "MSE"}, shrink=shrink, tol=1e-8, max_iter=100000, warm_start=True)
    if estimator in (plq_Ridge_Regressor, plq_ElasticNet_Regressor):
        options.update(fit_intercept=False)
    model = estimator(**options, constraint=[{"name": "fair", "sen_idx": [0], "tol_sen": 0.0}])
    reference = estimator(**options).fit(X, y)
    for _ in range(2):
        model.fit(X, y)
        assert model.converged_
        ratio = getattr(model, "l1_ratio", 0.0)
        optimum = (20 - ratio) / (3 - ratio)
        value = np.square(1 - 0.1 * optimum) * 100 + 0.5 * (1 - ratio) * optimum**2 + ratio * abs(optimum)
        actual = (
            np.square(y - X @ model.coef_).sum()
            + 0.5 * (1 - ratio) * (model.coef_ @ model.coef_)
            + ratio * abs(model.coef_).sum()
        )
        assert_objective_close(actual, value)
        assert_objective_close(model.objective_, reference.objective_)
        np.testing.assert_allclose(model.objective_ * (1 - ratio), actual, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("estimator", [plq_Ridge_Classifier, plq_ElasticNet_Classifier])
@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
@pytest.mark.parametrize("classes", [2, 4])
@pytest.mark.parametrize("intercept", [False, True])
def test_constant_fairness_with_weighted_binary_and_multiclass(
    estimator, strategy, classes, intercept, assert_objective_close
):
    rng = np.random.default_rng(16)
    X = rng.normal(size=(classes * 6, 3))
    X[:, 0] = 0.1
    y = np.array([f"label-{k}" for k in range(classes)] * 6)
    weight = rng.uniform(0.1, 3, len(y))
    weight[:classes] = 0
    model = estimator(
        loss={"name": "svm"},
        C=0.1,
        class_weight="balanced",
        multi_class=strategy,
        fit_intercept=intercept,
        intercept_scaling=2.0,
        warm_start=True,
        tol=1e-8,
        max_iter=100000,
        n_jobs=2,
    )
    reference = clone(model).fit(X, y, sample_weight=weight)
    model.set_params(constraint=[{"name": "fair", "sen_idx": [0], "tol_sen": 0.0}])
    for _ in range(2):
        model.fit(X, y, sample_weight=weight)
        assert np.all(model.converged_)
        assert_objective_close(model.objective_, reference.objective_)
        assert_objective_close(model.dual_objective_, reference.dual_objective_)
        assert np.max(model.scaled_constraint_violation_) <= 1e-8
        np.testing.assert_array_equal(model.to_inference().predict(X), model.predict(X))


@pytest.mark.parametrize("biased", [False, True])
def test_constant_fairness_mf_block_matches_analytic_ridge_solution(biased, assert_objective_close):
    design = np.full((3, 1), 0.1)
    if biased:
        design = np.column_stack((np.ones(3), design))
    model = plqMF_Ridge(1, 1, loss={"name": "MSE"}, rank=1, biased=biased, tol=1e-8, max_iter=100000)
    constraints = [{"name": "fair", "sen_idx": [-1], "tol_sen": 0.0}]
    y, weight = np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0])
    z, converged = model._solve_block(design, y, weight, np.zeros(3), constraints, 1.0, {})
    expected = np.linalg.solve(
        np.eye(design.shape[1]) + 2 * design.T @ (weight[:, None] * design), 2 * design.T @ (weight * y)
    )
    assert converged
    assert_objective_close(
        0.5 * (z @ z) + weight @ np.square(design @ z - y),
        0.5 * (expected @ expected) + weight @ np.square(design @ expected - y),
    )
