"""Finite inputs that overflow arithmetic must never yield false certificates."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from rehline import (
    CQR_Ridge,
    ReHLine,
    ReHLine_solver,
    plq_ElasticNet_Classifier,
    plq_ElasticNet_Regressor,
    plq_Ridge_Classifier,
    plq_Ridge_Regressor,
    plqERM_ElasticNet,
    plqERM_Ridge,
    plqMF_Ridge,
)
from rehline._internal import rehline_cqr_internal, rehline_internal, rehline_result


@pytest.mark.parametrize("shrink", [0, 1])
@pytest.mark.parametrize("quantiles", [0, 2])
@pytest.mark.parametrize("kind", ["constant_relu", "constant_rehu", "nonfinite_iterate"])
def test_native_and_python_entry_reject_nonfinite_computations(kind, quantiles, shrink):
    n = 2 * max(quantiles, 1)
    X = np.zeros((2, 1))
    U = V = S = T = Tau = np.empty((0, n))
    if kind == "constant_relu":
        U, V = np.zeros((1, n)), np.full((1, n), 1e308)
    elif kind == "constant_rehu":
        S, T, Tau = np.zeros((1, n)), np.full((1, n), 1e155), np.full((1, n), np.inf)
    else:
        X[:] = 1e308
        S, T, Tau = np.ones((1, n)), np.full((1, n), 1e308), np.full((1, n), np.inf)
    with pytest.raises(OverflowError, match="ReHLine numerical"):
        ReHLine_solver(X, U, V, S=S, T=T, Tau=Tau, _quantile_count=quantiles, max_iter=3, shrink=shrink, verbose=0)
    # Direct pybind callers get the same guard, including the implicit CQR entry.
    native = rehline_cqr_internal if quantiles else rehline_internal
    extra = (quantiles,) if quantiles else ()
    with pytest.raises(OverflowError, match="floating-point range"):
        native(
            rehline_result(),
            X,
            np.empty((0, 1 + quantiles)),
            np.empty(0),
            np.empty(0),
            U,
            V,
            S,
            T,
            Tau,
            *extra,
            3,
            1e-10,
            shrink,
            0,
            100,
        )


@pytest.mark.parametrize("estimator", [plqERM_Ridge, plqERM_ElasticNet, plq_Ridge_Regressor, plq_ElasticNet_Regressor])
@pytest.mark.parametrize("warm_start", [False, True])
def test_overflow_refit_rolls_back_and_first_fit_remains_unfitted(estimator, warm_start):
    options = dict(loss={"name": "MSE"}, C=0.1, warm_start=warm_start, tol=1e-10, max_iter=100000)
    if estimator in (plq_Ridge_Regressor, plq_ElasticNet_Regressor):
        options["fit_intercept"] = False
    X, y = np.zeros((4, 2)), np.arange(4.0)
    model = estimator(**options).fit(X, y)
    before = pickle.dumps(vars(model))
    with pytest.raises(OverflowError):
        model.fit(X, np.full(4, 1e155))
    assert pickle.dumps(vars(model)) == before
    first = clone(model)
    with pytest.raises(OverflowError):
        first.fit(X, np.full(4, 1e155))
    with pytest.raises(NotFittedError):
        check_is_fitted(first)
    model.fit(X, y)
    assert model.objective_ == pytest.approx(clone(model).fit(X, y).objective_, rel=1e-12)


@pytest.mark.parametrize("kind", ["raw", "cqr"])
def test_overflow_rolls_back_other_convex_estimators(kind):
    X, y = np.zeros((4, 2)), np.arange(4.0)
    model = ReHLine(U=np.zeros((1, 4)), V=np.ones((1, 4))) if kind == "raw" else CQR_Ridge([0.2, 0.8])
    model.fit(X) if kind == "raw" else model.fit(X, y)
    before = pickle.dumps(vars(model))
    # Weighting finite losses overflows their sum, without altering parameters.
    with pytest.raises(OverflowError):
        if kind == "raw":
            model.fit(X, sample_weight=1e308)
        else:
            model.fit(X, np.full(4, 1e308))
    assert pickle.dumps(vars(model)) == before


@pytest.mark.parametrize("estimator", [plq_Ridge_Classifier, plq_ElasticNet_Classifier])
def test_overflow_classification_weights_preserve_the_model(estimator):
    X, y = np.zeros((4, 2)), np.array([0, 1, 0, 1])
    options = dict(loss={"name": "svm"}, fit_intercept=False)
    if estimator is plq_ElasticNet_Classifier:
        options["l1_ratio"] = 0.2
    model = estimator(**options).fit(X, y)
    before = pickle.dumps(vars(model))
    with pytest.raises(OverflowError):
        model.fit(X, y, sample_weight=4e307 if estimator is plq_ElasticNet_Classifier else 1e308)
    assert pickle.dumps(vars(model)) == before


def test_overflow_mf_block_preserves_factors_and_history():
    X, y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.arange(4.0)
    model = plqMF_Ridge(2, 2, loss={"name": "MSE"}, rank=1, C=0.1, random_state=42, max_iter=100000, tol=1e-10).fit(
        X, y
    )
    before = pickle.dumps(vars(model))
    with pytest.raises(OverflowError):
        model.fit(X, np.full(4, 1e155))
    assert pickle.dumps(vars(model)) == before


@pytest.mark.parametrize("scale", [1e-150, 1e-50, 1.0, 1e50, 1e150])
@pytest.mark.parametrize("shrink", [0, 1])
def test_representable_rescaled_quadratic_preserves_analytic_objective(scale, shrink):
    # In every representation: 0.5*(beta-1)^2 + 0.5*beta^2; beta*=0.5.
    problem = dict(
        X=np.array([[scale]]),
        U=None,
        V=None,
        S=np.array([[1 / scale], [-1 / scale]]),
        T=np.array([[-1.0], [1.0]]),
        Tau=np.full((2, 1), np.inf),
        tol=1e-10,
        max_iter=100000,
        shrink=shrink,
        verbose=0,
    )
    result = ReHLine_solver(**problem)
    for _ in range(2):
        assert result.converged
        assert result.kkt_residual <= 1e-10
        np.testing.assert_allclose(result.beta, [0.5], rtol=1e-10, atol=1e-12)
        actual = 0.5 * (result.beta[0] - 1) ** 2 + 0.5 * result.beta[0] ** 2
        assert actual == pytest.approx(0.25, rel=1e-12)
        assert result.objective == pytest.approx(actual, rel=1e-12)
        assert result.dual_objective == pytest.approx(actual, rel=1e-12)
        result = ReHLine_solver(**problem, Gamma=result.Gamma)


def test_large_finite_objective_and_infeasible_gap_keep_their_meaning():
    result = ReHLine_solver(X=np.zeros((2, 1)), U=np.zeros((1, 2)), V=np.full((1, 2), 1e300), verbose=0)
    assert result.converged
    assert result.objective == result.dual_objective == 2e300
    assert result.dual_gap == 0
    result = ReHLine_solver(
        X=np.ones((1, 1)), U=None, V=None, A=np.array([[1.0], [-1.0]]), b=np.array([-1.0, 0.0]), max_iter=10, verbose=0
    )
    assert not result.converged
    assert result.constraint_violation > 0
    assert np.isinf(result.dual_gap)
    assert np.isfinite(result.objective)
