"""Previously stalled box problems, checked against analytic corner optima."""

from itertools import product

import numpy as np
import pytest
from scipy import sparse

from tests._helpers.core import constraint_violation, make_case, objective, solve_rehline


@pytest.mark.numerical_stress
@pytest.mark.parametrize("index", [1332, 2844])
@pytest.mark.filterwarnings("ignore:ReHLine failed to converge:sklearn.exceptions.ConvergenceWarning")
def test_primal_recovery_meets_requested_tolerance(index, assert_objective_close, record_property):
    case = make_case(index, seed=20260913)
    assert case["family"] == "mse" and case["geometry"] == "box"
    X, y, weight = case["X"], case["y"], case["weight"]
    d = X.shape[1]
    lower, upper = -case["b"][:d], case["b"][d:]
    corners = (np.where(mask, upper, lower) for mask in product((False, True), repeat=d))
    optimum = min(corners, key=lambda beta: objective(case, beta))
    gradient = 2 * case["C"] * (X.T @ (weight * (X @ optimum - y)))
    gradient += (1 - case["l1_ratio"]) * optimum + case["l1_ratio"] * case["omega"] * np.sign(optimum)
    # All feasible coordinate directions increase this convex objective,
    # proving this corner is optimal independently of either solver.
    directions = np.where(optimum == lower, 1, -1)
    assert np.min(directions * gradient) > 100
    expected = objective(case, optimum)
    for solution in solve_rehline(case, tol=1e-8, max_iter=1_000_000):
        label = f"shrink_{solution['shrink']}_warm_{solution['warm']}"
        record_property(label + "_converged", solution["converged"])
        record_property(label + "_kkt", solution["kkt_residual"])
        assert constraint_violation(case, solution["beta"]) <= 1e-8
        actual = objective(case, solution["beta"])
        assert_objective_close(actual, expected)
        # The same iterate's objective must still be evaluated accurately.
        np.testing.assert_allclose(solution["objective"], actual, rtol=1e-12, atol=1e-7)


@pytest.mark.parametrize("index", [1332, 2844])
@pytest.mark.parametrize("sparse_A", [False, True])
@pytest.mark.filterwarnings("ignore:ReHLine failed to converge:sklearn.exceptions.ConvergenceWarning")
def test_polished_primal_has_independently_verified_stationarity(index, sparse_A, record_property):
    from decimal import Decimal, localcontext

    from rehline import plqERM_ElasticNet

    case = make_case(index, seed=20260913)
    model = plqERM_ElasticNet(
        loss={"name": "MSE"},
        C=case["C"],
        l1_ratio=case["l1_ratio"],
        omega=case["omega"],
        constraint=[{"name": "custom", "A": sparse.csr_matrix(case["A"]) if sparse_A else case["A"], "b": case["b"]}],
        tol=1e-8,
        max_iter=1_000_000,
    ).fit(case["X"], case["y"], sample_weight=case["weight"])
    record_property("converged", model.converged_)
    record_property("kkt", model.kkt_residual_)
    effective_C = case["C"] / (1 - case["l1_ratio"])
    S = model._S * np.sqrt(effective_C * case["weight"])
    rho = case["l1_ratio"] * case["omega"] / (1 - case["l1_ratio"])
    # Decimal evaluates the exact stored binary64 products with enough precision
    # to expose stationarity errors hidden by cancellation in a double sum.
    with localcontext() as context:
        context.prec = 60
        D = Decimal.from_float
        for j in range(case["X"].shape[1]):
            recovered = sum(D(float(a)) * D(float(xi)) for a, xi in zip(case["A"][:, j], model._xi))
            recovered -= sum(
                D(float(case["X"][i, j])) * D(float(S[h, i])) * D(float(model._Gamma[h, i]))
                for i in range(len(case["X"]))
                for h in range(len(S))
            )
            recovered += 2 * D(float(model._mu[j])) - D(float(rho[j]))
            assert abs(recovered - D(float(model.coef_[j]))) <= Decimal("1e-8")
