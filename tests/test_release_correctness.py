"""Numerical and API regressions found during the release review."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, minimize
from scipy.special import huber
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.estimator_checks import parametrize_with_checks

from rehline import (
    CQR_Ridge,
    CQR_Ridge_path_sol,
    ReHLine,
    ReHLine_solver,
    plq_ElasticNet_Classifier,
    plq_ElasticNet_Regressor,
    plq_Ridge_Classifier,
    plq_Ridge_Regressor,
    plqERM_ElasticNet,
    plqERM_Ridge,
    plqERM_Ridge_path_sol,
)


@pytest.fixture
def regression():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(60, 3))
    y = X @ np.array([1.0, 2.0, 3.0]) - 2.0
    return X, y


@pytest.mark.parametrize("loss", [{"name": "MSE"}, {"name": "squared hinge"}, {"name": "MAE"}, {"name": "huber"}])
@pytest.mark.parametrize("estimator", [plqERM_Ridge, plq_Ridge_Regressor, plq_ElasticNet_Regressor])
def test_zero_weight_equals_removing_sample(regression, loss, estimator):
    X, y = regression
    weight = np.ones(len(y))
    weight[::3] = 0
    options = dict(loss=loss, C=0.2, max_iter=50000, tol=1e-8)
    weighted = estimator(**options).fit(X, y, sample_weight=weight)
    dropped = estimator(**options).fit(X[weight > 0], y[weight > 0])
    assert np.isfinite(weighted.coef_).all()
    np.testing.assert_allclose(weighted.coef_, dropped.coef_, atol=2e-6)
    assert weighted.converged_


@pytest.mark.parametrize("scale", [0.2, 1.0, 10.0])
@pytest.mark.parametrize("estimator", [plq_Ridge_Regressor, plq_ElasticNet_Regressor])
def test_scaled_intercept_matches_closed_form(regression, scale, estimator):
    X, y = regression
    options = dict(loss={"name": "MSE"}, C=0.1, intercept_scaling=scale, max_iter=100000, tol=1e-9)
    if estimator is plq_ElasticNet_Regressor:
        options["l1_ratio"] = 0
    model = estimator(**options).fit(X, y)
    augmented = np.column_stack((X, np.full(len(y), scale)))
    expected = np.linalg.solve(augmented.T @ augmented + 5 * np.eye(4), augmented.T @ y)
    np.testing.assert_allclose(model.coef_, expected[:-1], atol=1e-7)
    assert model.intercept_ == pytest.approx(scale * expected[-1], abs=1e-7)
    np.testing.assert_allclose(model.predict(X), augmented @ expected, atol=1e-7)


@pytest.mark.parametrize("estimator", [plq_Ridge_Classifier, plq_ElasticNet_Classifier])
@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
def test_multiclass_scaled_intercepts_and_threads(estimator, strategy):
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=3, n_redundant=0, n_classes=3, random_state=10
    )
    options = dict(loss={"name": "svm"}, C=0.1, multi_class=strategy, intercept_scaling=3.0, max_iter=50000, tol=1e-7)
    serial = estimator(**options, n_jobs=1).fit(X, y)
    parallel = estimator(**options, n_jobs=2).fit(X, y)
    np.testing.assert_allclose(serial.coef_, parallel.coef_, atol=1e-12)
    np.testing.assert_array_equal(serial.predict(X), parallel.predict(X))
    for k, submodel in enumerate(parallel._models_):
        assert parallel.intercept_[k] == pytest.approx(submodel.coef_[-1] * 3)


def test_monotonic_constraints_exclude_intercept(regression):
    X, y = regression
    options = dict(loss={"name": "MSE"}, C=0.1, max_iter=50000, tol=1e-8)
    unconstrained = plq_Ridge_Regressor(**options).fit(X, y)
    constrained = plq_Ridge_Regressor(**options, constraint=[{"name": "monotonic"}]).fit(X, y)
    assert np.all(np.diff(unconstrained.coef_) > 0)
    assert unconstrained.intercept_ < 0
    np.testing.assert_allclose(constrained.predict(X), unconstrained.predict(X), atol=1e-6)
    assert constrained.constraint_violation_ <= constrained.tol


def test_custom_constraints_support_features_and_explicit_intercept(regression):
    X, y = regression
    for A, b in [(np.eye(3), np.zeros(3)), (np.array([[0.0, 0.0, 0.0, -1.0]]), np.array([-1.0]))]:
        model = plq_Ridge_Regressor(
            loss={"name": "MSE"},
            C=0.1,
            intercept_scaling=3.0,
            constraint=[{"name": "custom", "A": A, "b": b}],
            max_iter=50000,
            tol=1e-8,
        ).fit(X, y)
        assert model.coef_.shape == (3,)
        beta = np.r_[model.coef_, model.intercept_] if A.shape[1] == 4 else model.coef_
        assert np.min(A @ beta + b) >= -1e-8
        assert model.predict(X).shape == y.shape


@pytest.mark.parametrize(
    "name,value",
    [
        ("C", 0),
        ("C", np.nan),
        ("tol", -1),
        ("max_iter", 1.5),
        ("trace_freq", 0),
        ("l1_ratio", 1),
        ("l1_ratio", -1),
        ("intercept_scaling", 0),
        ("omega", [1.0, -1.0, 1.0]),
    ],
)
def test_set_params_validated_at_fit(regression, name, value):
    model = plq_ElasticNet_Regressor().set_params(**{name: value})
    with pytest.raises(ValueError, match=name):
        model.fit(*regression)


@pytest.mark.parametrize(
    "bad",
    [
        {"U": np.ones((1, 4))},
        {"V": np.ones((2, 5))},
        {"A": np.ones((2, 3)), "b": np.ones(2)},
        {"A": np.eye(2), "b": np.ones(3)},
        {"rho": np.ones(3)},
        {"U": np.full((1, 5), np.nan)},
        {"Lambda": np.ones((2, 5))},
        {"A": np.zeros((1, 2)), "b": np.array([-1.0])},
    ],
)
def test_solver_rejects_malformed_inputs(bad):
    params = dict(X=np.ones((5, 2)), U=np.ones((1, 5)), V=np.ones((1, 5)), verbose=0)
    params.update(bad)
    with pytest.raises(ValueError):
        ReHLine_solver(**params)


@pytest.mark.parametrize("shrink", [0, 1])
def test_zero_rows_and_constant_relu_terms(shrink):
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = ReHLine_solver(
        X, -np.ones((1, 3)), np.ones((1, 3)), A=np.zeros((1, 2)), b=np.zeros(1), shrink=shrink, verbose=0
    )
    np.testing.assert_allclose(result.beta, [1.0, 1.0], atol=1e-8)
    assert result.converged
    assert result.objective == pytest.approx(2.0)
    assert result.dual_gap < 1e-8


@pytest.mark.parametrize("verbose", [0, 1])
def test_final_diagnostics_match_independent_objective(regression, verbose):
    X, y = regression
    model = plq_Ridge_Regressor(loss={"name": "MSE"}, C=0.1, verbose=verbose, max_iter=50000, tol=1e-8).fit(X, y)
    objective = 0.1 * np.sum((model.predict(X) - y) ** 2)
    objective += 0.5 * (np.sum(model.coef_**2) + model.intercept_**2)
    assert model.objective_ == pytest.approx(objective)
    assert model.dual_objective_ <= objective + 1e-8
    assert model.dual_gap_ == pytest.approx(objective - model.dual_objective_, abs=1e-8)
    assert model.kkt_residual_ <= model.tol
    assert model.converged_


def test_nonconvergence_and_infeasible_constraints_are_reported(regression):
    X, y = regression
    model = plq_Ridge_Regressor(loss={"name": "MSE"}, max_iter=1, tol=1e-12)
    with pytest.warns(ConvergenceWarning):
        model.fit(X, y)
    assert not model.converged_
    assert model.n_iter_ == 1
    result = ReHLine_solver(
        np.ones((3, 1)),
        np.ones((1, 3)),
        np.zeros((1, 3)),
        A=np.array([[1.0], [-1.0]]),
        b=np.array([-1.0, 0.0]),
        max_iter=100,
        verbose=0,
    )
    assert not result.converged
    assert result.constraint_violation > 0
    assert np.isinf(result.dual_gap)


def test_path_objective_and_cqr_snapshots(regression):
    X, y = regression
    Cs, _, objectives, _, coefs = plqERM_Ridge_path_sol(
        X, y, loss={"name": "MAE"}, Cs=[0.1, 2.0], max_iter=50000, tol=1e-8, return_time=False
    )
    expected = Cs * np.abs(X @ coefs - y[:, None]).sum(axis=0) + 0.5 * (coefs**2).sum(axis=0)
    np.testing.assert_allclose(objectives, expected)
    Cs, models, coefs, intercepts = CQR_Ridge_path_sol(
        X, y, quantiles=[0.25, 0.75], Cs=[0.1, 2.0], max_iter=50000, tol=1e-7, warm_start=True, return_time=False
    )
    assert models[0] is not models[1]
    assert models[0].C == Cs[0]
    for k, model in enumerate(models):
        np.testing.assert_allclose(model.predict(X), X @ coefs[k].T + intercepts[k])


def test_parallel_native_solves_have_independent_state(regression):
    X, y = regression

    def solve(C):
        return plqERM_Ridge(loss={"name": "MAE"}, C=C, max_iter=50000, tol=1e-8).fit(X, y).coef_

    Cs = [0.1, 0.3, 1.0, 2.0]
    expected = [solve(C) for C in Cs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(solve, Cs))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("shrink", [0, 1])
@pytest.mark.parametrize("penalty", [False, True])
def test_mixed_plq_and_constraints_against_independent_optimizer(shrink, penalty):
    """Epigraph QP/Huber reference, independent of the ReLU/ReHU dual solver."""
    rng = np.random.default_rng(30)
    n, d, C = 15, 3, 0.3
    X, y = rng.normal(size=(n, d)), rng.normal(size=n)
    rho = np.array([0.15, 0.0, 0.4]) if penalty else np.zeros(d)
    A = np.vstack((np.eye(d), [1.0, -1.0, 0.0]))
    b = np.full(len(A), 0.2)
    U = np.array([np.ones(n), -np.ones(n)])
    V = np.array([-y, y])
    result = ReHLine_solver(
        X,
        C * U,
        C * V,
        S=np.sqrt(C) * U,
        T=np.sqrt(C) * V,
        Tau=np.full((2, n), np.sqrt(C) * 0.7),
        A=A,
        b=b,
        rho=rho if penalty else None,
        shrink=shrink,
        verbose=0,
        max_iter=100000,
        tol=1e-9,
    )
    # z >= |X beta - y| and q >= |beta| linearize both absolute values.
    M = np.block(
        [
            [X, np.eye(n), np.zeros((n, d))],
            [-X, np.eye(n), np.zeros((n, d))],
            [np.eye(d), np.zeros((d, n)), np.eye(d)],
            [-np.eye(d), np.zeros((d, n)), np.eye(d)],
            [A, np.zeros((len(A), n)), np.zeros((len(A), d))],
        ]
    )
    lower = np.r_[y, -y, np.zeros(2 * d), -b]

    def objective(v):
        beta, z, q = v[:d], v[d : d + n], v[d + n :]
        return 0.5 * beta @ beta + C * (z.sum() + huber(0.7, X @ beta - y).sum()) + rho @ q

    def gradient(v):
        return np.r_[v[:d] + C * X.T @ np.clip(X @ v[:d] - y, -0.7, 0.7), np.full(n, C), rho]

    reference = minimize(
        objective,
        np.r_[np.zeros(d), np.abs(y), np.zeros(d)],
        jac=gradient,
        constraints=LinearConstraint(M, lower, np.inf),
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": 1000},
    )
    assert reference.success, reference.message
    assert result.converged
    np.testing.assert_allclose(result.beta, reference.x[:d], atol=2e-6)
    assert result.objective == pytest.approx(reference.fun, abs=1e-8)
    assert result.constraint_violation <= 1e-9
    assert result.dual_gap < 1e-7


def test_native_call_releases_gil():
    from rehline import rehline_internal, rehline_result

    rng = np.random.default_rng(33)
    X = rng.normal(size=(4000, 10))
    U = rng.choice([-1.0, 1.0], size=(1, len(X)))
    V = np.ones_like(U)
    empty = np.empty((0, len(X)))
    started = threading.Event()
    ran_at = []

    def worker():
        started.wait()
        time.sleep(0.02)
        ran_at.append(time.perf_counter())

    thread = threading.Thread(target=worker)
    thread.start()
    start = time.perf_counter()
    started.set()
    try:
        rehline_internal(
            rehline_result(),
            X,
            np.empty((0, 10)),
            np.empty(0),
            np.empty(0),
            U,
            V,
            empty,
            empty,
            empty,
            5000,
            1e-15,
            0,
            0,
            100,
        )
        end = time.perf_counter()
    finally:
        thread.join()
    if end - start < 0.04:
        pytest.skip("Native solve completed before the background scheduling window")
    assert ran_at[0] < end - 0.005, "Background Python thread was blocked throughout the native solve"


def test_constructor_and_fit_do_not_mutate_parameters(regression):
    omega = np.array([1.0, 2.0, 3.0])
    model = plq_ElasticNet_Regressor(omega=omega)
    copy = clone(model)
    model.fit(*regression)
    assert model.omega is omega
    np.testing.assert_array_equal(model.omega, copy.omega)
    assert model.loss is None


def test_warm_start_after_loss_dictionary_changes(regression):
    loss = {"name": "MSE"}
    model = plq_Ridge_Regressor(loss=loss, warm_start=True, max_iter=50000, tol=1e-8)
    model.fit(*regression)
    assert model._model_.loss is not loss
    loss["name"] = "MAE"
    model.fit(*regression)
    reference = plq_Ridge_Regressor(loss=loss, max_iter=50000, tol=1e-8).fit(*regression)
    np.testing.assert_allclose(model.predict(regression[0]), reference.predict(regression[0]), atol=1e-7)


@pytest.mark.parametrize("first,second", [(0.0, 0.5), (0.5, 0.0)])
@pytest.mark.parametrize("estimator", [plq_ElasticNet_Regressor, plqERM_ElasticNet])
def test_warm_start_when_enabling_or_disabling_l1(regression, first, second, estimator):
    model = estimator(loss={"name": "MSE"}, l1_ratio=first, C=0.1, warm_start=True, max_iter=50000, tol=1e-8)
    model.fit(*regression)
    model.set_params(l1_ratio=second).fit(*regression)
    reference = clone(model).set_params(warm_start=False).fit(*regression)
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-7)


@pytest.mark.parametrize("estimator", [plqERM_Ridge, plqERM_ElasticNet])
@pytest.mark.parametrize("change", ["samples", "loss", "constraints"])
def test_low_level_warm_start_after_problem_changes(regression, estimator, change):
    X, y = regression
    model = estimator(loss={"name": "MSE"}, C=0.1, warm_start=True, max_iter=50000, tol=1e-8).fit(X, y)
    if change == "samples":
        X, y = X[:30], y[:30]
    elif change == "loss":
        model.set_params(loss={"name": "MAE"})
    else:
        model.set_params(constraint=[{"name": "custom", "A": -np.eye(3), "b": np.full(3, 0.5)}])
    model.fit(X, y)
    reference = clone(model).set_params(warm_start=False).fit(X, y)
    assert model.converged_
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-7)


def test_raw_warm_start_after_sample_count_changes(regression):
    X, y = regression
    model = ReHLine(U=np.ones((1, len(y))), V=y[None, :], C=0.1, warm_start=True, max_iter=50000, tol=1e-8)
    model.fit(X)
    model._U, model._V = model._U[:, :30], model._V[:, :30]
    model.fit(X[:30])
    reference = ReHLine(U=model._U, V=model._V, C=model.C, max_iter=model.max_iter, tol=model.tol).fit(X[:30])
    assert model.converged_
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-7)


@pytest.mark.parametrize("estimator", [plqERM_Ridge, plqERM_ElasticNet, CQR_Ridge])
def test_compatible_refit_reuses_warm_start(regression, estimator):
    options = {"quantiles": [0.25, 0.75]} if estimator is CQR_Ridge else {"loss": {"name": "MSE"}}
    model = estimator(**options, C=0.1, warm_start=True, max_iter=50000, tol=1e-8).fit(*regression)
    n_iter, coef = model.n_iter_, model.coef_.copy()
    model.fit(*regression)
    assert model.converged_
    assert model.n_iter_ < n_iter
    np.testing.assert_allclose(model.coef_, coef, atol=1e-7)


def test_cqr_scalar_weight_and_changed_quantiles(regression):
    X, y = regression
    model = CQR_Ridge(quantiles=[0.25, 0.75], C=0.1, warm_start=True, max_iter=50000, tol=1e-8)
    model.fit(X, y, sample_weight=2.0)
    reference = clone(model).set_params(warm_start=False).fit(X, y, sample_weight=np.full(len(y), 2.0))
    np.testing.assert_allclose(model.predict(X), reference.predict(X), atol=1e-7)
    model.set_params(quantiles=[0.5]).fit(X, y)
    reference = clone(model).set_params(warm_start=False).fit(X, y)
    assert model.converged_
    np.testing.assert_allclose(model.predict(X), reference.predict(X), atol=1e-7)


@pytest.mark.parametrize("estimator", [plq_Ridge_Classifier, plq_ElasticNet_Classifier])
@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
@pytest.mark.parametrize("remaining_classes", [2, 3])
def test_zero_class_weights_equal_removing_samples(regression, estimator, strategy, remaining_classes):
    X, _ = regression
    y = np.arange(len(X)) % (remaining_classes + 2)
    options = dict(loss={"name": "svm"}, multi_class=strategy, C=0.1, max_iter=50000, tol=1e-8)
    weighted = estimator(**options, class_weight={0: 0, 1: 0}).fit(X, y)
    active = y >= 2
    reference = estimator(**options).fit(X[active], y[active])
    np.testing.assert_array_equal(weighted.classes_, reference.classes_)
    np.testing.assert_allclose(weighted.decision_function(X), reference.decision_function(X), atol=1e-7)
    np.testing.assert_array_equal(weighted.predict(X), reference.predict(X))


@pytest.mark.parametrize("class_weight", [{0: 0, 1: 0, 2: 0}, {0: 1, 1: 0, 2: 0}])
def test_zero_class_weights_require_two_active_classes(regression, class_weight):
    X, _ = regression
    y = np.arange(len(X)) % 3
    model = plq_Ridge_Classifier(loss={"name": "svm"}, multi_class="ovo", class_weight=class_weight)
    with pytest.raises(ValueError, match="positive weight"):
        model.fit(X, y)


@pytest.mark.parametrize("shrink", [0, 1])
@pytest.mark.parametrize("l1_ratio", [0.0, 0.5])
def test_correlated_quadratic_matches_exact_active_set_solution(shrink, l1_ratio):
    """Enumerate the 27 primal sign patterns independently of the dual solver."""
    rng = np.random.default_rng(0)
    X = 100 + rng.normal(size=(80, 2))
    y = rng.normal(size=len(X))
    augmented = np.column_stack((X, np.ones(len(X))))
    Q = 2 * augmented.T @ augmented + (1 - l1_ratio) * np.eye(3)
    target = 2 * augmented.T @ y
    best = None
    for pattern in product((-1, 0, 1), repeat=3):
        signs = np.array(pattern)
        active = signs != 0
        beta = np.zeros(3)
        beta[active] = np.linalg.solve(Q[np.ix_(active, active)], target[active] - l1_ratio * signs[active])
        if np.any(beta[active] * signs[active] < 0):
            continue
        gradient = Q @ beta - target
        if np.any(np.abs(gradient[~active]) > l1_ratio + 1e-9):
            continue
        best = beta
        break
    assert best is not None
    model = plq_ElasticNet_Regressor(
        loss={"name": "MSE"},
        l1_ratio=l1_ratio,
        shrink=shrink,
        max_iter=200,
        tol=1e-8,
        verbose=1,
        trace_freq=10,
    ).fit(X, y)
    assert model.converged_
    np.testing.assert_allclose(np.r_[model.coef_, model.intercept_], best, atol=2e-7)
    assert np.all(np.diff(model.dual_obj_) <= 1e-9)
    assert model.dual_gap_ < 1e-7


@parametrize_with_checks(
    [
        plq_Ridge_Regressor(loss={"name": "MSE"}, max_iter=50000, tol=1e-10),
        plq_ElasticNet_Regressor(loss={"name": "MSE"}, max_iter=50000, tol=1e-10),
        plq_Ridge_Classifier(loss={"name": "svm"}, max_iter=50000, tol=1e-10),
        plq_ElasticNet_Classifier(loss={"name": "svm"}, max_iter=50000, tol=1e-10),
    ],
)
def test_sklearn_estimator_contract(estimator, check):
    check(estimator)
