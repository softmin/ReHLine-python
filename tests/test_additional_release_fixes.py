"""Regressions for objective, parameter cloning, MF blocks and CQR prediction."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from rehline import CQR_Ridge, ReHLine, plqERM_ElasticNet, plqERM_Ridge, plqMF_Ridge


def mf_options(**kwargs):
    return dict(
        n_users=2,
        n_items=2,
        loss={"name": "MSE"},
        rank=1,
        C=0.3,
        random_state=42,
        max_iter=20000,
        tol=1e-8,
        max_iter_CD=100,
        tol_CD=1e-8,
        **kwargs,
    )


def mf_penalty(model):
    value = model.rho / model.n_users * np.square(model.P).sum()
    value += (1 - model.rho) / model.n_items * np.square(model.Q).sum()
    if model.biased:
        value += model.rho / model.n_users * np.square(model.bu).sum()
        value += (1 - model.rho) / model.n_items * np.square(model.bi).sum()
    return value


def test_mf_history_and_objective_use_training_weights(fit_mf):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y, weight = np.arange(1.0, 5.0), np.array([100.0, 0.1, 0.2, 1.0])
    model = fit_mf(plqMF_Ridge(**mf_options()), X, y, sample_weight=weight)
    expected_loss = weight @ np.square(y - model.decision_function(X))
    expected = model.C * expected_loss + mf_penalty(model)
    np.testing.assert_allclose(model.history[model.n_iter_], [expected_loss, expected], rtol=1e-12)
    assert model.objective_ == model.history[model.n_iter_, 1]
    np.testing.assert_allclose(model.obj(X, y, sample_weight=weight), [expected_loss, expected])
    # Evaluating a different dataset must not inherit training weights.
    test_X, test_y = X[[0, 2]], y[[0, 2]] + 1
    loss, obj = model.obj(test_X, test_y)
    assert loss == pytest.approx(np.square(test_y - model.decision_function(test_X)).sum())
    assert obj == pytest.approx(model.C * loss + mf_penalty(model))
    np.testing.assert_allclose(model.obj(test_X, test_y, sample_weight=0), [0, mf_penalty(model)])
    assert model.inner_converged_
    assert model.constraint_violation_ == 0


@pytest.mark.parametrize(
    "loss", [{"name": "MSE"}, {"name": "MAE"}, {"name": "QR", "qt": 0.3}, {"name": "huber", "tau": 0.7}]
)
def test_mf_scalar_weights_equal_explicit_weights(loss, fit_mf):
    X, y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.arange(1.0, 5.0)
    options = mf_options()
    options.update(loss=loss, max_iter_CD=5)
    a = fit_mf(plqMF_Ridge(**options), X, y, sample_weight=2.0)
    b = fit_mf(plqMF_Ridge(**options), X, y, sample_weight=np.full(4, 2.0))
    np.testing.assert_allclose(a.P, b.P)
    assert a.objective_ == pytest.approx(b.objective_)
    values = y - a.decision_function(X)
    if loss["name"] == "MSE":
        values = values**2
    elif loss["name"] == "MAE":
        values = abs(values)
    elif loss["name"] == "QR":
        values = np.maximum(0.3 * values, -0.7 * values)
    else:
        values = abs(values)
        values = np.where(values <= 0.7, 0.5 * values**2, 0.7 * (values - 0.35))
    assert a.objective_ == pytest.approx(a.C * 2 * values.sum() + mf_penalty(a))


def test_mf_one_pair_matches_analytic_weighted_optimum():
    model = plqMF_Ridge(
        n_users=1,
        n_items=1,
        loss={"name": "MSE"},
        rank=1,
        biased=False,
        C=0.5,
        random_state=1,
        max_iter_CD=1000,
        tol_CD=1e-13,
        tol=1e-10,
    ).fit([[0, 0]], [4.0], sample_weight=3.0)
    product = 4 - 1 / (2 * 0.5 * 3)
    expected = 0.5 * 3 * (4 - product) ** 2 + product
    assert model.objective_ == pytest.approx(expected, abs=1e-10)
    assert model.converged_


@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("zero_weight", [False, True])
def test_mf_empty_effective_blocks_satisfy_nonzero_constraints(biased, zero_weight):
    d = 1 + biased
    constraints = [{"name": "custom", "A": np.eye(d), "b": -np.ones(d)}]
    options = mf_options(biased=biased, constraint_user=constraints, constraint_item=constraints)
    X = np.array([[0, 0], [1, 1]]) if zero_weight else np.array([[0, 0]])
    y = np.full(len(X), 4.0)
    w = [1.0, 0.0] if zero_weight else [1.0]
    model = plqMF_Ridge(**options).fit(X, y, sample_weight=w)
    np.testing.assert_allclose(model.P[1], 1.0)
    np.testing.assert_allclose(model.Q[1], 1.0)
    if biased:
        assert model.bu[1] == pytest.approx(1.0)
        assert model.bi[1] == pytest.approx(1.0)
    assert model.constraint_violation_ <= model.tol
    assert np.min(model.P) >= 1 - model.tol
    assert np.min(model.Q) >= 1 - model.tol


def test_mf_zero_weight_user_is_a_valid_block():
    X, y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.arange(1.0, 5.0)
    model = plqMF_Ridge(**mf_options()).fit(X, y, sample_weight=[0, 0, 1, 1])
    np.testing.assert_array_equal(model.P[0], 0)
    assert model.bu[0] == 0
    assert np.isfinite(model.objective_)


def test_mf_zero_weight_observations_still_define_fairness_constraint():
    model = plqMF_Ridge(**mf_options(biased=False))
    # Both observations define covariance: Var([1, 10]) = 20.25, including the zero-weight row.
    beta, converged = model._solve_block(
        np.array([[1.0], [10.0]]),
        np.array([10.0, -100.0]),
        np.array([1.0, 0.0]),
        None,
        [{"name": "fair", "sen_idx": [0], "tol_sen": 1.0}],
        1.0,
        {},
    )
    assert converged
    assert beta[0] == pytest.approx(1 / 20.25, abs=1e-8)


def test_mf_empty_fairness_statistics_are_rejected():
    model = plqMF_Ridge(**mf_options(constraint_user=[{"name": "fair", "sen_idx": [0], "tol_sen": 1.0}]))
    with pytest.raises(ValueError, match="Fairness constraints require observations"):
        model.fit([[0, 0]], [1.0])


def test_mf_ids_and_unfitted_validation():
    model = plqMF_Ridge(**mf_options())
    with pytest.raises(NotFittedError):
        model.decision_function([[0, 0]])
    model.fit([[0.0, 0.0], [1.0, 1.0]], [1.0, 2.0])
    assert model.decision_function([[0.0, 0.0]]).shape == (1,)
    for X in ([[0.1, 0]], [[-0.1, 0]], [[-1, 0]], [[0, 2]], [[np.nan, 0]], [[np.inf, 0]]):
        with pytest.raises(ValueError):
            model.decision_function(X)
    with pytest.raises(ValueError):
        model.fit([[0, 0]], [1.0], sample_weight=[0])


def test_mf_iteration_budget_has_explicit_status(fit_mf):
    options = mf_options()
    options.update(max_iter_CD=1, tol_CD=1e-12)
    model = fit_mf(plqMF_Ridge(**options), [[0, 0], [1, 1]], [1.0, 2.0])
    assert model.n_iter_ == 1
    assert model.inner_converged_
    assert not model.converged_


def raw_problem():
    return dict(
        U=-np.ones((1, 2)),
        V=np.ones((1, 2)),
        S=np.ones((1, 2)),
        T=-np.ones((1, 2)),
        Tau=np.full((1, 2), np.inf),
        A=np.ones((1, 1)),
        b=np.array([-0.25]),
        tol=1e-10,
        max_iter=10000,
    )


def test_raw_clone_preserves_problem_and_separates_fitted_state():
    parameters = raw_problem()
    original = ReHLine(**parameters).fit(np.ones((2, 1)))
    copied = clone(original)
    assert copied.coef_ is None
    assert copied._Lambda.size == 0
    for name in ("U", "V", "S", "T", "Tau", "A", "b"):
        assert original.get_params()[name] is parameters[name]
        np.testing.assert_array_equal(copied.get_params()[name], parameters[name])
        assert copied.get_params()[name] is not parameters[name]
    copied.fit(np.ones((2, 1)))
    np.testing.assert_allclose(copied.coef_, original.coef_, atol=1e-12)
    assert copied.objective_ == pytest.approx(original.objective_, abs=1e-12)
    assert copied.constraint_violation_ <= copied.tol
    assert original.coef_[0] == pytest.approx(1.0)


def test_raw_legacy_assignment_and_set_params_work_with_clone():
    model = ReHLine(tol=1e-10)
    model._U = -np.ones((1, 2))
    model._V = np.ones((1, 2))
    copied = clone(model).set_params(A=np.ones((1, 1)), b=np.array([-1.5]))
    copied.fit(np.ones((2, 1)))
    assert copied.coef_[0] == pytest.approx(1.5)
    assert model.A is None
    assert clone(ReHLine()).get_params()["U"] is None


@pytest.mark.parametrize("estimator", [plqERM_Ridge, plqERM_ElasticNet])
def test_erm_clone_does_not_capture_generated_training_matrices(estimator):
    model = estimator(loss={"name": "MSE"}, C=0.1).fit(np.ones((4, 2)), np.arange(4.0))
    assert model._S.size > 0
    for name in ("U", "V", "S", "T", "Tau", "A", "b"):
        assert model.get_params()[name] is None
    copied = clone(model).fit(np.ones((3, 2)), np.arange(3.0))
    assert copied.converged_


@pytest.mark.parametrize("levels", [[0.5], [0.9, 0.1, 0.5]])
def test_cqr_prediction_preserves_quantile_order_and_objective(levels):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(25, 3))
    y = rng.normal(size=25)
    weight = rng.uniform(0.1, 2, 25)
    model = CQR_Ridge(levels, C=0.1, tol=1e-9, max_iter=20000).fit(X, y, sample_weight=weight)
    for data in (X, X[:1], X[::-1]):
        n, d = data.shape
        q = len(levels)
        expanded = np.zeros((n * q, d + q))
        for k in range(q):
            expanded[k * n : (k + 1) * n, :d] = data
            expanded[k * n : (k + 1) * n, d + k] = 1
        reference = (expanded @ np.r_[model.coef_, model.intercept_]).reshape(n, q, order="F")
        np.testing.assert_allclose(model.predict(data), reference, rtol=1e-13, atol=1e-13)
    residual = y[:, None] - model.predict(X)
    quantiles = np.array(levels)
    loss = np.maximum(residual * quantiles, residual * (quantiles - 1))
    expected = 0.1 * np.sum(weight[:, None] * loss) + 0.5 * (
        model.coef_ @ model.coef_ + model.intercept_ @ model.intercept_
    )
    assert model.objective_ == pytest.approx(expected, abs=1e-10)
    with pytest.raises(ValueError, match="3 features"):
        model.predict(X[:, :2])
