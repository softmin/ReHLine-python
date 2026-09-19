"""The outer MF feasibility check must use the same units as its block solves."""

import pickle
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from benchmarks.estimator_correctness import LOSSES, direct_loss
from rehline import plqMF_Ridge


def test_mf_equivalent_constraints_preserve_objective_and_stopping():
    X, y = np.array([(u, i) for u in range(2) for i in range(2)]), np.zeros(4)
    A, b = np.array([[1.0, 0.0], [0.5, 1.0]]), np.array([-1.0, -2.0])
    fitted = []
    for scale in (1.0, 1e6):
        constraints = [{"name": "custom", "A": scale * A, "b": scale * b}]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = plqMF_Ridge(
                2,
                2,
                loss={"name": "MSE"},
                rank=2,
                biased=False,
                C=0.01,
                random_state=42,
                tol=1e-8,
                tol_CD=1e-8,
                max_iter=100000,
                max_iter_CD=20,
                constraint_user=constraints,
                constraint_item=constraints,
            ).fit(X, y)
        assert not caught
        assert model.converged_ and model.inner_converged_
        assert model.scaled_constraint_violation_ <= model.tol
        fitted.append(model)
    assert fitted[0].n_iter_ == fitted[1].n_iter_ == 2
    np.testing.assert_allclose(fitted[0].objective_, fitted[1].objective_, atol=1e-10, rtol=0)


@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("family", range(4))
def test_scaled_mf_full_objective_cold_entities_and_zero_weight_blocks(biased, family):
    X = np.array([(u, i) for u in range(2) for i in range(3)])
    y, weight = np.zeros(len(X)), np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
    d = 2 + biased
    lower = np.linspace(0.2, 0.5, d)
    A, b = np.vstack((2 * np.eye(d), np.zeros(d))), np.r_[-2 * lower, 0.0]
    reference = None
    for exponent in (np.zeros(d + 1), np.full(d + 1, -40), np.full(d + 1, 40), np.arange(d + 1) * 40 - 40):
        scale = np.exp2(exponent)
        constraints = [{"name": "custom", "A": scale[:, None] * A, "b": scale * b}]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = plqMF_Ridge(
                3,
                4,
                loss=LOSSES[family],
                rank=2,
                biased=biased,
                C=0.01,
                random_state=42,
                tol=1e-9,
                tol_CD=1e-10,
                max_iter=100000,
                max_iter_CD=100,
                constraint_user=constraints,
                constraint_item=constraints,
            ).fit(X, y, sample_weight=weight)
        assert not caught
        assert model.converged_ and model.inner_converged_
        pred = np.sum(model.P[X[:, 0]] * model.Q[X[:, 1]], axis=1)
        penalty = model.rho / 3 * np.square(model.P).sum() + (1 - model.rho) / 4 * np.square(model.Q).sum()
        if biased:
            pred += model.bu[X[:, 0]] + model.bi[X[:, 1]]
            penalty += model.rho / 3 * np.square(model.bu).sum() + (1 - model.rho) / 4 * np.square(model.bi).sum()
        objective = model.C * (weight @ direct_loss(y - pred, family)) + penalty
        assert objective == pytest.approx(model.objective_, abs=1e-12)
        assert objective == pytest.approx(model.history[model.n_iter_, 1], abs=1e-12)
        violations, scaled_violations = [], []
        for factors, bias in ((model.P, model.bu), (model.Q, model.bi)):
            parameters = np.column_stack((bias, factors)) if biased else factors
            assert np.all(parameters > lower - model.tol)
            slack = parameters @ (scale[:, None] * A).T + scale * b
            violations.append(max(0, -slack.min()))
            scaled_violations.append(max(0, -(parameters - lower).min()))
        assert model.constraint_violation_ == pytest.approx(max(violations), abs=1e-12)
        assert model.scaled_constraint_violation_ == pytest.approx(max(scaled_violations), abs=1e-15)
        assert model.scaled_constraint_violation_ <= model.tol
        if reference is not None:
            assert model.n_iter_ == reference.n_iter_
            assert objective == pytest.approx(reference.objective_, abs=1e-10)
        reference = model


def test_tiny_raw_violation_does_not_pass_outer_feasibility_and_warning_rolls_back(monkeypatch):
    X, y = np.array([[0, 0], [1, 1]]), np.zeros(2)
    constraints = [{"name": "custom", "A": np.array([[1e-12]]), "b": np.array([-1e-12])}]
    options = dict(
        loss={"name": "MSE"},
        rank=1,
        biased=False,
        C=0.01,
        random_state=42,
        tol=1e-8,
        max_iter=100000,
        max_iter_CD=3,
        constraint_user=constraints,
        constraint_item=constraints,
    )
    fitted = plqMF_Ridge(2, 2, **options).fit(X, y)
    before = pickle.dumps(vars(fitted))
    # Inject an incorrectly certified block result. The outer check must still
    # reject its unit-sized violation of z >= 1, despite raw slack of -1e-12.
    monkeypatch.setattr(plqMF_Ridge, "_solve_block", lambda self, *args: (np.zeros(1), True))
    with pytest.warns(ConvergenceWarning, match="final constraints") as caught:
        invalid = plqMF_Ridge(2, 2, **options).fit(X, y)
    assert len(caught) == 1
    assert invalid.inner_converged_ and not invalid.converged_
    assert invalid.constraint_violation_ < invalid.tol
    assert invalid.scaled_constraint_violation_ == pytest.approx(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        with pytest.raises(ConvergenceWarning, match="final constraints"):
            fitted.fit(X, y)
    assert pickle.dumps(vars(fitted)) == before


def test_unrepresentable_final_diagnostics_raise_overflow():
    X, y = np.array([[0, 0]]), np.zeros(1)
    model = plqMF_Ridge(1, 1, loss={"name": "MSE"}, rank=1, biased=False).fit(X, y)
    model.constraint_user = [{"name": "custom", "A": np.array([[1e308]]), "b": np.zeros(1)}]
    model.P[:] = 2
    with pytest.raises(OverflowError, match="MF constraint diagnostics"):
        model._factor_constraint_violations(X)
