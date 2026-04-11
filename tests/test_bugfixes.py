"""Regression tests for bugs fixed in the codebase.

Each test targets a specific bug that was identified and fixed,
ensuring it does not regress in future changes.
"""

import numpy as np
import pytest

from rehline import (
    ReHLine,
    make_mf_dataset,
    plqERM_ElasticNet,
    plqERM_Ridge,
    plqERM_Ridge_path_sol,
)
from rehline._base import (
    _check_rehu,
    _check_relu,
    _make_constraint_rehline_param,
    _make_loss_rehline_param,
    _make_penalty_rehline_param,
    _rehu,
)
from rehline._loss import ReHLoss
from rehline._mf_class import plqMF_Ridge
from rehline._sklearn_mixin import plq_Ridge_Classifier, plq_Ridge_Regressor

# ---------------------------------------------------------------------------
# Helper data generators
# ---------------------------------------------------------------------------


def _make_classification_data(n=200, d=3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    beta0 = rng.randn(d)
    y = np.sign(X.dot(beta0) + rng.randn(n))
    return X, y


def _make_regression_data(n=200, d=3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    beta0 = rng.randn(d)
    y = X.dot(beta0) + 0.5 * rng.randn(n)
    return X, y


# ===========================================================================
# Bug #1: Mutable default arguments — instances must be independent
# ===========================================================================


class TestMutableDefaults:
    """Verify that two independently created instances do not share state."""

    def test_plqERM_Ridge_constraint_independence(self):
        """Mutating constraint on instance A must not affect instance B."""
        a = plqERM_Ridge(loss={"name": "svm"})
        a.constraint.append({"name": "nonnegative"})

        b = plqERM_Ridge(loss={"name": "svm"})
        assert b.constraint == [], f"Instance B should have empty constraint, got {b.constraint}"

    def test_plqERM_ElasticNet_constraint_independence(self):
        a = plqERM_ElasticNet(loss={"name": "svm"})
        a.constraint.append({"name": "nonnegative"})

        b = plqERM_ElasticNet(loss={"name": "svm"})
        assert b.constraint == []

    def test_ReHLine_U_independence(self):
        """Mutating _U on instance A must not affect instance B."""
        a = ReHLine()
        b = ReHLine()
        assert a._U is not b._U, "_U arrays should be distinct objects"

    def test_plq_Ridge_Classifier_multi_class_independence(self):
        plq_Ridge_Classifier(loss={"name": "svm"}, multi_class="ovr")
        b = plq_Ridge_Classifier(loss={"name": "svm"})
        # b should get a fresh default, not be affected by a's value
        assert b.multi_class == []

    def test_plq_Ridge_Regressor_loss_independence(self):
        """Default loss dict on instance A must not leak to instance B."""
        a = plq_Ridge_Regressor()
        a.loss["qt"] = 0.9  # mutate A's loss

        b = plq_Ridge_Regressor()
        assert b.loss["qt"] == 0.5, f"Instance B should have default qt=0.5, got {b.loss['qt']}"

    def test_plqMF_Ridge_constraint_independence(self):
        a = plqMF_Ridge(n_users=10, n_items=10, loss={"name": "MAE"})
        a.constraint_user.append({"name": "nonnegative"})

        b = plqMF_Ridge(n_users=10, n_items=10, loss={"name": "MAE"})
        assert b.constraint_user == []
        assert b.constraint_item == []


# ===========================================================================
# Bug #2: make_mf_dataset should return clipped ratings
# ===========================================================================


class TestMakeMfDataset:
    """Verify that make_mf_dataset returns ratings within [min, max]."""

    def test_ratings_within_bounds_default(self):
        data = make_mf_dataset(n_users=50, n_items=50, seed=0, noise_std=2.0)
        y = data["y"]
        assert np.all(y >= 1.0), f"Min rating {y.min():.2f} < 1.0"
        assert np.all(y <= 5.0), f"Max rating {y.max():.2f} > 5.0"

    def test_ratings_within_custom_bounds(self):
        data = make_mf_dataset(n_users=50, n_items=50, seed=0, noise_std=2.0, rating_min=0.0, rating_max=10.0)
        y = data["y"]
        assert np.all(y >= 0.0), f"Min rating {y.min():.2f} < 0.0"
        assert np.all(y <= 10.0), f"Max rating {y.max():.2f} > 10.0"

    def test_ratings_are_rounded_to_half(self):
        """Ratings should be rounded to nearest 0.5."""
        data = make_mf_dataset(n_users=50, n_items=50, seed=0)
        y = data["y"]
        # y * 2 should be integers (since rounded to 0.5 precision)
        assert np.allclose(y * 2, np.round(y * 2)), "Ratings should be rounded to nearest 0.5"


# ===========================================================================
# Bug #3: path_sol verbose>=1 with return_time=False should not crash
# ===========================================================================


class TestPathSolVerbose:
    """Verify that plqERM_Ridge_path_sol does not crash with verbose + no timing."""

    def test_verbose_without_return_time(self, capsys):
        """verbose=1 + return_time=False must not raise NameError."""
        X, y = _make_classification_data(n=100, d=3)
        loss = {"name": "svm"}
        Cs = np.array([0.1, 1.0])

        # This used to raise NameError: name 'total_time' is not defined
        result = plqERM_Ridge_path_sol(
            X,
            y,
            loss=loss,
            Cs=Cs,
            max_iter=5000,
            tol=1e-3,
            verbose=1,
            return_time=False,
        )

        # When return_time=False, result should have 5 elements (no times)
        assert len(result) == 5, f"Expected 5 return values, got {len(result)}"
        Cs_out, n_iters, loss_vals, l2_norms, coefs = result
        assert len(Cs_out) == 2
        assert len(n_iters) == 2
        captured = capsys.readouterr()
        assert "PLQ ERM Path Solution Results" in captured.out
        assert "Time (s)" not in captured.out

    def test_verbose_with_return_time(self, capsys):
        """verbose=1 + return_time=True should still work."""
        X, y = _make_classification_data(n=100, d=3)
        loss = {"name": "svm"}
        Cs = np.array([0.1, 1.0])

        result = plqERM_Ridge_path_sol(
            X,
            y,
            loss=loss,
            Cs=Cs,
            max_iter=5000,
            tol=1e-3,
            verbose=1,
            return_time=True,
        )

        assert len(result) == 6, f"Expected 6 return values, got {len(result)}"
        captured = capsys.readouterr()
        assert "PLQ ERM Path Solution Results" in captured.out
        assert "Total Time" in captured.out


# ===========================================================================
# Bug #4: ReHU cutpoints (_Tau) must be forwarded to _rehu
# ===========================================================================


class TestReHUCutpoints:
    """Verify that call_ReLHLoss correctly uses _Tau cutpoints."""

    def test_rehu_with_different_cutpoints(self):
        """_rehu(x, cut) should differ from _rehu(x, cut=1) when cut != 1."""
        x = np.array([[0.5, 1.5, 2.5]])
        cut_small = np.array([[0.3, 0.3, 0.3]])
        cut_default = 1

        result_small = _rehu(x, cut_small)
        result_default = _rehu(x, cut_default)

        # With cut=0.3, large x values should be capped differently than cut=1
        assert not np.allclose(result_small, result_default), (
            "ReHU with cut=0.3 should differ from cut=1 for large inputs"
        )

    def test_call_ReLHLoss_uses_tau(self):
        """call_ReLHLoss should produce different results with different _Tau."""
        n = 50
        rng = np.random.RandomState(42)

        # Create a ReHLine instance with ReHU terms (H > 0)
        clf = ReHLine(C=1.0)
        clf._U = np.empty(shape=(0, 0))  # no ReLU terms
        clf._V = np.empty(shape=(0, 0))
        clf._S = rng.randn(1, n)
        clf._T = rng.randn(1, n)
        clf.L = 0
        clf.H = 1

        score = rng.randn(n)

        # Tau = 0.1 (small cutpoints)
        clf._Tau = 0.1 * np.ones((1, n))
        loss_small_tau = clf.call_ReLHLoss(score)

        # Tau = 10.0 (large cutpoints)
        clf._Tau = 10.0 * np.ones((1, n))
        loss_large_tau = clf.call_ReLHLoss(score)

        assert not np.allclose(loss_small_tau, loss_large_tau), (
            "call_ReLHLoss should produce different values for different _Tau"
        )

    def test_rehu_cutpoint_clamps_output(self):
        """ReHU with small cut should clamp output more aggressively."""
        x = np.array([[3.0]])  # large positive input

        loss_cut_01 = _rehu(x, cut=0.1)
        loss_cut_10 = _rehu(x, cut=10.0)

        # huber(delta, x) for large x: delta * x - 0.5 * delta^2
        # So smaller delta = smaller output
        assert loss_cut_01 < loss_cut_10, f"ReHU(cut=0.1) = {loss_cut_01} should be < ReHU(cut=10) = {loss_cut_10}"


# ===========================================================================
# Bug #5 & #6: Proper exceptions (ValueError, NotImplementedError)
# ===========================================================================


class TestExceptionTypes:
    """Verify that validation errors raise ValueError (not AssertionError)
    and unsupported features raise NotImplementedError (not Exception)."""

    # --- _check_relu / _check_rehu: ValueError ---

    def test_check_relu_shape_mismatch(self):
        """_check_relu should raise ValueError on shape mismatch."""
        coef = np.ones((2, 5))
        intercept = np.ones((3, 5))
        with pytest.raises(ValueError, match="same shape"):
            _check_relu(coef, intercept)

    def test_check_rehu_shape_mismatch(self):
        """_check_rehu should raise ValueError on shape mismatch."""
        coef = np.ones((2, 5))
        intercept = np.ones((3, 5))
        cut = np.ones((2, 5))
        with pytest.raises(ValueError, match="same shape"):
            _check_rehu(coef, intercept, cut)

    def test_check_rehu_negative_cut(self):
        """_check_rehu should raise ValueError on negative cutpoints."""
        coef = np.ones((2, 5))
        intercept = np.ones((2, 5))
        cut = -np.ones((2, 5))
        with pytest.raises(ValueError, match="non-negative"):
            _check_rehu(coef, intercept, cut)

    def test_check_rehu_valid_input_no_error(self):
        """_check_rehu should not raise on valid input."""
        coef = np.ones((2, 5))
        intercept = np.ones((2, 5))
        cut = np.ones((2, 5))
        _check_rehu(coef, intercept, cut)  # should not raise

    # --- Unsupported loss: ValueError ---

    def test_unsupported_loss_raises_valueerror(self):
        """_make_loss_rehline_param should raise ValueError for unknown loss."""
        X, y = _make_classification_data(n=50)
        with pytest.raises(ValueError, match="does not support this loss"):
            _make_loss_rehline_param(loss={"name": "nonexistent_loss"}, X=X, y=y)

    # --- Unsupported constraint: ValueError ---

    def test_unsupported_constraint_raises_valueerror(self):
        """_make_constraint_rehline_param should raise ValueError for unknown constraint."""
        X, y = _make_classification_data(n=50)
        with pytest.raises(ValueError, match="does not support this constraint"):
            _make_constraint_rehline_param(constraint=[{"name": "nonexistent_constraint"}], X=X, y=y)

    # --- Unimplemented penalty: NotImplementedError ---

    def test_make_penalty_raises_not_implemented(self):
        """_make_penalty_rehline_param should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="under development"):
            _make_penalty_rehline_param()

    # --- ReHLoss validation: ValueError ---

    def test_rehloss_shape_mismatch_raises_valueerror(self):
        """ReHLoss should raise ValueError when relu and rehu have different n_samples."""
        relu_coef = np.ones((2, 10))
        relu_intercept = np.ones((2, 10))
        rehu_coef = np.ones((1, 5))  # different n_samples
        rehu_intercept = np.ones((1, 5))
        rehu_cut = np.ones((1, 5))

        loss = ReHLoss(relu_coef, relu_intercept, rehu_coef, rehu_intercept, rehu_cut)
        score = np.random.randn(10)

        with pytest.raises(ValueError, match="same shape"):
            loss(score)

    # --- Fairness constraint dim mismatch: ValueError ---

    def test_fair_constraint_dim_mismatch(self):
        """Fair constraint should raise ValueError when X_sen and tol_sen dims differ."""
        X, y = _make_classification_data(n=50, d=3)
        constraint = [
            {
                "name": "fair",
                "sen_idx": [0, 1],  # 2 sensitive features
                "tol_sen": [0.1],  # but only 1 tolerance → mismatch
            }
        ]
        with pytest.raises(ValueError, match="dim of X_sen and len of tol_sen"):
            _make_constraint_rehline_param(constraint=constraint, X=X, y=y)
