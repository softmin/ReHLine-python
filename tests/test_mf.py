"""
Test Matrix Factorization (plqMF_Ridge) on a small simulated dataset.

Dataset sizes are kept small (100 users, 300 items, 2000 interactions)
so the test suite runs quickly in CI.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
import pytest
from sklearn.model_selection import train_test_split

from rehline import make_mf_dataset, plqMF_Ridge


@pytest.fixture(scope="module")
def mf_data():
    """Small rating dataset used across multiple tests."""
    n_users, n_items = 100, 300
    ratings = make_mf_dataset(
        n_users=n_users,
        n_items=n_items,
        n_interactions=2000,
        seed=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        ratings["X"],
        ratings["y"],
        test_size=0.2,
        random_state=42,
    )
    return {
        "n_users": n_users,
        "n_items": n_items,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mf_mae_regression_fits(mf_data):
    """plqMF_Ridge with MAE loss should fit and return a finite objective."""
    d = mf_data
    model = plqMF_Ridge(
        loss={"name": "mae"},
        n_users=d["n_users"],
        n_items=d["n_items"],
        rank=5,
        C=0.001,
        max_iter=5000,
        tol=0.01,
    )
    model.fit(d["X_train"], d["y_train"])

    obj = model.obj(d["X_test"], d["y_test"])
    assert np.isfinite(obj[0]), f"Objective should be finite, got {obj}"


def test_mf_mse_regression_fits(mf_data):
    """plqMF_Ridge with MSE loss should fit and return a finite objective."""
    d = mf_data
    model = plqMF_Ridge(
        loss={"name": "MSE"},
        n_users=d["n_users"],
        n_items=d["n_items"],
        rank=5,
        C=0.001,
        max_iter=5000,
        tol=0.01,
    )
    model.fit(d["X_train"], d["y_train"])

    obj = model.obj(d["X_test"], d["y_test"])
    assert np.isfinite(obj[0]), f"Objective should be finite, got {obj}"


def test_mf_hinge_classification_fits(mf_data):
    """plqMF_Ridge with hinge loss should fit binary classification data."""
    d = mf_data
    y_train_bin = np.where(d["y_train"] > np.median(d["y_train"]), 1, -1)
    y_test_bin = np.where(d["y_test"] > np.median(d["y_train"]), 1, -1)

    model = plqMF_Ridge(
        loss={"name": "hinge"},
        n_users=d["n_users"],
        n_items=d["n_items"],
        rank=5,
        C=0.001,
        max_iter=5000,
        tol=0.01,
    )
    model.fit(d["X_train"], y_train_bin)

    # decision_function should return a 1-D array of length n_test
    scores = model.decision_function(d["X_test"])
    assert scores.shape == (len(d["X_test"]),), f"decision_function shape mismatch: {scores.shape}"

    # Accuracy should be above random guessing
    preds = np.where(scores > 0, 1, -1)
    accuracy = np.mean(preds == y_test_bin)
    assert accuracy > 0.5, f"Hinge-loss MF accuracy ({accuracy:.3f}) should be > 0.5"


def test_mf_data_validation_errors():
    """Test data validation raises appropriate errors."""
    # Test X with wrong shape (not 2 columns)
    with pytest.raises(ValueError, match="X must have shape"):
        model = plqMF_Ridge(n_users=10, n_items=10, loss={"name": "mae"})
        model.fit(np.array([[0, 0, 0]]), np.array([1.0]))

    # Test X and y mismatch
    with pytest.raises(ValueError, match="X and y must have the same number"):
        model = plqMF_Ridge(n_users=10, n_items=10, loss={"name": "mae"})
        model.fit(np.array([[0, 0]]), np.array([1.0, 2.0]))

    # Test invalid user ID (negative)
    with pytest.raises(ValueError, match="User IDs must be in"):
        model = plqMF_Ridge(n_users=10, n_items=10, loss={"name": "mae"})
        model.fit(np.array([[-1, 0]]), np.array([1.0]))

    # Test invalid user ID (>= n_users)
    with pytest.raises(ValueError, match="User IDs must be in"):
        model = plqMF_Ridge(n_users=10, n_items=10, loss={"name": "mae"})
        model.fit(np.array([[10, 0]]), np.array([1.0]))

    # Test invalid item ID (negative)
    with pytest.raises(ValueError, match="Item IDs must be in"):
        model = plqMF_Ridge(n_users=10, n_items=10, loss={"name": "mae"})
        model.fit(np.array([[0, -1]]), np.array([1.0]))

    # Test invalid item ID (>= n_items)
    with pytest.raises(ValueError, match="Item IDs must be in"):
        model = plqMF_Ridge(n_users=10, n_items=10, loss={"name": "mae"})
        model.fit(np.array([[0, 10]]), np.array([1.0]))


def test_mf_cold_start_users_items():
    """Test cold start handling: users/items with no interactions."""
    # Create data where user 0 and item 0 have no interactions
    # n_users=3, n_items=3, but only users 1,2 and items 1,2 interact
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    y = np.array([3.0, 4.0, 2.0, 5.0])

    model = plqMF_Ridge(
        n_users=3,
        n_items=3,
        loss={"name": "mae"},
        rank=2,
        C=0.1,
        max_iter=1000,
        tol=0.01,
    )
    model.fit(X, y)

    # Cold start user (user 0) should have zero factors and bias
    assert np.allclose(model.P[0, :], 0.0)
    assert model.bu[0] == 0.0

    # Cold start item (item 0) should have zero factors and bias
    assert np.allclose(model.Q[0, :], 0.0)
    assert model.bi[0] == 0.0


def test_mf_biased_false():
    """Test plqMF_Ridge with biased=False (no bias terms)."""
    n_users, n_items = 20, 30
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    model = plqMF_Ridge(
        n_users=n_users,
        n_items=n_items,
        loss={"name": "mae"},
        biased=False,
        rank=3,
        C=0.1,
        max_iter=1000,
        tol=0.01,
    )
    model.fit(X, y)

    # bu and bi should be None when biased=False
    assert model.bu is None
    assert model.bi is None

    # decision_function should work without biases
    scores = model.decision_function(X)
    assert scores.shape == (len(X),)

    # obj should work without biases
    loss_term, obj_val = model.obj(X, y)
    assert np.isfinite(loss_term)
    assert np.isfinite(obj_val)


def test_mf_verbose_output(capsys):
    """Test verbose printing (lines 308, 464-466)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    # Test verbose=1 (CD iteration progress)
    model = plqMF_Ridge(
        n_users=2,
        n_items=2,
        loss={"name": "mae"},
        rank=2,
        C=0.1,
        max_iter=500,
        tol=0.01,
        max_iter_CD=2,
        verbose=1,
    )
    model.fit(X, y)
    captured = capsys.readouterr()
    assert "Iteration" in captured.out
    assert "Average Loss" in captured.out


def test_mf_convergence_warning():
    """Test convergence warning when max_iter is too small."""
    from sklearn.exceptions import ConvergenceWarning

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    model = plqMF_Ridge(
        n_users=2,
        n_items=2,
        loss={"name": "mae"},
        rank=2,
        C=0.1,
        max_iter=1,  # Only 1 iteration to guarantee non-convergence
        tol=1e-10,
        max_iter_CD=1,
    )
    with pytest.warns(ConvergenceWarning, match="ReHLine failed to converge"):
        model.fit(X, y)


def test_mf_param_validation_errors():
    """Test parameter validation raises appropriate errors."""
    # Test invalid rho (must be between 0 and 1)
    with pytest.raises(ValueError, match="rho must be between 0 and 1"):
        model = plqMF_Ridge(n_users=10, n_items=10, loss={"name": "mae"}, rho=0.0)
        model.fit(np.array([[0, 0]]), np.array([1.0]))

    with pytest.raises(ValueError, match="rho must be between 0 and 1"):
        model = plqMF_Ridge(n_users=10, n_items=10, loss={"name": "mae"}, rho=1.0)
        model.fit(np.array([[0, 0]]), np.array([1.0]))

    # Test invalid C (must be positive)
    with pytest.raises(ValueError, match="C must be positive"):
        model = plqMF_Ridge(n_users=10, n_items=10, loss={"name": "mae"}, C=0.0)
        model.fit(np.array([[0, 0]]), np.array([1.0]))

    # Test invalid tol_CD (must be positive)
    with pytest.raises(ValueError, match="tol_CD must be positive"):
        model = plqMF_Ridge(n_users=10, n_items=10, loss={"name": "mae"}, tol_CD=0.0)
        model.fit(np.array([[0, 0]]), np.array([1.0]))

    # Test invalid tol (must be positive)
    with pytest.raises(ValueError, match="tol must be positive"):
        model = plqMF_Ridge(n_users=10, n_items=10, loss={"name": "mae"}, tol=0.0)
        model.fit(np.array([[0, 0]]), np.array([1.0]))


def test_mf_nonneg_constraint(mf_data):
    """plqMF_Ridge with non-negative constraints should produce non-negative factors."""
    d = mf_data
    model = plqMF_Ridge(
        loss={"name": "mae"},
        n_users=d["n_users"],
        n_items=d["n_items"],
        rank=3,
        C=0.001,
        max_iter=3000,
        tol=0.05,
        constraint_user=[{"name": ">=0"}],
        constraint_item=[{"name": ">=0"}],
    )
    model.fit(d["X_train"], d["y_train"])

    # P: user factor matrix, shape (n_users, rank)
    assert np.all(model.P >= -1e-4), "User factors (P) should be non-negative (within numerical tolerance)"
    # Ui: list of item factor vectors (length n_items); check each individually
    for i, ui in enumerate(model.Ui):
        if ui is not None and hasattr(ui, "__len__"):
            assert np.all(np.asarray(ui) >= -1e-4), f"Item factor Ui[{i}] should be non-negative"
