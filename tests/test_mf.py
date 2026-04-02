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
