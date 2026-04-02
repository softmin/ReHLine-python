"""Test regularization-path solution (plqERM_Ridge_path_sol)."""

import numpy as np
from sklearn.datasets import make_hastie_10_2

from rehline import plqERM_Ridge_path_sol


def test_path_sol_warm_start_shapes():
    """plqERM_Ridge_path_sol should return arrays with consistent shapes."""
    X, y = make_hastie_10_2(random_state=1)
    loss = {"name": "svm"}
    # Use a small number of C values so the test is fast
    Cs = np.logspace(-3, 3, 10, base=2)

    (Cs_out, times, n_iters, loss_vals, l2_norms, coefs) = plqERM_Ridge_path_sol(
        X,
        y,
        loss=loss,
        Cs=Cs,
        max_iter=50000,
        tol=1e-3,
        verbose=0,
        warm_start=True,
        constraint=[],
        return_time=True,
    )

    n_path = len(Cs)
    n_features = X.shape[1]

    assert len(Cs_out) == n_path, f"Cs_out length should be {n_path}, got {len(Cs_out)}"
    assert len(times) == n_path, f"times length should be {n_path}, got {len(times)}"
    assert len(n_iters) == n_path, f"n_iters length should be {n_path}, got {len(n_iters)}"
    assert len(loss_vals) == n_path, f"loss_vals length should be {n_path}, got {len(loss_vals)}"
    assert coefs.shape == (n_path, n_features), f"coefs shape should be ({n_path}, {n_features}), got {coefs.shape}"

    # All timing values should be non-negative
    assert np.all(np.array(times) >= 0), "All timing values should be non-negative"

    # Loss values should be finite
    assert np.all(np.isfinite(loss_vals)), "All loss values should be finite"


def test_path_sol_loss_range_with_larger_C():
    """Training loss should be generally lower at larger C (less regularisation)."""
    X, y = make_hastie_10_2(random_state=1)
    loss = {"name": "svm"}
    # Compare endpoints: a large C should give lower (or equal) training loss
    # than a small C, allowing generous tolerance for convergence noise.
    Cs = np.array([0.01, 0.1, 1.0, 10.0])

    _, _, _, loss_vals, _, _ = plqERM_Ridge_path_sol(
        X,
        y,
        loss=loss,
        Cs=Cs,
        max_iter=100000,
        tol=1e-3,
        verbose=0,
        warm_start=True,
        constraint=[],
        return_time=True,
    )

    # Loss at the largest C should be no more than 5% above loss at the smallest C
    # (allows for convergence noise, verifies the general trend)
    assert loss_vals[-1] <= loss_vals[0] * 1.05, (
        f"Loss at C=10 ({loss_vals[-1]:.2f}) should be ≤ 105% of loss at C=0.01 ({loss_vals[0]:.2f})"
    )
