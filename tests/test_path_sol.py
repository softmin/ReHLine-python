"""Test regularization-path solution (plqERM_Ridge_path_sol)."""

import numpy as np
from sklearn.datasets import make_hastie_10_2

from rehline import CQR_Ridge_path_sol, plqERM_Ridge_path_sol


def test_path_sol_warm_start_shapes():
    """plqERM_Ridge_path_sol should return arrays with consistent shapes."""
    X, y = make_hastie_10_2(random_state=1)
    loss = {"name": "svm"}
    # Use a small number of C values so the test is fast
    Cs = np.logspace(-3, 3, 7, base=2)

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
    assert coefs.shape == (n_features, n_path), f"coefs shape should be ({n_features}, {n_path}), got {coefs.shape}"

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


def test_path_sol_generates_default_Cs_when_not_provided():
    """plqERM_Ridge_path_sol should generate a sorted path when Cs is omitted."""
    X, y = make_hastie_10_2(random_state=1)
    loss = {"name": "svm"}

    Cs_out, n_iters, loss_vals, l2_norms, coefs = plqERM_Ridge_path_sol(
        X,
        y,
        loss=loss,
        eps=1e-2,
        n_Cs=4,
        max_iter=100000,
        tol=1e-3,
        verbose=0,
        warm_start=False,
        constraint=None,
        return_time=False,
    )

    assert len(Cs_out) == 4
    assert np.all(np.diff(Cs_out) >= 0), "Generated Cs should be sorted in ascending order"
    assert len(n_iters) == 4
    assert len(loss_vals) == 4
    assert len(l2_norms) == 4
    assert coefs.shape == (X.shape[1], 4)


def test_cqr_path_sol_shapes_without_times():
    """CQR_Ridge_path_sol should return consistently shaped outputs without timing."""
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = X @ np.array([1.0, 2.0]) + np.random.randn(200)
    quantiles = [0.1, 0.5, 0.9]
    Cs = np.array([0.1, 1.0])

    Cs_out, models, coefs, intercepts = CQR_Ridge_path_sol(
        X,
        y,
        quantiles=quantiles,
        Cs=Cs,
        max_iter=20000,
        tol=1e-3,
        verbose=0,
        warm_start=False,
        return_time=False,
    )

    assert np.array_equal(Cs_out, Cs)
    assert len(models) == len(Cs)
    assert coefs.shape == (len(Cs), len(quantiles), X.shape[1])
    assert intercepts.shape == (len(Cs), len(quantiles))


def test_cqr_path_sol_generates_default_Cs_with_times():
    """CQR_Ridge_path_sol should generate default Cs and return timing info."""
    np.random.seed(0)
    X = np.random.randn(120, 3)
    y = X @ np.array([1.0, -0.5, 2.0]) + np.random.randn(120)
    quantiles = [0.25, 0.5, 0.75]

    Cs_out, models, coefs, intercepts, fit_times = CQR_Ridge_path_sol(
        X,
        y,
        quantiles=quantiles,
        eps=1e-3,
        n_Cs=3,
        max_iter=20000,
        tol=1e-3,
        verbose=0,
        warm_start=True,
        return_time=True,
    )

    expected_Cs = np.power(10.0, np.linspace(np.log10(1e-3), np.log10(10), 3))

    assert np.allclose(Cs_out, expected_Cs)
    assert len(models) == 3
    assert coefs.shape == (3, len(quantiles), X.shape[1])
    assert intercepts.shape == (3, len(quantiles))
    assert len(fit_times) == 3
    assert np.all(np.array(fit_times) >= 0)


def test_cqr_path_sol_verbose_reports_progress(capsys):
    """CQR_Ridge_path_sol should print per-C progress when verbose is enabled."""
    np.random.seed(1)
    X = np.random.randn(80, 2)
    y = X @ np.array([1.5, -0.5]) + np.random.randn(80)

    CQR_Ridge_path_sol(
        X,
        y,
        quantiles=[0.2, 0.8],
        Cs=np.array([0.5]),
        max_iter=20000,
        tol=1e-3,
        verbose=1,
        warm_start=False,
        return_time=True,
    )

    captured = capsys.readouterr()
    assert "[OK] C=" in captured.out
