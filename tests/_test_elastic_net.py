"""
Test ElasticNet on simulated dataset
Tests PR #7fd2ab1: add ElasticNet penalty support to ReHLine solver

Data size controlled to ~5000 samples to prevent memory overflow

rehline objective:
    min_beta C * sum_i PLQ(y_i, x_i^T beta) + l1_ratio * ||beta||_1 + 0.5*(1-l1_ratio)*||beta||_2^2

sklearn ElasticNet objective:
    min_beta (1/2n) * sum_i (y_i - x_i^T beta)^2 + alpha*l1_ratio*||beta||_1 + (alpha/2)*(1-l1_ratio)*||beta||_2^2

Matching conditions (for MSE loss):
    - Loss term: C * n = 1/(2*alpha) => alpha = 1/(2*C*n)
    - L1 term: l1_ratio = alpha * l1_ratio => alpha = 1 (contradiction!)
    
The two parameterizations are NOT directly equivalent because:
    - rehline L2 penalty: 0.5*(1-l1_ratio)*||beta||^2
    - sklearn L2 penalty: (alpha/2)*(1-l1_ratio)*||beta||^2 = 1/(4*C*n)*(1-l1_ratio)*||beta||^2

This is a KNOWN ISSUE with the current ElasticNet implementation.
The test below documents this discrepancy.
"""
import time
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rehline import plqERM_ElasticNet, plqERM_Ridge


def test_elasticnet_vs_sklearn_mse():
    """
    Test ElasticNet against sklearn implementation (MSE loss, no intercept).
    
    ⚠️ KNOWN ISSUE: The rehline ElasticNet parameterization differs from sklearn.
    
    rehline objective:
        C * sum_i (y_i - x_i^T beta)^2 + l1_ratio * ||beta||_1 + 0.5*(1-l1_ratio)*||beta||_2^2
    
    sklearn objective:
        (1/2n) * sum_i (y_i - x_i^T beta)^2 + alpha*l1_ratio*||beta||_1 + (alpha/2)*(1-l1_ratio)*||beta||_2^2
    
    These are NOT equivalent because the L2 penalty scales differently.
    To match sklearn exactly, we would need alpha = 1/(2*C*n) for the loss term,
    but then the L2 penalty would be (1-l1_ratio)/(4*C*n)*||beta||^2 instead of
    0.5*(1-l1_ratio)*||beta||^2.
    """
    print("\n" + "="*60)
    print("Test 1: ElasticNet vs sklearn (MSE loss, no intercept)")
    print("="*60)
    
    # Controlled size dataset (~5000 samples)
    n = 5000
    n_features = 20
    C, l1_ratio = 0.1, 0.5
    
    np.random.seed(42)
    X, y = make_regression(n_samples=n, n_features=n_features, noise=0.1, 
                           random_state=42, n_informative=10)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset: {n} samples, {n_features} features")
    print(f"C={C}, l1_ratio={l1_ratio}")
    
    # sklearn ElasticNet (using original test's parameterization: alpha = 1/(C * 2 * n))
    # This matches the loss term but NOT the L2 penalty term
    clf_skl = ElasticNet(alpha=1 / (C * 2 * len(X_train_scaled)), 
                         l1_ratio=l1_ratio,
                         max_iter=10000,
                         tol=1e-5,
                         fit_intercept=False)
    
    clf_skl.fit(X_train_scaled, y_train)
    sol_skl = clf_skl.coef_.flatten()
    
    # ReHLine ElasticNet
    clf_reh = plqERM_ElasticNet(loss={'name': 'mse'}, 
                                C=C,
                                l1_ratio=l1_ratio,
                                max_iter=10000,
                                tol=1e-5)
    
    clf_reh.fit(X_train_scaled, y_train)
    sol_reh = clf_reh.coef_.flatten()
    sol_reh_thresh = np.where(np.abs(sol_reh) < 1e-8, 0, sol_reh)
    
    # Compare coefficients
    print("\nCoefficient Comparison:")
    print("=" * 70)
    print(f"{'Index':^8} {'sklearn':^20} {'rehline':^20} {'diff':^10}")
    print("=" * 70)
    
    max_diff = 0
    for i, (s, r) in enumerate(zip(sol_skl, sol_reh_thresh)):
        diff = abs(s - r)
        max_diff = max(max_diff, diff)
        print(f"{i:^8d} {s:^20.8f} {r:^20.8f} {diff:^10.2e}")
    
    print("=" * 70)
    print(f"Max coefficient difference: {max_diff:.6e}")
    
    # Check sparsity
    sklearn_zeros = np.sum(np.abs(sol_skl) < 1e-8)
    rehline_zeros = np.sum(np.abs(sol_reh_thresh) < 1e-8)
    print(f"\nSparsity (near-zero coefficients):")
    print(f"  sklearn: {sklearn_zeros}/{n_features}")
    print(f"  rehline: {rehline_zeros}/{n_features}")
    
    # Predictions
    y_pred_skl = X_test_scaled @ sol_skl
    y_pred_reh = X_test_scaled @ sol_reh
    
    mse_skl = mean_squared_error(y_test, y_pred_skl)
    mse_reh = mean_squared_error(y_test, y_pred_reh)
    
    print(f"\nTest MSE:")
    print(f"  sklearn: {mse_skl:.6f}")
    print(f"  rehline: {mse_reh:.6f}")
    
    # ⚠️ ASSERTION: Solutions should match within tol=1e-4
    # If this fails, it indicates a bug in the ElasticNet implementation
    print(f"\n⚠️  Checking if solutions match within tol=1e-4...")
    if max_diff > 1e-4:
        print(f"❌ FAIL: Max coefficient difference {max_diff:.6e} > 1e-4")
        print(f"   This indicates a DISCREPANCY between rehline and sklearn ElasticNet!")
        print(f"   Possible cause: Different parameterization of L2 penalty term.")
        print(f"   rehline L2: 0.5*(1-l1_ratio)*||beta||^2")
        print(f"   sklearn L2: (alpha/2)*(1-l1_ratio)*||beta||^2 = {1/(2*C*2*n):.6e}*(1-l1_ratio)*||beta||^2")
        raise AssertionError(
            f"ElasticNet solutions differ by {max_diff:.6e} (> 1e-4). "
            f"rehline MSE={mse_reh:.4f}, sklearn MSE={mse_skl:.4f}. "
            f"This is a known parameterization issue."
        )
    else:
        print(f"✓ Solutions match within tol=1e-4 (max_diff={max_diff:.6e})")
    
    return max_diff


def test_different_l1_ratios():
    """Test ElasticNet with different l1_ratio values."""
    print("\n" + "="*60)
    print("Test 2: Different l1_ratio values")
    print("="*60)
    
    n = 3000
    n_features = 15
    C = 0.01
    
    np.random.seed(42)
    X, y = make_regression(n_samples=n, n_features=n_features, noise=0.1, 
                           random_state=42, n_informative=8)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset: {n} samples, {n_features} features, C={C}")
    print("\nTesting different l1_ratio values:")
    print("-" * 60)
    print(f"{'l1_ratio':^12} {'nonzero':^12} {'train MSE':^15} {'test MSE':^15}")
    print("-" * 60)
    
    # Note: l1_ratio=1.0 would cause division by zero in rho = l1_ratio/(1-l1_ratio)
    l1_ratios = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    for l1_ratio in l1_ratios:
        clf = plqERM_ElasticNet(loss={'name': 'mse'}, 
                                C=C,
                                l1_ratio=l1_ratio,
                                max_iter=5000,
                                tol=1e-4)
        
        clf.fit(X_train_scaled, y_train)
        coef = clf.coef_.flatten()
        n_nonzero = np.sum(np.abs(coef) > 1e-8)
        
        # Compute MSE
        y_train_pred = X_train_scaled @ coef
        y_test_pred = X_test_scaled @ coef
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        print(f"{l1_ratio:^12.1f} {n_nonzero:^12d} {train_mse:^15.6f} {test_mse:^15.6f}")
        
        assert clf.coef_ is not None, f"Failed to fit with l1_ratio={l1_ratio}"
        assert clf.coef_.shape == (n_features,), f"Wrong coef_ shape for l1_ratio={l1_ratio}"
    
    print("-" * 60)
    print("\n✓ All l1_ratio tests passed!")


def test_different_losses():
    """Test ElasticNet with different loss functions."""
    print("\n" + "="*60)
    print("Test 3: Different loss functions with ElasticNet")
    print("="*60)
    
    n = 3000
    n_features = 12
    C = 0.01
    l1_ratio = 0.5
    
    np.random.seed(42)
    X, y = make_regression(n_samples=n, n_features=n_features, noise=0.1, 
                           random_state=42, n_informative=8)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset: {n} samples, {n_features} features")
    print(f"C={C}, l1_ratio={l1_ratio}")
    print("\nTesting different loss functions:")
    print("-" * 60)
    print(f"{'loss':^20} {'n_iter':^10} {'test MAE':^15} {'nonzero':^10}")
    print("-" * 60)
    
    losses = [
        {'name': 'mse'},
        {'name': 'mae'},
        {'name': 'huber', 'tau': 0.1},
        {'name': 'SVR', 'epsilon': 0.1},
    ]
    
    for loss in losses:
        clf = plqERM_ElasticNet(loss=loss, 
                                C=C,
                                l1_ratio=l1_ratio,
                                max_iter=10000,
                                tol=1e-4)
        
        clf.fit(X_train_scaled, y_train)
        coef = clf.coef_.flatten()
        n_nonzero = np.sum(np.abs(coef) > 1e-8)
        
        # Compute MAE
        y_test_pred = X_test_scaled @ coef
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        loss_name = loss['name']
        print(f"{loss_name:^20} {clf.n_iter_:^10d} {test_mae:^15.6f} {n_nonzero:^10d}")
        
        assert clf.coef_ is not None, f"Failed to fit with loss={loss}"
        assert clf.coef_.shape == (n_features,), f"Wrong coef_ shape for loss={loss}"
    
    print("-" * 60)
    print("\n✓ All loss function tests passed!")


def test_warm_start():
    """Test warm start functionality."""
    print("\n" + "="*60)
    print("Test 4: Warm start functionality")
    print("="*60)
    
    n = 5000
    n_features = 30
    l1_ratio = 0.5
    
    np.random.seed(42)
    X, y = make_regression(n_samples=n, n_features=n_features, noise=0.1, 
                           random_state=42, n_informative=15)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Dataset: {n} samples, {n_features} features")
    print(f"l1_ratio={l1_ratio}")
    print(f"\nFitting on sequence of C values: [0.001, 0.002, 0.003, 0.004, 0.005]")
    
    Cs = [0.001, 0.002, 0.003, 0.004, 0.005]
    
    # Cold start
    print("\n--- Cold start ---")
    clf_cold = plqERM_ElasticNet(loss={'name': 'mae'}, 
                                 C=1.0,
                                 l1_ratio=l1_ratio,
                                 max_iter=10000,
                                 tol=1e-4, 
                                 warm_start=False)
    
    start = time.perf_counter()
    iter_counts_cold = []
    for C_tmp in Cs:
        clf_cold.C = C_tmp
        clf_cold.fit(X_scaled, y)
        iter_counts_cold.append(clf_cold.n_iter_)
        print(f"  C={C_tmp:.4f}: n_iter={clf_cold.n_iter_}")
    end = time.perf_counter()
    time_cold = end - start
    print(f"Total time (cold): {time_cold:.4f} s")
    
    # Warm start
    print("\n--- Warm start ---")
    clf_warm = plqERM_ElasticNet(loss={'name': 'mae'}, 
                                 C=1.0,
                                 l1_ratio=l1_ratio,
                                 max_iter=10000,
                                 tol=1e-4, 
                                 warm_start=True)
    
    start = time.perf_counter()
    iter_counts_warm = []
    for C_tmp in Cs:
        clf_warm.C = C_tmp
        clf_warm.fit(X_scaled, y)
        iter_counts_warm.append(clf_warm.n_iter_)
        print(f"  C={C_tmp:.4f}: n_iter={clf_warm.n_iter_}")
    end = time.perf_counter()
    time_warm = end - start
    print(f"Total time (warm): {time_warm:.4f} s")
    
    # Compare
    print(f"\n--- Comparison ---")
    print(f"Cold start total iterations: {sum(iter_counts_cold)}")
    print(f"Warm start total iterations: {sum(iter_counts_warm)}")
    print(f"Speedup: {time_cold/time_warm:.2f}x")
    
    # Warm start should be faster or similar
    assert time_warm <= time_cold * 2.0, "Warm start should not be significantly slower"
    print("\n✓ Warm start test passed!")


def test_elasticnet_vs_ridge():
    """Compare ElasticNet (l1_ratio=0) vs Ridge regression."""
    print("\n" + "="*60)
    print("Test 5: ElasticNet (l1_ratio=0) vs Ridge")
    print("="*60)
    
    n = 3000
    n_features = 12
    C = 0.1
    
    np.random.seed(42)
    X, y = make_regression(n_samples=n, n_features=n_features, noise=0.1, 
                           random_state=42, n_informative=8)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset: {n} samples, {n_features} features, C={C}")
    print(f"l1_ratio=0 should give same results as Ridge (no L1 penalty)")
    
    # ElasticNet with l1_ratio=0 (pure L2)
    clf_en = plqERM_ElasticNet(loss={'name': 'mse'}, 
                               C=C,
                               l1_ratio=0.0,
                               max_iter=5000,
                               tol=1e-4)
    clf_en.fit(X_train_scaled, y_train)
    coef_en = clf_en.coef_.flatten()
    
    # Ridge
    clf_ridge = plqERM_Ridge(loss={'name': 'mse'}, 
                             C=C,
                             max_iter=5000,
                             tol=1e-4)
    clf_ridge.fit(X_train_scaled, y_train)
    coef_ridge = clf_ridge.coef_.flatten()
    
    # Compare
    max_diff = np.max(np.abs(coef_en - coef_ridge))
    print(f"\nMax coefficient difference: {max_diff:.6e}")
    
    # Compute test MSE
    y_pred_en = X_test_scaled @ coef_en
    y_pred_ridge = X_test_scaled @ coef_ridge
    mse_en = mean_squared_error(y_test, y_pred_en)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    
    print(f"Test MSE - ElasticNet(l1_ratio=0): {mse_en:.6f}")
    print(f"Test MSE - Ridge: {mse_ridge:.6f}")
    
    # With l1_ratio=0, ElasticNet should match Ridge within tol=1e-4
    assert max_diff < 1e-4, \
        f"ElasticNet with l1_ratio=0 should match Ridge within 1e-4, max_diff={max_diff:.6e}"
    print("\n✓ Test passed!")


def test_sparsity_increases_with_l1():
    """Test that sparsity increases as l1_ratio increases."""
    print("\n" + "="*60)
    print("Test 6: Sparsity increases with l1_ratio")
    print("="*60)
    
    n = 3000
    n_features = 20
    C = 0.001
    
    np.random.seed(42)
    X, y = make_regression(n_samples=n, n_features=n_features, noise=0.1, 
                           random_state=42, n_informative=8)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Dataset: {n} samples, {n_features} features, C={C}")
    print(f"Expecting more zeros as l1_ratio increases:")
    
    l1_ratios = [0.0, 0.3, 0.6, 0.9]
    n_zeros_list = []
    
    for l1_ratio in l1_ratios:
        clf = plqERM_ElasticNet(loss={'name': 'mse'}, 
                                C=C,
                                l1_ratio=l1_ratio,
                                max_iter=5000,
                                tol=1e-4)
        clf.fit(X_scaled, y)
        coef = clf.coef_.flatten()
        n_zeros = np.sum(np.abs(coef) < 1e-8)
        n_zeros_list.append(n_zeros)
        print(f"  l1_ratio={l1_ratio:.1f}: {n_zeros}/{n_features} near-zero coefficients")
    
    # Sparsity should generally increase with l1_ratio
    assert n_zeros_list[-1] >= n_zeros_list[0], \
        f"Expected more zeros at l1_ratio=0.9 than l1_ratio=0.0, got {n_zeros_list}"
    print("\n✓ Sparsity test passed!")


def test_dual_variable_mu():
    """Test that dual variable mu is properly initialized and updated."""
    print("\n" + "="*60)
    print("Test 7: Dual variable mu (ElasticNet-specific)")
    print("="*60)
    
    n = 2000
    n_features = 10
    C = 0.01
    l1_ratio = 0.5
    
    np.random.seed(42)
    X, y = make_regression(n_samples=n, n_features=n_features, noise=0.1, 
                           random_state=42, n_informative=6)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Dataset: {n} samples, {n_features} features")
    print(f"C={C}, l1_ratio={l1_ratio}")
    
    clf = plqERM_ElasticNet(loss={'name': 'mse'}, 
                            C=C,
                            l1_ratio=l1_ratio,
                            max_iter=5000,
                            tol=1e-4)
    
    clf.fit(X_scaled, y)
    
    # Check that mu dual variable is stored
    assert hasattr(clf, '_mu'), "clf should have _mu attribute"
    print(f"\nDual variable mu shape: {clf._mu.shape}")
    print(f"mu values (first 5): {clf._mu[:5]}")
    
    # mu should be in [0, rho]
    rho = clf.rho
    print(f"rho = {rho:.4f}")
    assert np.all(clf._mu >= 0), "mu should be non-negative"
    assert np.all(clf._mu <= rho + 1e-10), f"mu should be <= rho={rho}"
    
    print(f"\nmu is in [0, {rho:.4f}] ✓")
    print("\n✓ Dual variable mu test passed!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ELASTICNET TEST SUITE")
    print("Testing PR #7fd2ab1: add ElasticNet penalty support")
    print("="*70)
    
    # Run all tests
    # Test 1 will FAIL if rehline and sklearn solutions differ by more than 1e-4
    # This is expected to reveal a parameterization issue
    try:
        test_elasticnet_vs_sklearn_mse()
    except AssertionError as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        print("   This indicates a bug in the ElasticNet implementation!")
    
    test_different_l1_ratios()
    test_different_losses()
    test_warm_start()
    
    try:
        test_elasticnet_vs_ridge()
    except AssertionError as e:
        print(f"\n❌ Test 5 FAILED: {e}")
    
    test_sparsity_increases_with_l1()
    test_dual_variable_mu()
    
    print("\n" + "="*70)
    print("ElasticNet: TEST SUITE COMPLETE")
    print("="*70)
