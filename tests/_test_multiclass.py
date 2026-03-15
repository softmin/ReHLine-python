"""
Test Multi-class Classification on simulated dataset
Tests PR #55cc374: add multi-class classification function

Compares rehline's plq_Ridge_Classifier (OvR and OvO) with sklearn's LinearSVC.
All algorithms use tol=1e-5. Solutions must match within tol=1e-3.

Data size controlled to ~5000 samples to prevent memory overflow
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from rehline import plq_Ridge_Classifier


def test_binary_vs_sklearn():
    """
    Test binary classification against sklearn's LinearSVC.
    Both use tol=1e-5. Solutions (coef_) must match within tol=1e-3.
    """
    print("\n" + "="*60)
    print("Test 1: Binary Classification vs sklearn LinearSVC")
    print("="*60)
    
    # Generate binary dataset
    np.random.seed(42)
    n_samples = 3000
    n_features = 10
    C = 1.0
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        class_sep=1.5,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset: {n_samples} samples, {n_features} features, 2 classes")
    print(f"C={C}, tol=1e-5")
    
    # sklearn LinearSVC
    print(f"\n--- sklearn LinearSVC ---")
    clf_skl = LinearSVC(
        C=C,
        loss='hinge',
        fit_intercept=True,
        max_iter=1000000,
        tol=1e-5,
        random_state=42
    )
    clf_skl.fit(X_train_scaled, y_train)
    coef_skl = clf_skl.coef_.flatten()
    intercept_skl = float(clf_skl.intercept_[0])
    y_pred_skl = clf_skl.predict(X_test_scaled)
    acc_skl = accuracy_score(y_test, y_pred_skl)
    print(f"sklearn accuracy: {acc_skl:.4f}")
    print(f"sklearn coef_ (first 5): {coef_skl[:5]}")
    print(f"sklearn intercept_: {intercept_skl:.6f}")
    
    # rehline plq_Ridge_Classifier
    print(f"\n--- rehline plq_Ridge_Classifier ---")
    clf_reh = plq_Ridge_Classifier(
        loss={'name': 'svm'},
        C=C,
        max_iter=1000000,
        tol=1e-5,
        verbose=0
    )
    clf_reh.fit(X_train_scaled, y_train)
    coef_reh = clf_reh.coef_.flatten()
    intercept_reh = float(clf_reh.intercept_)
    y_pred_reh = clf_reh.predict(X_test_scaled)
    acc_reh = accuracy_score(y_test, y_pred_reh)
    print(f"rehline accuracy: {acc_reh:.4f}")
    print(f"rehline coef_ (first 5): {coef_reh[:5]}")
    print(f"rehline intercept_: {intercept_reh:.6f}")
    
    # Compare coefficients
    print(f"\n--- Coefficient Comparison ---")
    print("=" * 70)
    print(f"{'Index':^8} {'sklearn':^20} {'rehline':^20} {'diff':^10}")
    print("=" * 70)
    max_coef_diff = 0
    for i, (s, r) in enumerate(zip(coef_skl, coef_reh)):
        diff = abs(s - r)
        max_coef_diff = max(max_coef_diff, diff)
        print(f"{i:^8d} {s:^20.8f} {r:^20.8f} {diff:^10.2e}")
    print("=" * 70)
    print(f"Max coef difference: {max_coef_diff:.6e}")
    print(f"Intercept difference: {abs(intercept_skl - intercept_reh):.6e}")
    print(f"Accuracy: sklearn={acc_skl:.4f}, rehline={acc_reh:.4f}")
    
    # Solutions must match within tol=1e-3
    if max_coef_diff > 1e-3:
        print(f"\n❌ FAIL: Max coef difference {max_coef_diff:.6e} > 1e-3")
        print(f"   This indicates a discrepancy between rehline and sklearn binary SVM!")
    assert max_coef_diff <= 1e-3, \
        f"Binary coef_ difference {max_coef_diff:.6e} > 1e-3. " \
        f"sklearn={coef_skl}, rehline={coef_reh}"
    
    print("\n✓ Binary vs sklearn test passed!")
    return acc_skl, acc_reh, max_coef_diff


def test_multiclass_ovr_vs_sklearn():
    """
    Test One-vs-Rest multiclass classification against sklearn's LinearSVC (OvR).
    Both use tol=1e-5. Solutions (coef_) must match within tol=1e-3.
    """
    print("\n" + "="*60)
    print("Test 2: OvR Multi-class vs sklearn LinearSVC (OvR)")
    print("="*60)
    
    # Generate multiclass dataset (controlled size ~5000 samples)
    np.random.seed(42)
    n_samples = 5000
    n_features = 10
    n_classes = 4
    C = 1.0
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        n_classes=n_classes,
        class_sep=1.5,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    print(f"C={C}, tol=1e-5")
    
    # sklearn LinearSVC with OvR (default)
    print(f"\n--- sklearn LinearSVC (OvR) ---")
    clf_skl = LinearSVC(
        C=C,
        loss='hinge',
        multi_class='ovr',
        fit_intercept=True,
        max_iter=1000000,
        tol=1e-5,
        random_state=42
    )
    clf_skl.fit(X_train_scaled, y_train)
    coef_skl = clf_skl.coef_  # shape (n_classes, n_features)
    intercept_skl = clf_skl.intercept_  # shape (n_classes,)
    y_pred_skl = clf_skl.predict(X_test_scaled)
    acc_skl = accuracy_score(y_test, y_pred_skl)
    print(f"sklearn OvR accuracy: {acc_skl:.4f}")
    print(f"sklearn coef_ shape: {coef_skl.shape}")
    
    # rehline plq_Ridge_Classifier with OvR
    print(f"\n--- rehline plq_Ridge_Classifier (OvR) ---")
    clf_reh = plq_Ridge_Classifier(
        loss={'name': 'svm'},
        C=C,
        multi_class='ovr',
        max_iter=1000000,
        tol=1e-5,
        verbose=0
    )
    clf_reh.fit(X_train_scaled, y_train)
    coef_reh = clf_reh.coef_  # shape (n_classes, n_features)
    intercept_reh = clf_reh.intercept_  # shape (n_classes,)
    y_pred_reh = clf_reh.predict(X_test_scaled)
    acc_reh = accuracy_score(y_test, y_pred_reh)
    print(f"rehline OvR accuracy: {acc_reh:.4f}")
    print(f"rehline coef_ shape: {coef_reh.shape}")
    
    # Compare coefficients for each class
    print(f"\n--- Coefficient Comparison (per class) ---")
    max_coef_diff = 0
    for k in range(n_classes):
        diff = np.max(np.abs(coef_skl[k] - coef_reh[k]))
        max_coef_diff = max(max_coef_diff, diff)
        print(f"Class {k}: max coef diff = {diff:.6e}, intercept diff = {abs(float(intercept_skl[k]) - float(intercept_reh[k])):.6e}")
    
    print(f"\nOverall max coef difference: {max_coef_diff:.6e}")
    print(f"Accuracy: sklearn={acc_skl:.4f}, rehline={acc_reh:.4f}")
    
    # Verify shapes
    assert coef_reh.shape == (n_classes, n_features), \
        f"Expected coef_ shape ({n_classes}, {n_features}), got {coef_reh.shape}"
    
    # Solutions must match within tol=1e-3
    if max_coef_diff > 1e-3:
        print(f"\n❌ FAIL: Max coef difference {max_coef_diff:.6e} > 1e-3")
        print(f"   This indicates a discrepancy between rehline and sklearn OvR!")
    assert max_coef_diff <= 1e-3, \
        f"OvR coef_ difference {max_coef_diff:.6e} > 1e-3. " \
        f"This indicates a discrepancy between rehline and sklearn OvR!"
    
    print("\n✓ OvR vs sklearn test passed!")
    return acc_skl, acc_reh, max_coef_diff


def test_multiclass_ovo_vs_sklearn():
    """
    Test One-vs-One multiclass classification against sklearn's OneVsOneClassifier.
    Both use tol=1e-5. Solutions (coef_) must match within tol=1e-3.
    """
    print("\n" + "="*60)
    print("Test 3: OvO Multi-class vs sklearn OneVsOneClassifier")
    print("="*60)
    
    # Generate multiclass dataset (controlled size ~3000 samples for OvO)
    np.random.seed(42)
    n_samples = 3000
    n_features = 8
    n_classes = 3
    C = 1.0
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        n_classes=n_classes,
        class_sep=1.5,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    print(f"C={C}, tol=1e-5")
    
    # sklearn OneVsOneClassifier with LinearSVC
    print(f"\n--- sklearn OneVsOneClassifier(LinearSVC) ---")
    base_clf = LinearSVC(
        C=C,
        loss='hinge',
        fit_intercept=True,
        max_iter=1000000,
        tol=1e-5,
        random_state=42
    )
    clf_skl_ovo = OneVsOneClassifier(base_clf)
    clf_skl_ovo.fit(X_train_scaled, y_train)
    y_pred_skl = clf_skl_ovo.predict(X_test_scaled)
    acc_skl = accuracy_score(y_test, y_pred_skl)
    print(f"sklearn OvO accuracy: {acc_skl:.4f}")
    
    # Extract sklearn OvO coefficients
    n_estimators = n_classes * (n_classes - 1) // 2
    coef_skl_list = []
    intercept_skl_list = []
    for est in clf_skl_ovo.estimators_:
        coef_skl_list.append(est.coef_.flatten())
        intercept_skl_list.append(float(est.intercept_[0]))
    coef_skl = np.array(coef_skl_list)  # shape (n_estimators, n_features)
    intercept_skl = np.array(intercept_skl_list)  # shape (n_estimators,)
    print(f"sklearn coef_ shape: {coef_skl.shape}")
    
    # rehline plq_Ridge_Classifier with OvO
    print(f"\n--- rehline plq_Ridge_Classifier (OvO) ---")
    clf_reh = plq_Ridge_Classifier(
        loss={'name': 'svm'},
        C=C,
        multi_class='ovo',
        max_iter=1000000,
        tol=1e-5,
        verbose=0
    )
    clf_reh.fit(X_train_scaled, y_train)
    coef_reh = clf_reh.coef_  # shape (n_estimators, n_features)
    intercept_reh = clf_reh.intercept_  # shape (n_estimators,)
    y_pred_reh = clf_reh.predict(X_test_scaled)
    acc_reh = accuracy_score(y_test, y_pred_reh)
    print(f"rehline OvO accuracy: {acc_reh:.4f}")
    print(f"rehline coef_ shape: {coef_reh.shape}")
    
    # Compare coefficients for each binary classifier
    print(f"\n--- Coefficient Comparison (per binary classifier) ---")
    max_coef_diff = 0
    for k in range(n_estimators):
        diff = np.max(np.abs(coef_skl[k] - coef_reh[k]))
        max_coef_diff = max(max_coef_diff, diff)
        print(f"Classifier {k}: max coef diff = {diff:.6e}, intercept diff = {abs(float(intercept_skl[k]) - float(intercept_reh[k])):.6e}")
    
    print(f"\nOverall max coef difference: {max_coef_diff:.6e}")
    print(f"Accuracy: sklearn={acc_skl:.4f}, rehline={acc_reh:.4f}")
    
    # Verify shapes
    assert coef_reh.shape == (n_estimators, n_features), \
        f"Expected coef_ shape ({n_estimators}, {n_features}), got {coef_reh.shape}"
    
    # Solutions must match within tol=1e-3
    if max_coef_diff > 1e-3:
        print(f"\n❌ FAIL: Max coef difference {max_coef_diff:.6e} > 1e-3")
        print(f"   This indicates a discrepancy between rehline and sklearn OvO!")
    assert max_coef_diff <= 1e-3, \
        f"OvO coef_ difference {max_coef_diff:.6e} > 1e-3. " \
        f"This indicates a discrepancy between rehline and sklearn OvO!"
    
    print("\n✓ OvO vs sklearn test passed!")
    return acc_skl, acc_reh, max_coef_diff


def test_decision_function_shapes():
    """Test that decision_function returns correct shapes."""
    print("\n" + "="*60)
    print("Test 4: Decision Function Shapes")
    print("="*60)
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    
    # Test binary case
    y_binary = np.random.randint(0, 2, n_samples)
    clf_binary = plq_Ridge_Classifier(loss={'name': 'svm'}, C=1.0, tol=1e-5)
    clf_binary.fit(X, y_binary)
    
    scores_binary = clf_binary.decision_function(X)
    assert scores_binary.shape == (n_samples,), \
        f"Binary: Expected shape ({n_samples},), got {scores_binary.shape}"
    print(f"Binary decision_function shape: {scores_binary.shape} ✓")
    
    # Test OvR case
    y_multi = np.random.randint(0, 4, n_samples)
    clf_ovr = plq_Ridge_Classifier(loss={'name': 'svm'}, C=1.0, multi_class='ovr', tol=1e-5)
    clf_ovr.fit(X, y_multi)
    
    scores_ovr = clf_ovr.decision_function(X)
    assert scores_ovr.shape == (n_samples, 4), \
        f"OvR: Expected shape ({n_samples}, 4), got {scores_ovr.shape}"
    print(f"OvR decision_function shape: {scores_ovr.shape} ✓")
    
    # Test OvO case
    clf_ovo = plq_Ridge_Classifier(loss={'name': 'svm'}, C=1.0, multi_class='ovo', tol=1e-5)
    clf_ovo.fit(X, y_multi)
    
    scores_ovo = clf_ovo.decision_function(X)
    n_estimators = 4 * 3 // 2  # 6
    assert scores_ovo.shape == (n_samples, n_estimators), \
        f"OvO: Expected shape ({n_samples}, {n_estimators}), got {scores_ovo.shape}"
    print(f"OvO decision_function shape: {scores_ovo.shape} ✓")
    
    print("\n✓ All decision_function shape tests passed!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-CLASS CLASSIFICATION TEST SUITE")
    print("Testing PR #55cc374: add multi-class classification function")
    print("Comparing with sklearn's LinearSVC (OvR and OvO)")
    print("All algorithms use tol=1e-5. Solutions must match within tol=1e-3.")
    print("="*70)
    
    # Run all tests
    acc_skl_bin, acc_reh_bin, diff_bin = test_binary_vs_sklearn()
    acc_skl_ovr, acc_reh_ovr, diff_ovr = test_multiclass_ovr_vs_sklearn()
    acc_skl_ovo, acc_reh_ovo, diff_ovo = test_multiclass_ovo_vs_sklearn()
    test_decision_function_shapes()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"{'Test':^30} {'sklearn acc':^12} {'rehline acc':^12} {'max coef diff':^15}")
    print("-" * 70)
    print(f"{'Binary Classification':^30} {acc_skl_bin:^12.4f} {acc_reh_bin:^12.4f} {diff_bin:^15.2e}")
    print(f"{'OvR Multi-class':^30} {acc_skl_ovr:^12.4f} {acc_reh_ovr:^12.4f} {diff_ovr:^15.2e}")
    print(f"{'OvO Multi-class':^30} {acc_skl_ovo:^12.4f} {acc_reh_ovo:^12.4f} {diff_ovo:^15.2e}")
    print("="*70)
    print("\n✓ All tests passed successfully!")
