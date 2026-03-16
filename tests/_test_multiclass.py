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


def test_ovo_coef_sign_convention():
    """
    Test 5: OvO coefficient sign convention (regression test for the sign bug).

    The previous bug assigned cls_i -> +1 and cls_j -> -1 in each OvO subproblem,
    which is opposite to sklearn's LabelEncoder convention (cls_i -> -1, cls_j -> +1)
    since combinations() always yields sorted pairs (cls_i < cls_j).
    This caused every subproblem's coef_ to be fully negated (diff ≈ 2 * |β|).

    This test directly checks the sign direction of each OvO subproblem's coef_
    via dot product, rather than relying solely on accuracy, so the bug cannot
    silently reappear.
    """
    print("\n" + "="*60)
    print("Test 5: OvO Coefficient Sign Convention")
    print("="*60)

    np.random.seed(0)
    n_samples = 2000
    n_features = 6
    n_classes = 3
    C = 1.0

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=1,
        n_classes=n_classes,
        class_sep=2.0,
        random_state=0
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # sklearn OvO reference
    base_clf = LinearSVC(C=C, loss='hinge', fit_intercept=True,
                         max_iter=1000000, tol=1e-5, random_state=0)
    clf_skl = OneVsOneClassifier(base_clf)
    clf_skl.fit(X, y)

    # rehline OvO
    clf_reh = plq_Ridge_Classifier(
        loss={'name': 'svm'}, C=C, multi_class='ovo',
        max_iter=1000000, tol=1e-5, verbose=0
    )
    clf_reh.fit(X, y)

    n_estimators = n_classes * (n_classes - 1) // 2
    print(f"\n{'Estimator':^12} {'dot(skl,reh)':^16} {'||skl||':^12} {'||reh||':^12} {'sign OK':^10}")
    print("-" * 65)

    all_positive_dot = True
    for k, est in enumerate(clf_skl.estimators_):
        coef_skl = est.coef_.flatten()
        coef_reh = clf_reh.coef_[k]
        dot = np.dot(coef_skl, coef_reh)
        norm_skl = np.linalg.norm(coef_skl)
        norm_reh = np.linalg.norm(coef_reh)
        # If signs agree the dot product is positive; if reversed it is negative.
        sign_ok = dot > 0
        all_positive_dot = all_positive_dot and sign_ok
        print(f"{k:^12d} {dot:^16.4f} {norm_skl:^12.4f} {norm_reh:^12.4f} {'✓' if sign_ok else '❌':^10}")

    assert all_positive_dot, \
        "OvO coef_ sign convention mismatch: at least one subproblem has reversed sign. " \
        "This is the sign-convention bug (cls_i/cls_j label encoding mismatch)."

    print("\n✓ OvO sign convention test passed!")


def test_ovo_predict_consistency():
    """
    Test 6: OvO predict / decision_function consistency.

    Verifies that predict() produces exactly the same result as manually
    reconstructing predictions from decision_function() using the voting logic,
    ensuring the sign convention in fit and predict are perfectly aligned.
    """
    print("\n" + "="*60)
    print("Test 6: OvO predict / decision_function Consistency")
    print("="*60)

    np.random.seed(7)
    n_samples = 1500
    n_features = 5
    n_classes = 4
    C = 1.0

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=0,
        n_classes=n_classes,
        class_sep=1.5,
        random_state=7
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = plq_Ridge_Classifier(
        loss={'name': 'svm'}, C=C, multi_class='ovo',
        max_iter=1000000, tol=1e-5, verbose=0
    )
    clf.fit(X, y)

    # Predictions from predict()
    y_pred = clf.predict(X)

    # Manually reconstruct predictions from decision_function (mirrors predict internals)
    scores = clf.decision_function(X)
    n_cls = len(clf.classes_)
    votes = np.zeros((n_samples, n_cls))
    confidences = np.zeros((n_samples, n_cls))
    for k, (_, _, cls_i, cls_j) in enumerate(clf.estimators_):
        i = np.where(clf.classes_ == cls_i)[0][0]
        j = np.where(clf.classes_ == cls_j)[0][0]
        pred = (scores[:, k] > 0).astype(int)
        votes[:, j] += pred
        votes[:, i] += 1 - pred
        confidences[:, j] += scores[:, k]
        confidences[:, i] -= scores[:, k]
    transformed = confidences / (3 * (np.abs(confidences) + 1))
    y_manual = clf.classes_[np.argmax(votes + transformed, axis=1)]

    n_disagree = np.sum(y_pred != y_manual)
    print(f"Disagreements between predict() and manual reconstruction: {n_disagree}")

    assert n_disagree == 0, \
        f"predict() and decision_function() are inconsistent: {n_disagree} samples disagree. " \
        "This indicates a mismatch between the sign convention in fit and predict."

    print("✓ OvO predict / decision_function consistency test passed!")


def test_ovo_fit_intercept_false():
    """
    Test 7: OvO with fit_intercept=False — correct coef_ shape and accuracy.

    Ensures that disabling the intercept still produces the correct coef_ shape,
    sets intercept_ to all zeros, and matches sklearn's solution.
    """
    print("\n" + "="*60)
    print("Test 7: OvO with fit_intercept=False")
    print("="*60)

    np.random.seed(13)
    n_samples = 2000
    n_features = 6
    n_classes = 3
    C = 1.0

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=1,
        n_classes=n_classes,
        class_sep=2.0,
        random_state=13
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # sklearn OvO, no intercept
    base_clf = LinearSVC(C=C, loss='hinge', fit_intercept=False,
                         max_iter=1000000, tol=1e-5, random_state=13)
    clf_skl = OneVsOneClassifier(base_clf)
    clf_skl.fit(X, y)

    # rehline OvO, no intercept
    clf_reh = plq_Ridge_Classifier(
        loss={'name': 'svm'}, C=C, multi_class='ovo',
        fit_intercept=False, max_iter=1000000, tol=1e-5, verbose=0
    )
    clf_reh.fit(X, y)

    n_estimators = n_classes * (n_classes - 1) // 2

    # Shape checks
    assert clf_reh.coef_.shape == (n_estimators, n_features), \
        f"Expected coef_ shape ({n_estimators}, {n_features}), got {clf_reh.coef_.shape}"
    assert np.all(clf_reh.intercept_ == 0.0), \
        "intercept_ should be all zeros when fit_intercept=False"

    # Accuracy checks
    max_diff = 0
    for k, est in enumerate(clf_skl.estimators_):
        diff = np.max(np.abs(est.coef_.flatten() - clf_reh.coef_[k]))
        max_diff = max(max_diff, diff)
        print(f"Estimator {k}: max coef diff = {diff:.6e}")

    print(f"Overall max coef diff: {max_diff:.6e}")
    assert max_diff <= 1e-3, \
        f"fit_intercept=False OvO coef_ diff {max_diff:.6e} > 1e-3"

    print("✓ OvO fit_intercept=False test passed!")


def test_multiclass_invalid_multi_class():
    """
    Test 8: Invalid multi_class parameter should raise ValueError.

    Ensures that passing an unrecognised multi_class value causes fit() to raise
    a clear ValueError rather than silently failing or producing wrong results.
    """
    print("\n" + "="*60)
    print("Test 8: Invalid multi_class Parameter")
    print("="*60)

    np.random.seed(42)
    X = np.random.randn(200, 4)
    y = np.random.randint(0, 3, 200)

    clf = plq_Ridge_Classifier(
        loss={'name': 'svm'}, C=1.0, multi_class='invalid_option'
    )

    raised = False
    try:
        clf.fit(X, y)
    except ValueError as e:
        raised = True
        print(f"ValueError raised as expected: {e}")

    assert raised, "Expected ValueError for invalid multi_class parameter, but none was raised."
    print("✓ Invalid multi_class parameter test passed!")


def test_ovo_more_classes():
    """
    Test 9: OvO correctness with 5 classes (10 subproblems).

    Verifies that the number of subproblems, coef_ shape, and coefficient
    accuracy are all correct when the number of classes grows, guarding against
    errors in the combinatorial subproblem construction logic.
    """
    print("\n" + "="*60)
    print("Test 9: OvO with 5 Classes (10 subproblems)")
    print("="*60)

    np.random.seed(99)
    n_samples = 3000
    n_features = 8
    n_classes = 5
    C = 1.0
    n_estimators = n_classes * (n_classes - 1) // 2  # 10

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=1,
        n_classes=n_classes,
        class_sep=1.5,
        random_state=99
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=99, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # sklearn
    base_clf = LinearSVC(C=C, loss='hinge', fit_intercept=True,
                         max_iter=1000000, tol=1e-5, random_state=99)
    clf_skl = OneVsOneClassifier(base_clf)
    clf_skl.fit(X_train, y_train)
    acc_skl = accuracy_score(y_test, clf_skl.predict(X_test))

    # rehline
    clf_reh = plq_Ridge_Classifier(
        loss={'name': 'svm'}, C=C, multi_class='ovo',
        max_iter=1000000, tol=1e-5, verbose=0
    )
    clf_reh.fit(X_train, y_train)
    acc_reh = accuracy_score(y_test, clf_reh.predict(X_test))

    # 形状检查
    assert clf_reh.coef_.shape == (n_estimators, n_features), \
        f"Expected coef_ shape ({n_estimators}, {n_features}), got {clf_reh.coef_.shape}"
    assert clf_reh.intercept_.shape == (n_estimators,), \
        f"Expected intercept_ shape ({n_estimators},), got {clf_reh.intercept_.shape}"
    assert len(clf_reh.estimators_) == n_estimators, \
        f"Expected {n_estimators} estimators, got {len(clf_reh.estimators_)}"

    # 精度检查
    max_diff = 0
    for k, est in enumerate(clf_skl.estimators_):
        diff = np.max(np.abs(est.coef_.flatten() - clf_reh.coef_[k]))
        max_diff = max(max_diff, diff)
    print(f"5-class OvO: {n_estimators} subproblems, max coef diff = {max_diff:.6e}")
    print(f"Accuracy: sklearn={acc_skl:.4f}, rehline={acc_reh:.4f}")

    assert max_diff <= 1e-3, \
        f"5-class OvO coef_ diff {max_diff:.6e} > 1e-3"

    print("✓ OvO 5-class test passed!")
    return acc_skl, acc_reh, max_diff


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
    test_ovo_coef_sign_convention()
    test_ovo_predict_consistency()
    test_ovo_fit_intercept_false()
    test_multiclass_invalid_multi_class()
    acc_skl_ovo5, acc_reh_ovo5, diff_ovo5 = test_ovo_more_classes()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"{'Test':^30} {'sklearn acc':^12} {'rehline acc':^12} {'max coef diff':^15}")
    print("-" * 70)
    print(f"{'Binary Classification':^30} {acc_skl_bin:^12.4f} {acc_reh_bin:^12.4f} {diff_bin:^15.2e}")
    print(f"{'OvR Multi-class':^30} {acc_skl_ovr:^12.4f} {acc_reh_ovr:^12.4f} {diff_ovr:^15.2e}")
    print(f"{'OvO Multi-class (3cls)':^30} {acc_skl_ovo:^12.4f} {acc_reh_ovo:^12.4f} {diff_ovo:^15.2e}")
    print(f"{'OvO Multi-class (5cls)':^30} {acc_skl_ovo5:^12.4f} {acc_reh_ovo5:^12.4f} {diff_ovo5:^15.2e}")
    print(f"{'Decision Func Shapes':^30} {'—':^12} {'—':^12} {'—':^15}")
    print(f"{'OvO Sign Convention':^30} {'—':^12} {'—':^12} {'—':^15}")
    print(f"{'OvO Predict Consistency':^30} {'—':^12} {'—':^12} {'—':^15}")
    print(f"{'OvO No Intercept':^30} {'—':^12} {'—':^12} {'—':^15}")
    print(f"{'Invalid multi_class':^30} {'—':^12} {'—':^12} {'—':^15}")
    print("="*70)
    print("\n✓ All 9 tests passed successfully!")