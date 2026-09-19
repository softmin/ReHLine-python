"""Zero is handled like sklearn's linear classifiers in binary and OvO fits."""

import numpy as np
import pytest
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

from rehline import plq_ElasticNet_Classifier, plq_Ridge_Classifier


@pytest.mark.parametrize("estimator", [plq_Ridge_Classifier, plq_ElasticNet_Classifier])
@pytest.mark.parametrize("count", [2, 3, 4, 13])
def test_zero_margin_fits_match_sklearn_ovo_wrapper(estimator, count):
    X, y = np.zeros((count * 4, 2)), np.repeat(np.arange(count), 4)
    options = dict(loss={"name": "svm"}, fit_intercept=False, max_iter=100000, tol=1e-9)
    model = estimator(multi_class="ovo", **options).fit(X, y)
    reference = OneVsOneClassifier(estimator(**options)).fit(X, y)
    np.testing.assert_array_equal(model.coef_, 0)
    np.testing.assert_array_equal(model.decision_function(X), reference.decision_function(X))
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))
    np.testing.assert_array_equal(model.predict(X), y[0])
    before = model.objective_.copy() if count > 2 else model.objective_
    model.set_params(decision_function_shape="ovo")
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))
    np.testing.assert_array_equal(model.objective_, before)
    # All zero features imply beta=0 and the original objective is C times
    # the number of observations in each binary problem.
    ratio = getattr(model, "l1_ratio", 0)
    np.testing.assert_allclose(model.objective_ * (1 - ratio), 8.0, atol=1e-10)


@pytest.mark.parametrize("estimator", [plq_Ridge_Classifier, plq_ElasticNet_Classifier])
def test_binary_signed_zero_and_near_zero_follow_linear_sklearn(estimator):
    model = estimator(loss={"name": "svm"}, fit_intercept=False).fit([[-1.0], [1.0]], ["a", "b"])
    model.coef_, model.intercept_ = np.array([1.0]), 0.0
    reference = LinearSVC(fit_intercept=False).fit([[-1.0], [1.0]], ["a", "b"])
    reference.coef_, reference.intercept_ = np.array([[1.0]]), np.array([0.0])
    # The smallest normal values are finite nonzero decisions: do not use a
    # numerical epsilon to widen the zero tie or change optimization outputs.
    tiny = np.finfo(float).tiny
    X = np.array([[-tiny], [-0.0], [0.0], [tiny]])
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))
    np.testing.assert_array_equal(model.predict(X), ["a", "a", "a", "b"])
