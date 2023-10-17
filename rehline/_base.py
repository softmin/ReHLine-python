"""Base functions for ReHLine."""

# Authors: Ben Dai <bendai@cuhk.edu.hk>
#          Yixuan Qiu <yixuanq@gmail.com>

# License: MIT License

import numpy as np
from scipy.special import huber
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

def relu(x):
    """
    Evaluation of ReLU given a vector.

    Parameters
    ----------

    x: {array-like} of shape (n_samples, )
    Training vector, where `n_samples` is the number of samples


    Returns
    -------
    array of shape (n_samples, )
        An array with ReLU applied, i.e., all negative values are replaced with 0.

    """
    return np.maximum(x, 0)


def rehu(x, cut=1):
    """
    Evaluation of ReHU given a vector.

    Parameters
    ----------

    x: {array-like} of shape (n_samples, )
        Training vector, where `n_samples` is the number of samples

    cut: {array-like} of shape (n_samples, )
        Cutpoints of ReHU, where `n_samples` is the number of samples

    Returns
    -------
    array of shape (n_samples, ) 
        The result of the ReHU function.

    """
    n_samples = x.shape[0]
    cut = cut * np.ones_like(x)

    u = np.maximum(x, 0)
    return huber(cut, u)

def _check_relu(relu_coef, relu_intercept):
    assert relu_coef.shape == relu_intercept.shape, "`relu_coef` and `relu_intercept` should be the same shape!"

def _check_rehu(rehu_coef, rehu_intercept, rehu_cut):
    assert rehu_coef.shape == rehu_intercept.shape, "`rehu_coef` and `rehu_intercept` should be the same shape!"
    if len(rehu_coef) > 0:
        assert (rehu_cut >= 0.0).all(), "`rehu_cut` must be non-negative!"

def make_fair_classification(n_samples=100, n_features=5, ind_sensitive=0):
    """
    Generate a random binary fair classification problem.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=5
        The total number of features. 

    ind_sensitive : int, default=0
        The index of the sensitive feature.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The +/- labels for class membership of each sample.

    X_sen: ndarray of shape (n_samples,)
        The centered samples of the sensitive feature.
    """

    X, y = make_classification(n_samples, n_features)
    y = 2*y - 1

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_sen = X[:, ind_sensitive]

    return X, y, X_sen