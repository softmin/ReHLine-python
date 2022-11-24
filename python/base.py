"""Base functions for ReHLine."""

# Authors: Ben Dai <bendai@cuhk.edu.hk>
#          Yixuan Qiu <yixuanq@gmail.com>

# License: MIT License

import numpy as np
from scipy.special import huber 

def relu(x):
    """
    Evaluation of ReLU given a vector

    Parameters
    ----------

    x: {array-like} of shape (n_samples, )
    Training vector, where `n_samples` is the number of samples

    """
    return np.maximum(x, 0)


def rehu(x, cut=1):
    """
    Evaluation of ReHU given a vector

    Parameters
    ----------

    x: {array-like} of shape (n_samples, )
        Training vector, where `n_samples` is the number of samples

    cut: {array-like} of shape (n_samples, )
        Cutpoints of ReHU, where `n_samples` is the number of samples

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

