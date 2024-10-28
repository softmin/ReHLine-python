""" ReHLoss: Convert a piecewise quadratic loss function to a ReHLoss. """

# Authors: Ben Dai <bendai@cuhk.edu.hk>
#          Yixuan Qiu <qiuyixuan@sufe.edu.cn>

# License: MIT License


import numpy as np

from ._base import _check_rehu, _check_relu, _rehu, _relu


class ReHLoss(object):
    """
    A ReHLine loss function composed of one or multiple ReLU and ReHU components.
    
    Parameters
    ----------

    relu_coef : {array-like} of shape (n_relu, n_samples)
        ReLU coeff matrix, where `n_loss` is the number of losses and
        `n_relu` is the number of relus.

    relu_intercept : {array-like} of shape (n_relu, n_samples)
        ReLU intercept matrix, where `n_loss` is the number of losses and
        `n_relu` is the number of relus.

    rehu_coef : {array-like} of shape (n_rehu, n_samples)
        ReHU coeff matrix, where `n_loss` is the number of losses and
        `n_relu` is the number of rehus.

    rehu_intercept : {array-like} of shape (n_rehu, n_samples)
        ReHU coeff matrix, where `n_loss` is the number of losses and
        `n_relu` is the number of rehus.

    rehu_cut : {array-like} of shape (n_rehu, n_samples)
        ReHU cutpoints matrix, where `n_loss` is the number of losses and
        `n_relu` is the number of rehus.

    Example
    -------

    >>> import numpy as np
    >>> L, H, n = 3, 2, 100
    >>> relu_coef, relu_intercept = np.random.randn(L,n), np.random.randn(L,n)
    >>> rehu_cut, rehu_coef, rehu_intercept = abs(np.random.randn(H,n)), np.random.randn(H,n), np.random.randn(H,n)
    >>> x = np.random.randn(n)
    >>> random_loss = ReHLoss(relu_coef, relu_intercept, rehu_coef, rehu_intercept, rehu_cut)
    >>> random_loss(x)
    """

    def __init__(self, relu_coef, relu_intercept,
                       rehu_coef=np.empty(shape=(0,0)), rehu_intercept=np.empty(shape=(0,0)), rehu_cut=1):
        self.relu_coef = relu_coef
        self.relu_intercept = relu_intercept
        self.rehu_cut = rehu_cut * np.ones_like(rehu_coef)
        self.rehu_coef = rehu_coef
        self.rehu_intercept = rehu_intercept
        self.H = rehu_coef.shape[0]
        self.L = relu_coef.shape[0]
        self.n = relu_coef.shape[1]

    def __call__(self, x):
        """Evaluate ReHLoss given a data matrix

        x: {array-like} of shape (n_samples, )
            Training vector, where `n_samples` is the number of samples
        """
        if (self.L > 0) and (self.H > 0):
            assert self.relu_coef.shape[1] == self.rehu_coef.shape[1], "n_samples for `relu_coef` and `rehu_coef` should be the same shape!"

        _check_relu(self.relu_coef, self.relu_intercept)
        _check_rehu(self.rehu_coef, self.rehu_intercept, self.rehu_cut)

        self.L, self.H, self.n = self.relu_coef.shape[0], self.rehu_coef.shape[0], self.relu_coef.shape[1]

        ans = 0
        if len(self.relu_coef) > 0:
            relu_input = (self.relu_coef.T * x[:,np.newaxis]).T + self.relu_intercept 
            ans += np.sum(relu(relu_input), 0).sum()
        if len(self.rehu_coef) > 0:
            rehu_input = (self.rehu_coef.T * x[:,np.newaxis]).T + self.rehu_intercept
            ans += np.sum(rehu(rehu_input, cut=self.rehu_cut), 0).sum()

        return ans
