""" ReHLoss: Convert a piecewise quadratic loss function to a ReHLoss. """

# Authors: Ben Dai <bendai@cuhk.edu.hk>
#          Yixuan Qiu <yixuanq@gmail.com>

# License: MIT License


import numpy as np
import base

class ReHLoss(object):
    """
    A series of ReHLoss

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

        base._check_relu(self.relu_coef, self.relu_intercept)
        base._check_rehu(self.rehu_coef, self.rehu_intercept, self.rehu_cut)

        self.L, self.H, self.n = self.relu_coef.shape[0], self.rehu_coef.shape[0], self.relu_coef.shape[1]
        relu_input = (self.relu_coef.T * x[:,np.newaxis]).T + self.relu_intercept
        rehu_input = (self.rehu_coef.T * x[:,np.newaxis]).T + self.rehu_intercept

        return np.sum(base.relu(relu_input), 0) + np.sum(base.rehu(rehu_input), 0)


class PQLoss(object):
    """ PQLoss: continuous convex piecewise quandratic function (with a function converting to ReHLoss).

    Parameters
    ----------

    cutpoints : list of cutpoints
        cutpoints of the PQLoss, except -np.inf and np.inf

    quad_coef : {dict-like} of {'a': [], 'b': [], 'c': []}
        The quandratic coefficients in pieces of the PQLoss
        The i-th piece Q is: a[i]**2 * x**2 + b[i] * x + c[i]

    Example
    -------
    >>> import numpy as np
    >>> cutpoints = [0., 1.]
    >>> quad_coef = {'a': np.array([0., .5, 0.]), 'b': np.array([-1, 0., 1]), 'c': np.array([0., 0., -.5])}
    >>> test_loss = PQLoss(cutpoints, quad_coef)
    >>> x = np.arange(-2,2,.05)
    >>> test_loss(x)
    """


    def __init__(self, cutpoints, quad_coef):
        self.cutpoints = np.concatenate(([-np.inf], cutpoints, [np.inf]))
        self.quad_coef = quad_coef
        self.n_pieces = len(self.cutpoints) - 1
        self.min = 0

    def __call__(self, x):
        """ Evaluation of PQLoss

        out = quad_coef['a'][i]*x**2 + quad_coef['b'][i]*x + quad_coef['c'][i], if cutpoints[i] < x < cutpoints[i+1]
        """
        condlist, funclist = [], []
        x = np.array(x)
        assert len(self.quad_coef['a']) == self.n_pieces, "`cutpoints` and `quad_coef` are mismatched."
        assert len(self.quad_coef['b']) == self.n_pieces, "`cutpoints` and `quad_coef` are mismatched."
        assert len(self.quad_coef['c']) == self.n_pieces, "`cutpoints` and `quad_coef` are mismatched."

        y = np.zeros_like(x)
        for i in range(self.n_pieces):
            cond_tmp = (x > self.cutpoints[i])&(x <= self.cutpoints[i+1])
            y[cond_tmp] = self.quad_coef['a'][i]*x[cond_tmp]**2 + self.quad_coef['b'][i]*x[cond_tmp] + self.quad_coef['c'][i]
        return y

    def _2ReHLoss(self):
        relu_coef, relu_intercept = [], []
        rehu_coef, rehu_intercept, rehu_cut = [], [], []
        quad_coef = self.quad_coef.copy()
        cutpoints = self.cutpoints.copy()
        cutpoints = cutpoints[1:-1]

        ## evaluate cutpoints
        out_cut = self(cutpoints)
        ## move the loss function to the x-axis;
        if min(out_cut) > 0:
            self.min = min(out_cut)
            quad_coef['c'] -= self.min
            out_cut -= self.min

        ## remove a ReLU/ReHU function from this point; i-th point -> i-th or (i+1)-th interval
        ind_tmp = np.argmax(out_cut == 0.)
        ## Left
        if quad_coef['a'][ind_tmp] == 0.:
            ## +relu
            relu_coef.append(quad_coef['b'][ind_tmp])
            relu_intercept.append(quad_coef['c'][ind_tmp])
            if len(quad_coef['b'][:ind_tmp]) > 0:
                # remove relu for left intervals
                quad_coef['b'][:ind_tmp] -= quad_coef['b'][ind_tmp]
                quad_coef['c'][:ind_tmp] -= quad_coef['c'][ind_tmp]
            quad_coef['b'][ind_tmp], quad_coef['c'][ind_tmp] = 0., 0.
        else:
            ## +rehu
            if ind_tmp == 0:
                rehu_coef.append(-np.sqrt(2*quad_coef['a'][ind_tmp]))
                rehu_intercept.append(-quad_coef['b'][ind_tmp]/np.sqrt(2*quad_coef['a'][ind_tmp]))
                rehu_cut.append(np.inf)
            else:
                rehu_coef.append(-np.sqrt(2*quad_coef['a'][ind_tmp]))
                rehu_intercept.append(-quad_coef['b'][ind_tmp]/np.sqrt(2*quad_coef['a'][ind_tmp]))
                rehu_cut.append(-np.sqrt(2*quad_coef['a'][ind_tmp])*cutpoints[ind_tmp-1] - quad_coef['b'][ind_tmp]/np.sqrt(2*quad_coef['a'][ind_tmp]))
            if len(quad_coef['a'][:ind_tmp]) > 0:
                quad_coef['b'][:ind_tmp] -= 2*quad_coef['a'][ind_tmp]*cutpoints[ind_tmp] + quad_coef['b'][ind_tmp]
                quad_coef['c'][:ind_tmp] -= - quad_coef['a'][ind_tmp]*cutpoints[ind_tmp]**2 + quad_coef['c'][ind_tmp]
            quad_coef['a'][ind_tmp], quad_coef['b'][ind_tmp], quad_coef['c'][ind_tmp] = 0., 0., 0.

        ## Right
        if quad_coef['a'][ind_tmp+1] == 0.:
            ## +relu
            relu_coef.append(quad_coef['b'][ind_tmp+1])
            relu_intercept.append(quad_coef['c'][ind_tmp+1])
            if len(quad_coef['b'][:ind_tmp]) > 0:
                # remove relu for right intervals
                quad_coef['b'][ind_tmp+2:] -= quad_coef['b'][ind_tmp+1]
                quad_coef['c'][ind_tmp+2:] -= quad_coef['c'][ind_tmp+1]
            quad_coef['b'][ind_tmp+1], quad_coef['c'][ind_tmp+1] = 0., 0.
        else:
            ## +rehu
            if ind_tmp == len(cutpoints)-1:
                rehu_coef.append(np.sqrt(2*quad_coef['a'][ind_tmp]))
                rehu_intercept.append(quad_coef['b'][ind_tmp]/np.sqrt(2*quad_coef['a'][ind_tmp]))
                rehu_cut.append(np.inf)
            else:
                rehu_coef.append(np.sqrt(2*quad_coef['a'][ind_tmp]))
                rehu_intercept.append(quad_coef['b'][ind_tmp]/np.sqrt(2*quad_coef['a'][ind_tmp]))
                rehu_cut.append(np.sqrt(2*quad_coef['a'][ind_tmp])*cutpoints[ind_tmp-1] + quad_coef['b'][ind_tmp]/np.sqrt(2*quad_coef['a'][ind_tmp]))
            if len(quad_coef['a'][ind_tmp+2:]) > 0:
                quad_coef['b'][ind_tmp+2:] -= 2*quad_coef['a'][ind_tmp+1]*cutpoints[ind_tmp+1] + quad_coef['b'][ind_tmp+1]
                quad_coef['c'][ind_tmp+2:] -= - quad_coef['a'][ind_tmp+1]*cutpoints[ind_tmp+1]**2 + quad_coef['c'][ind_tmp+1]
            quad_coef['a'][ind_tmp+1], quad_coef['b'][ind_tmp+1], quad_coef['c'][ind_tmp+1] = 0., 0., 0.
