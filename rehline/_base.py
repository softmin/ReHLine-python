"""Base functions for ReHLine."""

# Authors: Ben Dai <bendai@cuhk.edu.hk>
#          Yixuan Qiu <yixuanq@gmail.com>

# License: MIT License

from abc import abstractmethod

import numpy as np
from scipy.special import huber
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._internal import rehline_internal, rehline_result


class _BaseReHLine(BaseEstimator):
    r"""Base Class of ReHLine Formulation.

    .. math::

        \min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}( u_{li} \mathbf{x}_i^\intercal \mathbf{\beta} + v_{li}) + \sum_{i=1}^n \sum_{h=1}^H {\text{ReHU}}_{\tau_{hi}}( s_{hi} \mathbf{x}_i^\intercal \mathbf{\beta} + t_{hi}) + \frac{1}{2} \| \mathbf{\beta} \|_2^2, \\ \text{ s.t. } 
        \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},
        
    where :math:`\mathbf{U} = (u_{li}),\mathbf{V} = (v_{li}) \in \mathbb{R}^{L \times n}` 
    and :math:`\mathbf{S} = (s_{hi}),\mathbf{T} = (t_{hi}),\mathbf{\tau} = (\tau_{hi}) \in \mathbb{R}^{H \times n}` 
    are the ReLU-ReHU loss parameters, and :math:`(\mathbf{A},\mathbf{b})` are the constraint parameters.
    
    Parameters
    ----------

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. 

    U, V: array of shape (L, n_samples), default=np.empty(shape=(0, 0))
        The parameters pertaining to the ReLU part in the loss function.

    Tau, S, T: array of shape (H, n_samples), default=np.empty(shape=(0, 0))
        The parameters pertaining to the ReHU part in the loss function.
    
    A: array of shape (K, n_features), default=np.empty(shape=(0, 0))
        The coefficient matrix in the linear constraint.

    b: array of shape (K, ), default=np.empty(shape=0)
        The intercept vector in the linear constraint.

    """

    def __init__(self, C=1.,
                       U=np.empty(shape=(0,0)), V=np.empty(shape=(0,0)),
                       Tau=np.empty(shape=(0,0)),
                       S=np.empty(shape=(0,0)), T=np.empty(shape=(0,0)),
                       A=np.empty(shape=(0,0)), b=np.empty(shape=(0))):
        self.C = C
        self.U = U
        self.V = V
        self.S = S
        self.T = T
        self.Tau = Tau
        self.A = A
        self.b = b
        self.L = U.shape[0]
        self.n = U.shape[1]
        self.H = S.shape[0]
        self.K = A.shape[0]

    def auto_shape(self):
        """
        Automatically generate the shape of the parameters of the ReHLine loss function.
        """
        self.L = self.U.shape[0]
        self.n = self.U.shape[1]
        self.H = self.S.shape[0]
        self.K = self.A.shape[0]

    def call_ReLHLoss(self, score):
        """
        Return the value of the ReHLine loss of the `score`.

        Parameters
        ----------
        score : ndarray of shape (n_samples, )
            The input score that will be evaluated through the ReHLine loss.

        Returns
        -------
        float
            ReHLine loss evaluation of the given score.
        """

        relu_input = np.zeros((self.L, self.n))
        rehu_input = np.zeros((self.H, self.n))
        if self.L > 0:
            relu_input = (self.U.T * score[:,np.newaxis]).T + self.V
        if self.H > 0:
            rehu_input = (self.S.T * score[:,np.newaxis]).T + self.T
        return np.sum(_relu(relu_input), 0) + np.sum(_rehu(rehu_input), 0)

    @abstractmethod
    def fit(self, X, y, sample_weight):
        """Fit model."""

    @abstractmethod
    def decision_function(self, X):
        """The decision function evaluated on the given dataset

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        ndarray of shape (n_samples, )
            Returns the decision function of the samples.
        """
        # Check if fit has been called
        check_is_fitted(self)

        X = check_array(X)

def _relu(x):
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


def _rehu(x, cut=1):
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

def ReHLine_solver(X, U, V,
        Tau=np.empty(shape=(0, 0)),
        S=np.empty(shape=(0, 0)), T=np.empty(shape=(0, 0)),
        A=np.empty(shape=(0, 0)), b=np.empty(shape=(0)),
        max_iter=1000, tol=1e-4, shrink=1, verbose=1, trace_freq=100):
    result = rehline_result()
    rehline_internal(result, X, A, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq)
    return result