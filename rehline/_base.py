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

    def __init__(self, *, C=1.,
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
        self.H = S.shape[0]
        self.K = A.shape[0]

    def auto_shape(self):
        """
        Automatically generate the shape of the parameters of the ReHLine loss function.
        """
        self.L = self.U.shape[0]
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
        n = len(score)
        relu_input = np.zeros((self.L, n))
        rehu_input = np.zeros((self.H, n))
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


def _make_loss_rehline_param(loss, X, y):
    """The `_make_loss_rehline_param` function generates parameters for the ReHLine solver, based on the provided training data.

    The function supports various loss functions, including:
        - 'hinge'
        - 'svm' or 'SVM'
        - 'check' or 'quantile' or 'quantile regression' or 'QR'
        - 'sSVM' or 'smooth SVM' or 'smooth hinge'
        - 'TV'
        - 'huber' or 'Huber'
        - 'SVR' or 'svr'
        - Custom loss functions (manual setup required)

    Parameters
    ----------
    loss : dict
        A dictionary containing the loss function parameters.
        
        Keys:
            - 'name' : str, the name of the loss function (e.g. 'hinge', 'svm', 'QR', etc.)
            - 'loss_kwargs': more keys and values for loss parameters

    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The +/- labels for class membership of each sample.
    """

    n, d = X.shape

    ## initialization of ReHLine params
    U=np.empty(shape=(0,0))
    V=np.empty(shape=(0,0))
    Tau=np.empty(shape=(0,0))
    S=np.empty(shape=(0,0))
    T=np.empty(shape=(0,0))

    # _dummy_X = False

    if (loss['name'] == 'hinge') or (loss['name'] == 'svm')\
        or (loss['name'] == 'SVM'):
        U = -y.reshape(1,-1)
        V = (np.array(np.ones(n))).reshape(1,-1)
    
    elif (loss['name'] == 'check') \
            or (loss['name'] == 'quantile') \
            or (loss['name'] == 'quantile regression') \
            or (loss['name'] == 'QR'):

        qt = loss['qt']

        U = np.ones((2, n))
        V = np.ones((2, n))

        U[0] = - qt*U[0]
        U[1] = (1-qt)*U[1]
        V[0] = qt*V[0]*y
        V[1] = -(1-qt)*V[1]*y

    # elif (loss['name'] == 'CQR') \

    #     n_qt = len(loss['qt'])
    #     U = np.ones((2, n*n_qt))
    #     V = np.ones((2, n*n_qt))
    #     X_fake = np.zeros((n*n_qt, d+n_qt))

    #     for l,qt_tmp in enumerate(loss['qt']):
    #         U[0,l*n:(l+1)*n] = - (qt_tmp*U[0,l*n:(l+1)*n])
    #         U[1,l*n:(l+1)*n] = ((1.-qt_tmp)*U[1,l*n:(l+1)*n])

    #         V[0,l*n:(l+1)*n] = qt_tmp*V[0,l*n:(l+1)*n]*y
    #         V[1,l*n:(l+1)*n] = - (1.-qt_tmp)*V[1,l*n:(l+1)*n]*y

    #         X_fake[l*n:(l+1)*n,:d] = X
    #         X_fake[l*n:(l+1)*n,d+l] = 1.
        
    elif (loss['name'] == 'sSVM') \
            or (loss['name'] == 'smooth SVM') \
            or (loss['name'] == 'smooth hinge'):
        S = np.ones((1, n))
        T = np.ones((1, n))
        Tau = np.ones((1, n))
        S[0] = - y

    elif loss['name'] == 'TV':
        U = np.ones((2, n))
        V = np.ones((2, n))
        U[1] = - U[1]

        V[0] = - X.dot(y)
        V[1] = X.dot(y)

    elif (loss['name'] == 'huber') or (loss['name'] == 'Huber'):
        S = np.ones((2, n))
        T = np.ones((2, n))
        Tau = loss['tau'] * np.ones((2, n))

        S[0] = -S[0]
        T[0] = y
        T[1] = -y

    elif (loss['name'] in ['SVR', 'svr']):
        U = np.ones((2, n))
        V = np.ones((2, n))
        U[1] = -U[1]

        V[0] = -(y + loss['epsilon'])
        V[1] =  (y - loss['epsilon'])

    else:
        raise Exception("Sorry, ReHLine currently does not support this loss function, \
                        but you can manually set ReHLine params to solve the problem via `ReHLine` class.")

    return U, V, Tau, S, T

def _make_constraint_rehline_param(constraint, X, y=None):
    """The `_make_constraint_rehline_param` function generates constraint parameters for the ReHLine solver.
    
    Parameters
    ----------
    constraint : list of dict
        A list of dictionaries, where each dictionary represents a constraint.
        Each dictionary must contain a 'name' key, which specifies the type of constraint.
        The following constraint types are supported:
            * 'nonnegative' or '>=0': A non-negativity constraint.
            * 'fair' or 'fairness': A fairness constraint.
            * 'custom': A custom constraint, where the user must provide the constraint matrix 'A' and vector 'b'.

    X : array-like of shape (n_samples, n_features)
        The design matrix.

    y : array-like of shape (n_samples,), default=None
        The target variable. Not used in this function.

    Returns
    -------
    A : array-like of shape (n_constraints, n_features)
        The constraint matrix.

    b : array-like of shape (n_constraints,)
        The constraint vector.

    Notes
    -----
    This function iterates over the list of constraints and generates the constraint matrix 'A' and vector 'b' accordingly.
    For 'nonnegative' and 'fair' constraints, the function generates the constraint parameters automatically.
    For 'custom' constraints, the user must provide the constraint matrix 'A' and vector 'b' explicitly.    
    """

    n, d = X.shape

    ## initialization
    A=np.empty(shape=(0, 0))
    b=np.empty(shape=(0))

    for constr_tmp in constraint:
        if (constr_tmp['name'] == 'nonnegative') or (constr_tmp['name'] == '>=0'):
            A_tmp = np.identity(d)
            b_tmp = np.zeros(d)
        elif (constr_tmp['name'] == 'fair') or (constr_tmp['name'] == 'fairness'):
            X_sen = constr_tmp['X_sen']
            tol_sen = constr_tmp['tol_sen']
            tol_sen = np.array(tol_sen).reshape(-1)

            assert len(X_sen) == len(X), "X and X_sen must have the same length"
            X_sen = X_sen.reshape(n,-1)

            assert X_sen.shape[1] == len(tol_sen), "dim of X_sen and len of tol_sen must be equal"
            d_sen = X_sen.shape[1]

            A_tmp = np.repeat(X_sen.T @ X, repeats=[2], axis=0) / n
            A_tmp[::2] = -A_tmp[::2]
            b_tmp = np.repeat(tol_sen, repeats=[2], axis=0)
        elif (constr_tmp['name'] == 'custom'):
            A_tmp = constr_tmp['A']
            b_tmp = constr_tmp['b']
        else:
            raise Exception("Sorry, ReHLine currently does not support this constraint, \
                        but you can add it by manually setting A and b via {'name': 'custom', 'A': A, 'b': b}")

        A = np.vstack([A, A_tmp]) if A.size else A_tmp
        b = np.hstack([b, b_tmp]) if b.size else b_tmp

    return A, b

def _make_penalty_rehline_param(self, penalty=None, X=None):
    """The `_make_penalty_rehline_param` function generates penalty parameters for the ReHLine solver.
    """
    raise Exception("Sorry, `_make_penalty_rehline_param` feature is currently under development.")


# def append_l1(self, X, l1_pen=1.0):
#     r"""
#     This function appends the l1 penalty to the ReHLine problem. The formulation becomes:

#     .. math::

#         \min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}( u_{li} \mathbf{x}_i^\intercal \mathbf{\beta} + v_{li}) + \sum_{i=1}^n \sum_{h=1}^H {\text{ReHU}}_{\tau_{hi}}( s_{hi} \mathbf{x}_i^\intercal \mathbf{\beta} + t_{hi}) + \frac{1}{2} \| \mathbf{\beta} \|_2^2 + \lambda_1 \| \mathbf{\beta} \|_1, \\ \text{ s.t. } 
#         \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},

#     where :math:`\lambda_1` is associated with `l1_pen`.

#     Parameters
#     ----------

#     X : ndarray of shape (n_samples, n_features)
#         The generated samples.

#     l1_pen : float, default=1.0
#         The l1 penalty level, which controls the complexity or sparsity of the resulting model.

#     Returns
#     -------

#     X_fake: ndarray of shape (n_samples+n_features, n_features)
#         The manipulated data matrix. It has been padded with 
#         identity matrix, allowing the correctly structured data to be input 
#         into `self.fit` or other modelling processes.

#     Examples
#     --------

#     >>> import numpy as np
#     >>> from rehline import ReHLine

#     >>> # simulate classification dataset
#     >>> n, d, C, lam1 = 1000, 3, 0.5, 1.0
#     >>> np.random.seed(1024)
#     >>> X = np.random.randn(1000, 3)
#     >>> beta0 = np.random.randn(3)
#     >>> y = np.sign(X.dot(beta0) + np.random.randn(n))

#     >>> clf = ReHLine(loss={'name': 'svm'}, C=C)
#     >>> clf.make_ReLHLoss(X=X, y=y, loss={'name': 'svm'})
#     >>> # save and fit with the manipulated data matrix
#     >>> X_fake = clf.append_l1(X, l1_pen=lam1)
#     >>> clf.fit(X=X_fake)
#     >>> print('sol privided by rehline: %s' %clf.coef_)
#     >>> sol privided by rehline: [ 7.17796629e-01 -1.87075728e-06  2.61965622e+00] #sparse sol
#     >>> print(clf.decision_function([[.1,.2,.3]]))
#     >>> [0.85767616]
#     """

#     n, d = X.shape
#     l1_pen = l1_pen*np.ones(d)
#     U_new = np.zeros((self.L+2, n+d))
#     V_new = np.zeros((self.L+2, n+d))
#     ## Block 1
#     if len(self.U):
#         U_new[:self.L, :n] = self.U
#         V_new[:self.L, :n] = self.V
#     ## Block 2
#     U_new[-2,n:] = l1_pen
#     U_new[-1,n:] = -l1_pen

#     if len(self.S):
#         S_new = np.zeros((self.H, n+d))
#         T_new = np.zeros((self.H, n+d))
#         Tau_new = np.zeros((self.H, n+d))

#         S_new[:,:n] = self.S
#         T_new[:,:n] = self.T
#         Tau_new[:,:n] = self.Tau

#         self.S = S_new
#         self.T = T_new
#         self.Tau = Tau_new

#     ## fake X
#     X_fake = np.zeros((n+d, d))
#     X_fake[:n,:] = X
#     X_fake[n:,:] = np.identity(d)

#     self.U = U_new
#     self.V = V_new
#     self.auto_shape()
#     return X_fake