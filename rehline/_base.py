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
        self._U = U
        self._V = V
        self._S = S
        self._T = T
        self._Tau = Tau
        self._A = A
        self._b = b
        self.L = self._U.shape[0]
        self.H = self._S.shape[0]
        self.K = self._A.shape[0]

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        
        Override the default get_params to exclude computation-only parameters.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            if key not in ['U', 'V', 'S', 'T', 'Tau', 'A', 'b', 'Lambda', 'Gamma', 'xi']:
                value = getattr(self, key)
                if deep and hasattr(value, 'get_params') and not isinstance(value, type):
                    deep_items = value.get_params().items()
                    out.update((key + '__' + k, val) for k, val in deep_items)
                out[key] = value
        return out

    def auto_shape(self):
        """
        Automatically generate the shape of the parameters of the ReHLine loss function.
        """
        self.L = self._U.shape[0]
        self.H = self._S.shape[0]
        self.K = self._A.shape[0]

    def cast_sample_weight(self, sample_weight=None):
        """
        Cast the sample weight to the ReHLine parameters.

        Parameters
        ----------
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        U_weight : array-like of shape (L, n_samples)
            Weighted ReLU coefficient matrix.

        V_weight : array-like of shape (L, n_samples)
            Weighted ReLU intercept vector.

        Tau_weight : array-like of shape (H, n_samples)
            Weighted ReHU cutpoint matrix.

        S_weight : array-like of shape (H, n_samples)
            Weighted ReHU coefficient vector.

        T_weight : array-like of shape (H, n_samples)
            Weighted ReHU intercept vector.

        Notes
        -----
        This method casts the sample weight to the ReHLine parameters by multiplying
        the sample weight with the ReLU and ReHU parameters. If sample_weight is None,
        then the sample weight is set to the weight parameter C.
        """
        
        self.auto_shape()
        
        sample_weight = self.C*sample_weight

        if self.L > 0:
            U_weight = self._U * sample_weight
            V_weight = self._V * sample_weight
        else:
            U_weight = self._U
            V_weight = self._V

        if self.H > 0:
            sqrt_sample_weight = np.sqrt(sample_weight)
            Tau_weight = self._Tau * sqrt_sample_weight
            S_weight = self._S * sqrt_sample_weight
            T_weight = self._T * sqrt_sample_weight
        else:
            Tau_weight = self._Tau
            S_weight = self._S
            T_weight = self._T

        return U_weight, V_weight, Tau_weight, S_weight, T_weight

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
            relu_input = (self._U.T * score[:,np.newaxis]).T + self._V
        if self.H > 0:
            rehu_input = (self._S.T * score[:,np.newaxis]).T + self._T
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
        Lambda=np.empty(shape=(0, 0)),
        Gamma=np.empty(shape=(0, 0)),
        xi=np.empty(shape=(0, 0)),
        max_iter=1000, tol=1e-4, shrink=1, verbose=1, trace_freq=100):
    result = rehline_result()
    if len(Lambda)>0:
        result.Lambda = np.maximum(0, np.minimum(Lambda, 1.0))
    if len(Gamma)>0:
        result.Gamma = np.maximum(0, np.minimum(Gamma, Tau))
    if len(xi)>0:
        result.xi = np.maximum(xi, 0.0)
    rehline_internal(result, X, A, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq)
    return result


def _make_loss_rehline_param(loss, X, y):
    """The `_make_loss_rehline_param` function generates parameters for the ReHLine solver, based on the provided training data.

    The function supports various loss functions, including:
        - 'hinge'
        - 'svm' or 'SVM'
        - 'mae' or 'MAE' or 'mean absolute error'
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

    # n, d = X.shape
    n = len(y)

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

    
    elif (loss['name'] == 'MAE') \
            or (loss['name'] == 'mae') \
            or (loss['name'] == 'mean absolute error'):
        U = np.array([[1.0] * n, [-1.0] * n])
        V = np.array([-y , y])

    elif (loss['name'] == 'SVM square') \
            or (loss['name'] == 'svm square') \
            or (loss['name'] == 'hinge square'):
        Tau = np.inf * np.ones((1, n)) 
        S = - np.sqrt(2) * y.reshape(1,-1)
        T = np.sqrt(2) * np.ones((1, n)) 

    elif (loss['name'] == 'MSE') \
            or (loss['name'] == 'mse') \
            or (loss['name'] == 'mean square error'):
        Tau = np.inf * np.ones((2, n)) 
        S = np.array([[np.sqrt(2)] * n, [-np.sqrt(2)] * n])
        T = np.array([-np.sqrt(2) * y , np.sqrt(2) * y])

    
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
            * 'fair' or 'fairness': A fairness constraint using 'sen_idx' and 'tol_sen'.
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
    """

    n, d = X.shape

    ## initialization
    A = np.empty(shape=(0, 0))
    b = np.empty(shape=(0))

    for constr_tmp in constraint:
        if (constr_tmp['name'] == 'nonnegative') or (constr_tmp['name'] == '>=0'):
            A_tmp = np.identity(d)
            b_tmp = np.zeros(d)

        elif (constr_tmp['name'] == 'fair') or (constr_tmp['name'] == 'fairness'):
            sen_idx = constr_tmp['sen_idx']   # list of indices
            tol_sen = constr_tmp['tol_sen']
            tol_sen = np.array(tol_sen).reshape(-1)

            X_sen = X[:, sen_idx]
            X_sen = X_sen.reshape(n, -1)

            assert X_sen.shape[1] == len(tol_sen), "dim of X_sen and len of tol_sen must be equal"

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


def _cast_sample_bias(U, V, Tau, S, T, sample_bias=None):
    """Cast sample bias to ReHLine parameters by injecting bias into V and T.
    
    This function modifies the ReHLine parameters to incorporate individual
    sample biases through linear transformations of the intercept parameters.

    Parameters
    ----------
    U : array-like of shape (L, n_samples)
        ReLU coefficient matrix.

    V : array-like of shape (L, n_samples)
        ReLU intercept vector.

    Tau : array-like of shape (H, n_samples)
        ReHU cutpoint matrix.

    S : array-like of shape (H, n_samples)
        ReHU coefficient vector.

    T : array-like of shape (H, n_samples)
        ReHU intercept vector.

    sample_bias : array-like of shape (n_samples, 1)
        Individual sample bias vector. If None, parameters are returned unchanged.

    Returns
    -------
    U_bias : array-like of shape (L, n_samples)
        Biased coefficient matrix, actually doesn't change

    V_bias : array-like of shape (L, n_samples)
        Biased ReLU intercept vector: V + U * sample_bias

    Tau_bias : array-like of shape (H, n_samples)    
        Biased ReHU cutpoint matrix, actually doesn't change

    S_bias : array-like of shape (H, n_samples)
        Biased ReHU coefficient vector, actually doesn't change

    T_bias : array-like of shape (H, n_samples)
        Biased ReHU intercept vector: T + S * sample_bias

    Notes
    -----
    The transformation applies the sample bias through:
    - V_bias = V + U ⊙ sample_bias
    - T_bias = T + S ⊙ sample_bias
    
    where ⊙ denotes element-wise multiplication with broadcasting.
    """
    if sample_bias is None:
        return U, V, Tau, S, T
    
    else:
        sample_bias = sample_bias.reshape(1, -1)
        U_bias = U
        V_bias = V + (U * sample_bias if U.shape[0] > 0 else 0)
        Tau_bias = Tau
        S_bias = S
        T_bias = T + (S * sample_bias if S.shape[0] > 0 else 0)

        return U_bias, V_bias, Tau_bias, S_bias, T_bias


def _cast_sample_weight(U, V, Tau, S, T, C=1.0, sample_weight=None):
    """Apply sample weights and regularization to ReHLine parameters.

    Parameters
    ----------
    U : array-like of shape (L, n_samples)
        ReLU coefficient matrix.

    V : array-like of shape (L, n_samples)
        ReLU intercept vector.

    Tau : array-like of shape (H, n_samples)
        ReHU cutpoint matrix.

    S : array-like of shape (H, n_samples)
        ReHU coefficient vector.

    T : array-like of shape (H, n_samples)
        ReHU intercept vector.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. 

    sample_weight : array-like of shape (n_samples,), default=None
        Individual sample weight. If None, then samples are equally weighted.

    Returns
    -------
    U_weight : array-like of shape (L, n_samples)
        Weighted ReLU coefficient matrix.

    V_weight : array-like of shape (L, n_samples)
        Weighted ReLU intercept vector.

    Tau_weight : array-like of shape (H, n_samples)
        Weighted ReHU cutpoint matrix.

    S_weight : array-like of shape (H, n_samples)
        Weighted ReHU coefficient vector.

    T_weight : array-like of shape (H, n_samples)
        Weighted ReHU intercept vector.

    Notes
    -----
    This function casts the sample weight to the ReHLine parameters by multiplying
    the sample weight with the ReLU and ReHU parameters. If sample_weight is None,
    then the sample weight is set to the regularization parameter C.
    """
    sample_weight = C * sample_weight

    if U.shape[0] > 0:
        U_weight = U * sample_weight
        V_weight = V * sample_weight
    else:
        U_weight = U
        V_weight = V

    if S.shape[0] > 0:
        sqrt_sample_weight = np.sqrt(sample_weight)
        Tau_weight = Tau * sqrt_sample_weight
        S_weight = S * sqrt_sample_weight
        T_weight = T * sqrt_sample_weight
    else:
        Tau_weight = Tau
        S_weight = S
        T_weight = T

    return U_weight, V_weight, Tau_weight, S_weight, T_weight


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
