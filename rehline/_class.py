""" ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence """

# Authors: Ben Dai <bendai@cuhk.edu.hk>
#          C++ support by Yixuan Qiu <qiuyixuan@sufe.edu.cn>

# License: MIT License

import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import (_check_sample_weight, check_array,
                                      check_is_fitted, check_X_y)

from ._base import (ReHLine_solver, _BaseReHLine,
                    _make_constraint_rehline_param, _make_loss_rehline_param)


class ReHLine(_BaseReHLine, BaseEstimator):
    r"""ReHLine Minimization.

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

    verbose : int, default=0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=1000
        The maximum number of iterations to be run.

    Attributes
    ----------
    coef\_ : array-like
        The optimized model coefficients.

    n_iter\_ : int
        The number of iterations performed by the ReHLine solver.

    opt_result\_ : object
        The optimization result object.

    dual_obj\_ : array-like
        The dual objective function values.

    primal_obj\_ : array-like
        The primal objective function values.

    Examples
    --------

    >>> ## test SVM on simulated dataset
    >>> import numpy as np
    >>> from rehline import ReHLine 

    >>> # simulate classification dataset
    >>> n, d, C = 1000, 3, 0.5
    >>> np.random.seed(1024)
    >>> X = np.random.randn(1000, 3)
    >>> beta0 = np.random.randn(3)
    >>> y = np.sign(X.dot(beta0) + np.random.randn(n))

    >>> # Usage of ReHLine
    >>> n, d = X.shape
    >>> U = -(C*y).reshape(1,-1)
    >>> L = U.shape[0]
    >>> V = (C*np.array(np.ones(n))).reshape(1,-1)
    >>> clf = ReHLine(C=C)
    >>> clf.U, clf.V = U, V
    >>> clf.fit(X=X)
    >>> print('sol privided by rehline: %s' %clf.coef_)
    >>> sol privided by rehline: [ 0.7410154  -0.00615574  2.66990408]
    >>> print(clf.decision_function([[.1,.2,.3]]))
    >>> [0.87384162]

    References
    ----------
    .. [1] `Dai, B., Qiu, Y,. (2023). ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence <https://openreview.net/pdf?id=3pEBW2UPAD>`_
    """

    def __init__(self, C=1.,
                       U=np.empty(shape=(0,0)), V=np.empty(shape=(0,0)),
                       Tau=np.empty(shape=(0,0)),
                       S=np.empty(shape=(0,0)), T=np.empty(shape=(0,0)),
                       A=np.empty(shape=(0,0)), b=np.empty(shape=(0)), 
                       max_iter=1000, tol=1e-4, shrink=1, verbose=0, trace_freq=100):
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
        self.max_iter = max_iter
        self.tol = tol
        self.shrink = shrink
        self.verbose = verbose
        self.trace_freq = trace_freq

    def fit(self, X, sample_weight=None):
        """Fit the model based on the given training data.

        Parameters
        ----------

        X: {array-like} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual
            samples. If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            An instance of the estimator.
        """
        # X = check_array(X)

        
        if sample_weight is None:
            sample_weight = self.C
        else:
            sample_weight = self.C*_check_sample_weight(sample_weight, X, dtype=X.dtype)

        if self.L > 0:
            U_weight = self.U * sample_weight
            V_weight = self.V * sample_weight
        else:
            U_weight = self.U
            V_weight = self.V

        if self.H > 0:
            sqrt_sample_weight = np.sqrt(sample_weight)
            Tau_weight = self.Tau * sqrt_sample_weight
            S_weight = self.S * sqrt_sample_weight
            T_weight = self.T * sqrt_sample_weight
        else:
            Tau_weight = self.Tau
            S_weight = self.S
            T_weight = self.T

        result = ReHLine_solver(X=X,
                                U=U_weight, V=V_weight,
                                Tau=Tau_weight,
                                S=S_weight, T=T_weight,
                                A=self.A, b=self.b,
                                max_iter=self.max_iter, tol=self.tol,
                                shrink=self.shrink, verbose=self.verbose,
                                trace_freq=self.trace_freq)

        self.coef_ = result.beta
        self.opt_result_ = result
        self.n_iter_ = result.niter
        self.dual_obj_ = result.dual_objfns
        self.primal_obj_ = result.primal_objfns

        if self.n_iter_ >= self.max_iter:
            warnings.warn(
                "ReHLine failed to converge, increase the number of iterations: `max_iter`.",
                ConvergenceWarning,
            )

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
        return np.dot(X, self.coef_)

class plqERM_Ridge(_BaseReHLine, BaseEstimator):
    r"""Empirical Risk Minimization (ERM) with a piecewise linear-quadratic (PLQ) objective with a ridge penalty.

    .. math::

        \min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \text{PLQ}(y_i, \mathbf{x}_i^T \mathbf{\beta}) + \frac{1}{2} \| \mathbf{\beta} \|_2^2, \ \text{ s.t. } \ 
        \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},

    The function supports various loss functions, including:
        - 'hinge', 'svm' or 'SVM'
        - 'check' or 'quantile' or 'quantile regression' or 'QR'
        - 'sSVM' or 'smooth SVM' or 'smooth hinge'
        - 'TV'
        - 'huber' or 'Huber'
        - 'SVR' or 'svr'

    The following constraint types are supported:
        * 'nonnegative' or '>=0': A non-negativity constraint.
        * 'fair' or 'fairness': A fairness constraint.
        * 'custom': A custom constraint, where the user must provide the constraint matrix 'A' and vector 'b'.

    Parameters
    ----------
    loss : dict
        A dictionary specifying the loss function parameters. 
    
    constraint : list of dict
        A list of dictionaries, where each dictionary represents a constraint.
        Each dictionary must contain a 'name' key, which specifies the type of constraint.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. 
        `C` will be absorbed by the ReHLine parameters when `self.make_ReLHLoss` is conducted.

    verbose : int, default=0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=1000
        The maximum number of iterations to be run.

    U, V: array of shape (L, n_samples), default=np.empty(shape=(0, 0))
        The parameters pertaining to the ReLU part in the loss function.

    Tau, S, T: array of shape (H, n_samples), default=np.empty(shape=(0, 0))
        The parameters pertaining to the ReHU part in the loss function.
    
    A: array of shape (K, n_features), default=np.empty(shape=(0, 0))
        The coefficient matrix in the linear constraint.

    b: array of shape (K, ), default=np.empty(shape=0)
        The intercept vector in the linear constraint.
    
    Attributes
    ----------
    coef\_ : array-like
        The optimized model coefficients.

    n_iter\_ : int
        The number of iterations performed by the ReHLine solver.

    opt_result\_ : object
        The optimization result object.

    dual_obj\_ : array-like
        The dual objective function values.

    primal_obj\_ : array-like
        The primal objective function values.

    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit the model based on the given training data.

    decision_function(X)
        The decision function evaluated on the given dataset.

    Notes
    -----
    The `plqERM_Ridge` class is a subclass of `_BaseReHLine` and `BaseEstimator`, which suggests that it is part of a larger framework for implementing ReHLine algorithms.

    """

    def __init__(self, loss,
                       constraint=[],
                       C=1.,
                       U=np.empty(shape=(0,0)), V=np.empty(shape=(0,0)),
                       Tau=np.empty(shape=(0,0)),
                       S=np.empty(shape=(0,0)), T=np.empty(shape=(0,0)),
                       A=np.empty(shape=(0,0)), b=np.empty(shape=(0)), 
                       max_iter=1000, tol=1e-4, shrink=1, verbose=0, trace_freq=100):
        self.loss = loss
        self.constraint = constraint
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
        self.max_iter = max_iter
        self.tol = tol
        self.shrink = shrink
        self.verbose = verbose
        self.trace_freq = trace_freq
        self.dummy_n = 0

    def fit(self, X, y, sample_weight=None):
        """Fit the model based on the given training data.

        Parameters
        ----------

        X: {array-like} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable.

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual
            samples. If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            An instance of the estimator.


        """
        n, d = X.shape
        
        ## loss -> rehline params
        self.U, self.V, self.Tau, self.S, self.T = _make_loss_rehline_param(loss=self.loss, X=X, y=y)
        
        ## constrain -> rehline params
        self.A, self.b = _make_constraint_rehline_param(constraint=self.constraint, X=X, y=y)
        self.auto_shape()

        ## sample weight -> rehline params
        if sample_weight is None:
            sample_weight = self.C
        else:
            sample_weight = self.C*_check_sample_weight(sample_weight, X, dtype=X.dtype)

        if self.L > 0:
            U_weight = self.U * sample_weight
            V_weight = self.V * sample_weight
        else:
            U_weight = self.U
            V_weight = self.V

        if self.H > 0:
            sqrt_sample_weight = np.sqrt(sample_weight)
            Tau_weight = self.Tau * sqrt_sample_weight
            S_weight = self.S * sqrt_sample_weight
            T_weight = self.T * sqrt_sample_weight
        else:
            Tau_weight = self.Tau
            S_weight = self.S
            T_weight = self.T

        result = ReHLine_solver(X=X,
                                U=U_weight, V=V_weight,
                                Tau=Tau_weight,
                                S=S_weight, T=T_weight,
                                A=self.A, b=self.b,
                                max_iter=self.max_iter, tol=self.tol,
                                shrink=self.shrink, verbose=self.verbose,
                                trace_freq=self.trace_freq)

        self.coef_ = result.beta
        self.opt_result_ = result
        self.n_iter_ = result.niter
        self.dual_obj_ = result.dual_objfns
        self.primal_obj_ = result.primal_objfns

        if self.n_iter_ >= self.max_iter:
            warnings.warn(
                "ReHLine failed to converge, increase the number of iterations: `max_iter`.",
                ConvergenceWarning,
            )

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
        return np.dot(X, self.coef_)


# # ReHLine estimator with an option of additional linear term
# class ReHLineLinear(ReHLine, _BaseReHLine, BaseEstimator):
#     r"""ReHLine Minimization with additional linear terms. 

#     .. math::

#         \min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}( u_{li} \mathbf{x}_i^\intercal \mathbf{\beta} + v_{li}) + \sum_{i=1}^n \sum_{h=1}^H {\text{ReHU}}_{\tau_{hi}}( s_{hi} \mathbf{x}_i^\intercal \mathbf{\beta} + t_{hi}) + \mathbf{\mu}^T \mathbf{\beta} + \frac{1}{2} \| \mathbf{\beta} \|_2^2, \\ \text{ s.t. } 
#         \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},
        
#     where :math:`\mathbf{U} = (u_{li}),\mathbf{V} = (v_{li}) \in \mathbb{R}^{L \times n}` 
#     and :math:`\mathbf{S} = (s_{hi}),\mathbf{T} = (t_{hi}),\mathbf{\tau} = (\tau_{hi}) \in \mathbb{R}^{H \times n}` 
#     are the ReLU-ReHU loss parameters, and :math:`(\mathbf{A},\mathbf{b})` are the constraint parameters.
    
#     Parameters
#     ----------

#     C : float, default=1.0
#         Regularization parameter. The strength of the regularization is
#         inversely proportional to C. Must be strictly positive. 
#         `C` will be absorbed by the ReHLine parameters when `self.make_ReLHLoss` is conducted.

#     verbose : int, default=0
#         Enable verbose output. Note that this setting takes advantage of a
#         per-process runtime setting in liblinear that, if enabled, may not work
#         properly in a multithreaded context.

#     max_iter : int, default=1000
#         The maximum number of iterations to be run.

#     U, V: array of shape (L, n_samples), default=np.empty(shape=(0, 0))
#         The parameters pertaining to the ReLU part in the loss function.

#     Tau, S, T: array of shape (H, n_samples), default=np.empty(shape=(0, 0))
#         The parameters pertaining to the ReHU part in the loss function.
    
#     mu: array of shape (n_features, ), default=np.empty(shape=0)
#         The parameters pertaining to the linear part in the loss function.

#     A: array of shape (K, n_features), default=np.empty(shape=(0, 0))
#         The coefficient matrix in the linear constraint.

#     b: array of shape (K, ), default=np.empty(shape=0)
#         The intercept vector in the linear constraint.
    

#     Attributes
#     ----------

#     coef_ : array of shape (n_features,)
#         Weights assigned to the features (coefficients in the primal
#         problem).

#     n_iter_: int
#         Maximum number of iterations run across all classes.

#     """
#     def __init__(self, *, mu=np.empty(shape=(0))):
#         _BaseReHLine().__init__(C, U, V, Tau, S, T, A, b, mu, max_iter, tol, shrink, verbose, trace_freq)

#     # @override
#     def fit(self, X, sample_weight=None):
#         """Fit the model based on the given training data.

#         Parameters
#         ----------

#         X: {array-like} of shape (n_samples, n_features)
#             Training vector, where `n_samples` is the number of samples and
#             `n_features` is the number of features.

#         sample_weight : array-like of shape (n_samples,), default=None
#             Array of weights that are assigned to individual
#             samples. If not provided, then each sample is given unit weight.

#         Returns
#         -------
#         self : object
#             An instance of the estimator.
#         """

#         # X = check_array(X)
#         if sample_weight is None:
#             sample_weight = np.ones(X.shape[0])

#         if self.L > 0:
#             U_weight = self.U * sample_weight
#             V_weight = self.V * sample_weight
#         else:
#             U_weight = self.U
#             V_weight = self.V

#         if self.H > 0:
#             sqrt_sample_weight = np.sqrt(sample_weight)
#             Tau_weight = self.Tau * sqrt_sample_weight
#             S_weight = self.S * sqrt_sample_weight
#             T_weight = self.T * sqrt_sample_weight
#         else:
#             Tau_weight = self.Tau
#             S_weight = self.S
#             T_weight = self.T


#         Xtmu = X @ self.mu
#         if self.mu.size > 0:
#             # v_li' = v_li + u_li * (x_i.T @ mu)
#             V_weight = V_weight + U_weight * Xtmu.reshape(1, -1) if self.L > 0 else V_weight
#             # t_hi' = t_hi + s_hi * (x_i.T @ mu)
#             T_weight = T_weight + S_weight * Xtmu.reshape(1, -1) if self.H > 0 else T_weight
#             b_shifted = self.b + self.A @ self.mu
#         else:
#             b_shifted = self.b

#         result = ReHLine_solver(X=X,
#                         U=U_weight, V=V_weight,
#                         Tau=Tau_weight,
#                         S=S_weight, T=T_weight,
#                         A=self.A, b=b_shifted,
#                         max_iter=self.max_iter, tol=self.tol,
#                         shrink=self.shrink, verbose=self.verbose,
#                         trace_freq=self.trace_freq)

#         # unshift results
#         result.beta = result.beta + self.mu
#         result.dual_objfns = [dual_func + 0.5*self.mu.T @ self.mu for dual_func in result.dual_objfns]
#         result.primal_objfns = [primal_func - 0.5*self.mu.T @ self.mu for primal_func in result.primal_objfns]

#         self.coef_ = result.beta
#         self.opt_result_ = result
#         self.n_iter_ = result.niter
#         self.dual_obj_ = result.dual_objfns
#         self.primal_obj_ = result.primal_objfns
