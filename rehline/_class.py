""" ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence """

# Authors: Ben Dai <bendai@cuhk.edu.hk>
#          C++ support by Yixuan Qiu <qiuyixuan@sufe.edu.cn>

# License: MIT License

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from ._base import relu, rehu
from ._internal import rehline_internal, rehline_result

def ReHLine_solver(X, U, V,
        Tau=np.empty(shape=(0, 0)),
        S=np.empty(shape=(0, 0)), T=np.empty(shape=(0, 0)),
        A=np.empty(shape=(0, 0)), b=np.empty(shape=(0)),
        max_iter=1000, tol=1e-4, shrink=1, verbose=1, trace_freq=100):
    result = rehline_result()
    rehline_internal(result, X, A, b, U, V, S, T, Tau, max_iter, tol, shrink, verbose, trace_freq)
    return result

class ReHLine(BaseEstimator):
    r"""**(main class)** ReHLine Minimization. (draft version v1.0)

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

    coef_ : array of shape (n_features,)
        Weights assigned to the features (coefficients in the primal
        problem).

    n_iter_: int
        Maximum number of iterations run across all classes.

    References
    ----------
    .. [1] `Dai, B., Qiu, Y,. (2023). ReHLine: Regularized Composite ReLU-ReHU Loss Minimization with Linear Computation and Linear Convergence 
        <https://openreview.net/pdf?id=3pEBW2UPAD>`_

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

    >>> # Usage 1: build-in loss
    >>> clf = ReHLine(loss={'name': 'svm'}, C=C)
    >>> clf.make_ReLHLoss(X=X, y=y, loss={'name': 'svm'})
    >>> clf.fit(X=X)
    >>> print('sol privided by rehline: %s' %clf.coef_)
    >>> sol privided by rehline: [ 0.74104604 -0.00622664  2.66991198]
    >>> print(clf.decision_function([[.1,.2,.3]]))
    >>> [0.87383287]

    >>> # Usage 2: manually specify params
    >>> n, d = X.shape
    >>> U = -(C*y).reshape(1,-1)
    >>> L = U.shape[0]
    >>> V = (C*np.array(np.ones(n))).reshape(1,-1)
    >>> clf = ReHLine(loss={'name': 'svm'}, C=C)
    >>> clf.U, clf.V = U, V
    >>> clf.fit(X=X)
    >>> print('sol privided by rehline: %s' %clf.coef_)
    >>> sol privided by rehline: [ 0.7410154  -0.00615574  2.66990408]
    >>> print(clf.decision_function([[.1,.2,.3]]))
    >>> [0.87384162]
    """

    def __init__(self, loss={'name':'QR', 'qt':[.25, .75]}, C=1.,
                       U=np.empty(shape=(0,0)), V=np.empty(shape=(0,0)),
                       Tau=np.empty(shape=(0,0)),
                       S=np.empty(shape=(0,0)), T=np.empty(shape=(0,0)),
                       A=np.empty(shape=(0,0)), b=np.empty(shape=(0)),
                       max_iter=1000, tol=1e-4, shrink=1, verbose=0, trace_freq=100):
        self.loss = loss
        self.C = C
        self.U = U
        self.V = V
        self.S = S
        self.T = T
        self.Tau = Tau
        self.A = A
        self.b = b
        self.max_iter = max_iter
        self.tol = tol
        self.shrink = shrink
        self.verbose = verbose
        self.trace_freq = trace_freq
        self.L = U.shape[0]
        self.n = U.shape[1]
        self.H = S.shape[0]
        self.K = A.shape[0]

    def make_ReLHLoss(self, X, y, loss={}):
        """The `make_ReLHLoss` function generates parameters for the ReLoss, based on the provided training data.

        The function matches the specific ReLoss (self.loss) with loss functions 
        like 'hinge', 'svm', 'SVM', 'check', 'quantile', 'quantile regression', 
        'QR', 'sSVM', 'smooth SVM', 'smooth hinge', 'TV', 'huber', and 'custom'.

        Parameters
        ----------

        X : ndarray of shape (n_samples, n_features)
            The generated samples.

        y : ndarray of shape (n_samples,)
            The +/- labels for class membership of each sample.

        loss: dictionary
            A dictionary that provides the loss function type and properties (optional).
        """
        
        if (loss=={}) or (loss==self.loss):
            pass
        else:
            print('Loss has been updated!')
            self.loss.update(loss)

        n, d = X.shape

        if (self.loss['name'] == 'hinge') or (self.loss['name'] == 'svm')\
            or (self.loss['name'] == 'SVM'):
            self.U = -(self.C*y).reshape(1,-1)
            self.V = (self.C*np.array(np.ones(n))).reshape(1,-1)
        elif (self.loss['name'] == 'check') \
                or (self.loss['name'] == 'quantile') \
                or (self.loss['name'] == 'quantile regression') \
                or (self.loss['name'] == 'QR'):

            n_qt = len(loss['qt'])
            self.U = np.ones((2, n*n_qt))
            self.V = np.ones((2, n*n_qt))
            X_fake = np.zeros((n*n_qt, d+n_qt))

            for l,qt_tmp in enumerate(loss['qt']):
                self.U[0,l*n:(l+1)*n] = - (self.C*qt_tmp*self.U[0,l*n:(l+1)*n])
                self.U[1,l*n:(l+1)*n] = (self.C*(1.-qt_tmp)*self.U[1,l*n:(l+1)*n])

                self.V[0,l*n:(l+1)*n] = self.C*qt_tmp*self.V[0,l*n:(l+1)*n]*y
                self.V[1,l*n:(l+1)*n] = - self.C*(1.-qt_tmp)*self.V[1,l*n:(l+1)*n]*y

                X_fake[l*n:(l+1)*n,:d] = X
                X_fake[l*n:(l+1)*n,d+l] = 1.

            self.auto_shape()
            return X_fake

        elif (self.loss['name'] == 'sSVM') \
                or (self.loss['name'] == 'smooth SVM') \
                or (self.loss['name'] == 'smooth hinge'):
            self.S = np.ones((1, n))
            self.T = np.ones((1, n))
            self.Tau = np.ones((1, n))

            self.S[0] = - np.sqrt(self.C)*y
            self.T[0] = np.sqrt(self.C)
            self.Tau[0] = np.sqrt(self.C)
        elif self.loss['name'] == 'TV':
            self.U = np.ones((2, n))*self.C
            self.V = np.ones((2, n))*self.C
            self.U[1] = -self.U[1]

            self.V[0] = - X.dot(y)*self.C
            self.V[1] = X.dot(y)*self.C
        elif (self.loss['name'] == 'huber'):
            self.S = np.ones((2, n))
            self.T = np.ones((2, n))
            self.Tau = np.sqrt(self.C) * loss['tau'] * np.ones((2, n))

            self.S[0] = - np.sqrt(self.C)
            self.S[1] =   np.sqrt(self.C)
            self.T[0] = np.sqrt(self.C)*y
            self.T[1] = -np.sqrt(self.C)*y
        elif (self.loss['name'] == 'custom'):
            pass
        else:
            raise Exception("Sorry, ReHLine currently does not support this loss function, \
                            but you can manually set ReLoss params to solve the problem.")
        self.auto_shape()

    def append_l1(self, X, l1_pen=1.0):
        r"""
        This function appends the l1 penalty to the ReHLine problem. The formulation becomes:

        .. math::

            \min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}( u_{li} \mathbf{x}_i^\intercal \mathbf{\beta} + v_{li}) + \sum_{i=1}^n \sum_{h=1}^H {\text{ReHU}}_{\tau_{hi}}( s_{hi} \mathbf{x}_i^\intercal \mathbf{\beta} + t_{hi}) + \frac{1}{2} \| \mathbf{\beta} \|_2^2 + \lambda_1 \| \mathbf{\beta} \|_1, \\ \text{ s.t. } 
            \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},

        where :math:`\lambda_1` is associated with `l1_pen`.

        Parameters
        ----------

        X : ndarray of shape (n_samples, n_features)
            The generated samples.

        l1_pen : float, default=1.0
            The l1 penalty level, which controls the complexity or sparsity of the resulting model.

        Returns
        -------

        X_fake: ndarray of shape (n_samples+n_features, n_features)
            The manipulated data matrix. It has been padded with 
            identity matrix, allowing the correctly structured data to be input 
            into `self.fit` or other modelling processes.

        Examples
        --------

        >>> import numpy as np
        >>> from rehline import ReHLine

        >>> # simulate classification dataset
        >>> n, d, C, lam1 = 1000, 3, 0.5, 1.0
        >>> np.random.seed(1024)
        >>> X = np.random.randn(1000, 3)
        >>> beta0 = np.random.randn(3)
        >>> y = np.sign(X.dot(beta0) + np.random.randn(n))

        >>> clf = ReHLine(loss={'name': 'svm'}, C=C)
        >>> clf.make_ReLHLoss(X=X, y=y, loss={'name': 'svm'})
        >>> # save and fit with the manipulated data matrix
        >>> X_fake = clf.append_l1(X, l1_pen=lam1)
        >>> clf.fit(X=X_fake)
        >>> print('sol privided by rehline: %s' %clf.coef_)
        >>> sol privided by rehline: [ 7.17796629e-01 -1.87075728e-06  2.61965622e+00] #sparse sol
        >>> print(clf.decision_function([[.1,.2,.3]]))
        >>> [0.85767616]
        """

        n, d = X.shape
        l1_pen = l1_pen*np.ones(d)
        U_new = np.zeros((self.L+2, n+d))
        V_new = np.zeros((self.L+2, n+d))
        ## Block 1
        if len(self.U):
            U_new[:self.L, :n] = self.U
            V_new[:self.L, :n] = self.V
        ## Block 2
        U_new[-2,n:] = l1_pen
        U_new[-1,n:] = -l1_pen

        if len(self.S):
            S_new = np.zeros((self.H, n+d))
            T_new = np.zeros((self.H, n+d))
            Tau_new = np.zeros((self.H, n+d))

            S_new[:,:n] = self.S
            T_new[:,:n] = self.T
            Tau_new[:,:n] = self.Tau

            self.S = S_new
            self.T = T_new
            self.Tau = Tau_new

        ## fake X
        X_fake = np.zeros((n+d, d))
        X_fake[:n,:] = X
        X_fake[n:,:] = np.identity(d)

        self.U = U_new
        self.V = V_new
        self.auto_shape()
        return X_fake

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
        return np.sum(relu(relu_input), 0) + np.sum(rehu(rehu_input), 0)


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
            sample_weight = np.ones(X.shape[0])

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
