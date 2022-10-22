""" ReMin: Regularized ReLU/ReHU Composite Loss Minimization """

# Authors: Ben Dai <bendai@cuhk.edu.hk>
#          C++ support by Yixuan Qiu <yixuanq@gmail.com>

# License: MIT License

import numpy as np
from sklearn.base import BaseEstimator
import l3solver
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

def ReMin_solver(X, U, V, 
        Tau=np.empty(shape=(0, 0)),
        S=np.empty(shape=(0, 0)), T=np.empty(shape=(0, 0)), 
        A=np.empty(shape=(0, 0)), b=np.empty(shape=(0)), 
        max_iter=1000, tol=1e-4, verbose=True):

    result = l3solver.L3Result()
    l3solver.l3solver_internal(result, X, A, b, U, V, S, T, Tau, max_iter, tol, verbose)
    return result

class ReMin(BaseEstimator):
    """Regularized ReLU/ReHU Minimization. (draft version v1.0)
    
    Parameters
    ----------

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
    
    verbose : int, default=0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, default=1000
        The maximum number of iterations to be run.

    Attributes
    ----------

    coef_ : array of shape (n_features,) 
        Weights assigned to the features (coefficients in the primal
        problem).
    
    n_iter_: int
        Maximum number of iterations run across all classes.
    """

    def __init__(self, U, V, C = 1., 
                    Tau=np.empty(shape=(0,0)),
                    S=np.empty(shape=(0,0)), T=np.empty(shape=(0,0)),
                    A=np.empty(shape=(0,0)), b=np.empty(shape=(0)),
                    max_iter=1000, tol=1e-4, verbose=False):
        self.U = U
        self.V = V
        self.S = S
        self.T = T
        self.Tau = Tau
        self.A = A
        self.b = b
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.L = U.shape[0]
        self.n = U.shape[1]
        self.H = S.shape[0]
        self.K = A.shape[0]
        self.C = C

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
        X = check_array(X)
        if sample_weight is None:
            sample_weight = self.C

        U_weight = self.U * sample_weight
        V_weight = self.V * sample_weight
        if self.H > 0:
            sqrt_sample_weight = np.sqrt(sample_weight)
            Tau_weight = self.Tau * sqrt_sample_weight
            S_weight = self.S * sqrt_sample_weight
            T_weight = self.T * sqrt_sample_weight
        else:
            Tau_weight = self.Tau
            S_weight = self.S
            T_weight = self.T
            
        result = ReMin_solver(X=X, 
                            U=U_weight, V=V_weight,
                            Tau=Tau_weight,
                            S=S_weight, T=T_weight,
                            A=self.A, b=self.b,
                            max_iter=self.max_iter, tol=self.tol, verbose=self.verbose)

        self.coef_ = result.beta
        self.opt_result_ = result
        self.n_iter_ = result.niter
        self.dual_obj_ = result.dual_objfns

    def decision_function(self, X):
        """The decision function evaluated on the given dataset

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        dec : ndarray of shape (n_samples,)
            Returns the decision function of the samples.
        """


        # Check if fit has been called
        check_is_fitted(self)
        
        X = check_array(X)
        return np.dot(X, self.coef_)









