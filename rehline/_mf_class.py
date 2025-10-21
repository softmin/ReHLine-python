'''Matrix Factorization Optimization with Various Loss Functions Based on ReHLine'''

import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_sample_weight
from sklearn.exceptions import ConvergenceWarning
from ._base import  (_BaseReHLine, ReHLine_solver,
                    _make_loss_rehline_param,  _make_constraint_rehline_param,
                    _cast_sample_bias, _cast_sample_weight)


class plqMF_Ridge(_BaseReHLine, BaseEstimator):
    r"""Matrix Factorization (MF) with a piecewise linear-quadratic objective and ridge penalty.

    .. math::

        \min_{\substack{
            \mathbf{P} \in \mathbb{R}^{n \times r}, 
            \mathbf{a} \in \mathbb{R}^n \\
            \mathbf{Q} \in \mathbb{R}^{m \times r}, 
            \mathbf{b} \in \mathbb{R}^m
        }} 
        \left[
            \sum_{(u,i)\in \Omega} C \cdot \phi(\mathbf{p}_u^\top \mathbf{q}_i + a_u + b_i) 
        \right]  
        + 
        \left[ 
            \frac{\rho}{n}\sum_{u=1}^n(\|\mathbf{p}_u\|_2^2 + a_u^2) 
            + \frac{1-\rho}{m}\sum_{i=1}^m(\|\mathbf{q}_i\|_2^2 + b_i^2) 
        \right]
        
    The function supports various loss functions, including:
        - 'hinge', 'svm' or 'SVM'
        - 'MAE' or 'mae' or 'mean absolute error'

    The following constraint types are supported:
        * 'nonnegative' or '>=0': A non-negativity constraint.
        * 'fair' or 'fairness': A fairness constraint.
        * 'custom': A custom constraint, where the user must provide the constraint matrix 'A' and vector 'b'.

    Parameters
    ----------
    n_users : int
        Number of unique users in the dataset(or number of rows in target sparse matrix).

    n_items : int
        Number of unique items in the dataset(or number of columns in target sparse matrix).

    loss : dict
        A dictionary specifying the loss function parameters. 
    
    constraint : list of dict
        A list of dictionaries, where each dictionary represents a constraint.
        Each dictionary must contain a 'name' key, which specifies the type of constraint.

    biased : bool, default=True
            Whether to include user and item bias terms in the model.

    rank : int, default=10
        Dimensionality of the latent factor vectors (number of factors).

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. 
        `C` will be absorbed by the ReHLine parameters when `_cast_sample_weight()` is conducted.

    rho : float, default=0.5
        Regularization strength ratio between user and item factors. Must be within the range of (0,1).
        user_reg = rho / n_users, item_reg = (1 - rho) / n_items

    init_mean : float, default=0.0
        Mean of the Gaussian distribution for initializing latent factors.

    init_sd : float, default=0.1
        Standard deviation of the Gaussian distribution for initializing latent factors. 

    random_state : int or RandomState, default=None
        Random seed for reproducible initialization of latent factors.

    max_iter : int, default=1000
        The maximum number of iterations to be run.
    
    tol : float, default=1e-4
        The tolerance for the stopping criterion.

    shrink : float, default=1
        The shrinkage of dual variables for the ReHLine algorithm.

    trace_freq : int, default=100
        The frequency at which to print the optimization trace.

    max_iter_CD : int, default=10
        Maximum number of iterations for coordinate descent steps.

    tol_CD : float, default=1e-3
        Tolerance for convergence checking in coordinate descent steps.

    verbose : int, default=0
        Verbosity level:
        - 0: No output
        - 1: CD algorithm progress information
        - 2: ReHLine Solver optimization information
        - 3: All information of CD and ReHLine

    Attributes
    ----------
    n_users : int
        Number of users in the training data.

    n_items : int
        Number of items in the training data.

    n_ratings : int
        Number of ratings in the training data (available after fitting).

    P : ndarray of shape (n_users, rank)
        User latent factor matrix. Learned during fitting.

    Q : ndarray of shape (n_items, rank)  
        Item latent factor matrix. Learned during fitting.

    bu : ndarray of shape (n_users,) or None
        User bias terms. Only available if `biased=True`.

    bi : ndarray of shape (n_items,) or None
        Item bias terms. Only available if `biased=True`.

    Iu : list of ndarray
        List where each element contains indices of items rated by the corresponding user.
        Available after fitting.

    Ui : list of ndarray  
        List where each element contains indices of users who rated the corresponding item.
        Available after fitting.

    history : ndarray of shape (max_iter_CD + 1, 2)
        Optimization history containing loss and objective values at each coordinate descent iteration.
        First column: loss term values, Second column: objective function values.

    sample_weight : ndarray of shape (n_ratings,)
        Sample weights used during fitting. Available after fitting.
        
    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit the model based on the given training data.

    decision_function(X)
        The decision function evaluated on the given dataset.

    obj(X, y, loss))
        Compute the values of loss term and objective function.

    Notes
    -----
    The `plqMF_Ridge` class is a subclass of `_BaseReHLine` and `BaseEstimator`, which suggests that it is part of a larger framework for implementing ReHLine algorithms.

    """

    def __init__(self, n_users, n_items, loss, constraint=[], biased=True,
                    rank=10, C=1, rho=0.5,
                    init_mean=0, init_sd=0.1, random_state=None,
                    max_iter=10000, tol=1e-3, shrink=1, trace_freq=100, 
                    max_iter_CD=10, tol_CD=1e-3, verbose=0):
        # check input
        errors = []
        checks = [
            (0 < rho < 1, "rho must be between 0 and 1"),
            (C > 0, "C must be positive"),
            (tol_CD > 0, "tol_CD must be positive"),
            (tol > 0, "tol must be positive")
        ]
        for condition, error_msg in checks:
            if not condition:
                errors.append(error_msg)
        if errors:
            raise ValueError("; ".join(errors))

        # parameter initialization
        ## -----------------------------basic perameters-----------------------------
        self.n_users = n_users 
        self.n_items = n_items 
        self.loss = loss
        self.constraint = constraint
        self.biased = biased
        ## -----------------------------hyper perameters-----------------------------
        self.rank = rank
        self.C = C
        self.rho = rho
        ## --------------------------coefficient perameters--------------------------
        self.init_mean = init_mean
        self.init_sd = init_sd
        self.random_state = random_state
        if self.random_state:
            np.random.seed(random_state)
        self.P = np.random.normal(loc=init_mean, scale=init_sd, size=(n_users, rank)) 
        self.Q = np.random.normal(loc=init_mean, scale=init_sd, size=(n_items, rank))
        self.bu = np.zeros(n_users) if self.biased else None 
        self.bi = np.zeros(n_items) if self.biased else None 
        ## ----------------------------fitting parameters----------------------------
        self.max_iter_CD = max_iter_CD
        self.tol_CD = tol_CD
        self.verbose = verbose
        self.tol = tol
        self.max_iter = max_iter
        self.shrink = shrink
        self.trace_freq = trace_freq



    def fit(self, X, y, sample_weight=None):
        """Fit the model based on the given training data.
        
        Parameters
        ----------
        X : array-like of shape (n_ratings, 2)
            Input data where first column contains user id and 
            second column contains item id.

        y : array-like of shape (n_ratings,)
            Target rating values.

        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
            
         Returns
        -------
        self : object
            An instance of the estimator.
            
        """
        # Preparation
        self.n_ratings = len(y)
        self.history = np.nan * np.zeros((self.max_iter_CD + 1, 2))
        self.sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)


        X_df = pd.DataFrame(X, columns=['user', 'item'])
        uidx_map = X_df.groupby('user').indices
        iidx_map = X_df.groupby('item').indices
        self.Iu = [uidx_map.get(u, np.array([], dtype=int)) for u in range(self.n_users)]
        self.Ui = [iidx_map.get(i, np.array([], dtype=int)) for i in range(self.n_items)]


        C_user = self.C * self.n_users / (self.rho) / 2
        C_item = self.C * self.n_items / (1-self.rho) / 2

        if self.verbose in (1, 3):
            print("{:<12} {:<20} {:<20}".format("Iteration", f"Average Loss({self.loss['name']})", "Objective Function"))


        # CD algorithm
        self.history[0] = self.obj(X, y, loss=self.loss)
        for l in range(self.max_iter_CD):
            ## User side update
            for user in range(self.n_users):

                ### item indices given current user
                index_tmp = self.Iu[user]
                len_tmp = len(index_tmp)

                ### if lack of interaction(cold start)
                if len_tmp == 0:
                    self.P[user,:] = 0.0
                    if self.biased:
                        self.bu[user] = 0.0
                    continue

                ### prepare sub-optimization data
                y_tmp = y[index_tmp]
                item_tmp = X[index_tmp][:,1]
                Q_tmp = np.c_[np.ones((len_tmp, 1)), self.Q[item_tmp]] if self.biased else self.Q[item_tmp]
                bias_tmp = self.bi[item_tmp] if self.biased  else None
                weight_tmp = self.sample_weight[index_tmp]

                ### prepare rehline parameter
                U, V, Tau, S, T = _make_loss_rehline_param(loss=self.loss, X=Q_tmp, y=y_tmp)
                U_bias, V_bias, Tau_bias, S_bias,  T_bias = _cast_sample_bias(U, V, Tau, S, T, sample_bias=bias_tmp)
                U_weight, V_weight, Tau_weight, S_weight, T_weight = _cast_sample_weight(U_bias, V_bias, Tau_bias, S_bias, T_bias, C=C_user, sample_weight=weight_tmp)
                A, b = _make_constraint_rehline_param(constraint=self.constraint, X=Q_tmp, y=y_tmp)

                ### solve and update
                result_tmp = ReHLine_solver(X=Q_tmp, 
                                            U=U_weight, V=V_weight,
                                            Tau=Tau_weight, S=S_weight, T=T_weight,
                                            A=A, b=b,
                                            max_iter=self.max_iter, 
                                            tol=self.tol, 
                                            shrink=self.shrink, 
                                            verbose=(self.verbose == 2 or self.verbose == 3), 
                                            trace_freq=self.trace_freq)

                if self.biased:
                    self.bu[user], self.P[user,:] = result_tmp.beta[0], result_tmp.beta[1:]
                else:
                    self.P[user,:] = result_tmp.beta
                 
                ### algo convergence
                if result_tmp.niter >= self.max_iter:
                    warnings.warn(
                        "ReHLine failed to converge, increase the number of iterations: `max_iter`.",
                        ConvergenceWarning,
                    )


            ## Item side update
            for item in range(self.n_items):
                
                ### user indices given current item
                index_tmp = self.Ui[item]
                len_tmp = len(index_tmp)

                ### if lack of interaction(cold start)
                if len_tmp == 0:
                    self.Q[item, :] = 0.0
                    if self.biased:
                        self.bi[item] = 0.0
                    continue

                ### prepare sub-optimization data
                y_tmp = y[index_tmp]
                user_tmp = X[index_tmp][:,0]
                P_tmp = np.c_[np.ones((len_tmp, 1)) , self.P[user_tmp]] if self.biased else self.P[user_tmp]
                weight_tmp = self.sample_weight[index_tmp]
                bias_tmp = self.bu[user_tmp] if self.biased else None
                
                ### prepare rehline parameter
                U, V, Tau, S, T = _make_loss_rehline_param(loss=self.loss, X=P_tmp, y=y_tmp)
                U_bias, V_bias, Tau_bias, S_bias,  T_bias = _cast_sample_bias(U, V, Tau, S, T, sample_bias=bias_tmp)
                U_weight, V_weight, Tau_weight, S_weight, T_weight = _cast_sample_weight(U_bias, V_bias, Tau_bias, S_bias, T_bias, C=C_item, sample_weight=weight_tmp)
                A, b = _make_constraint_rehline_param(constraint=self.constraint, X=P_tmp, y=y_tmp)
                
                ### solve and update
                result_tmp = ReHLine_solver(X=P_tmp, 
                                            U=U_weight, V=V_weight,
                                            Tau=Tau_weight, S=S_weight, T=T_weight,
                                            A=A, b=b,
                                            max_iter=self.max_iter, 
                                            tol=self.tol, 
                                            shrink=self.shrink, 
                                            verbose=(self.verbose == 2 or self.verbose == 3), 
                                            trace_freq=self.trace_freq)

                if self.biased:
                    self.bi[item], self.Q[item,:] = result_tmp.beta[0], result_tmp.beta[1:]
                else:
                    self.Q[item,:] = result_tmp.beta
                 
                ### algo convergence
                if result_tmp.niter >= self.max_iter:
                    warnings.warn(
                        "ReHLine failed to converge, increase the number of iterations: `max_iter`.",
                        ConvergenceWarning,
                    )


            ## Check convergence
            self.history[l+1] = self.obj(X, y, loss=self.loss)
            obj_diff = (self.history[l] - self.history[l+1])[1]

            
            if self.verbose in (1, 3):
                mean_loss = f"{self.history[l+1][0] / self.n_ratings:.6f}"
                obj = f"{self.history[l+1][1]:.6f}"
                print("{:<12} {:<20} {:<20}".format(l + 1, mean_loss, obj))
            
            if obj_diff < self.tol_CD:
                break

        return self



    def decision_function(self, X):
        """The decision function evaluated on the given dataset

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            Training data where first column contains user id and 
            second column contains item id.

        Returns
        -------
        prediction : ndarray of shape (n_samples,)
            Predicted ratings for the input pairs.
        """
        users = X[:, 0]
        items = X[:, 1]
        dot_products = np.einsum('ij,ij->i', self.P[users], self.Q[items])  
        
        if self.biased:
            user_biases = self.bu[users]
            item_biases = self.bi[items]
            return user_biases + item_biases + dot_products  
        else:
            return dot_products  



    def obj(self, X, y, loss):
        """
        Compute the values of loss term and objective function.
        
        Parameters
        ----------
        X : array-like of shape (n_ratings, 2)
            User-item rating pairs.

        y : array-like of shape (n_ratings,)
            Actual rating values.

        loss : dict
            A dictionary specifying the loss function parameters. 
            
        Returns
        -------
        loss_term : float
            The data fitting term (sum of loss values).
            
        objective_value : float
            The total objective value including regularization.
    
        """
        
        if self.biased:
            user_penalty = ( np.sum(self.P ** 2) + np.sum(self.bu ** 2) ) * self.rho / self.n_users
            item_penalty = ( np.sum(self.Q ** 2) + np.sum(self.bi ** 2) ) * (1 - self.rho) / self.n_items
            penalty = user_penalty + item_penalty
        else:
            user_penalty = np.sum(self.P ** 2) * self.rho / self.n_users
            item_penalty = np.sum(self.Q ** 2) * (1 - self.rho) / self.n_items
            penalty = user_penalty + item_penalty

        if (loss['name'] == 'mae') \
            or (loss['name'] == 'MAE') \
            or (loss['name'] == 'mean absolute error'):
            loss_term =  np.sum( np.abs(self.decision_function(X) - y) )

        elif (loss['name'] == 'MSE') \
            or (loss['name'] == 'mse') \
            or (loss['name'] == 'mean square error'):
            loss_term =  np.sum( (self.decision_function(X) - y) ** 2 )
            
        elif (loss['name'] == 'hinge') \
            or (loss['name'] == 'svm') \
            or (loss['name'] == 'SVM'):
            loss_term = np.sum( np.maximum(0, 1 - y * self.decision_function(X)) )
        
        elif (loss['name'] == 'hinge square') \
            or (loss['name'] == 'svm square') \
            or (loss['name'] == 'SVM square'):
            loss_term = np.sum( np.maximum(0, 1 - y * self.decision_function(X)) ** 2 )

        else:
            raise ValueError(f"Unsupported loss function: {loss['name']}. "
                            f"Supported losses are: 'mae', 'mse', 'hinge', 'hinge square'")

        return loss_term, self.C * loss_term + penalty
