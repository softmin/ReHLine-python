
rehline
=======

.. py:module:: rehline


Overview
--------

.. list-table:: Classes
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`ReHLine <rehline.ReHLine>`
     - ReHLine Minimization.
   * - :py:obj:`plqERM_Ridge <rehline.plqERM_Ridge>`
     - Empirical Risk Minimization (ERM) with a piecewise linear-quadratic (PLQ) objective with a ridge penalty.
   * - :py:obj:`CQR_Ridge <rehline.CQR_Ridge>`
     - Composite Quantile Regressor (CQR) with a ridge penalty.


.. list-table:: Function
   :header-rows: 0
   :widths: auto
   :class: summarytable

   * - :py:obj:`ReHLine_solver <rehline.ReHLine_solver>`\ (X, U, V, Tau, S, T, A, b, Lambda, Gamma, xi, max_iter, tol, shrink, verbose, trace_freq)
     - \-



Classes
-------

.. py:class:: ReHLine(C=1.0, U=np.empty(shape=(0, 0)), V=np.empty(shape=(0, 0)), Tau=np.empty(shape=(0, 0)), S=np.empty(shape=(0, 0)), T=np.empty(shape=(0, 0)), A=np.empty(shape=(0, 0)), b=np.empty(shape=0), max_iter=1000, tol=0.0001, shrink=1, warm_start=0, verbose=0, trace_freq=100)

   Bases: :py:obj:`rehline._base._BaseReHLine`, :py:obj:`sklearn.base.BaseEstimator`

   ReHLine Minimization.

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
       Enable verbose output. 

   max_iter : int, default=1000
       The maximum number of iterations to be run.

   tol : float, default=1e-4
       The tolerance for the stopping criterion.

   shrink : float, default=1
       The shrinkage of dual variables for the ReHLine algorithm.

   warm_start : bool, default=False
       Whether to use the given dual params as an initial guess for the
       optimization algorithm.

   trace_freq : int, default=100
       The frequency at which to print the optimization trace.

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

   Lambda: array-like
       The optimized dual variables for ReLU parts.

   Gamma: array-like
       The optimized dual variables for ReHU parts.

   xi: array-like
       The optimized dual variables for linear constraints.

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


   Overview
   ========


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`fit <rehline.ReHLine.fit>`\ (X, sample_weight)
        - Fit the model based on the given training data.
      * - :py:obj:`decision_function <rehline.ReHLine.decision_function>`\ (X)
        - The decision function evaluated on the given dataset


   Members
   =======

   .. py:method:: fit(X, sample_weight=None)

      Fit the model based on the given training data.

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


   .. py:method:: decision_function(X)

      The decision function evaluated on the given dataset

      Parameters
      ----------
      X : array-like of shape (n_samples, n_features)
          The data matrix.

      Returns
      -------
      ndarray of shape (n_samples, )
          Returns the decision function of the samples.




.. py:class:: plqERM_Ridge(loss, constraint=[], C=1.0, U=np.empty(shape=(0, 0)), V=np.empty(shape=(0, 0)), Tau=np.empty(shape=(0, 0)), S=np.empty(shape=(0, 0)), T=np.empty(shape=(0, 0)), A=np.empty(shape=(0, 0)), b=np.empty(shape=0), max_iter=1000, tol=0.0001, shrink=1, warm_start=0, verbose=0, trace_freq=100)

   Bases: :py:obj:`rehline._base._BaseReHLine`, :py:obj:`sklearn.base.BaseEstimator`

   Empirical Risk Minimization (ERM) with a piecewise linear-quadratic (PLQ) objective with a ridge penalty.

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



   Overview
   ========


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`fit <rehline.plqERM_Ridge.fit>`\ (X, y, sample_weight)
        - Fit the model based on the given training data.
      * - :py:obj:`decision_function <rehline.plqERM_Ridge.decision_function>`\ (X)
        - The decision function evaluated on the given dataset


   Members
   =======

   .. py:method:: fit(X, y, sample_weight=None)

      Fit the model based on the given training data.

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




   .. py:method:: decision_function(X)

      The decision function evaluated on the given dataset

      Parameters
      ----------
      X : array-like of shape (n_samples, n_features)
          The data matrix.

      Returns
      -------
      ndarray of shape (n_samples, )
          Returns the decision function of the samples.




.. py:class:: CQR_Ridge(quantiles, C=1.0, max_iter=1000, tol=0.0001, shrink=1, warm_start=0, verbose=0, trace_freq=100)

   Bases: :py:obj:`rehline._base._BaseReHLine`, :py:obj:`sklearn.base.BaseEstimator`

   Composite Quantile Regressor (CQR) with a ridge penalty.

   It allows for the fitting of a linear regression model that minimizes a composite quantile loss function.

   .. math::

       \min_{\mathbf{\beta} \in \mathbb{R}^d, \mathbf{\beta_0} \in \mathbb{R}^K} \sum_{k=1}^K \sum_{i=1}^n \text{PLQ}(y_i, \mathbf{x}_i^T \mathbf{\beta} + \mathbf{\beta_0k}) + \frac{1}{2} \| \mathbf{\beta} \|_2^2.


   Parameters
   ----------
   quantiles : list of float (n_quantiles,)
       The quantiles to be estimated.

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

   tol : float, default=1e-4
       The tolerance for the stopping criterion.

   shrink : float, default=1
       The shrinkage of dual variables for the ReHLine algorithm.

   warm_start : bool, default=False
       Whether to use the given dual params as an initial guess for the
       optimization algorithm.

   trace_freq : int, default=100
       The frequency at which to print the optimization trace.
       
   Attributes
   ----------
   coef\_ : array-like
       The optimized model coefficients.

   intercept\_ : array-like
       The optimized model intercepts.

   quantiles\_: array-like
       The quantiles to be estimated.

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

   predict(X)
       The prediction for the given dataset.


   Overview
   ========


   .. list-table:: Methods
      :header-rows: 0
      :widths: auto
      :class: summarytable

      * - :py:obj:`fit <rehline.CQR_Ridge.fit>`\ (X, y, sample_weight)
        - Fit the model based on the given training data.
      * - :py:obj:`predict <rehline.CQR_Ridge.predict>`\ (X)
        - The decision function evaluated on the given dataset


   Members
   =======

   .. py:method:: fit(X, y, sample_weight=None)

      Fit the model based on the given training data.

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




   .. py:method:: predict(X)

      The decision function evaluated on the given dataset

      Parameters
      ----------
      X : array-like of shape (n_samples, n_features)
          The data matrix.

      Returns
      -------
      ndarray of shape (n_samples, n_quantiles)
          Returns the decision function of the samples.




Functions
---------
.. py:function:: ReHLine_solver(X, U, V, Tau=np.empty(shape=(0, 0)), S=np.empty(shape=(0, 0)), T=np.empty(shape=(0, 0)), A=np.empty(shape=(0, 0)), b=np.empty(shape=0), Lambda=np.empty(shape=(0, 0)), Gamma=np.empty(shape=(0, 0)), xi=np.empty(shape=(0, 0)), max_iter=1000, tol=0.0001, shrink=1, verbose=1, trace_freq=100)




