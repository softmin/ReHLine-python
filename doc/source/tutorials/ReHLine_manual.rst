Manual ReHLine Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

`ReHLine` is designed to address the regularized ReLU-ReHU minimization problem, named *ReHLine optimization*, of the following form:

.. math::

  \min_{\mathbf{\beta} \in \mathbb{R}^d} \sum_{i=1}^n \sum_{l=1}^L \text{ReLU}( u_{li} \mathbf{x}_i^\intercal \mathbf{\beta} + v_{li}) + \sum_{i=1}^n \sum_{h=1}^H {\text{ReHU}}_{\tau_{hi}}( s_{hi} \mathbf{x}_i^\intercal \mathbf{\beta} + t_{hi}) + \frac{1}{2} \| \mathbf{\beta} \|_2^2, \ \text{ s.t. } \mathbf{A} \mathbf{\beta} + \mathbf{b} \geq \mathbf{0},


where :math:`\mathbf{U} = (u_{li}),\mathbf{V} = (v_{li}) \in \mathbb{R}^{L \times n}` 
and :math:`\mathbf{S} = (s_{hi}),\mathbf{T} = (t_{hi}),\mathbf{\tau} = (\tau_{hi}) \in \mathbb{R}^{H \times n}` 
are the ReLU-ReHU loss parameters, and :math:`(\mathbf{A},\mathbf{b})` are the constraint parameters. 

The key to using `ReHLine`` to solve any problem lies in utilizing custom ReHLine parameters to represent the problem, we illustrate this with following examples. Suppose that we have `X` and `y` as our data. 

.. code-block:: python

  ## Data
  ## X : [n x d]
  ## y : [n]
  import numpy as np
  n, d = X.shape

.. note:: 

  Most of the examples below can be directly implemented by `ReHLine: Empirical Risk Minimization <./tutorials/ReHLine_ERM.rst>`_; we are simply illustrating how to convert the problem to the ReHLine formulation.
  
SVM
---

SVMs solve the following optimization problem:

.. math::
  \min_{\mathbf{\beta} \in \mathbb{R}^d} \frac{C}{n} \sum_{i=1}^n ( 1 - y_i \mathbf{\beta}^\intercal \mathbf{x}_i )_+ + \frac{1}{2} \| \mathbf{\beta} \|_2^2

where :math:`\mathbf{x}_i \in \mathbb{R}^d` is a feature vector, and :math:`y_i \in \{-1, 1\}` is a binary label. Note that the SVM can be rewritten as a ReHLine optimization with

.. math::
  \mathbf{U} \leftarrow -C \mathbf{y}^\intercal/n, \quad
  \mathbf{V} \leftarrow C \mathbf{1}^\intercal_n/n,

where :math:`\mathbf{1}_n = (1, \cdots, 1)^\intercal` is the $n$-length one vector, :math:`\mathbf{X} \in \mathbb{R}^{n \times d}` is the feature matrix, and :math:`\mathbf{y} = (y_1, \cdots, y_n)^\intercal` is the response vector.

The python implementation is:

.. code-block:: python

  ## SVM ReHLine parameters
  clf = ReHLine()
  ## U
  clf.U = -(C*y).reshape(1,-1)
  ## V
  clf.V = (C*np.array(np.ones(n))).reshape(1,-1)
  ## Fit
  clf.fit(X)

Smooth SVM
----------

Smoothed SVMs solve the following optimization problem:

.. math::
  \min_{\mathbf{\beta} \in \mathbb{R}^d} \frac{C}{n} \sum_{i=1}^n V( y_i \mathbf{\beta}^\intercal \mathbf{x}_i ) + \frac{1}{2} \| \mathbf{\beta} \|_2^2

where :math:`\mathbf{x}_i \in \mathbb{R}^d` is a feature vector, and :math:`y_i \in \{-1, 1\}` is a binary label, and :math:`V(\cdot)` is the modified Huber loss or the smoothed hinge loss:

.. math::
  \begin{equation*}
    V(z) =
    \begin{cases}
    \ 0, & z \geq 1, \\
    \ (1-z)^2/2,                  & 0 < z \leq 1, \\
    \ (1/2 - z ),   & z < 0.
    \end{cases}
  \end{equation*}

Smoothed SVM can be rewritten as a ReHLine optimization with

.. math::
  \mathbf{S} \leftarrow -\sqrt{C/n} \mathbf{y}^\intercal, \quad
  \mathbf{T} \leftarrow \sqrt{C/n} \mathbf{1}^\intercal_n, \quad
  \mathbf{\tau} \leftarrow \sqrt{C/n} \mathbf{1}^\intercal_n.

where :math:`\mathbf{1}_n = (1, \cdots, 1)^\intercal` is the $n$-length one vector, :math:`\mathbf{X} \in \mathbb{R}^{n \times d}` is the feature matrix, and :math:`\mathbf{y} = (y_1, \cdots, y_n)^\intercal` is the response vector.

The python implementation is:

.. code-block:: python

  ## sSVM ReHLine parameters
  clf = ReHLine()
  ## S
  clf.S = -(np.sqrt(C/n)*y).reshape(1,-1)
  ## T
  clf.T = (np.sqrt(C/n)*np.ones(n)).reshape(1,-1)
  ## Tau
  clf.Tau = (np.sqrt(C/n)*np.ones(n)).reshape(1,-1)
  ## Fit
  clf.fit(X)

FairSVM
-------

The SVM with fairness constraints (FairSVM) solves the following optimization problem:

.. math::
  \begin{align}
    & \min_{\mathbf{\beta} \in \mathbb{R}^d} \frac{C}{n} \sum_{i=1}^n ( 1 - y_i \mathbf{\beta}^\intercal \mathbf{x}_i )_+ + \frac{1}{2} \| \mathbf{\beta} \|_2^2, \nonumber \\
    \text{subj. to } & \quad \frac{1}{n} \sum_{i=1}^n \mathbf{z}_i \mathbf{\beta}^\intercal \mathbf{x}_i \leq \mathbf{\rho}, \quad \frac{1}{n} \sum_{i=1}^n \mathbf{z}_i \mathbf{\beta}^\intercal \mathbf{x}_i \geq -\mathbf{\rho},
  \end{align}

where :math:`\mathbf{x}_i \in \mathbb{R}^d` is a feature vector, and :math:`y_i \in \{-1, 1\}` is a binary label, $\mathbf{z}_i$ is a collection of centered sensitive features

.. math::
  \sum_{i=1}^n z_{ij} = 0,

such as gender and/or race. The constraints limit the correlation between the $d_0$-length sensitive features :math:`\mathbf{z}_ i \in \mathbb{R}^{d_0}` and the decision function :math:`\mathbf{\beta}^\intercal \mathbf{x}`, and the constants :math:`\mathbf{\rho} \in \mathbb{R}_+^{d_0}` trade-offs predictive accuracy and fairness. Note that the FairSVM can be rewritten as a ReHLine optimization with

.. math::
  \mathbf{U} \leftarrow -C \mathbf{y}^\intercal/n, \quad
  \mathbf{V} \leftarrow C \mathbf{1}^\intercal_n/n, \quad
  \mathbf{A} \leftarrow
  \begin{pmatrix}
    \mathbf{Z}^\intercal \mathbf{X} / n \\
    -\mathbf{Z}^\intercal \mathbf{X} / n
    \end{pmatrix}, \quad
  \mathbf{b} \leftarrow
  \begin{pmatrix}
    \mathbf{\rho} \\
    \mathbf{\rho}
    \end{pmatrix}

The python implementation is:

.. code-block:: python

  ## FairSVM ReHLine parameters
  clf = ReHLine()
  ## U
  clf.U = -(C*y).reshape(1,-1)
  ## V
  clf.V = (C*np.array(np.ones(n))).reshape(1,-1)
  ## A
  ## we illustrate that the first column of X as sensitive features, and tol is 0.1
  X_sen = X[:,0]
  tol_sen = 0.1
  clf.A = np.repeat([X_sen @ X], repeats=[2], axis=0) / n
  clf.A[1] = -clf.A[1]
  ## b
  clf.b = np.array([tol_sen, tol_sen])
  ## Fit
  clf.fit(X)

Ridge Huber regression
----------------------

The Ridge regularized Huber minimization (RidgeHuber) solves the following optimization problem:

.. math::
   \min_{\mathbf{\beta}} \frac{1}{n} \sum_{i=1}^n H_\kappa( y_i - \mathbf{x}_i^\intercal \mathbf{\beta} ) + \frac{\lambda}{2} \| \mathbf{\beta} \|_2^2,

where :math:`H_\kappa(\cdot)` is the Huber loss with a given parameter :math:`\kappa`:

.. math::
   H_\kappa(z) =
   \begin{cases}
   z^2/2,                  & 0 < |z| \leq \kappa, \\
   \ \kappa ( |z| - \kappa/2 ),   & |z| > \kappa.
   \end{cases}

In this case, the RidgeHuber can be rewritten as a ReHLine optimization with:

.. math::
   \mathbf{S} \leftarrow
   \begin{pmatrix}
   -\sqrt{\frac{1}{n\lambda}} \mathbf{1}^\intercal_n \\
   \sqrt{\frac{1}{n\lambda}} \mathbf{1}^\intercal_n \\
   \end{pmatrix}, \quad
   \mathbf{T} \leftarrow
   \begin{pmatrix}
   \sqrt{\frac{1}{n\lambda}} \mathbf{y}^\intercal  \\
   -\sqrt{\frac{1}{n\lambda}} \mathbf{y}^\intercal \\
   \end{pmatrix}, \quad
   \mathbf{\tau} \leftarrow
   \begin{pmatrix}
   \kappa \sqrt{\frac{1}{n\lambda}} \mathbf{1}^\intercal_n \\
   \\
   \kappa \sqrt{\frac{1}{n\lambda}} \mathbf{1}^\intercal_n \\
   \end{pmatrix}.

The python implementation is:

.. code-block:: python

  ## Huber ReHLine parameters
  clf = ReHLine()
  ## S
  clf.S = -np.repeat([np.sqrt(1/n/lam)*np.ones(n)], repeats=[2], axis=0)
  clf.S[1] = -clf.S[1]
  ## T
  clf.T = np.repeat([np.sqrt(1/n/lam)*y], repeats=[2], axis=0)
  clf.T[1] = -clf.T[1]
  ## Tau
  clf.Tau = np.repeat([kappa*np.sqrt(1/n/lam)*np.ones(n)], repeats=[2], axis=0)
  ## Fit
  clf.fit(X)