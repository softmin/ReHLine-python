Warm-Starting with ReHLine
==========================

This tutorial explains how to use warm-starting with ReHLine, a Python library for regression with hinge loss, to enhance the efficiency of solving similar optimization problems.

Introduction
------------

Warm-starting is a technique used to accelerate the convergence of optimization algorithms by initializing them with a solution from a previous run. This is particularly beneficial when you have a sequence of related problems to solve.

Setup
-----

Before you begin, ensure you have the necessary packages installed. You need the `rehline` library, which is used for regression with hinge loss, and `numpy` for numerical operations. Install these packages using pip if you haven't already:

.. code-block:: bash

    pip install rehline numpy

Simulating the Dataset
----------------------

We first create a random dataset for classification:

.. code-block:: python

    import numpy as np

    n, d, C = 1000, 3, 0.5
    X = np.random.randn(n, d)
    beta0 = np.random.randn(d)
    y = np.sign(X.dot(beta0) + np.random.randn(n))

- **n** is the number of samples.
- **d** is the number of features.
- **C** is a regularization parameter.
- **X** is the feature matrix.
- **y** is the target vector, generated as a sign function of a linear combination of features plus some noise.

Using ReHLine Solver
--------------------

The `ReHLine_solver` is tested first with a cold start and then with a warm start:

.. code-block:: python

    from rehline._base import ReHLine_solver

    U = -(C*y).reshape(1,-1)
    V = (C*np.array(np.ones(n))).reshape(1,-1)
    res = ReHLine_solver(X, U, V)  # Cold start
    res_ws = ReHLine_solver(X, U, V, Lambda=res.Lambda)  # Warm start

- **Cold Start**: The solver starts from scratch without any prior information.
- **Warm Start**: The solver uses the solution from the cold start (`res.Lambda`) as the initial point for the next run.

Using ReHLine Class
-------------------

The `ReHLine` class is used to fit a model:

.. code-block:: python

    from rehline import ReHLine

    clf = ReHLine(verbose=1)
    clf.C = C
    clf.U = -y.reshape(1,-1)
    clf.V = np.array(np.ones(n)).reshape(1,-1)
    clf.fit(X)  # Cold start

    clf.C = 2*C
    clf.warm_start = 1
    clf.fit(X)  # Warm start

- **Cold Start**: The class is fitted with the initial data.
- **Warm Start**: The class is fitted again with a different regularization parameter (`2*C`), using the previous solution as a starting point.

Using plqERM_Ridge
------------------

Finally, the `plqERM_Ridge` class is tested similarly:

.. code-block:: python

    from rehline import plqERM_Ridge

    clf = plqERM_Ridge(loss={'name': 'svm'}, C=C, verbose=1)
    clf.fit(X=X, y=y)  # Cold start

    clf.C = 2*C
    clf.warm_start = 1
    clf.fit(X=X, y=y)  # Warm start

