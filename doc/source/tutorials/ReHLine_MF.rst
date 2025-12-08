ReHLine: Matrix Factorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tutorial illustrates how to conduct Matrix Factorization (MF) with multiple PLQ loss functions through ReHLine. 

Mathematical Formulation
------------------------

Considering a User-Item-Rating triplet dataset :math:`(u, i, r_{ui})` derived from target sparse matrix, the optimization problem corresponding to this scenario is:

.. math::
        \min_{\substack{
            \mathbf{P} \in \mathbb{R}^{n \times k}\ 
            \pmb{\alpha} \in \mathbb{R}^n \\
            \mathbf{Q} \in \mathbb{R}^{m \times k}\ 
            \pmb{\beta} \in \mathbb{R}^m
        }} 
        \left[
            \sum_{(u,i)\in \Omega} C \cdot \text{PLQ}(r_{ui}, \ \mathbf{p}_u^T \mathbf{q}_i + \alpha_u + \beta_i) 
        \right]  
        + 
        \left[ 
            \frac{\rho}{n}\sum_{u=1}^n(\|\mathbf{p}_u\|_2^2 + \alpha_u^2) 
            + \frac{1-\rho}{m}\sum_{i=1}^m(\|\mathbf{q}_i\|_2^2 + \beta_i^2) 
        \right]

.. math::
        \ \text{ s.t. } \ 
        \mathbf{A}_{\text{user}} \begin{pmatrix} \alpha_u \\ \mathbf{p}_u \end{pmatrix} + \mathbf{b}_{\text{user}} \geq \mathbf{0},\ u = 1,\dots,n
        \quad \text{and} \quad
        \mathbf{A}_{\text{item}} \begin{pmatrix} \beta_i \\ \mathbf{q}_i \end{pmatrix} + \mathbf{b}_{\text{item}} \geq \mathbf{0},\ i = 1,\dots,m

where

- :math:`\text{PLQ}(\cdot , \cdot)` 
  is a convex piecewise linear-quadratic loss function. You can find built-in loss functions in the `Loss <./loss.rst>`_ section.
  
- :math:`\mathbf{A}_{\text{user}}` is a :math:`d \times (k+1)` matrix and :math:`\mathbf{b}_{\text{user}}` is a :math:`d`-dimensional vector 
  representing :math:`d` linear constraints to user side parameters. See `Constraints <./constraint.rst>`_ for more details.

- :math:`\mathbf{A}_{\text{item}}` is a :math:`d \times (k+1)` matrix and :math:`\mathbf{b}_{\text{item}}` is a :math:`d`-dimensional vector 
  representing :math:`d` linear constraints to item side parameters. See `Constraints <./constraint.rst>`_ for more details.

- :math:`\Omega`
  is a user-item collection that records all training data

- :math:`n` is number of users, :math:`m` is number of items

- :math:`k` is length of latent factors (rank of MF) 

- :math:`C` is regularization parameter, :math:`\rho` balances regularization strength between user and item

- :math:`\mathbf{p}_u` and :math:`\alpha_u`
  are latent vector and individual bias of u-th user. Specifically, :math:`\mathbf{p}_u` is the u-th row of :math:`\mathbf{P}`, and :math:`\alpha_u` is the u-th element of :math:`\pmb{\alpha}`
  
- :math:`\mathbf{q}_i` and :math:`\beta_i`
  are latent vector and individual bias of i-th item. Specifically, :math:`\mathbf{q}_i` is the i-th row of :math:`\mathbf{Q}`, and :math:`\beta_i` is the i-th element of :math:`\pmb{\beta}`


Implementation Guide
--------------------

A simple synthetic dataset is used for illustration. The implementation can be easily adapted to your specific triplet data, allowing you to experiment with various loss functions.

Setup
^^^^^

To proceed, ensure that you have already installed :code:`rehline`:

.. code-block:: bash

    pip install rehline

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   # 1. Necessary Packages
   import numpy as np
   from rehline import plqMF_Ridge, make_mf_dataset
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_absolute_error


   # 2. Data Preparation
   # Generate synthetic data (replace with your own data in practice)
   user_num, item_num = 1200, 4000 
   ratings = make_mf_dataset(n_users=user_num, n_items=item_num, 
                             n_interactions=50000, seed=42)
   
   # Split into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(
       ratings['X'], ratings['y'], test_size=0.3, random_state=42)


   # 3. Model Construction
   clf = plqMF_Ridge(
       C=0.001,                             ## Regularization strength
       rank=6,                              ## Latent factor dimension
       loss={'name': 'mae'},                ## Use absolute loss
       n_users=user_num,                    ## Number of users
       n_items=item_num,                    ## Number of items
   )
   clf.fit(X_train, y_train)


   # 4. Evaluation
   y_pred = clf.decision_function(X_test)
   mae_score = mean_absolute_error(y_test, y_pred)
   print(f"Test MAE: {mae_score:.3f}")
 
Advanced Configuration
^^^^^^^^^^^^^^^^^^^^^^

Choosing different `loss functions <./loss.rst>`_ through :code:`loss`:

.. code-block:: python

   # Square loss
   clf_mse = plqMF_Ridge(
        C=0.001, 
        rank=6, 
        loss={'name': 'mse'},               ## Choose square loss
        n_users=user_num, 
        n_items=item_num)
   
   # Hinge loss (suitable for binary data)
   clf_hinge = plqMF_Ridge(
        C=0.001, 
        rank=6, 
        loss={'name': 'hinge'},             ## Choose hinge loss
        n_users=user_num, 
        n_items=item_num)

`Linear constraints <./constraint.rst>`_ can be applied via :code:`constraint_user` and :code:`constraint_item`:

.. code-block:: python

   # Implement a linear constraint 
   clf_nonnegative = plqMF_Ridge(
        C=0.001, 
        rank=6, 
        loss={'name': 'mae'},
        n_users=user_num, 
        n_items=item_num,
        constraint_user=[{'name': '>=0'}],  ## Use nonnegative constraint
        constraint_item=[{'name': '>=0'}]
    )
  
The algorithm includes bias terms :math:`\mathbf{\alpha}` and :math:`\mathbf{\beta}` by default. To disable them, that is, :math:`\mathbf{\alpha} = \mathbf{0}` and :math:`\mathbf{\beta} = \mathbf{0}`, set: :code:`biased=False`:

.. code-block:: python

   # Exclude user and item biases
   clf_unbiased = plqMF_Ridge(
        C=0.001, 
        rank=6, 
        loss={'name': 'mae'},
        n_users=user_num, 
        n_items=item_num,
        biased=False                        ## Disable bias terms
    )
  
Imposing different strengths of regularization on items/users through :code:`rho`:

.. code-block:: python

   # Imbalanced penalty 
   clf_asymmetric = plqMF_Ridge(
        C=0.001, 
        rank=6, 
        loss={'name': 'mae'},
        n_users=user_num, 
        n_items=item_num,
        rho=0.7                             ## Add heavier penalties for user parameters
    )

Parameter Tuning
^^^^^^^^^^^^^^^^

The model complexity is mainly controlled by :code:`C` and :code:`rank`. 

.. code-block:: python

   
   for C_value in [0.0002, 0.001, 0.005]:
       clf = plqMF_Ridge(
            C=C_value,                      ## Try different regularization strengths
            rank=6, 
            loss={'name': 'mae'},
            n_users=user_num, 
            n_items=item_num
        )
       clf.fit(X_train, y_train)
       y_pred = clf.decision_function(X_test)
       mae = mean_absolute_error(y_test, y_pred)
       print(f"C={C_value}: MAE = {mae:.3f}")


   for rank_value in [4, 8, 12]:
       clf = plqMF_Ridge(
            C=0.001, 
            rank=rank_value,                ## Try different latent factor dimensions
            loss={'name': 'mae'},
            n_users=user_num, 
            n_items=item_num
        )
       clf.fit(X_train, y_train)
       y_pred = clf.decision_function(X_test)
       mae = mean_absolute_error(y_test, y_pred)
       print(f"rank={rank_value}: MAE = {mae:.3f}")

Practical Guidance
^^^^^^^^^^^^^^^^^^

- The first column of :code:`X` corresponds to **users**, and the second column corresponds to **items**. Please ensure this aligns with your :code:`n_users` and :code:`n_items` parameters.
- The default penalty strength is relatively weak; it is recommended to set a relatively small :code:`C` value initially.
- When using larger :code:`C` values, consider increasing :code:`max_iter` to avoid ConvergenceWarning.

Example
-------

.. nblinkgallery::
   :caption: Empirical Risk Minimization
   :name: rst-link-gallery

   ../examples/NMF.ipynb
