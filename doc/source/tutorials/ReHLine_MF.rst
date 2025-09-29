ReHLine: Matrix Factorization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This tutorial illustrates how to conduct Matrix Factorization (MF) with multiple objective functions through ReHLine. 
Currently supported PLQ losses include:

.. math::
   \begin{aligned}
   &\text{Hinge Loss:}     && \phi(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y}) \\
   &\text{Hinge Square:}   && \phi(y, \hat{y}) = [\max(0, 1 - y \cdot \hat{y})]^2 \\
   &\text{Absolute Loss:}  && \phi(y, \hat{y}) = |y - \hat{y}| \\
   &\text{Square Loss:}    && \phi(y, \hat{y}) = (y - \hat{y})^2
   \end{aligned}




1. Problem Description
----------------------

Considering an triplet format dataset (User ID, Item ID, Ratings) derived from target sparse matrix, the optimization problem corresponding to this scenario is:

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

where

- :math:`\phi(\cdot)` 
  gives loss values corresponding to the ground truth.
  
- :math:`\Omega`
  is a user-item collection that records all training data

- :math:`n` is number of users, :math:`m` is number of items

- :math:`C` is regularization parameter, :math:`\rho` balances regularization strength between user and item

- :math:`\mathbf{p}_u` and :math:`a_u`
  are latent vector and individual bias of u-th user. Specifically, :math:`\mathbf{p}_u` is the u-th row of :math:`\mathbf{P}`, and :math:`a_u` is the u-th element of :math:`\mathbf{a}`
  
- :math:`\mathbf{q}_i` and :math:`b_i`
  are latent vector and individual bias of i-th item. Specifically, :math:`\mathbf{q}_i` is the i-th row of :math:`\mathbf{Q}`, and :math:`b_i` is the i-th element of :math:`\mathbf{b}`


Or you can choose to use an unbiased version of this algorithm, which simply optimizes:

.. math::
        \min_{\substack{
            \mathbf{P} \in \mathbb{R}^{n \times r}\\
            \mathbf{Q} \in \mathbb{R}^{m \times r}
        }} 
        \left[
            \sum_{(u,i)\in \Omega} C \cdot \phi(\mathbf{p}_u^\top \mathbf{q}_i) 
        \right]  
        + 
        \left[ 
            \frac{\rho}{n}\sum_{u=1}^n\|\mathbf{p}_u\|_2^2 
            + \frac{1-\rho}{m}\sum_{i=1}^m\|\mathbf{q}_i\|_2^2 
        \right]
        




2. Algorithm Explanation
------------------------

Within the Coordinate Descent (CD) framework, this algorithm conducts optimization by alternately updating the user side parameters and the item side parameters. To formulate the optimization problem, we define:

- :math:`U_i`: Users who rated item :math:`i`
- :math:`I_u`: Items rated by user :math:`u`


🛠️Biased Version: :math:`\hat{y}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i + a_u + b_i`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**STEP1: User Side Update**

With item side fixed, the objective function for updating user side parameters reduces to:

.. math::
        \min_{
            \mathbf{P} \in \mathbb{R}^{n \times r}, 
            \mathbf{a} \in \mathbb{R}^n
        }
       \sum_{u=1}^n 
       \left\{
          \left[ \sum_{i \in I_u} C \cdot \phi(\, \mathbf{q}_i^\top \mathbf{p}_u + a_u + b_i \,) \right]
          + \frac{\rho}{n} ( \lVert \mathbf{p}_u \rVert_2^2 + a_u^2 )
       \right\}
       + \text{const}

That is, solving the following sub-optimization for each user:

.. math::
        \min_{
            \mathbf{p}_u \in \mathbb{R}^r, 
            a_u \in \mathbb{R}
        } 
         \sum_{i \in I_u} \frac{Cn}{2\rho} \cdot \phi(\, \mathbf{q}_i^\top \mathbf{p}_u + a_u + b_i \,)
        + \frac{1}{2} ( \lVert \mathbf{p}_u \rVert_2^2 + a_u^2 )

Denoting :math:`\beta_u = \begin{bmatrix} a_u \\ \mathbf{p}_u \end{bmatrix}`, :math:`\mathbf{x}_i = \begin{bmatrix} 1 \\ \mathbf{q}_i \end{bmatrix}` and :math:`C_{user}=\frac{Cn}{2\rho}`, sub-optimization now becomes:

.. math::
        \min_{
            \beta_u \in \mathbb{R}^{r+1}
        } 
         \sum_{i \in I_u} C_{user} \cdot \phi(\, \mathbf{x}_i^\top \beta_u + b_i \,)
        + \frac{1}{2}\lVert \beta_u \rVert_2^2

By applying a transformation on the intercept term after ReLU-ReHU decomposition, the above sub-optimization problem is actually equivalent to a ReHLine optimization (see proof at Appendix). 

After each sub-optimization, denoting result as :math:`\beta^*_u`, user side parameters will be updated by: 

.. math::
  a_u \leftarrow \beta^*_u[0], \quad
  \mathbf{p}_u \leftarrow \beta^*_u[1:]

Here the intercept term of the ReHLine optimization is used as user bias, and remaining coefficient part is used as user latent vector.

**STEP2: Item Side Update**

With user side fixed, the objective function for updating item side parameters reduces to:

.. math::
        \min_{
            \mathbf{Q} \in \mathbb{R}^{m \times r}, 
            \mathbf{b} \in \mathbb{R}^m
        }
       \sum_{i=1}^m 
       \left\{
          \left[ \sum_{u \in U_i} C \cdot \phi(\, \mathbf{p}_u^\top \mathbf{q}_i + b_i + a_u \,) \right]
          + \frac{1-\rho}{m} ( \lVert \mathbf{q}_i \rVert_2^2 + b_i^2 )
       \right\}
       + \text{const}

By denoting :math:`\beta_i = \begin{bmatrix} b_i \\ \mathbf{q}_i \end{bmatrix}`, :math:`\mathbf{x}_u = \begin{bmatrix} 1 \\ \mathbf{p}_u \end{bmatrix}` and :math:`C_{item}=\frac{Cm}{2(1-\rho)}`, sub-optimization for each item becomes:

.. math::
        \min_{
            \beta_i \in \mathbb{R}^{r+1}
        } 
         \sum_{u \in U_i} C_{item} \cdot \phi(\, \mathbf{x}_u^\top \beta_i + a_u \,)
        + \frac{1}{2}\lVert \beta_i \rVert_2^2

Similarly, this is also a ReHLine optimization (see proof at Appendix). 

After each sub-optimization, denoting result as :math:`\beta^*_i`, item side parameters will be updated by: 

.. math::
  b_i \leftarrow \beta^*_i[0], \quad
  \mathbf{q}_i \leftarrow \beta^*_i[1:]

Here the intercept term of the ReHLine optimization is used as item bias, and remaining coefficient part is used as item latent vector.


🔧Unbiased Version: :math:`\hat{y}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**STEP1: User Side Update**

With item side fixed, the objective function for updating the user side parameters reduces to:

.. math::
        \min_{
            \mathbf{P} \in \mathbb{R}^{n \times r}
        }
       \sum_{u=1}^n 
       \left[
           \sum_{i \in I_u} C \cdot \phi(\, \mathbf{q}_i^\top \mathbf{p}_u \,) 
          + \frac{\rho}{n} \lVert \mathbf{p}_u \rVert_2^2
       \right]
       + \text{const}

By denoting :math:`C_{user}=\frac{Cn}{2\rho}`, it's quite intuitive that sub-optimization for each user is a ReHLine optimization:

.. math::
        \min_{
            \mathbf{p}_u \in \mathbb{R}^{r}
        } 
        \sum_{i \in I_u} C_{user} \cdot \phi(\, \mathbf{q}_i^\top \mathbf{p}_u \,)
        + \frac{1}{2}\lVert \mathbf{p}_u \rVert_2^2

After each sub-optimization, denoting result as :math:`\mathbf{p}^*_u`, user side parameters will be updated by: 

.. math::
  \mathbf{p}_u \leftarrow \mathbf{p}^*_u

**STEP2: Item Side Update**

With user side fixed, the objective function for updating the item side parameters reduces to:

.. math::
        \min_{
            \mathbf{Q} \in \mathbb{R}^{m \times r}
        }
       \sum_{i=1}^m 
       \left[
           \sum_{u \in U_i} C \cdot \phi(\, \mathbf{p}_u^\top \mathbf{q}_i \,) 
          + \frac{1-\rho}{m} \lVert \mathbf{q}_i \rVert_2^2
       \right]
       + \text{const}

By denoting :math:`C_{item}=\frac{Cm}{2(1-\rho)}`, it's quite intuitive that sub-optimization for each item is a ReHLine optimization:

.. math::
        \min_{
            \mathbf{q}_i \in \mathbb{R}^{r}
        } 
        \sum_{u \in U_i} C_{item} \cdot \phi(\, \mathbf{p}_u^\top \mathbf{q}_i \,)
        + \frac{1}{2}\lVert \mathbf{q}_i \rVert_2^2

After each sub-optimization, denoting result as :math:`\mathbf{q}^*_i`, item side parameters will be updated by: 

.. math::
  \mathbf{q}_i \leftarrow \mathbf{q}^*_i








3. Regularization Strength Conversion
-------------------------------------
The regularization in this algorithm is tuned via :math:`C` and :math:`\rho`. For users who prefer to set the penalty strength directly, the following equivalents can be used:

.. math::
        \lambda_{\text{user}} = \frac{\rho}{Cn}
        \quad\text{and}\quad  
        \lambda_{\text{item}} = \frac{(1 - \rho)}{Cm}


.. math::
        C = \frac{1}{m \cdot \lambda_{\text{item}} + n \cdot \lambda_{\text{user}}}
        \quad\text{and}\quad  
        \rho = \frac{1}{\frac{m \cdot \lambda_{\text{item}}}{ n \cdot \lambda_{\text{user}}}+1}


4. Implementation Guide
-----------------------

To get started, ReHLine provides `MovieLens 100K <https://grouplens.org/datasets/movielens/100k/>`_ dataset. The implementation can be easily adapted to your specific **User-Item-Rating** data, allowing you to experiment with various loss functions.

.. code-block:: python

  # Packages
  import numpy as np
  from rehline import plqMF_Ridge, load_dataset
  from sklearn.model_selection import train_test_split

  # Data Preparation
  X, y = load_dataset("ml-100k", return_X_y=True) # load MovieLens-100k dataset
  user_num, item_num = np.max(X, axis=0) + 1 # user number & item number
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42) # split data into training set & testing set
  
  # Model Construction
  clf = plqMF_Ridge(C = 0.0001, # Default penalty strength is weak, it is recommended to set a relatively small C value
                    rank = 6,
                    loss={'name': 'mse'},
                    n_users=user_num, n_items=item_num)
  clf.fit(X_train, y_train) # fit the model
  
  # Evaluation
  training_rmse = np.sqrt( clf.history[-1, 0] / clf.n_ratings )
  y_pred = clf.decision_function(X_test)
  testing_rmse = np.sqrt( np.mean((y_pred - y_test)**2) )
  
  print(f"Training RMSE: {training_rmse:.3f}")
  print(f"Testing  RMSE: {testing_rmse:.3f}")
  

5. Appendix
-----------

This section provides the proof of each sub-optimization in user side & item side is still a ReHLine problem. Consider a PLQ-ERM optimization, but now each observation has a unique individual bias:

.. math::
    \min_{
        \beta \in \mathbb{R}^{d}
    } 
     \sum_{i=1}^n L_i(\, X_i^\top \beta + \gamma_i \,)
    + \frac{1}{2}\lVert \beta \rVert_2^2

where :math:`\gamma_i` is individual bias. By applying ReLU-ReHU decomposition to :math:`L_i(\, X_i^\top \beta + \gamma_i \,)`, we have:

.. math::
   \begin{align*}
   L_i(X_i \beta + \gamma_i) 
   &= \sum_{l=1}^L \text{ReLU}\big[ \mathtt{u}_{li} (X_i \beta + \gamma_i) + \mathtt{v}_{li} \big] 
      + \sum_{h=1}^H \text{ReHU}_{\tau_{hi}}\big[ \mathtt{s}_{hi} (X_i \beta + \gamma_i) + \mathtt{t}_{hi} \big] \\
   &= \sum_{l=1}^L \text{ReLU}\big( \mathtt{u}_{li} X_i \beta + \mathtt{v}^*_{li} \big) 
      + \sum_{h=1}^H \text{ReHU}_{\tau_{hi}}\big( \mathtt{s}_{hi} X_i \beta + \mathtt{t}^*_{hi} \big)
   \end{align*}

where

.. math::
   \begin{align*}
   \mathtt{v^*}_{li} &= \mathtt{v}_{li} + \mathtt{u}_{li} \gamma_i \\
   \mathtt{t^*}_{hi} &= \mathtt{t}_{hi} + \mathtt{s}_{hi} \gamma_i
   \end{align*}

Plug above transformation of :math:`L_i(\cdot)` into the objective function to obtain:

.. math::
    \min_{
        \beta \in \mathbb{R}^{d}
    } 
    \left\{
        \sum_{i=1}^n
        \left[
            \sum_{l=1}^L ReLU(\mathtt{u}_{li} X_i \beta + \mathtt{v}^*_{li}) + \sum_{h=1}^H ReHU_{\tau_{hi}}(\mathtt{s}_{hi} X_i \beta + \mathtt{t}^*_{hi})  
        \right]
        + \frac{1}{2}\lVert \beta \rVert_2^2
    \right\}

Above optimization is still a ReHLine problem.





