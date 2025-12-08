Constraint
**********

Supported linear constraints in ReHLine are listed in the table below.

Usage
-----

.. code:: python
   
   # list of 
   # name (str): name of the custom linear constraints
   # loss_kwargs: more keys and values for constraint parameters
   constraint = [{'name': <constraint_name>, <**constraint_kwargs>}, ...]

.. list-table::
 :align: left
 :widths: 5 20 15
 :header-rows: 1

 * - constraint
   - | args
   - | Example 

 * - **nonnegative**
   - | ``name``: 'nonnegative' or '>=0'
   - | ``constraint=[{'name': '>=0'}]``

 * - **fair**
   - | ``name``: 'fair' or 'fairness'
     | ``sen_idx``: a list contains column indices for sensitive attributes
     | ``tol_sen``: 1d array [p] of tolerance for fairness
   - | ``constraint=[{'name': 'fair', 'sen_idx': sen_idx, 'tol_sen': tol_sen}]``

 * - **custom**
   - | ``name``: 'custom'
     | ``A``: 2d array [K x d] for linear constraint coefficients
     | ``b``: 1d array [K] of constraint intercepts
   - | ``constraint=[{'name': 'custom', 'A': A, 'b': b}]``

Related Examples
----------------

.. nblinkgallery::
   :caption: Constraints
   :name: rst-link-gallery

   ../examples/FairSVM.ipynb
   ../examples/NMF.ipynb
