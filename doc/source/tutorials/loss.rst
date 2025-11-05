Loss
****

Supported loss functions in ReHLine are listed in the table below.

Usage
-----

.. code:: python
   
   # name (str): name of the custom loss function
   # loss_kwargs: more keys and values for loss parameters
   loss = {'name': <loss_name>, <**loss_kwargs>}



Classification loss
~~~~~~~~~~~~~~~~~~~

.. list-table::
 :align: left
 :widths: 5 20 15
 :header-rows: 1

 * - loss
   - | args
   - | Example 

 * - **SVM**
   - | ``name``: 'hinge' / 'svm' / 'SVM'
   - | ``loss={'name': 'SVM'}``

 * - **Smooth SVM**
   - | ``name``: 'sSVM' / 'smooth SVM' / 'smooth hinge'
   - | ``loss={'name': 'sSVM'}``

 * - **Squared SVM**
   - | ``name``: 'squared SVM' / 'squared svm' / 'squared hinge'
   - | ``loss={'name': 'squared SVM'}``

Regression loss
~~~~~~~~~~~~~~~

.. list-table::
 :align: left
 :widths: 5 20 15
 :header-rows: 1

 * - loss
   - | args
   - | Example 

 * - **Quantile Reg**
   - | ``name``: 'check' / 'quantile' / 'QR'
     | ``qt`` (*float*): qt
   - | ``loss={'name': 'QR', 'qt': 0.25}``

 * - **Huber**
   - | ``name``: 'huber' / 'Huber'
   - | ``loss={'name': 'huber'}``

 * - **SVR**
   - | ``name``: 'SVR' / 'svr'
     | ``epsilon`` (*float*): 0.1
   - | ``loss={'name': 'svr', 'epsilon': 0.1}``

 * - **MAE**
   - | ``name``: 'MAE' / 'mae' / 'mean absolute error'
   - | ``loss={'name': 'mae'}``

 * - **MSE**
   - | ``name``: 'MSE' / 'mse' / 'mean squared error'
   - | ``loss={'name': 'mse'}``

Related Examples
----------------

.. nblinkgallery::
   :caption: Constraints
   :name: rst-link-gallery

   ../examples/QR.ipynb
   ../examples/SVM.ipynb
