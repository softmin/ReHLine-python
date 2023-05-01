'''
This example shows that `hqreg` provides a suboptimal solution in terms of regularized QR. 
The simulated case is not one that has been specially constructed for the purpose of analysis, 
but rather has been chosen to more aptly reflect the underlying issue. 
Noted that `hqreg` is a approximation algorithm to QR, and does not provide an assurance with regards to optimality (under the authors' choice of tuning parameters).
'''

from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_samples, n_features': [
            (500, 100),
        ]
    }

    def __init__(self, n_samples=10, n_features=50, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):
        np.random.seed(self.random_state)
        n, d = self.n_samples, self.n_features
        q = np.random.rand()
        X = np.random.randn(n, d)
        beta0 = 10*np.random.randn(d)
        out = X@beta0
        y = out + np.random.randn(n)

        data = dict(X=X, y=y, q=q)

        return self.n_features, data
