from benchopt import BaseDataset, safe_import_context
from sklearn.datasets import make_classification

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_samples, n_features': [
            (500, 100),
            (5000, 100),
            (50000, 100),
            
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
        X = np.random.randn(n,d)
        beta0 = np.random.randn(d)
        y = X@beta0 + 0.1*np.random.randn(n)

        data = dict(X=X, y=y, q=q)

        return self.n_features, data
