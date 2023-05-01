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
        n, d = self.n_samples, self.n_features
        X, y = make_classification(n_samples=n, n_features=d, n_informative=d-2, random_state=self.random_state)
        y = 2*y - 1.0
        data = dict(X=X, y=y)

        return self.n_features, data
