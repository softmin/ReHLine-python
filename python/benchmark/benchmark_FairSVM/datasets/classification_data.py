from benchopt import BaseDataset, safe_import_context
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "classification_data"

    # steel-plates-fault: 1941 x 34
    # philippine: 5832 x 309
    # sylva_prior: 14395 x 109
    # creditcard: 284807 x 31
    parameters = {
        'dataset_name': ['steel-plates-fault', 'philippine', 'sylva_prior', 'creditcard']
    }

    def __init__(self, dataset_name='philippine', random_state=0):
        self.dataset_name = dataset_name
        self.random_state = random_state

    def get_data(self):
        np.random.seed(self.random_state)
        dataset = fetch_openml(name=self.dataset_name)
        X = dataset.data.values
        if self.dataset_name == 'steel-plates-fault':
            y = dataset.target.values.map({'1':-1., '2':1.})
        elif self.dataset_name in ['philippine', 'creditcard']:
            y = dataset.target.values.map({'0':-1., '1':1.})
        else:
            y = dataset.target.values

        y = np.array(y, dtype=float)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        n, d = X.shape
        x_sensitive = X[:,np.random.randint(d)]
        x_sensitive = x_sensitive - np.mean(x_sensitive)
        data = dict(X=X, y=y, Z=x_sensitive)

        return self.dataset_name, data
