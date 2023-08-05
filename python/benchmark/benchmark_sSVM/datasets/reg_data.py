from benchopt import BaseDataset, safe_import_context
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "reg_data"

    # liver-disorders: 345 x 6
    # kin8nm: 8191 x 9
    # topo_2_1: 8885 x 267
    # house_8L: 22784 x 9
    # Buzzinsocialmedia_Twitter: 583250 x 78
    parameters = {
        'dataset_name': ['liver-disorders', 'kin8nm', 'house_8L', 'topo_2_1', 'Buzzinsocialmedia_Twitter']
    }

    def __init__(self, dataset_name='liver-disorders', random_state=0):
        self.dataset_name = dataset_name
        self.random_state = random_state

    def get_data(self):
        np.random.seed(self.random_state)
        dataset = fetch_openml(name=self.dataset_name)
        scaler = StandardScaler()
        X = dataset.data.values
        y = dataset.target.values

        X = scaler.fit_transform(X)
        q = 0.1
        data = dict(X=X, y=y, q=q)

        return self.dataset_name, data
