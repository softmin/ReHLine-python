import numpy as np
from benchopt import BaseObjective

class Objective(BaseObjective):
    min_benchopt_version = "1.3"
    name = "ElasticQR"

    parameters = {
        'lam1': [1.0],
        'lam2': [1.0]
    }

    def __init__(self, lam1, lam2):
        self.lam1 = lam1
        self.lam2 = lam2

    def set_data(self, X, y, q):
        self.X, self.y, self.q = X, y, q

    def compute(self, beta):
        n, d = self.X.shape
        out = np.dot(self.X, beta[:-1]) + beta[-1]
        loss = self.q * np.maximum(self.y - out, 0) + (1 - self.q) * np.maximum(out - self.y, 0)
        reg = self.lam1 * np.sum(np.abs(beta)) + self.lam2 * np.sum(beta**2) / 2
        return np.mean(loss) + reg

    def get_objective(self):
        return dict(X=self.X, y=self.y, q=self.q, lam1=self.lam1, lam2=self.lam2)
