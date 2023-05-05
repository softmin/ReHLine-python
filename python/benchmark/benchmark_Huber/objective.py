import numpy as np
from benchopt import BaseObjective

class Objective(BaseObjective):
    min_benchopt_version = "1.3"
    name = "ElasticHuber"

    parameters = {
        'lam1': [0.],
        'lam2': [1.],
        'tau': [1.0]
    }

    def __init__(self, tau, lam1, lam2):
        self.tau = tau
        self.lam1 = lam1
        self.lam2 = lam2

    def set_data(self, X, y):
        self.X, self.y = X, y

    def compute(self, beta):
        n, d = self.X.shape
        out = np.dot(self.X, beta)
        res = abs(self.y - out)
        loss = np.where(res>self.tau, self.tau * res - self.tau**2 / 2, res**2/2)
        reg = self.lam1 * np.sum(np.abs(beta)) + self.lam2 * np.sum(beta**2) / 2
        return np.mean(loss) + reg

    def get_objective(self):
        return dict(X=self.X, y=self.y, tau=self.tau, lam1=self.lam1, lam2=self.lam2)
