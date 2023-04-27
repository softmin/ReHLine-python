import numpy as np
from benchopt import BaseObjective

class Objective(BaseObjective):
    min_benchopt_version = "1.3"
    name = "FairSVM"

    parameters = {
        'obj': [0,1],
        'C': [1.],
        'rho': [0.01]
    }

    def __init__(self, obj, C, rho):
        self.obj = obj
        self.C = C
        self.rho = rho

    def set_data(self, X, y, Z):
        self.X, self.y, self.Z = X, y, Z

    def compute(self, beta):
        n,d = self.X.shape
        s = np.dot(self.X, beta)
        if self.obj:
            return self.C * np.sum(np.maximum(1.0 - self.y * s, 0.)) / len(self.X) + 0.5 * np.dot(beta, beta)
        else:
            constrain = abs(np.dot(self.Z, s) / len(self.X)) - self.rho
            return max(constrain, 1e-6)

    def get_objective(self):
        return dict(X=self.X, y=self.y, Z=self.Z, rho=self.rho, C=self.C)