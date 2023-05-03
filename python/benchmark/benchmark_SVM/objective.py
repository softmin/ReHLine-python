import numpy as np
from benchopt import BaseObjective

class Objective(BaseObjective):
    min_benchopt_version = "1.3"
    name = "SVM"

    parameters = {
        'C': [1.],
    }

    def __init__(self, C):
        self.C = C

    def set_data(self, X, y):
        self.X, self.y = X, y

    def compute(self, beta):
        n,d = self.X.shape
        s = np.dot(self.X, beta)
        return self.C * np.sum(np.maximum(1.0 - self.y * s, 0.)) / len(self.X) + 0.5 * np.dot(beta, beta)

    def get_objective(self):
        return dict(X=self.X, y=self.y, C=self.C)