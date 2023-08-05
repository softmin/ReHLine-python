import numpy as np
from benchopt import BaseObjective

class Objective(BaseObjective):
    min_benchopt_version = "1.3"
    name = "smooth-SVM"

    parameters = {
        'C': [1.],
    }

    def __init__(self, C):
        self.C = C

    def set_data(self, X, y):
        self.X, self.y = X, y

    def compute(self, beta):
        n,d = self.X.shape
        s = self.y * np.dot(self.X, beta)
        return self.C * smooth_hinge(s) + 0.5 * np.dot(beta, beta)

    def get_objective(self):
        return dict(X=self.X, y=self.y, C=self.C)

def smooth_hinge(u, gamma=1.0):
    loss = (1.0 - u)**2 / (2*gamma)
    loss[u>=1] = 0.0
    loss[u<=(1-gamma)] = 1 - u[u<=(1-gamma)] - gamma/2
    return np.mean(loss)