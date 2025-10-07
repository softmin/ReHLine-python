## Test SVM on simulated dataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from rehline import plqERM_Ridge

np.random.seed(1024)
# simulate classification dataset
n, d, C = 100, 5, 0.5
X, y = make_classification(n, d)
y = 2*y - 1

scaler = StandardScaler()
X = scaler.fit_transform(X)
sen_idx = [0]

## solution provided by ReHLine
# build-in hinge loss for svm
clf = plqERM_Ridge(loss={'name': 'svm'}, C=C)

# specific the param of FairSVM
X_sen = X[:,sen_idx]
A = np.repeat([X_sen.flatten() @ X], repeats=[2], axis=0) / n
A[1] = -A[1]
# suppose the fair tolerance is 0.01
b = np.array([.01, .01])
clf._A, clf._b = A, b
clf.fit(X=X, y=y)

print('solution privided by rehline: %s' %clf.coef_)
score = X@clf.coef_
cor_sen = np.mean(score * X_sen)
print('correlation btw score and X_sen is: %.3f' %cor_sen)
