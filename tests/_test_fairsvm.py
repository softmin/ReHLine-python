## Test SVM on simulated dataset
import numpy as np
from rehline import ReHLine 
from rehline import make_fair_classification

np.random.seed(1024)
# simulate classification dataset
X, y, X_sen = make_fair_classification()
n, d = X.shape
C = 0.5

## solution provided by ReHLine
# build-in hinge loss for svm
clf = ReHLine(loss={'name': 'svm'}, C=C)
clf.make_ReLHLoss(X=X, y=y, loss={'name': 'svm'})

# specific the param of FairSVM
A = np.repeat([X_sen @ X], repeats=[2], axis=0) / n
A[1] = -A[1]
# suppose the fair tolerance is 0.01
b = np.array([.01, .01])
clf.A, clf.b = A, b
clf.fit(X=X)

print('solution privided by rehline: %s' %clf.coef_)
score = X@clf.coef_
cor_sen = np.mean(score * X_sen)
print('correlation btw score and X_sen is: %.3f' %cor_sen)
