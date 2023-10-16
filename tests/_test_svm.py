## Test SVM on simulated dataset
import numpy as np
from rehline import ReHLine

np.random.seed(1024)
# simulate classification dataset
n, d, C = 1000, 3, 0.5
X = np.random.randn(1000, 3)
beta0 = np.random.randn(3)
y = np.sign(X.dot(beta0) + np.random.randn(n))

## solution provided by sklearn
from sklearn.svm import LinearSVC
clf = LinearSVC(C=C, loss='hinge', fit_intercept=False, 
                random_state=0, tol=1e-6, max_iter=1000000)
clf.fit(X, y)
sol = clf.coef_.flatten()

print('solution privided by liblinear: %s' %sol)

## solution provided by ReHLine
# build-in loss
clf = ReHLine(loss={'name': 'svm'}, C=C)
clf.make_ReLHLoss(X=X, y=y, loss={'name': 'svm'})
clf.fit(X=X)

print('solution privided by rehline: %s' %clf.coef_)
print(clf.decision_function([[.1,.2,.3]]))

# manually specify params
n, d = X.shape
U = -(C*y).reshape(1,-1)
L = U.shape[0]
V = (C*np.array(np.ones(n))).reshape(1,-1)

clf = ReHLine(loss={'name': 'svm'}, C=C)
clf.U, clf.V = U, V
clf.fit(X=X)

print('solution privided by rehline: %s' %clf.coef_)
print(clf.decision_function([[.1,.2,.3]]))