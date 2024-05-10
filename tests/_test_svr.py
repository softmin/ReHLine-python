## Test SVR on simulated dataset
import numpy as np
from rehline import ReHLine

np.random.seed(1024)
# simulate regression dataset
n, d, C = 10000, 5, 0.5
X = np.random.randn(n, d)
beta0 = np.random.randn(d)
print(beta0)
y = X.dot(beta0) + np.random.randn(n)
new_sample = np.random.randn(d)

## solution provided by sklearn
from sklearn.svm import LinearSVR
reg = LinearSVR(C=C, loss='epsilon_insensitive', fit_intercept=False, epsilon=1e-5,
                random_state=0, tol=1e-6, max_iter=1000000, dual='auto')
reg.fit(X, y)
sol = reg.coef_.flatten()

print('solution privided by liblinear: %s' %sol)
print(reg.predict([new_sample]))

## solution provided by ReHLine
# build-in loss
loss_dict = {'name': 'svr', 'epsilon': 1e-5}
reg = ReHLine(loss=loss_dict, C=C)
reg.make_ReLHLoss(X=X, y=y, loss=loss_dict)
reg.fit(X=X)

print('solution privided by rehline: %s' %reg.coef_)
print(reg.decision_function([new_sample]))

# manually specify params
n, d = X.shape

U = np.ones((2, n))*C
V = np.ones((2, n))
U[1] = -U[1]
V[0] = -C*(y + loss_dict['epsilon'])
V[1] = C*(y - loss_dict['epsilon'])

reg = ReHLine(loss=loss_dict, C=C)
reg.U, reg.V = U, V
reg.fit(X=X)

print('solution privided by rehline (manually specified params): %s' %reg.coef_)
print(reg.decision_function([new_sample]))

# Output:
# [-0.26024832 -0.29394989  0.05549916  2.24410393 -1.47306613]
# solution privided by liblinear: [-0.266752   -0.28534044  0.05883864  2.24027556 -1.4996596 ]
# [3.65233956]
# solution privided by rehline: [-0.26742525 -0.28575844  0.05840424  2.24040812 -1.50018069]
# [3.65242688]
# solution privided by rehline (manually specified params): [-0.26742525 -0.28575844  0.05840424  2.24040812 -1.50018069]
# [3.65242688]