import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

## test classifier
from rehline import plq_Ridge_Classifier
# generate the dataset
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=8,
    n_redundant=4,
    n_repeated=0,
    n_classes=2,
    weights=[0.7, 0.3],   # imbalance
    class_sep=1.2,
    flip_y=0.01,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# set the pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", plq_Ridge_Classifier(loss={"name": "svm"})),
])

# set the parameter grid
param_grid = {
    "clf__loss": [{"name": "svm"}, {"name": "sSVM"}],
    "clf__C": [0.1, 1.0, 3.0],
    "clf__fit_intercept": [True, False],
    "clf__intercept_scaling": [0.5, 1.0, 2.0],
    "clf__max_iter": [5000, 10000],
    "clf__class_weight": [None, "balanced", {0: 1.0, 1: 2.0}],
    "clf__constraint": [
        [],                              # no constraint
        [{"name": "nonnegative"}],
        [{"name": "fair", "sen_idx": [0], "tol_sen": 0.1}],
    ],
}

# cross_val_score function
cv_scores = cross_val_score(
    pipe,
    X_train, y_train,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
print("CV scores:", cv_scores)

# perform GridSearchCV to tune the hyperparameter
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    refit=True,
    verbose=1,
)

grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print("Test accuracy:", test_acc)
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=4))

## test regressor
from rehline import plq_Ridge_Regressor

# generate the data
X, y = make_regression(
    n_samples=1500,
    n_features=15,
    n_informative=10,
    noise=10.0,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# set the pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", plq_Ridge_Regressor(loss={"name": "QR", "qt": 0.5})),
])

# set the param_grid
param_grid = {
    "reg__loss": [
        {"name": "QR", "qt": 0.5},
        {"name": "huber", "tau": 1.0},    # Huber needs tau
        {"name": "SVR", "epsilon": 0.1},  # SVR needs epsilon
    ],
    "reg__C": [0.1, 1.0, 10.0],
    "reg__fit_intercept": [True, False],
    "reg__intercept_scaling": [0.5, 1.0],
    "reg__max_iter": [5000, 8000],
    "reg__constraint": [
        [],                              # no constraint
        [{"name": "nonnegative"}],
        [{"name": "fair", "sen_idx": [0], "tol_sen": 0.1}],
    ],
}

# cross_val_score function
cv_scores = cross_val_score(
    pipe,
    X_train, y_train,
    cv=5,
    scoring="r2",
    n_jobs=-1,
)
print("CV R^2 scores:", cv_scores)
print("Mean CV R^2:", np.mean(cv_scores))

# use GridSearchCV to tune the hyperparameters
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="r2",
    cv=5,
    n_jobs=-1,
    refit=True,
    verbose=1,
)

grid.fit(X_train, y_train)
# print the best parameters and the best CV R^2 score
print("Best params:", grid.best_params_)
print("Best CV R^2:", grid.best_score_)
# use the best estimator to fit and predict the model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print("Test R^2:", r2_score(y_test, y_pred))
print("Test MSE:", mean_squared_error(y_test, y_pred))