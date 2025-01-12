## Test CQR on simulated dataset
import numpy as np
from rehline._class import CQR_Ridge

def test_CQR_Ridge():
    # Simulate dataset
    np.random.seed(42)
    X = np.random.randn(1000, 2)
    beta = np.array([1, 2])
    y = X @ beta + np.random.randn(1000)
    sample_weight = np.random.rand(1000)

    # Define loss parameters
    quantiles = [0.05, 0.5, 0.95]
    n_qt = len(quantiles)

    # Initialize and fit the model
    cqr = CQR_Ridge(quantiles=quantiles)
    cqr.fit(X, y, sample_weight=sample_weight)

    # Check if the model coefficients are not None
    assert cqr.coef_.shape == (len(beta),), "Model shape is incorrect"
    assert cqr.intercept_.shape == (n_qt,), "Intercept shape is incorrect"
    assert cqr.quantiles_.shape == (n_qt,), "Quantiles shape is incorrect"

    # Check decision function output
    pred = cqr.predict(X[:5])
    assert pred.shape == (5,n_qt), "Decision function output shape is incorrect"

    print(f"Coefficients of CQR_Ridge: {cqr.coef_}; Intercepts of CQR_Ridge: {cqr.intercept_}")

    print("All tests passed.")

if __name__ == "__main__":
    test_CQR_Ridge()
