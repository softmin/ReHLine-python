
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


def make_fair_classification(n_samples=100, n_features=5, ind_sensitive=0):
    """
    Generate a random binary fair classification problem.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=5
        The total number of features. 

    ind_sensitive : int, default=0
        The index of the sensitive feature.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The +/- labels for class membership of each sample.

    X_sen: ndarray of shape (n_samples,)
        The centered samples of the sensitive feature.
    """

    X, y = make_classification(n_samples, n_features)
    y = 2*y - 1

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_sen = X[:, ind_sensitive]

    return X, y, X_sen


def make_ratings(n_users, n_items, n_factors=20,
             n_interactions=None, density=0.01, 
             noise_std=0.1, seed=None, 
             rating_min=1.0, rating_max=5.0, return_params=True):
    """Generate synthetic rating data."""
    rng = np.random.RandomState(seed)
    
    # Calculate interactions
    total_pairs = n_users * n_items
    n_interactions = n_interactions or int(total_pairs * density)
    n_interactions = min(n_interactions, total_pairs)
    
    # Generate factors and biases
    scale = 1 / np.sqrt(n_factors)
    P = rng.normal(0, scale, (n_users, n_factors))
    Q = rng.normal(0, scale, (n_items, n_factors))
    bu = rng.normal(0, 0.5, n_users)
    bi = rng.normal(0, 0.5, n_items)
    
    # Sample interactions
    flat_idx = rng.choice(total_pairs, n_interactions, False)
    users, items = flat_idx // n_items, flat_idx % n_items
    
    # Compute ratings
    dot_vals = (P[users] * Q[items]).sum(axis=1)
    noise = rng.normal(0, noise_std, n_interactions)
    mu = (rating_min + rating_max) / 2
    y = np.clip(mu + bu[users] + bi[items] + dot_vals + noise, rating_min, rating_max)
    
    # Return results
    result = {"X": np.column_stack([users, items]), "y": y}
    if return_params:
        result["params"] = {"P": P, "Q": Q, "bu": bu, "bi": bi, "mu": mu}

return result
