
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
    """
    Generate synthetic rating data using matrix factorization model.
    
    Creates synthetic user-item rating data based on the matrix factorization
    approach commonly used in recommender systems. The ratings are generated
    as: rating = mu + user_bias + item_bias + user_factor * item_factor + noise
    
    Parameters
    ----------
    n_users : int
        Number of users in the synthetic dataset
    n_items : int
        Number of items in the synthetic dataset
    n_factors : int, default=20
        Number of latent factors for user and item embeddings
    n_interactions : int, optional
        Exact number of user-item interactions. If None, calculated as density * total_pairs
    density : float, default=0.01
        Density of the rating matrix (ignored if n_interactions is specified)
    noise_std : float, default=0.1
        Standard deviation of Gaussian noise added to ratings
    seed : int, optional
        Random seed for reproducible results
    rating_min : float, default=1.0
        Minimum possible rating value
    rating_max : float, default=5.0
        Maximum possible rating value
    return_params : bool, default=True
        If True, returns the underlying model parameters (P, Q, bu, bi, mu)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'X' : ndarray of shape (n_interactions, 2)
            User-item pairs where X[:, 0] are user indices and X[:, 1] are item indices
        - 'y' : ndarray of shape (n_interactions,)
            Synthetic ratings for each user-item pair
        - 'params' : dict, optional
            Only returned if return_params=True. Contains:
            * 'P' : ndarray of shape (n_users, n_factors) - User factor matrix
            * 'Q' : ndarray of shape (n_items, n_factors) - Item factor matrix  
            * 'bu' : ndarray of shape (n_users,) - User biases
            * 'bi' : ndarray of shape (n_items,) - Item biases
            * 'mu' : float - Global mean rating
    
    Notes
    -----
    The rating generation follows the standard matrix factorization model:
        r_ui = μ + b_u + b_i + p_u · q_i^T + ε
    where ε ~ N(0, noise_std²)
    
    The generated ratings are clipped to stay within [rating_min, rating_max] range.
    """
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
    y = mu + bu[users] + bi[items] + dot_vals + noise
    y_rounded = np.round(y * 2) / 2
    y_clipped = np.clip(y_rounded, rating_min, rating_max)
    
    # Return results
    result = {"X": np.column_stack([users, items]), "y": y}
    if return_params:
        result["params"] = {"P": P, "Q": Q, "bu": bu, "bi": bi, "mu": mu}

    return result
