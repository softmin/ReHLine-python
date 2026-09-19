from numbers import Integral

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

from ._validation import positive_real


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
    y = 2 * y - 1

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_sen = X[:, ind_sensitive]

    return X, y, X_sen


def _sample_pair_indices(rng, population, size):
    """Uniform ordered sampling without replacement using O(size) storage."""
    if size == 0:
        return np.empty(0, dtype=np.int64)
    if size * 10 >= population:
        # Here the population is at most ten times the requested output size.
        return rng.choice(population, size, replace=False).astype(np.int64, copy=False)
    # Partial Fisher-Yates: store only positions displaced by earlier draws.
    # Choosing j uniformly from [i, population) gives every remaining item the
    # same probability, including when the virtual population is enormous.
    result = np.empty(size, dtype=np.int64)
    displaced = {}
    for i in range(size):
        j = int(rng.randint(i, population, dtype=np.int64))
        result[i] = displaced.get(j, j)
        if j != i:
            displaced[j] = displaced.get(i, i)
        displaced.pop(i, None)
    return result


def make_mf_dataset(
    n_users,
    n_items,
    n_factors=20,
    n_interactions=None,
    density=0.01,
    noise_std=0.1,
    seed=None,
    rating_min=1.0,
    rating_max=5.0,
    return_params=True,
):
    """
    Generate synthetic rating data using matrix factorization model.

    Creates synthetic user-item rating data based on the matrix factorization
    approach commonly used in recommender systems. The ratings are generated
    as: rating = mu + user_bias + item_bias + user_factor * item_factor + noise

    Parameters
    ----------
    n_users : int
        Non-negative number of users in the synthetic dataset.

    n_items : int
        Non-negative number of items. n_users * n_items must fit in int64.

    n_factors : int, default=20
        Positive number of latent factors for user and item embeddings.

    n_interactions : int, optional
        Non-negative number of unique user-item pairs, capped at total_pairs.
        Zero produces empty data. If None, uses int(density * total_pairs).

    density : float, default=0.01
        Finite density in [0, 1], ignored if n_interactions is specified.

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

        - **X** : ndarray of shape (n_interactions, 2)
            User-item pairs where X[:, 0] are user indices and X[:, 1] are item indices
        - **y** : ndarray of shape (n_interactions,)
            Synthetic ratings for each user-item pair
        - **params** : dict, optional
            Only returned if return_params=True. Contains:

            * **P** : ndarray of shape (n_users, n_factors)
                User factor matrix
            * **Q** : ndarray of shape (n_items, n_factors)
                Item factor matrix
            * **bu** : ndarray of shape (n_users,)
                User biases
            * **bi** : ndarray of shape (n_items,)
                Item biases
            * **mu** : float
                Global mean rating

    Notes
    -----
    The rating generation follows the standard matrix factorization model:

        r_ui = μ + b_u + b_i + p_u · q_i^T + ε

        where ε ~ N(0, noise_std²)

    The generated ratings are clipped to stay within [rating_min, rating_max] range.

    Pairs are sampled uniformly without replacement. Pair-index storage is
    O(n_interactions); sparse sampling does not allocate all n_users * n_items
    pairs. Factors still require O((n_users + n_items) * n_factors) storage,
    even when return_params=False.

    The same seed and arguments reproduce results within this implementation.
    Sparse sampling (fewer than 10% of pairs) uses a different random draw
    sequence from the previous full-population sampler, so those datasets and
    their ratings can differ from earlier releases for the same seed.
    """
    # Use Python integers for the population product, avoiding NumPy integer
    # overflow before choosing an int64 flat index.
    counts = {}
    for name, value, minimum in (("n_users", n_users, 0), ("n_items", n_items, 0), ("n_factors", n_factors, 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        counts[name] = int(value)
    n_users, n_items, n_factors = (counts[name] for name in ("n_users", "n_items", "n_factors"))
    total_pairs = n_users * n_items
    if total_pairs > np.iinfo(np.int64).max:
        raise ValueError("n_users * n_items must fit in int64")
    if n_interactions is None:
        positive_real(density, "density", allow_zero=True)
        if density > 1:
            raise ValueError("density must be in [0, 1]")
        n_interactions = int(total_pairs * density)
    elif isinstance(n_interactions, (bool, np.bool_)) or not isinstance(n_interactions, Integral) or n_interactions < 0:
        raise ValueError("n_interactions must be a non-negative integer")
    n_interactions = int(n_interactions)
    n_interactions = min(n_interactions, total_pairs)
    rng = np.random.RandomState(seed)

    # Generate factors and biases
    scale = 1 / np.sqrt(n_factors)
    P = rng.normal(0, scale, (n_users, n_factors))
    Q = rng.normal(0, scale, (n_items, n_factors))
    bu = rng.normal(0, 0.5, n_users)
    bi = rng.normal(0, 0.5, n_items)

    # Sample interactions
    flat_idx = _sample_pair_indices(rng, total_pairs, n_interactions)
    users, items = flat_idx // n_items, flat_idx % n_items

    # Compute ratings
    dot_vals = (P[users] * Q[items]).sum(axis=1)
    noise = rng.normal(0, noise_std, n_interactions)
    mu = (rating_min + rating_max) / 2
    y = mu + bu[users] + bi[items] + dot_vals + noise
    y_rounded = np.round(y * 2) / 2
    y_clipped = np.clip(y_rounded, rating_min, rating_max)

    # Return results
    result = {"X": np.column_stack([users, items]), "y": y_clipped}
    if return_params:
        result["params"] = {"P": P, "Q": Q, "bu": bu, "bi": bi, "mu": mu}

    return result
