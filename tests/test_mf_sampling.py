"""Uniform ordered sampling, reproducible ratings, and sparse memory bounds."""

import tracemalloc
from itertools import permutations, product

import numpy as np
import pytest

from rehline import make_mf_dataset
from rehline._data import _sample_pair_indices


def test_every_sparse_ordered_sample_has_the_same_number_of_draw_sequences():
    # Exhaust all equally likely pairs of draws in partial Fisher-Yates.
    # Each ordered pair of distinct population members must occur exactly once.
    seen = set()

    class Draws:
        def __init__(self, choices):
            self.choices = iter(choices)

        def randint(self, low, high, dtype):
            draw = next(self.choices)
            assert low <= draw < high
            return draw

        def choice(self, *args, **kwargs):
            raise AssertionError("This small sample must not allocate the population")

    for choices in product(range(21), range(1, 21)):
        result = tuple(_sample_pair_indices(Draws(choices), 21, 2))
        assert result not in seen
        seen.add(result)
    assert seen == set(permutations(range(21), 2))


@pytest.mark.parametrize(
    "population,size",
    [(0, 0), (1, 0), (1, 1), (20, 1), (20, 2), (20, 19), (20, 20), (10**12, 100), (np.iinfo(np.int64).max, 10)],
)
def test_sampler_bounds_uniqueness_and_seed_reproducibility(population, size):
    first = _sample_pair_indices(np.random.RandomState(42), population, size)
    second = _sample_pair_indices(np.random.RandomState(42), population, size)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.int64 and first.shape == (size,)
    assert len(np.unique(first)) == size
    assert np.all((0 <= first) & (first < population))
    if size == population:
        np.testing.assert_array_equal(np.sort(first), np.arange(population))


@pytest.mark.parametrize("size", [0, 5, 25, 30])
@pytest.mark.parametrize("return_params", [False, True])
def test_public_generator_shapes_unique_pairs_ratings_and_repeatability(size, return_params):
    options = dict(
        n_users=5, n_items=5, n_factors=2, n_interactions=size, seed=8, noise_std=0.0, return_params=return_params
    )
    first, second = make_mf_dataset(**options), make_mf_dataset(**options)
    X, y = first["X"], first["y"]
    assert X.shape == (min(size, 25), 2)
    assert np.issubdtype(X.dtype, np.integer)
    assert len(np.unique(X, axis=0)) == len(X)
    np.testing.assert_array_equal(X, second["X"])
    np.testing.assert_array_equal(y, second["y"])
    assert np.all((0 <= X) & (X < 5)) and np.all((1 <= y) & (y <= 5))
    assert ("params" in first) == return_params
    if return_params:
        p = first["params"]
        expected = p["mu"] + p["bu"][X[:, 0]] + p["bi"][X[:, 1]]
        expected += (p["P"][X[:, 0]] * p["Q"][X[:, 1]]).sum(axis=1)
        np.testing.assert_array_equal(y, np.clip(np.round(2 * expected) / 2, 1, 5))
        for key in p:
            np.testing.assert_array_equal(p[key], second["params"][key])


def test_density_zero_and_zero_requested_count_produce_empty_data():
    for kwargs in [dict(density=0.0), dict(n_interactions=0), dict(n_users=0), dict(n_items=0)]:
        options = dict(n_users=3, n_items=4, seed=42)
        options.update(kwargs)
        result = make_mf_dataset(**options)
        assert result["X"].shape == (0, 2)
        assert result["y"].shape == (0,)
    assert len(make_mf_dataset(10, 10, density=0.12, seed=42)["y"]) == 12
    # Explicit counts take precedence over density.
    assert len(make_mf_dataset(2, 2, n_interactions=1, density="unused", seed=42)["y"]) == 1


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("n_users", -1),
        ("n_users", True),
        ("n_users", 1.5),
        ("n_items", -1),
        ("n_factors", 0),
        ("n_factors", 1.5),
        ("n_interactions", -1),
        ("n_interactions", 1.5),
        ("n_interactions", True),
        ("density", -0.1),
        ("density", 1.1),
        ("density", np.nan),
        ("density", np.inf),
    ],
)
def test_invalid_counts_fail_before_allocating(parameter, value):
    options = dict(n_users=3, n_items=4, seed=42)
    options[parameter] = value
    with pytest.raises(ValueError, match=parameter):
        make_mf_dataset(**options)


def test_population_product_is_computed_without_integer_overflow():
    with pytest.raises(ValueError, match="int64"):
        make_mf_dataset(np.int64(2**32), np.int64(2**32), n_interactions=1)


def test_huge_virtual_population_never_reaches_dense_sampling(monkeypatch):
    random_state = np.random.RandomState

    class BoundedRandomState:
        def __init__(self, seed):
            self.rng = random_state(seed)

        def choice(self, population, *args, **kwargs):
            # Prevent an allocation even if a regression restores the old path.
            assert population < 10**6, "Dense population sampling would exhaust memory"
            return self.rng.choice(population, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.rng, name)

    monkeypatch.setattr(np.random, "RandomState", BoundedRandomState)
    tracemalloc.start()
    try:
        result = make_mf_dataset(100000, 100000, n_factors=2, n_interactions=10, seed=42)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(result["y"]) == 10
    assert len(np.unique(result["X"], axis=0)) == 10
    assert peak < 16 * 1024**2
