import numpy as np


def generate_random_clock_drifts(min_value, max_value, size=10_000, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Setting the seed for reproducibility
    random_clock_drifts = min_value + ((max_value - min_value) * np.random.rand(size))
    for drift in random_clock_drifts:
        yield drift


def get_random_clock_drifts(min_value, max_value, size=10_000, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Setting the seed for reproducibility

    return min_value + ((max_value - min_value) * np.random.rand(size))
