import numpy as np


def add_noise(data, noise_level, wave_std=None, seed=None):
    noise = get_noise(data, noise_level, wave_std=wave_std, seed=seed)
    data += noise
    return data


def get_noise(data, noise_level, wave_std=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if wave_std is None:
        wave_std = np.std(data)
    noise_amplitude = noise_level * wave_std
    noise = np.random.normal(0, noise_amplitude, len(data))
    return noise
