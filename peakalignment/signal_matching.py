from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import distance

from peakalignment.constants import match_window_size, matches_per_window


def get_matches(signal_a_times: np.ndarray, signal_b_times: np.ndarray, max_offset: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find matches between two time series signals, represented as numpy arrays.

    :param np.ndarray signal_a_times: The time series of the first signal.
    :param np.ndarray signal_b_times: The time series of the second signal.
    :param int max_offset: The maximum offset to consider when matching signals.
                            Defaults to the difference between maximum and minimum timestamps.

    :return: Two matrices:
            1. Indices of matches for each sliding window in signal A.
            2. Corresponding euclidean distances between matched windows.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    # Calculate the difference between consecutive elements in signal_a_times
    signal_a_diff = np.diff(signal_a_times)

    # Calculate the difference between consecutive elements in signal_b_times
    signal_b_diff = np.diff(signal_b_times)

    # If max_offset is not given, consider the maximum possible offset
    if max_offset is None:
        max_offset = max(signal_a_times[-1], signal_b_times[-1]) - min(signal_a_times[0], signal_b_times[0])

    # Check if there are sufficient beats in both signals for the match window
    if signal_a_diff.size < match_window_size or signal_b_diff.size < match_window_size:
        raise ValueError("Not enough beats in one or both of the signals")

    # Create a sliding window view of signal_a_diff
    signal_a_sliding = sliding_window_view(signal_a_diff, window_shape=match_window_size)

    # Create a sliding window view of signal_b_diff
    signal_b_sliding = sliding_window_view(signal_b_diff, window_shape=match_window_size)

    # Initialize a matrix to store the indices of matches
    match_indices_matrix = np.zeros((signal_a_sliding.shape[0], matches_per_window), dtype=np.int64)

    # Initialize a matrix to store the distances of matches
    match_distance_matrix = np.zeros((signal_a_sliding.shape[0], matches_per_window))

    # Iterate through each window in signal_a_sliding
    for n_i in range(signal_a_sliding.shape[0]):
        # Calculate the start time of the window
        signal_a_window_start_time = signal_a_times[n_i]

        # Calculate the left and right boundaries for searching matches in signal_b_times
        signal_b_left = np.searchsorted(signal_b_times, signal_a_window_start_time - max_offset, side='left')
        signal_b_right = np.searchsorted(signal_b_times, signal_a_window_start_time + max_offset, side='right')

        # Skip this iteration if there are not enough potential matches in signal_b_times
        if signal_b_right - signal_b_left < matches_per_window:
            continue

        # Calculate the distance between each window in signal_a_sliding and signal_b_sliding
        distance_matrix = distance.cdist(signal_a_sliding[n_i:n_i+1],
                                         signal_b_sliding[signal_b_left:signal_b_right],
                                         'euclidean')

        # Get the indices of matches sorted by their distances
        sorted_distance_inds = np.argsort(distance_matrix[0])

        # Skip this iteration if there are not enough matches
        if sorted_distance_inds.size < matches_per_window:
            continue

        # Store the indices of matches in match_indices_matrix
        match_indices_matrix[n_i] = sorted_distance_inds[:matches_per_window] + signal_b_left

        # Store the distances of matches in match_distance_matrix
        match_distance_matrix[n_i] = distance_matrix[0][sorted_distance_inds[:matches_per_window]]

    # Return the indices of matches and their distances
    return match_indices_matrix, match_distance_matrix

