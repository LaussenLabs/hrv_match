from typing import Tuple

import numpy as np


def apply_alignment(input_time_array: np.ndarray, line_data: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply alignment to the given input time array based on the provided line data.

    :param np.ndarray input_time_array: An array of input time values.
    :param list line_data: A list of tuples containing information about each line segment
                           (start_time, end_time, (b, m), score, num_inliers).

    :return: A tuple of three NumPy arrays:
                1. aligned_indices: An array of aligned indices.
                2. aligned_original_times: An array of aligned original time values.
                3. alignment_corrected_times: An array of alignment-corrected time values.
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]

    >>> # Example usage:
    >>> input_time_array = np.array([0, 1, 2, 3, 4])
    >>> line_data = [(0, 2, (1, 1), 0.9, 2), (2, 4, (2, 1), 0.95, 2)]
    >>> aligned_indices, aligned_original_times, alignment_corrected_times = apply_alignment(input_time_array, line_data)
    >>> # print(aligned_indices)
    ... [0 1 4 3 2]
    >>> # print(aligned_original_times)
    ... [0 1 2 3 4]
    >>> # print(alignment_corrected_times)
    ... [0 1 4 3 2]
    """
    aligned_indices = []
    aligned_original_times = []
    alignment_corrected_times = []

    for start_time, end_time, (b, m), score, num_inliers in line_data:
        # Find the left and right indices for slicing the input_time_array
        left = np.searchsorted(input_time_array, start_time, side='left')
        right = np.searchsorted(input_time_array, end_time, side='right')

        # If left < right, process the data accordingly
        if left < right:
            # Append the range of indices to aligned_indices
            aligned_indices.append(np.arange(left, right))

            # Slice the input_time_array and append it to aligned_original_times
            natus_slice = input_time_array[left:right]
            aligned_original_times.append(natus_slice)

            # Calculate the offset and apply it to the input_time_array slice
            offset = (m * natus_slice) + b
            alignment_corrected_times.append(natus_slice + offset)

    # Concatenate the results into final NumPy arrays
    aligned_indices = np.concatenate(aligned_indices) if \
        len(aligned_indices) > 0 else np.array([], dtype=int)
    aligned_original_times = np.concatenate(aligned_original_times) if \
        len(aligned_original_times) > 0 else np.array([])
    alignment_corrected_times = np.concatenate(alignment_corrected_times) if \
        len(alignment_corrected_times) > 0 else np.array([])

    return aligned_indices, aligned_original_times, alignment_corrected_times
