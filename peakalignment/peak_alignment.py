import numpy as np

from peakalignment.apply_alignment import apply_alignment
from peakalignment.density_filter import get_density_filtered_matches
from peakalignment.regression import get_regression_lines
from peakalignment.signal_matching import get_matches


def align_peaks(signal_a_peak_times: np.ndarray, signal_b_peak_times: np.ndarray, signal_a_times: np.ndarray,
                max_offset=None):
    match_indices_matrix, match_distance_matrix = get_matches(signal_a_peak_times, signal_b_peak_times, max_offset=max_offset)

    # Replace with Density Filter if Needed:
    # offset_t, offset_v = get_density_filtered_matches(signal_a_times, signal_b_times, match_indices_matrix,
    #                                                   max_offset=None)
    offset_matrix = signal_b_peak_times[match_indices_matrix] - signal_a_peak_times[:match_indices_matrix.shape[0]].reshape(-1, 1)

    offset_t = signal_a_peak_times[np.repeat(np.arange(offset_matrix.shape[0]), offset_matrix.shape[1])]
    offset_v = offset_matrix.ravel()

    line_data = get_regression_lines(offset_t, offset_v)

    aligned_indices, aligned_original_times, alignment_corrected_times = \
        apply_alignment(signal_a_times, line_data)

    sorted_indices = alignment_corrected_times.argsort()

    return alignment_corrected_times[sorted_indices]
