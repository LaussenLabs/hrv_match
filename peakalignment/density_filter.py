import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from peakalignment.constants import density_time_step_size, density_time_steps_per_window, coarse_density_bin_size_ms, \
    coarse_density_accepted_distance_from_max_ms, density_offset_step_dur_ms, density_offset_steps_per_window


def get_density_filtered_matches(signal_a_times: np.ndarray, signal_b_times: np.ndarray, match_indices_matrix,
                                 max_offset=None):
    if max_offset is None:
        max_offset = max(signal_a_times[-1], signal_b_times[-1]) - min(signal_a_times[0], signal_b_times[0])

    offset_matrix = signal_b_times[match_indices_matrix] - signal_a_times[:match_indices_matrix.shape[0]].reshape(-1, 1)

    if offset_matrix.shape[0] < density_time_step_size * density_time_steps_per_window:
        return None

    offset_t = signal_a_times[np.repeat(np.arange(offset_matrix.shape[0]), offset_matrix.shape[1])]
    offset_v = offset_matrix.ravel()

    coarse_offset_bins = \
        np.arange(-max_offset, max_offset + coarse_density_bin_size_ms, coarse_density_bin_size_ms)

    coarse_digital_indices = np.digitize(offset_v, coarse_offset_bins[:-1], right=False)
    coarse_offset_hist = np.bincount(coarse_digital_indices, minlength=coarse_offset_bins.size)
    coarse_bin_max = coarse_offset_bins[np.argmax(coarse_offset_hist)]

    coarse_indices = np.logical_and((coarse_bin_max - coarse_density_accepted_distance_from_max_ms) < offset_v,
                                    offset_v < (coarse_bin_max + coarse_density_accepted_distance_from_max_ms))

    offset_t, offset_v = offset_t[coarse_indices], offset_v[coarse_indices]

    fine_bin_start = int(coarse_bin_max - coarse_density_accepted_distance_from_max_ms)

    fine_bin_dur = (2 * coarse_density_accepted_distance_from_max_ms) + density_offset_step_dur_ms
    fine_bin_dur = round_up_multiple(fine_bin_dur, density_offset_steps_per_window * density_offset_step_dur_ms)

    offset_bins = np.arange(fine_bin_start,
                            fine_bin_start + fine_bin_dur,
                            density_offset_step_dur_ms).reshape(-1, density_offset_steps_per_window).T

    counter_bins = np.zeros(offset_bins.shape, dtype=np.int64)

    accepted_bool_1 = np.zeros(offset_v.shape, dtype=bool)

    density_time_window_size = density_time_step_size * density_time_steps_per_window

    if offset_v.size < density_time_window_size:
        return None

    sliding_offset_matrix = sliding_window_view(
        offset_v,
        window_shape=density_time_window_size,
        axis=None)[::density_time_step_size]

    # Slice by Slice Digitized
    density_offset_window_dur = density_offset_step_dur_ms * density_offset_steps_per_window
    for time_i, time_slice in enumerate(sliding_offset_matrix):
        true_time_i = time_i * density_time_step_size
        counter_bins[:] = 0
        for bin_i in range(offset_bins.shape[0]):
            digital_slice_indices = np.digitize(time_slice, offset_bins[bin_i][1:])
            slice_bin_counts = np.bincount(digital_slice_indices, minlength=offset_bins.shape[1])
            counter_bins[bin_i] += slice_bin_counts

        slice_max_bin_ind = np.unravel_index(np.argmax(counter_bins, axis=None), counter_bins.shape)
        slice_max_bin = offset_bins[slice_max_bin_ind]

        accepted_bool_1[true_time_i:true_time_i+density_time_window_size] = np.logical_or(
            accepted_bool_1[true_time_i:true_time_i+density_time_window_size],
            np.logical_and(
                slice_max_bin <= time_slice,
                time_slice < slice_max_bin + density_offset_window_dur)
        )

    density_data = {'offset_matrix': offset_matrix,
                    'fine_offset_t': offset_t[accepted_bool_1],
                    'fine_offset_v': offset_v[accepted_bool_1],
                    'n_t': signal_a_times,
                    'p_t': signal_b_times,
                    'coarse_offset_bins': coarse_offset_bins,
                    'coarse_offset_hist': coarse_offset_hist,
                    'coarse_bin_max': coarse_bin_max,
                    'coarse_bin_mean': np.mean(coarse_offset_bins[coarse_digital_indices]),
                    'coarse_bin_std': np.std(coarse_offset_bins[coarse_digital_indices]),
                    'offset_t': offset_t,
                    'offset_v': offset_v
                    }

    return density_data


def round_up_multiple(num, multiple):
    return int(((num + multiple - 1) // multiple) * multiple)
