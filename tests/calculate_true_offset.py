import numpy as np
from scipy.signal import resample
import random


def generate_true_line_data_from_scratch(a_t, a_x, b_t, b_x, period_a, period_b, clock_drift_a, clock_drift_b,
                                         signal_a_sync_dur=600_000, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    slope = clock_drift_b - clock_drift_a

    # Apply Clock Drifts
    new_a_size = int(len(a_t) * (1 + clock_drift_a))
    resampled_a_x, true_a_t = resample(a_x, new_a_size, t=a_t)
    false_a_t = np.arange(a_t[0], a_t[0] + (resampled_a_x.size * period_a), period_a)

    new_b_size = int(len(b_t) * (1 + clock_drift_b))
    resampled_b_x, true_b_t = resample(b_x, new_b_size, t=b_t)
    false_b_t = np.arange(b_t[0], b_t[0] + (resampled_b_x.size * period_b), period_b)

    start_time, end_time = int(a_t[0]), int(a_t[-1])
    # We start with the simplest possible
    line_data = [[start_time, end_time, [0, slope]]]

    # Adding Gaps in signal A
    sync_duration_num_samples = int(signal_a_sync_dur / period_a)
    false_a_t, sync_indices_a, sync_times_a, sync_durations_a = sync_correction(
        false_a_t, true_a_t, period_a, sync_duration_num_samples)

    # Adjust the line_data for A gaps
    for sync_i, sync_t, sync_dur in zip(sync_indices_a, sync_times_a, sync_durations_a):
        line_i = find_position(sync_t, line_data)
        if line_i == -1:
            raise ValueError(f"sync_i, sync_t, sync_dur: {sync_i, sync_t, sync_dur} "
                             f"not found in line_data: {line_data}")
        old_start, old_end, (old_b, old_m) = line_data[line_i]

        left_line = [old_start, sync_t - sync_dur, [old_b, old_m]]
        right_line = [sync_t, old_end, [old_b - sync_dur, old_m]]
        line_data = line_data[:line_i] + [left_line, right_line] + line_data[line_i + 1:]

    return line_data, resampled_a_x, true_a_t, false_a_t, resampled_b_x, true_b_t, false_b_t


def find_position(t, line_data):
    for i, line in enumerate(line_data):
        start, end, _ = line
        if start <= t < end:
            return i
    return -1


def sync_correction(false_a_t, true_a_t, period_a, sync_duration_num_samples):
    # compute differences between true and false timestamps
    diff_a_t = true_a_t - false_a_t

    # compute corrections rounded to nearest multiple of period_a
    corrections = period_a * np.round(diff_a_t / period_a)

    # list to keep track of sync information
    sync_indices = []
    sync_times = []
    sync_durations = []

    # iterate over every sync_duration
    for i in range(sync_duration_num_samples, len(false_a_t) - sync_duration_num_samples + 1, sync_duration_num_samples):
        # add correction to the current and successive points
        false_a_t[i:] += corrections[i]

        sync_indices.append(i)
        sync_times.append(false_a_t[i])
        sync_durations.append(corrections[i])

    return false_a_t, sync_indices, sync_times, sync_durations


def generate_true_line_data(signal_times, intervals, slope, period, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_gaps = random.randint(2, 4)

    # Generate random indices for gaps
    signal_b_gap_times = np.random.choice(signal_times, num_gaps, replace=False)
    signal_b_gap_times = np.sort(signal_b_gap_times)

    line_data = []
    b_i = 0
    total_gap = 0
    last_end = 0
    for a_start, a_end in intervals:
        last_gap = a_start - last_end
        start = a_start
        while b_i < len(signal_b_gap_times) and signal_b_gap_times[b_i] < a_end:
            end = signal_b_gap_times[b_i]

            # Record Data
            intercept = total_gap
            line_data.append([start, end, (intercept, slope)])

            # Update
            gap = random.randint(1, signal_times.size // (num_gaps + 1)) * period
            start = end
            total_gap += gap
            b_i += 1

        end = a_end

        # Record Data
        intercept = total_gap - (start * slope)
        line_data.append([start, end, (intercept, slope)])

        # Update
        # total_gap -= last_gap
        last_end = end

    return line_data


def compute_linear_eqn_old(clock_a, clock_b, gap_indices_a, gap_durations_a, gap_indices_b, gap_durations_b,
                           clock_drift_a, clock_drift_b):
    slope = clock_drift_b - clock_drift_a
    linear_eqns = []  # Initialize a list to store the equations

    gap_end_times_a = clock_a[gap_indices_a]
    gap_end_times_b = clock_b[gap_indices_b]

    gap_start_times_a = clock_a[gap_indices_a - 1]
    gap_start_times_b = clock_b[gap_indices_b - 1]

    combined_gaps = list(zip(gap_start_times_a, gap_end_times_a, ['a'] * len(gap_end_times_a), gap_durations_a)) + list(
        zip(gap_start_times_b / (1 + slope), gap_end_times_b / (1 + slope), ['b'] * len(gap_end_times_b), gap_durations_b / (1 + slope)))

    # Sort by gap time
    combined_gaps.sort(key=lambda x: x[0])

    # Initialize relative_gap and previous time
    relative_gap = 0
    prev_time = 0

    for gap_start_time, gap_end_time, source, gap_duration in combined_gaps:
        # Compute start and end times for the current region
        start_time = prev_time
        end_time = gap_start_time

        intercept = (relative_gap * (1 + slope)) - (slope * start_time)

        # Store the equation parameters
        linear_eqns.append([start_time, end_time, (intercept, slope)])

        # Compute slope and intercept based on the relative_gap and source
        if source == 'a':
            relative_gap -= gap_duration
        else:
            relative_gap += gap_duration

        # Update previous time for the next iteration
        prev_time = gap_end_time

    # Don't forget the final segment
    start_time = prev_time
    end_time = clock_a[-1]

    intercept = relative_gap - (slope * start_time)

    # Store the equation parameters
    linear_eqns.append([start_time, end_time, (intercept, slope)])

    return linear_eqns


def compute_linear_eqn(clock_a, clock_b, gap_indices_a, gap_durations_a, gap_indices_b, gap_durations_b,
                       clock_drift_a, clock_drift_b):

    gap_end_times_a = clock_a[gap_indices_a]
    gap_end_times_b = clock_b[gap_indices_b]

    gap_start_times_a = clock_a[gap_indices_a - 1]
    gap_start_times_b = clock_b[gap_indices_b - 1]

    relative_gap = 0
    start_time = 0

    gap_list_b = list(zip(gap_start_times_b, gap_end_times_b, gap_durations_b))
    gap_list_b_i = 0

    slope = clock_drift_b - clock_drift_a
    intercept = slope * clock_a[0]
    linear_eqns = [[clock_a[0], clock_a[-1], (intercept, slope)]]
    for gap_start_time_a, gap_end_time_a, gap_duration_a in zip(gap_start_times_a, gap_end_times_a, gap_durations_a):
        while gap_list_b_i < len(gap_list_b) and (
                gap_list_b[gap_list_b_i][0] / slope) < gap_start_time_a + relative_gap:
            gap_start_times_b, gap_end_times_b, gap_durations_b = gap_list_b[gap_list_b_i]
            end_time = (gap_list_b[gap_list_b_i][0] / (1 + slope)) - relative_gap

            # Record line
            intercept = relative_gap - (slope * start_time)
            linear_eqns.append([start_time, end_time, (intercept, slope)])

            # Update Variables
            start_time = (gap_end_times_b / (1 + slope)) - relative_gap
            relative_gap += gap_durations_b / (1 + slope)
            gap_list_b_i += 1

        # Record line
        intercept = relative_gap - (slope * start_time)
        linear_eqns.append([start_time, gap_start_time_a, (intercept, slope)])

        # Update Variables
        start_time = gap_end_time_a
        relative_gap -= gap_duration_a

    # Rest of the signal b gaps
    while gap_list_b_i < len(gap_list_b):
        gap_start_times_b, gap_end_times_b, gap_durations_b = gap_list_b[gap_list_b_i]
        end_time = (gap_list_b[gap_list_b_i][0] / (1 + slope)) - relative_gap

        # Record line
        intercept = relative_gap - (slope * start_time)
        linear_eqns.append([start_time, end_time, (intercept, slope)])

        # Update Variables
        start_time = (gap_end_times_b / (1 + slope)) - relative_gap
        relative_gap += gap_durations_b / (1 + slope)
        gap_list_b_i += 1

    # Don't forget the final segment
    end_time = clock_a[-1]

    intercept = relative_gap - (slope * start_time)

    # Store the equation parameters
    linear_eqns.append([start_time, end_time, (intercept, slope)])

    return linear_eqns
