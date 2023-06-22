import numpy as np


def add_gaps(times: np.ndarray, period: int, gap_density=0.001, seed=42):
    np.random.seed(seed)

    # Calculate number of gaps to insert
    num_gaps = int(gap_density * times.size)

    # Generate random indices for gaps
    gap_indices = np.random.choice(np.arange(1, times.size), num_gaps, replace=False)

    # Calculate actual gaps
    gaps = period * np.random.randint(1, (times.size // 10) // (num_gaps + 1), size=num_gaps)

    # Sort indices for sequential insertion
    gap_sorted_idx = np.argsort(gap_indices)
    gap_indices, gaps = gap_indices[gap_sorted_idx], gaps[gap_sorted_idx]

    for gap_index, gap_duration in zip(gap_indices, gaps):
        times[gap_index:] += gap_duration

    return gap_indices, gaps


def convert_gaps_to_intervals(start_time, end_time, gaps):
    intervals = []

    interval_start = start_time

    for g_start, g_end in gaps:
        intervals.append([interval_start, g_start])
        interval_start = g_end

    intervals.append([interval_start, end_time])

    return np.array(intervals, dtype=int)


def apply_gaps(signal_times, gaps):
    for gap_s, gap_e in gaps:
        gap_dur = gap_e - gap_s
        left = np.searchsorted(signal_times, gap_s)
        if 0 <= left < signal_times.size:
            signal_times[left:] += gap_dur
