from matplotlib import pyplot as plt
from scipy.signal import resample

from peakalignment.apply_alignment import apply_alignment
from peakalignment.density_filter import get_density_filtered_matches
from peakalignment.line_filter import get_filtered_lines
from peakalignment.regression import get_regression_lines
from plotting.plot_rpeak_detector import plot_artefacts_on_signal
from tests.add_noise import add_noise
from tests.calculate_true_offset import compute_linear_eqn, compute_linear_eqn_old, generate_true_line_data, \
    generate_true_line_data_from_scratch
from tests.rpeak_detector import neurokit_rpeak_detect_fast
from tests.generate_wfdb import get_records
from peakalignment.signal_matching import get_matches
from tests.add_gaps import add_gaps, convert_gaps_to_intervals, apply_gaps
import numpy as np

from consensus_peaks import consensus_detect

def test_calculate_alignment():
    # Set up raw data
    records = get_records()
    desired_record_num = 1
    for _ in range(desired_record_num):
        record = next(records)

    signal_a = record.p_signal[:, 0]
    signal_b = np.copy(signal_a)

    # Resample original signal
    freq_hz_a = 500  # Desired Signal A Freq
    freq_hz_b = 256  # Desired Signal B Freq
    original_freq_hz = record.fs

    signal_a_new_num_values = (signal_a.size * freq_hz_a) // original_freq_hz
    signal_a = resample(signal_a, signal_a_new_num_values)

    signal_b_new_num_values = (signal_b.size * freq_hz_b) // original_freq_hz
    signal_b = resample(signal_b, signal_b_new_num_values)

    # Add Noise
    add_noise(signal_a, 0.01, seed=None)
    add_noise(signal_b, 0.02, seed=None)

    # Create Times
    start_time_ms = 0
    period_ms_b = 1000 / freq_hz_b
    period_ms_a = 1000 / freq_hz_a

    signal_a_signal_times = np.arange(start_time_ms, start_time_ms + (signal_a.size * period_ms_a), period_ms_a)
    signal_b_signal_times = np.arange(start_time_ms, start_time_ms + (signal_b.size * period_ms_b), period_ms_b)

    # Create Linear Transformation
    # gap_indices_a, gap_durations_a = add_gaps(signal_a_signal_times, int(period_ms_a), gap_density=0.000005, seed=42)
    # gap_end_times_a = signal_a_signal_times[gap_indices_a]
    # gap_start_times_a = signal_a_signal_times[gap_indices_a - 1]

    # gaps_a = np.column_stack((gap_start_times_a, gap_end_times_a))
    # gaps_a = np.array([], dtype=int)
    # intervals_a = convert_gaps_to_intervals(signal_a_signal_times[0], signal_a_signal_times[-1], gaps_a)
    clock_drift_a = -400 / (10 ** 6)  # -400 ppm
    clock_drift_b = 50 / (10 ** 6)  # 50 ppm

    # true_line_data = generate_true_line_data(signal_a_signal_times, intervals_a, clock_drift, period_ms_b, seed=None)
    true_line_data, resampled_a_x, true_a_t, false_a_t, resampled_b_x, true_b_t, false_b_t = \
        generate_true_line_data_from_scratch(
            signal_a_signal_times, signal_a, signal_b_signal_times, signal_b, period_ms_a, period_ms_b, clock_drift_a, clock_drift_b, seed=None)

    signal_a = resampled_a_x
    signal_b = resampled_b_x

    signal_a_signal_times = false_a_t
    signal_b_signal_times = false_b_t

    # Apply A Gaps to B
    # apply_gaps(signal_b_signal_times, gaps_a)

    # Apply Transformation
    true_line_data = [row + [None, None] for row in true_line_data]

    # b_indices, a_original_times, signal_b_signal_times = \
    #     apply_alignment(signal_b_signal_times, true_line_data)
    # signal_b = signal_b[b_indices]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(signal_a_signal_times, signal_a, color='blue')
    ax1.set_title(f'Signal A - {freq_hz_a}Hz - {(10**6)*clock_drift_a}ppm Drift')

    ax2.plot(signal_b_signal_times, signal_b, color='red')
    ax2.set_title(f'Signal B - {freq_hz_b}Hz - {(10**6)*clock_drift_b}ppm Drift')

    plt.xlabel('Time (ms)')

    plt.tight_layout()
    plt.show()

    peak_indices_a = neurokit_rpeak_detect_fast(signal_a, freq_hz_a)
    peak_indices_b = neurokit_rpeak_detect_fast(signal_b, freq_hz_b)

    # peak_indices_a = consensus_detect(signal_a, freq_hz_a)
    # peak_indices_b = consensus_detect(signal_b, freq_hz_b)

    signal_a_peak_times = signal_a_signal_times[peak_indices_a]
    signal_b_peak_times = signal_b_signal_times[peak_indices_b]

    plot_artefacts_on_signal(signal_a_signal_times[:10_000],
                             signal_a[:10_000],
                             [peak_t for peak_t in signal_a_peak_times if peak_t < signal_a_signal_times[10_000]])

    match_indices_matrix, match_distance_matrix = get_matches(signal_a_peak_times, signal_b_peak_times, max_offset=None)

    # offset_matrix = signal_b_peak_times[match_indices_matrix] - signal_a_peak_times[:match_indices_matrix.shape[0]].reshape(-1, 1)
    #
    # offset_t = signal_a_peak_times[np.repeat(np.arange(offset_matrix.shape[0]), offset_matrix.shape[1])]
    # offset_v = offset_matrix.ravel()

    density_data = get_density_filtered_matches(signal_a_peak_times, signal_b_peak_times, match_indices_matrix, max_offset=None)
    offset_t, offset_v = density_data['fine_offset_t'], density_data['fine_offset_v']

    line_data = get_regression_lines(offset_t, offset_v)
    line_data = get_filtered_lines(line_data)

    # times, distances = distances_between_lines(line_data, true_line_data)
    # indices = np.argsort(times)
    # times, distances = times[indices], distances[indices]

    # plt.scatter(times, distances)
    # plt.ylabel("distance, (ms)")
    # plt.xlabel("Times (ms)")
    # plt.show()

    # plt.hist(distances, bins=np.linspace(-1, 1, num=100))
    # plt.xlabel("distance, (ms)")
    # plt.show()
    # plt.scatter(offset_t, offset_v)
    plt.scatter(offset_t, offset_v, color='black', s=50, alpha=0.5)

    for line_i, (start_time, end_time, (b, m), _, _) in enumerate(true_line_data):
        line_x = np.array([start_time, end_time], dtype=np.int64)
        line_y = (m * line_x) + b
        plt.plot(line_x, line_y, linestyle='dotted', label=f"True Offset: {line_i}")
        av_x = np.mean(line_x)
        av_y = np.mean(line_y)
        plt.text(av_x, av_y, str(line_i), c='blue')

    for line_i, (start_time, end_time, (b, m), score, num_inliers) in enumerate(line_data):
        line_x = np.array([start_time, end_time], dtype=np.int64)
        line_y = (m * line_x) + b

        label = f"{line_i} {round(m * (10 ** 6), 2)}ppm: {round(score, 4)}ms"
        plt.plot(line_x, line_y, label=label)
        av_x = np.mean(line_x)
        av_y = np.mean(line_y)
        plt.text(av_x, av_y, str(line_i), c='red')

    plt.legend(loc='upper right')
    plt.title("Matching Offset Over Time - Legend (Slope PPM: Mean Distance From Line)")
    plt.xlabel("Clock A Time (ms)")
    plt.ylabel("Offset To Matching Clock B Time (ms)")

    plt.show()

    aligned_indices, aligned_original_times, alignment_corrected_times = \
        apply_alignment(signal_a_signal_times, line_data)

    sorted_indices = alignment_corrected_times.argsort()

    perf_aligned_indices, perf_aligned_original_times, perf_alignment_corrected_times = \
        apply_alignment(signal_a_signal_times, true_line_data)

    perf_sorted_indices = perf_alignment_corrected_times.argsort()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

    ax1.plot(signal_a_signal_times, signal_a, color='blue')
    ax1.set_title(f'Signal A - {freq_hz_a}Hz - {(10 ** 6) * clock_drift_a}ppm Drift')

    ax2.plot(alignment_corrected_times[sorted_indices], signal_a[aligned_indices][sorted_indices], color='green')
    ax2.set_title('Signal A - Aligned To The Signal B Clock')

    ax3.plot(perf_alignment_corrected_times[perf_sorted_indices], signal_a[perf_aligned_indices][perf_sorted_indices], color='orange')
    ax3.set_title(f'Signal A - Perfectly Aligned To The Signal B Clock')

    ax4.plot(signal_b_signal_times, signal_b, color='red')
    ax4.set_title(f'Signal B - {freq_hz_b}Hz - {(10 ** 6) * clock_drift_b}ppm Drift')

    plt.xlabel('Time (ms)')

    plt.tight_layout()
    plt.show()


def distance_between_points(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def distances_between_lines(line_data_1, line_data_2, step_size=1):
    times = []
    distances = []
    actual = []
    predicted = []

    for line1 in line_data_1:
        start1, end1, (b1, m1), _, _ = line1
        for line2 in line_data_2:
            start2, end2, (b2, m2), _, _ = line2
            common_start = max(start1, start2)
            common_end = min(end1, end2)
            for t in range(int(common_start), int(common_end), step_size):
                times.append(t)
                distances.append((m1*t + b1) - (m2*t + b2))
                actual.append(m1*t + b1)
                predicted.append(m2 * t + b2)
                # point1 = (t, m1*t + b1)
                # point2 = (t, m2*t + b2)
                # distances.append(distance_between_points(point1, point2))
    return np.array(times), np.array(distances), np.array(actual), np.array(predicted)

