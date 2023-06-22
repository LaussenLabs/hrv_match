import random
import time
from tqdm import tqdm

# Uncomment for consensus peak detection
# from consensus_peaks import consensus_detect
from matplotlib import pyplot as plt
from scipy.signal import resample
from sklearn.metrics import mean_squared_error

from peakalignment.apply_alignment import apply_alignment
from peakalignment.density_filter import get_density_filtered_matches
from peakalignment.line_filter import get_filtered_lines
from peakalignment.regression import get_regression_lines
from tests.add_noise import add_noise, get_noise
from tests.calculate_true_offset import generate_true_line_data, generate_true_line_data_from_scratch
from tests.generate_random import generate_random_clock_drifts, get_random_clock_drifts
from tests.rpeak_detector import neurokit_rpeak_detect_fast
from tests.generate_wfdb import get_records
from peakalignment.signal_matching import get_matches
from tests.add_gaps import add_gaps, convert_gaps_to_intervals
import numpy as np
import pandas as pd

from tests.test_calculate_alignment import distances_between_lines


def experiment_comprehensive():
    total_slope_id = []
    total_slopes_start = []
    total_slopes_end = []
    total_slopes_mean_dist = []
    total_slopes_count = []
    total_slopes = []
    total_offset_actual = []
    total_offset_predicted = []
    total_slope_rmse = []
    total_slope_records = []
    total_slope_sync_time = []
    total_expected_slopes = []
    total_clock_a_drift = []
    total_clock_b_drift = []
    total_noise_level = []
    total_iteration_number = []

    total_o_time = []
    total_o_offset = []
    total_o_predicted = []
    total_o_slopes = []
    total_o_iteration_number = []

    clock_a_drifts = get_random_clock_drifts(-600, -290, size=48, seed=42)
    clock_b_drifts = get_random_clock_drifts(10, 100, size=48, seed=42)

    overall_std_signal = calculate_overall_std_signal()
    print(f"overall_std_signal: {overall_std_signal}")

    iteration_number = 0
    slope_id = 0

    noise_level_list = np.arange(0, 0.05, 0.05)
    # noise_level_list = [0]

    for noise_level in noise_level_list:
        records = get_records()
        print(f"noise_level: {noise_level}")
        records_num = 0
        for record in tqdm(records, total=48):
            records_num += 1
            # if records_num > 1:
            #     break
            # print(records_num)
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
            add_noise(signal_a, noise_level, seed=None, wave_std=overall_std_signal)
            add_noise(signal_b, noise_level, seed=None, wave_std=overall_std_signal)

            # Create Times
            start_time_ms = 0
            period_ms_b = 1000 / freq_hz_b
            period_ms_a = 1000 / freq_hz_a

            signal_a_signal_times = np.arange(start_time_ms, start_time_ms + (signal_a.size * period_ms_a), period_ms_a)
            signal_b_signal_times = np.arange(start_time_ms, start_time_ms + (signal_b.size * period_ms_b), period_ms_b)

            for signal_a_sync_dur in np.arange(180_000, 600_000 + 60_000, 60_000):
                iteration_number += 1
                # signal_a_sync_dur -= 15_000
                # signal_a_sync_dur += random.randint(0, 30_000)
                clock_drift_a = clock_a_drifts[records_num-1] / (10 ** 6)
                clock_drift_b = clock_b_drifts[records_num-1] / (10 ** 6)

                true_line_data, resampled_a_x, true_a_t, false_a_t, resampled_b_x, true_b_t, false_b_t = \
                    generate_true_line_data_from_scratch(
                        signal_a_signal_times, signal_a, signal_b_signal_times, signal_b, period_ms_a, period_ms_b,
                        clock_drift_a, clock_drift_b, signal_a_sync_dur=signal_a_sync_dur, seed=None)

                true_line_data = [row + [None, None] for row in true_line_data]

                peak_indices_a = neurokit_rpeak_detect_fast(resampled_a_x, freq_hz_a)
                peak_indices_b = neurokit_rpeak_detect_fast(resampled_b_x, freq_hz_b)

                # peak_indices_a = consensus_detect(resampled_a_x, freq_hz_a)
                # peak_indices_b = consensus_detect(resampled_b_x, freq_hz_b)

                signal_a_peak_times = false_a_t[peak_indices_a]
                signal_b_peak_times = false_b_t[peak_indices_b]

                match_indices_matrix, match_distance_matrix = get_matches(signal_a_peak_times, signal_b_peak_times,
                                                                          max_offset=None)

                # offset_matrix = signal_b_peak_times[match_indices_matrix] - signal_a_peak_times[
                #                                                             :match_indices_matrix.shape[0]].reshape(-1, 1)
                #
                # offset_t = signal_a_peak_times[np.repeat(np.arange(offset_matrix.shape[0]), offset_matrix.shape[1])]
                # offset_v = offset_matrix.ravel()

                density_data = get_density_filtered_matches(signal_a_peak_times, signal_b_peak_times,
                                                            match_indices_matrix,
                                                            max_offset=None)
                if density_data is not None:
                    offset_t, offset_v = density_data['fine_offset_t'], density_data['fine_offset_v']

                    line_data = get_regression_lines(offset_t, offset_v)
                    line_data = get_filtered_lines(line_data)

                else:
                    line_data = []

                expected_slope = true_line_data[0][2][1]
                for s_true, e_true, (b_true, m_true), _, _ in true_line_data:
                    line_i = -1
                    mid_true = (s_true + e_true) / 2

                    # Find the matching line
                    for i, (s, e, (b, m), score, count) in enumerate(line_data):
                        if s <= mid_true <= e:
                            line_i = i
                            break

                    if line_i >= 0:
                        s, e, (b, m), score, count = line_data[line_i]
                        times, distances, actual, predicted = distances_between_lines(
                            [[s_true, e_true, [b_true, m_true], score, count]], line_data, step_size=100)
                        rmse = mean_squared_error(actual, predicted, squared=False)

                        total_slopes.append(m)
                        total_slopes_mean_dist.append(score)
                        total_slopes_count.append(count)
                        total_offset_actual.append(actual[0])
                        total_offset_predicted.append(predicted[0])
                        total_slope_rmse.append(rmse)

                        o_times = times[::100]
                        o_offset = actual[::100]
                        o_predicted = predicted[::100]
                        o_slopes = np.zeros(o_times.size, dtype=int) + slope_id
                        o_iteration = np.zeros(o_times.size, dtype=int) + iteration_number

                        total_o_time.append(o_times)
                        total_o_offset.append(o_offset)
                        total_o_predicted.append(o_predicted)
                        total_o_slopes.append(o_slopes)
                        total_o_iteration_number.append(o_iteration)

                    else:
                        total_slopes.append(None)
                        total_slopes_mean_dist.append(None)
                        total_slopes_count.append(None)
                        total_offset_actual.append(None)
                        total_offset_predicted.append(None)
                        total_slope_rmse.append(None)

                    pass
                    total_slope_id.append(slope_id)
                    total_slopes_start.append(s_true)
                    total_slopes_end.append(e_true)
                    total_slope_records.append(records_num)
                    total_expected_slopes.append(expected_slope)
                    total_slope_sync_time.append(signal_a_sync_dur)
                    total_clock_a_drift.append(clock_drift_a)
                    total_clock_b_drift.append(clock_drift_b)
                    total_noise_level.append(noise_level)
                    total_iteration_number.append(iteration_number)

                    slope_id += 1

    data = {
        'slope_id': total_slope_id,
        'start': total_slopes_start,
        'end': total_slopes_end,
        'slope': total_slopes,
        'offset': total_offset_actual,
        'offset_expected': total_offset_predicted,
        'rmse_from_expected': total_slope_rmse,
        'clock_a_drift': total_clock_a_drift,
        'clock_b_drift': total_clock_b_drift,
        'expected_slope': total_expected_slopes,
        'count': total_slopes_count,
        'mean_dist_from_regression': total_slopes_mean_dist,
        'noise_level_percentage': total_noise_level,
        'milliseconds_until_synchronization': total_slope_sync_time,
        'record': total_slope_records,
        'iteration_number': total_iteration_number
    }

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    df.to_csv('experiment_7_slopes.csv', index=False)

    total_o_time = np.concatenate(total_o_time, axis=None)
    total_o_offset = np.concatenate(total_o_offset, axis=None)
    total_o_predicted = np.concatenate(total_o_predicted, axis=None)
    total_o_slopes = np.concatenate(total_o_slopes, axis=None)
    total_o_iteration_number = np.concatenate(total_o_iteration_number, axis=None)

    data = {
        'Timestamp_ms': total_o_time,
        'actual_offset': total_o_offset,
        'true_offset': total_o_predicted,
        'slope_id': total_o_slopes,
        'iteration_number': total_o_iteration_number
    }

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    df.to_csv('experiment_7_offsets.csv', index=False)


def experiment_segment_length():
    records = get_records()

    total_slopes_start = []
    total_slopes_end = []
    total_slopes_mean_dist = []
    total_slopes_count = []
    total_slopes = []
    total_offset_actual = []
    total_offset_predicted = []
    total_slope_rmse = []
    total_slope_records = []
    total_slope_sync_time = []
    total_expected_slopes = []
    total_clock_a_drift = []
    total_clock_b_drift = []

    clock_a_drifts = generate_random_clock_drifts(-600, -290, size=1_000_000, seed=42)
    clock_b_drifts = generate_random_clock_drifts(10, 100, size=1_000_000, seed=42)

    records_num = 0
    for record in tqdm(records, total=48):
        records_num += 1
        # if records_num > 1:
        #     break
        print(records_num)
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

        for signal_a_sync_dur in np.arange(60_000, 600_000 + 15_000, 15_000):
            signal_a_sync_dur -= 15_000
            signal_a_sync_dur += random.randint(0, 30_000)
            clock_drift_a = next(clock_a_drifts) / (10 ** 6)
            clock_drift_b = next(clock_b_drifts) / (10 ** 6)

            true_line_data, resampled_a_x, true_a_t, false_a_t, resampled_b_x, true_b_t, false_b_t = \
                generate_true_line_data_from_scratch(
                    signal_a_signal_times, signal_a, signal_b_signal_times, signal_b, period_ms_a, period_ms_b,
                    clock_drift_a, clock_drift_b, signal_a_sync_dur=signal_a_sync_dur, seed=None)

            true_line_data = [row + [None, None] for row in true_line_data]

            peak_indices_a = neurokit_rpeak_detect_fast(resampled_a_x, freq_hz_a)
            peak_indices_b = neurokit_rpeak_detect_fast(resampled_b_x, freq_hz_b)

            # peak_indices_a = consensus_detect(resampled_a_x, freq_hz_a)
            # peak_indices_b = consensus_detect(resampled_b_x, freq_hz_b)

            signal_a_peak_times = false_a_t[peak_indices_a]
            signal_b_peak_times = false_b_t[peak_indices_b]

            match_indices_matrix, match_distance_matrix = get_matches(signal_a_peak_times, signal_b_peak_times,
                                                                      max_offset=None)

            offset_matrix = signal_b_peak_times[match_indices_matrix] - signal_a_peak_times[
                                                                        :match_indices_matrix.shape[0]].reshape(-1, 1)

            offset_t = signal_a_peak_times[np.repeat(np.arange(offset_matrix.shape[0]), offset_matrix.shape[1])]
            offset_v = offset_matrix.ravel()

            line_data = get_regression_lines(offset_t, offset_v)
            line_data = get_filtered_lines(line_data)

            expected_slope = true_line_data[0][2][1]
            for s, e, (b, m), score, count in line_data:
                times, distances, actual, predicted = distances_between_lines(
                    [[s, e, [b, m], score, count]], true_line_data, step_size=100)
                rmse = mean_squared_error(actual, predicted, squared=False)

                total_slopes.append(m)
                total_slopes_start.append(s)
                total_slopes_end.append(e)
                total_slopes_mean_dist.append(score)
                total_slopes_count.append(count)
                total_slope_records.append(records_num)
                total_expected_slopes.append(expected_slope)
                total_offset_actual.append(actual[0])
                total_offset_predicted.append(predicted[0])

                total_slope_rmse.append(rmse)
                total_slope_sync_time.append(signal_a_sync_dur)
                total_clock_a_drift.append(clock_drift_a)
                total_clock_b_drift.append(clock_drift_b)

    data = {
        'start': total_slopes_start,
        'end': total_slopes_end,
        'slope': total_slopes,
        'offset': total_offset_actual,
        'offset_expected': total_offset_predicted,
        'rmse_from_expected': total_slope_rmse,
        'clock_a_drift': total_clock_a_drift,
        'clock_b_drift': total_clock_b_drift,
        'expected_slope': total_expected_slopes,
        'count': total_slopes_count,
        'mean_dist_from_regression': total_slopes_mean_dist,
        'milliseconds_until_synchronization': total_slope_sync_time,
        'record': total_slope_records
    }

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    df.to_csv('experiement_3_slopes.csv', index=False)

    plt.scatter(np.array(total_slopes_end) - np.array(total_slopes_start), total_slope_rmse)
    plt.title("Alignment Line Duration Vs RMSE Between Measured Line and Theory")
    plt.xlabel("Duration (ms)")
    plt.ylabel("RMSE")
    plt.show()


def experiment_noise():
    records = get_records()
    total_noise_level = []
    total_benchmark = []
    total_alignment_duration_ms = []
    total_inliers = []
    total_rmse = []
    total_records = []

    records_num = 0
    for record in records:
        records_num += 1
        if records_num > 100:
            break
        print(records_num)
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

        # Create Times
        start_time_ms = 0
        period_ms_b = 1000 / freq_hz_b
        period_ms_a = 1000 / freq_hz_a

        signal_a_signal_times = np.arange(start_time_ms, start_time_ms + (signal_a.size * period_ms_a), period_ms_a)
        signal_b_signal_times = np.arange(start_time_ms, start_time_ms + (signal_b.size * period_ms_b), period_ms_b)

        clock_drift_a = -400 / (10 ** 6)  # -400 ppm
        clock_drift_b = 50 / (10 ** 6)  # 50 ppm

        true_line_data, resampled_a_x, true_a_t, false_a_t, resampled_b_x, true_b_t, false_b_t = \
            generate_true_line_data_from_scratch(
                signal_a_signal_times, signal_a, signal_b_signal_times, signal_b, period_ms_a, period_ms_b,
                clock_drift_a, clock_drift_b, seed=None)

        signal_a = resampled_a_x
        signal_b = resampled_b_x

        signal_a_signal_times = false_a_t
        signal_b_signal_times = false_b_t

        true_line_data = [row + [None, None] for row in true_line_data]

        noise_options = np.arange(0, 0.2, 0.01)

        for noise_amplitude_percentage in tqdm(noise_options):
            # Add Noise
            start_bench = time.perf_counter()
            noise_a = get_noise(signal_a, noise_amplitude_percentage, seed=None)
            noise_b = get_noise(signal_b, noise_amplitude_percentage, seed=None)

            signal_a_noise = signal_a + noise_a
            signal_b_noise = signal_b + noise_b

            peak_indices_a = neurokit_rpeak_detect_fast(signal_a_noise, freq_hz_a)
            peak_indices_b = neurokit_rpeak_detect_fast(signal_b_noise, freq_hz_b)

            # peak_indices_a = consensus_detect(signal_a, freq_hz_a)
            # peak_indices_b = consensus_detect(signal_b, freq_hz_b)

            signal_a_peak_times = signal_a_signal_times[peak_indices_a]
            signal_b_peak_times = signal_b_signal_times[peak_indices_b]

            match_indices_matrix, match_distance_matrix = get_matches(signal_a_peak_times, signal_b_peak_times,
                                                                      max_offset=None)

            # offset_matrix = signal_b_peak_times[match_indices_matrix] - signal_a_peak_times[
            #                                                             :match_indices_matrix.shape[0]].reshape(-1, 1)
            #
            # offset_t = signal_a_peak_times[np.repeat(np.arange(offset_matrix.shape[0]), offset_matrix.shape[1])]
            # offset_v = offset_matrix.ravel()

            density_data = get_density_filtered_matches(signal_a_peak_times, signal_b_peak_times, match_indices_matrix,
                                                        max_offset=None)
            offset_t, offset_v = density_data['fine_offset_t'], density_data['fine_offset_v']

            line_data = get_regression_lines(offset_t, offset_v)
            line_data = get_filtered_lines(line_data)
            end_bench = time.perf_counter()

            if len(line_data) > 0:
                times, distances, actual, predicted = distances_between_lines(line_data, true_line_data, step_size=1000)
                rmse = mean_squared_error(actual, predicted, squared=False)
            else:
                rmse = -1

            this_inliers = 0
            this_duration = 0

            for start, end, (b, m), score, count in line_data:
                this_duration += end - start
                this_inliers += count

            total_noise_level.append(noise_amplitude_percentage)
            total_benchmark.append(end_bench - start_bench)
            total_alignment_duration_ms.append(this_duration)
            total_inliers.append(this_inliers)
            total_rmse.append(rmse)
            total_records.append(records_num)

    data = {
        'noise amplitude percentage': total_noise_level,
        'benchmark seconds': total_benchmark,
        'Alignment Duration': total_alignment_duration_ms,
        'Alignment Num Inliers': total_inliers,
        'RMSE': total_rmse,
        'record': total_records
    }

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    df.to_csv('experiement_noise_1.csv', index=False)

    plt.scatter(total_noise_level, total_rmse)
    plt.title("Noise Level % VS RMSE")
    plt.ylabel("RMSE")
    plt.xlabel("Noise Level (As a % of Total Amplitude)")
    plt.show()


def experiment_clock_drift():
    records = get_records()
    total_distances = []
    total_times = []
    total_records = []

    total_slopes_start = []
    total_slopes_end = []
    total_slopes_mean_dist = []
    total_slopes_count = []
    total_slopes = []
    total_slope_records = []
    total_expected_slopes = []

    records_num = 0
    for record in records:
        records_num += 1
        print(records_num)
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

        clock_drift_a = -400 / (10 ** 6)  # -400 ppm
        clock_drift_b = 50 / (10 ** 6)  # 50 ppm

        true_line_data, resampled_a_x, true_a_t, false_a_t, resampled_b_x, true_b_t, false_b_t = \
            generate_true_line_data_from_scratch(
                signal_a_signal_times, signal_a, signal_b_signal_times, signal_b, period_ms_a, period_ms_b,
                clock_drift_a, clock_drift_b, seed=None)

        signal_a = resampled_a_x
        signal_b = resampled_b_x

        signal_a_signal_times = false_a_t
        signal_b_signal_times = false_b_t

        true_line_data = [row + [None, None] for row in true_line_data]

        peak_indices_a = neurokit_rpeak_detect_fast(signal_a, freq_hz_a)
        peak_indices_b = neurokit_rpeak_detect_fast(signal_b, freq_hz_b)

        # peak_indices_a = consensus_detect(signal_a, freq_hz_a)
        # peak_indices_b = consensus_detect(signal_b, freq_hz_b)

        signal_a_peak_times = signal_a_signal_times[peak_indices_a]
        signal_b_peak_times = signal_b_signal_times[peak_indices_b]

        match_indices_matrix, match_distance_matrix = get_matches(signal_a_peak_times, signal_b_peak_times,
                                                                  max_offset=None)

        offset_matrix = signal_b_peak_times[match_indices_matrix] - signal_a_peak_times[
                                                                    :match_indices_matrix.shape[0]].reshape(-1, 1)

        offset_t = signal_a_peak_times[np.repeat(np.arange(offset_matrix.shape[0]), offset_matrix.shape[1])]
        offset_v = offset_matrix.ravel()

        line_data = get_regression_lines(offset_t, offset_v)
        line_data = get_filtered_lines(line_data)

        times, distances = distances_between_lines(line_data, true_line_data, step_size=1000)

        indices = np.argsort(times)
        times, distances = times[indices], distances[indices]

        total_distances.append(distances)
        total_times.append(times)
        total_records.append([records_num for _ in range(len(distances))])

        expected_slope = true_line_data[0][2][1]
        for s, e, (b, m), score, count in line_data:
            total_slopes.append(m)
            total_slopes_start.append(s)
            total_slopes_end.append(e)
            total_slopes_mean_dist.append(score)
            total_slopes_count.append(count)
            total_slope_records.append(records_num)
            total_expected_slopes.append(expected_slope)

    total_distances = np.concatenate(total_distances, axis=None)
    total_times = np.concatenate(total_times, axis=None)
    total_records = np.concatenate(total_records, axis=None)

    data = {
        'distance': total_distances,
        'time': total_times,
        'record': total_records
    }

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    df.to_csv('experiement_1_consensus_distances.csv', index=False)

    data = {
        'start': total_slopes_start,
        'end': total_slopes_end,
        'slope': total_slopes,
        'expected_slope': total_expected_slopes,
        'count': total_slopes_count,
        'mean_dist_from_regression': total_slopes_mean_dist,
        'record': total_slope_records
    }

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    df.to_csv('experiement_1_consensus_slopes.csv', index=False)

    plt.hist(total_distances, bins=np.linspace(-2, 6, num=800))
    plt.xlabel("Distance (ms)")
    plt.show()


def calculate_alignment():
    # Set up raw data
    records = get_records()
    total_distances = []
    total_times = []
    total_records = []
    records_num = 0
    for record in records:
        records_num += 1
        # if total_records > 10:
        #     break
        print(records_num)
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
        signal_b_signal_times = np.arange(start_time_ms, start_time_ms + (signal_a.size * period_ms_b), period_ms_b)

        # Create Linear Transformation
        # gap_indices_a, gap_durations_a = add_gaps(signal_a_signal_times, int(period_ms_a), gap_density=0.000005, seed=42)
        # gap_end_times_a = signal_a_signal_times[gap_indices_a]
        # gap_start_times_a = signal_a_signal_times[gap_indices_a - 1]

        # gaps_a = np.column_stack((gap_start_times_a, gap_end_times_a))
        gaps_a = np.array([], dtype=int)
        intervals_a = convert_gaps_to_intervals(signal_a_signal_times[0], signal_a_signal_times[-1], gaps_a)
        clock_drift = 400 / (10 ** 6)  # 400 ppm
        true_line_data = generate_true_line_data(signal_a_signal_times, intervals_a, clock_drift, period_ms_b,
                                                 seed=None)

        # Apply A Gaps to B
        # apply_gaps(signal_b_signal_times, gaps_a)

        # Apply Transformation
        true_line_data = [row + [None, None] for row in true_line_data]
        b_indices, a_original_times, signal_b_signal_times = \
            apply_alignment(signal_b_signal_times, true_line_data)

        signal_b = signal_b[b_indices]

        # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        #
        # ax1.plot(signal_a_signal_times, signal_a, color='blue')
        # ax1.set_title(f'Signal A - {freq_hz_a}Hz - {0}ppm Drift')
        #
        # ax2.plot(signal_b_signal_times, signal_b, color='red')
        # ax2.set_title(f'Signal B - {freq_hz_b}Hz - {(10 ** 6) * clock_drift}ppm Drift')
        #
        # plt.xlabel('Time (ms)')
        #
        # plt.tight_layout()
        # plt.show()

        peak_indices_a = neurokit_rpeak_detect_fast(signal_a, freq_hz_a)
        peak_indices_b = neurokit_rpeak_detect_fast(signal_b, freq_hz_b)

        signal_a_peak_times = signal_a_signal_times[peak_indices_a]
        signal_b_peak_times = signal_b_signal_times[peak_indices_b]

        match_indices_matrix, match_distance_matrix = get_matches(signal_a_peak_times, signal_b_peak_times,
                                                                  max_offset=None)

        offset_matrix = signal_b_peak_times[match_indices_matrix] - signal_a_peak_times[
                                                                    :match_indices_matrix.shape[0]].reshape(-1, 1)

        offset_t = signal_a_peak_times[np.repeat(np.arange(offset_matrix.shape[0]), offset_matrix.shape[1])]
        offset_v = offset_matrix.ravel()

        line_data = get_regression_lines(offset_t, offset_v)
        line_data = get_filtered_lines(line_data)

        times, distances = distances_between_lines(line_data, true_line_data, step_size=1000)

        indices = np.argsort(times)
        times, distances = times[indices], distances[indices]

        total_distances.append(distances)
        total_times.append(times)
        total_records.append([records_num for _ in range(len(distances))])

        # plt.scatter(times, distances)
        # plt.ylabel("distance, (ms)")
        # plt.xlabel("Times (ms)")
        # plt.show()

    total_distances = np.concatenate(total_distances, axis=None)
    total_times = np.concatenate(total_times, axis=None)
    total_records = np.concatenate(total_records, axis=None)

    data = {
        'distance': total_distances,
        'time': total_times,
        'record': total_records
    }

    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    df.to_csv('output_2.csv', index=False)


    plt.hist(total_distances, bins=np.linspace(-2, 2, num=200))
    plt.xlabel("Distance (ms)")
    plt.show()


def calculate_overall_std_signal():
    total_signal = []
    for record in get_records():
        total_signal.append(record.p_signal[:, 0])

    return np.std(np.concatenate(total_signal, axis=None))


if __name__ == "__main__":
    experiment_comprehensive()
    # experiment_segment_length()
    # experiment_noise()
    # experiment_clock_drift()
    # calculate_alignment()
