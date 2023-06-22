import numpy as np

from peakalignment.peak_alignment import align_peaks
from tests.add_gaps import add_gaps
from tests.generate_wfdb import get_records
from tests.rpeak_detector import neurokit_rpeak_detect_fast


def test_total_process():
    # Set up input data
    records = get_records()
    record = next(records)
    signal = record.p_signal[:, 0]  # Assuming the first channel is the ECG signal
    freq_hz = 500

    start_time_ms = 0
    period_ms = 1000 / freq_hz

    signal_a_signal_times = np.arange(start_time_ms, start_time_ms + (signal.size * period_ms), period_ms)

    signal_b_signal_times = signal_a_signal_times.copy()
    gap_indices, gap_durations = add_gaps(signal_b_signal_times, int(period_ms), gap_density=0.00001)

    peak_indices = neurokit_rpeak_detect_fast(signal, freq_hz)

    signal_a_peak_times = signal_a_signal_times[peak_indices]
    signal_b_peak_times = signal_b_signal_times[peak_indices]

    aligned_times = align_peaks(signal_a_peak_times, signal_b_peak_times, signal_a_signal_times, max_offset=None)
