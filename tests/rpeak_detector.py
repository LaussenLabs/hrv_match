import biosppy
import neurokit2 as nk
import numpy as np


def neurokit_rpeak_detect_fast(signal_values, freq_hz: int):
    clean_signal = nk.ecg_clean(signal_values, sampling_rate=int(freq_hz))
    signals, info = nk.ecg_peaks(clean_signal, sampling_rate=int(freq_hz))

    peak_indices = info["ECG_R_Peaks"]
    corrected_indices = biosppy.signals.ecg.correct_rpeaks(signal_values, peak_indices, int(freq_hz))
    return corrected_indices
