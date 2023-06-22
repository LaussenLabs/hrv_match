# HRV Match

HRV Match is a Python library for aligning heart rate variability (HRV) signals recorded by different devices with different clocks. It provides an algorithm that calculates the mapping function between the time spaces of two HRV signals, allowing for precise alignment and synchronization.

## Overview

The main goal of HRV Match is to take two signals from which HRV can be derived, recorded by different devices with different clock behaviors, and provide a mapping function from one clock's time space to the other's. The library achieves this by performing the following steps:

1. Windowed Signal Matching: Matches between windows of R-peak time points of the two signals are found using the Euclidean distance between windows of R-peak times. The matches with the smallest distances are selected.
2. Density Filtering: Clusters of similar matches are given preference over isolated outliers.
3. Robust Linear Regression: Clusters are then quantified by Robust Linear Regression (RANSAC). Each cluster represents a part of a piecewise linear function to map time from timespace A to timespace B.
4. Regression Filtering: The regression lines are filtered based on their inlier density, slope range, and collision with other lines.
5. Application: Timestamp arrays can then be given as input to the linear function defined by the regression step, their output represents a transformation into the timespace of the other device.

## Installation

To install HRV Match, you can use pip:

```bash
cd /path/to/repo
pip install .
```

You can also install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

If you're looking to better understand the algorithm, we recommend walking through the provided 
[Notebook Example](./algorithm_explantation.ipynb) which shows the key steps of the algorithm in detail.

If you just want to quickly get started, here is an example of using HRV Match to align two ECG signals pulled from 
MIT-BIH:

```python
import numpy as np
from tests.generate_wfdb import get_records
from tests.add_noise import add_noise
from tests.calculate_true_offset import generate_true_line_data_from_scratch
from tests.rpeak_detector import neurokit_rpeak_detect_fast
from peakalignment import align_peaks

# Set up raw data
records = get_records()
record = next(records)
signal_a = record.p_signal[:, 0]
signal_b = np.copy(signal_a)
freq_hz_b = freq_hz_a = record.fs

add_noise(signal_a, 0.20)
add_noise(signal_b, 0.30)

# Create Times
start_time_ms = 0
period_ms_b = 1000 / freq_hz_b
period_ms_a = 1000 / freq_hz_a

signal_a_signal_times = np.arange(start_time_ms, start_time_ms + (signal_a.size * period_ms_a), period_ms_a)
signal_b_signal_times = np.arange(start_time_ms, start_time_ms + (signal_b.size * period_ms_b), period_ms_b)

# Add Clock Drift
clock_drift_a = -420 / (10 ** 6)  # - 420 ppm
clock_drift_b = -20 / (10 ** 6)  # - 20 ppm

# Simulate Server Resynchronization.
signal_a_sync_dur = 600_000  # Sync every 10 min or 600k ms.
_, signal_a, _, signal_a_signal_times, signal_b, _, signal_b_signal_times = \
    generate_true_line_data_from_scratch(
        signal_a_signal_times, signal_a, signal_b_signal_times, signal_b, period_ms_a, period_ms_b,
        clock_drift_a, clock_drift_b, signal_a_sync_dur=signal_a_sync_dur, seed=None)

peak_indices_a = neurokit_rpeak_detect_fast(signal_a, freq_hz_a)
peak_indices_b = neurokit_rpeak_detect_fast(signal_b, freq_hz_b)

signal_a_peak_times = signal_a_signal_times[peak_indices_a]
signal_b_peak_times = signal_b_signal_times[peak_indices_b]

# Align signal A to the signal B clock.
aligned_signal_a_times = align_peaks(signal_a_peak_times, signal_b_peak_times, signal_a_signal_times)
```

## Documentation

For detailed documentation and usage examples...

## Contributing

Contributions are welcome! If you encounter any bugs or have suggestions for new features, please open an issue on this repo.

## License

This project is licensed under the Apache 2.0 - see the LICENSE file for details.

## Acknowledgements

Authors: William Dixon and Andrew Goodwin. Research Group: LaussenLabs. Institution: The Hospital For Sick Children.

## References

- Reference 1
- Reference 2
- Reference 3