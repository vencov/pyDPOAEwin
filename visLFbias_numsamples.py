# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 17:12:14 2025

This codes yields useful information on biasing, the number of samples for processing, window length


@author: audiobunka
"""

import numpy as np
from math import gcd
from functools import reduce

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

def lcm_multiple(numbers):
    return reduce(lcm, numbers)

def suggest_window(f_bias, f1=None, f2=None, fs=44100, max_periods=1000):
    """
    Suggest a window length (in samples) that contains an integer number of periods of the bias tone,
    and optionally f1 and f2.

    Parameters:
    - f_bias: Bias tone frequency (Hz)
    - f1, f2: Optional additional tones (Hz)
    - fs: Sampling frequency (Hz)
    - max_periods: Maximum number of bias tone periods to search

    Returns:
    - (n_samples, n_periods, window_duration_seconds)
    """
    base_ratio = fs / f_bias  # Not necessarily integer
    freqs = [f_bias]

    if f1: freqs.append(f1)
    if f2: freqs.append(f2)

    # Determine common time base for all frequencies (LCM of denominators)
    freq_ratios = [fs / f for f in freqs]
    denominators = [ratio.as_integer_ratio()[1] for ratio in freq_ratios]
    common_multiple = lcm_multiple(denominators)

    # Find smallest number of bias tone periods that results in integer sample count
    for n_periods in range(1, max_periods + 1):
        window_duration = n_periods / f_bias
        n_samples = window_duration * fs
        if n_samples.is_integer():
            return int(n_samples), n_periods, window_duration

    return None, None, None  # No match found in range

# === Example usage ===
f_bias = 96
f1 = 2000
f2 = 2400
fs = 44100

samples, periods, duration = suggest_window(f_bias, f1, f2, fs)
if samples:
    print(f"Suggested window:")
    print(f"- Samples: {samples}")
    print(f"- Bias tone periods: {periods}")
    print(f"- Duration: {duration:.6f} seconds")
else:
    print("No suitable window found in the given search range.")
