"""Stage 1: EEG Standardization Module

Converts heterogeneous EEG data into a uniform format:
- Resample → 200Hz
- Re-reference → Common Average Reference
- Bandpass filter → 0.5~75Hz
- Segment → 2-second windows
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from math import gcd


def resample(signal: np.ndarray, orig_srate: int, target_srate: int = 200) -> np.ndarray:
    """Resample EEG signal to target sampling rate.

    Args:
        signal: [C, T] array
        orig_srate: original sampling rate in Hz
        target_srate: target sampling rate in Hz
    Returns:
        [C, T'] resampled array
    """
    if orig_srate == target_srate:
        return signal
    g = gcd(orig_srate, target_srate)
    up = target_srate // g
    down = orig_srate // g
    return resample_poly(signal, up, down, axis=-1)


def common_average_reference(signal: np.ndarray) -> np.ndarray:
    """Apply Common Average Reference.

    Args:
        signal: [C, T] array
    Returns:
        [C, T] re-referenced array
    """
    return signal - signal.mean(axis=0, keepdims=True)


def bandpass_filter(
    signal: np.ndarray,
    srate: int,
    low: float = 0.5,
    high: float = 75.0,
    order: int = 4,
) -> np.ndarray:
    """Apply bandpass filter using second-order sections.

    Args:
        signal: [C, T] array
        srate: sampling rate in Hz
        low: low cutoff frequency
        high: high cutoff frequency
        order: filter order
    Returns:
        [C, T] filtered array
    """
    sos = butter(order, [low, high], btype="band", fs=srate, output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def segment_windows(
    signal: np.ndarray,
    srate: int,
    window_sec: float = 2.0,
    stride_sec: float | None = None,
) -> np.ndarray:
    """Segment signal into fixed-length windows.

    Args:
        signal: [C, T] array
        srate: sampling rate in Hz
        window_sec: window length in seconds
        stride_sec: stride in seconds (defaults to window_sec, i.e. no overlap)
    Returns:
        [num_windows, C, window_samples] array
    """
    if stride_sec is None:
        stride_sec = window_sec

    window_samples = int(srate * window_sec)
    stride_samples = int(srate * stride_sec)
    num_channels, total_samples = signal.shape

    windows = []
    start = 0
    while start + window_samples <= total_samples:
        windows.append(signal[:, start : start + window_samples])
        start += stride_samples

    if len(windows) == 0:
        raise ValueError(
            f"Signal too short ({total_samples} samples) for window "
            f"({window_samples} samples)"
        )

    return np.stack(windows, axis=0)


def standardize_eeg(
    signal: np.ndarray,
    orig_srate: int,
    target_srate: int = 200,
    bandpass_low: float = 0.5,
    bandpass_high: float = 75.0,
    window_sec: float = 2.0,
    stride_sec: float | None = None,
) -> np.ndarray:
    """Full standardization pipeline.

    Args:
        signal: [C, T] raw EEG array
        orig_srate: original sampling rate
        target_srate: target sampling rate
        bandpass_low: low cutoff Hz
        bandpass_high: high cutoff Hz
        window_sec: window size in seconds
        stride_sec: stride in seconds
    Returns:
        [num_windows, C, window_samples] standardized array
    """
    x = resample(signal, orig_srate, target_srate)
    x = common_average_reference(x)
    x = bandpass_filter(x, target_srate, bandpass_low, bandpass_high)
    x = segment_windows(x, target_srate, window_sec, stride_sec)
    return x
