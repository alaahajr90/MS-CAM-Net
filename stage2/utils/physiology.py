"""Signal-processing and physiological metrics for Stage 2 evaluation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import signal
from scipy.stats import pearsonr

FS = 35.0
CARDIAC_BAND = (0.7, 3.5)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    a, b = a[:n], b[:n]
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(pearsonr(a, b)[0])


def polarity_corrected_pearson(pred: np.ndarray, ref: np.ndarray) -> float:
    r = safe_pearson(pred, ref)
    return float(abs(r)) if np.isfinite(r) else r


def lag_aware_pearson(pred: np.ndarray, ref: np.ndarray, max_lag_seconds: float = 1.0, fs: float = FS) -> float:
    pred, ref = zscore(pred), zscore(ref)
    max_lag = int(round(max_lag_seconds * fs))
    best = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = pred[-lag:], ref[: len(ref) + lag]
        elif lag > 0:
            a, b = pred[: len(pred) - lag], ref[lag:]
        else:
            a, b = pred, ref
        r = safe_pearson(a, b)
        if np.isfinite(r):
            best = max(best, abs(r))
    return float(best) if np.isfinite(best) else float("nan")


def artifact_masked_nrmse(pred: np.ndarray, ref: np.ndarray) -> float:
    pred, ref = zscore(pred), zscore(ref)
    n = min(len(pred), len(ref))
    pred, ref = pred[:n], ref[:n]
    robust_scale = np.median(np.abs(ref - np.median(ref))) * 1.4826 + 1e-8
    mask = np.abs(ref - np.median(ref)) <= 5.0 * robust_scale
    if mask.sum() < 3:
        mask = np.ones(n, dtype=bool)
    rmse = np.sqrt(np.mean((pred[mask] - ref[mask]) ** 2))
    denom = np.std(ref[mask]) + 1e-8
    return float(rmse / denom)


def cardiac_band_snr_db(x: np.ndarray, fs: float = FS) -> float:
    x = zscore(x)
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(len(x), int(20 * fs)))
    band = (freqs >= CARDIAC_BAND[0]) & (freqs <= CARDIAC_BAND[1])
    signal_power = float(np.trapz(psd[band], freqs[band])) if np.any(band) else 0.0
    total_power = float(np.trapz(psd, freqs)) + 1e-12
    noise_power = max(total_power - signal_power, 1e-12)
    return float(10.0 * np.log10(max(signal_power, 1e-12) / noise_power))


def cardiac_band_coherence(pred: np.ndarray, ref: np.ndarray, fs: float = FS) -> float:
    pred, ref = zscore(pred), zscore(ref)
    n = min(len(pred), len(ref))
    freqs, coh = signal.coherence(pred[:n], ref[:n], fs=fs, nperseg=min(n, int(20 * fs)))
    band = (freqs >= CARDIAC_BAND[0]) & (freqs <= CARDIAC_BAND[1])
    return float(np.mean(coh[band])) if np.any(band) else float("nan")


def bandpass(x: np.ndarray, fs: float = FS) -> np.ndarray:
    x = zscore(x)
    sos = signal.butter(4, CARDIAC_BAND, btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def hr_from_waveform(x: np.ndarray, fs: float = FS) -> float:
    x = bandpass(x, fs=fs)
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(len(x), int(20 * fs)), nfft=max(4096, len(x)))
    band = (freqs >= CARDIAC_BAND[0]) & (freqs <= CARDIAC_BAND[1])
    if not np.any(band):
        return float("nan")
    return float(freqs[band][np.argmax(psd[band])] * 60.0)


def rmssd_from_waveform(x: np.ndarray, fs: float = FS) -> float:
    x = bandpass(x, fs=fs)
    peaks, _ = signal.find_peaks(x, distance=max(1, int(fs * 0.3)), prominence=0.1)
    if len(peaks) < 3:
        return float("nan")
    ibi_ms = np.diff(peaks) / fs * 1000.0
    return float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2))) if len(ibi_ms) >= 2 else float("nan")


def weighted_overlap_add(windows: Iterable[tuple[int, np.ndarray]], total_length: int) -> np.ndarray:
    """Reconstruct a signal with Hann-weighted overlap-add."""
    output = np.zeros(total_length, dtype=np.float64)
    weights = np.zeros(total_length, dtype=np.float64)
    for start, values in windows:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        end = min(start + len(values), total_length)
        values = values[: end - start]
        if len(values) == 0:
            continue
        window = np.hanning(len(values))
        if np.max(window) <= 0:
            window = np.ones(len(values))
        output[start:end] += values * window
        weights[start:end] += window
    valid = weights > 1e-8
    output[valid] /= weights[valid]
    if not np.all(valid) and np.any(valid):
        valid_idx = np.flatnonzero(valid)
        output[~valid] = np.interp(np.flatnonzero(~valid), valid_idx, output[valid])
    return output


def summarize(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
