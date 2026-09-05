"""Deterministic statistical features for interleaver/FEC classifiers."""

import numpy as np


def bit_features(bits: np.ndarray, *, max_lag: int = 32) -> np.ndarray:
    values = np.asarray(bits, dtype=np.float32).ravel()
    if values.size == 0:
        return np.zeros(max_lag + 8, dtype=np.float32)
    centered = values - values.mean()
    scale = np.dot(centered, centered) + 1e-12
    autocorrelation = np.array([
        np.dot(centered[:-lag] if lag else centered, centered[lag:]) / scale
        for lag in range(max_lag)
    ], dtype=np.float32)
    transitions = np.diff(values).astype(bool)
    runs = np.diff(np.flatnonzero(np.r_[True, transitions, True])).astype(np.float32)
    probabilities = np.bincount(values.astype(np.uint8), minlength=2) / values.size
    entropy = float(-sum(probability * np.log2(probability) for probability in probabilities if probability > 0))
    summary = [values.mean(), values.std(), runs.mean(), runs.std(), runs.max(), entropy, values.size, np.mean(np.abs(np.fft.rfft(centered)))]
    return np.concatenate([autocorrelation, summary]).astype(np.float32)
