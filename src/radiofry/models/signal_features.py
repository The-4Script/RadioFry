"""Feature-channel construction shared by training and runtime inference."""

import numpy as np


def add_signal_features(iq_channels: np.ndarray, *, include_engineered: bool) -> np.ndarray:
    """Return I/Q channels with optional amplitude and phase-difference channels."""

    values = np.asarray(iq_channels, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("iq_channels must have shape (2, samples)")
    if not include_engineered:
        return values
    complex_values = values[0] + 1j * values[1]
    amplitude = np.abs(complex_values).astype(np.float32)
    phase_difference = np.angle(complex_values[1:] * np.conj(complex_values[:-1]))
    phase_difference = np.concatenate(([0.0], phase_difference)).astype(np.float32)
    channels = np.vstack((values, amplitude, phase_difference))
    channel_power = np.sqrt(np.mean(channels**2, axis=1, keepdims=True))
    return channels / np.maximum(channel_power, 1e-6)


def add_signal_features_batch(frames: np.ndarray, *, include_engineered: bool) -> np.ndarray:
    """Apply feature construction to frames shaped (batch, channels, samples)."""

    values = np.asarray(frames, dtype=np.float32)
    if not include_engineered:
        return values
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("frames must have shape (batch, 2, samples)")
    return np.stack([add_signal_features(frame, include_engineered=True) for frame in values]).astype(np.float32)
