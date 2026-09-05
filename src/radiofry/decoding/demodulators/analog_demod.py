"""Basic analytic-signal demodulators for analog benchmark classes."""

import numpy as np


def demodulate_am(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.complex64)
    envelope = np.abs(values)
    return (envelope - np.mean(envelope)).astype(np.float32)


def demodulate_ssb(
    samples: np.ndarray,
    sample_rate: float,
    carrier_frequency: float | None,
) -> np.ndarray:
    """Use a product detector for SSB; an envelope detector needs a carrier."""

    values = np.asarray(samples, dtype=np.complex64)
    if values.size == 0:
        return np.array([], dtype=np.float32)
    carrier = float(carrier_frequency or 0.0)
    time = np.arange(values.size, dtype=np.float64) / sample_rate
    baseband = values * np.exp(-2j * np.pi * carrier * time)
    return np.real(baseband).astype(np.float32)


def demodulate_fm(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.complex64)
    return np.diff(np.unwrap(np.angle(values))).astype(np.float32)
