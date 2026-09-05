"""Basic analytic-signal demodulators for analog benchmark classes."""

import numpy as np


def demodulate_am(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.complex64)
    envelope = np.abs(values)
    return (envelope - np.mean(envelope)).astype(np.float32)


def demodulate_fm(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.complex64)
    return np.diff(np.unwrap(np.angle(values))).astype(np.float32)
