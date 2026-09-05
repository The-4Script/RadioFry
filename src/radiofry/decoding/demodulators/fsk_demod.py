"""Binary FSK demodulation from instantaneous frequency."""

import numpy as np

from .common import DemodulationResult


def demodulate_fsk(samples: np.ndarray) -> DemodulationResult:
    values = np.asarray(samples, dtype=np.complex64)
    frequency = np.diff(np.unwrap(np.angle(values)))
    threshold = float(np.median(frequency)) if frequency.size else 0.0
    bits = (frequency >= threshold).astype(np.uint8)
    return DemodulationResult(frequency, bits, "2FSK", {"threshold": threshold})
