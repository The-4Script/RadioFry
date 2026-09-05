"""Nearest-point PSK demodulation for already symbol-timed samples."""

import numpy as np

from .common import DemodulationResult


def demodulate_psk(samples: np.ndarray, order: int = 2) -> DemodulationResult:
    if order not in {2, 4, 8}:
        raise ValueError("PSK order must be 2, 4, or 8")
    values = np.asarray(samples, dtype=np.complex64)
    phases = np.mod(np.angle(values), 2 * np.pi)
    indices = np.floor((phases + np.pi / order) / (2 * np.pi / order)).astype(np.uint8) % order
    bits_per_symbol = int(np.log2(order))
    bits = ((indices[:, None] >> np.arange(bits_per_symbol - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    return DemodulationResult(indices, bits, f"{order}PSK", {"order": order})
