"""Square-QAM nearest-grid demodulation."""

import numpy as np

from .common import DemodulationResult


def demodulate_qam(samples: np.ndarray, order: int = 16) -> DemodulationResult:
    side = int(np.sqrt(order))
    if side * side != order or side not in {4, 8}:
        raise ValueError("QAM order must be 16 or 64")
    values = np.asarray(samples, dtype=np.complex64)
    levels = np.arange(-(side - 1), side, 2, dtype=np.float32)
    real_index = np.argmin(np.abs(values.real[:, None] - levels), axis=1)
    imag_index = np.argmin(np.abs(values.imag[:, None] - levels), axis=1)
    indices = (real_index * side + imag_index).astype(np.uint8)
    bits_per_symbol = int(np.log2(order))
    bits = ((indices[:, None] >> np.arange(bits_per_symbol - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    symbols = levels[real_index] + 1j * levels[imag_index]
    return DemodulationResult(symbols, bits, f"QAM{order}", {"order": order})
