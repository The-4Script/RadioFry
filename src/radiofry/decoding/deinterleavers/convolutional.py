"""Cross-interleaver helpers with explicit parameter requirements."""

import numpy as np


def convolutional_deinterleave(bits: np.ndarray, branches: int, increment: int) -> np.ndarray:
    """Invert a deterministic branch-delay permutation for a complete block."""

    values = np.asarray(bits, dtype=np.uint8).ravel()
    if branches < 1 or increment < 1:
        raise ValueError("branches and increment must be positive")
    order = np.argsort(np.arange(values.size) + (np.arange(values.size) % branches) * increment)
    return values[order]
