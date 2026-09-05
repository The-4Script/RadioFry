"""Cross-interleaver helpers with explicit parameter requirements."""

import numpy as np


def convolutional_deinterleave(bits: np.ndarray, branches: int, increment: int) -> np.ndarray:
    """Invert the deterministic branch-delay permutation used by generation."""

    values = np.asarray(bits, dtype=np.uint8).ravel()
    if branches < 1 or increment < 1:
        raise ValueError("branches and increment must be positive")
    indices = np.arange(values.size)
    permutation = np.argsort(indices + (indices % branches) * increment, kind="stable")
    return values[np.argsort(permutation, kind="stable")]
