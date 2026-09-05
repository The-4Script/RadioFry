"""Simple diagonal interleaver inversion."""

import numpy as np


def diagonal_deinterleave(bits: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).ravel()
    if width < 1 or values.size % width:
        raise ValueError("bitstream length must be divisible by width")
    matrix = values.reshape(-1, width)
    return np.concatenate([np.roll(row, -(index % width)) for index, row in enumerate(matrix)])
