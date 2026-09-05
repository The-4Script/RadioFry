"""Rectangular block deinterleaving."""

import numpy as np


def block_deinterleave(bits: np.ndarray, rows: int, columns: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).ravel()
    size = rows * columns
    if rows < 1 or columns < 1 or values.size % size:
        raise ValueError("bitstream length must be divisible by rows * columns")
    return values.reshape(-1, columns, rows).transpose(0, 2, 1).reshape(-1)
