"""Bit-level interleavers used for classifier and decoder fixtures."""

import numpy as np


def block_interleave(bits: np.ndarray, rows: int, columns: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).ravel()
    size = rows * columns
    if values.size % size:
        raise ValueError("bitstream length must be divisible by rows * columns")
    return values.reshape(-1, rows, columns).transpose(0, 2, 1).reshape(-1)


def diagonal_interleave(bits: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).ravel()
    if values.size % width:
        raise ValueError("bitstream length must be divisible by width")
    matrix = values.reshape(-1, width)
    return np.concatenate([np.roll(row, index) for row in matrix for index in [0]]) if width == 1 else matrix[:, ::-1].ravel()


def pseudo_random_interleave(bits: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).ravel()
    generator = np.random.default_rng(seed)
    return values[generator.permutation(values.size)]
