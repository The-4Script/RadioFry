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
    return np.concatenate([np.roll(row, index % width) for index, row in enumerate(matrix)])


def diagonal_deinterleave(bits: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).ravel()
    if width < 1 or values.size % width:
        raise ValueError("bitstream length must be divisible by width")
    matrix = values.reshape(-1, width)
    return np.concatenate([np.roll(row, -(index % width)) for index, row in enumerate(matrix)])


def _convolutional_permutation(size: int, branches: int, increment: int) -> np.ndarray:
    if branches < 1 or increment < 1:
        raise ValueError("branches and increment must be positive")
    indices = np.arange(size)
    return np.argsort(indices + (indices % branches) * increment, kind="stable")


def convolutional_interleave(bits: np.ndarray, branches: int, increment: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).ravel()
    return values[_convolutional_permutation(values.size, branches, increment)]


def convolutional_deinterleave(bits: np.ndarray, branches: int, increment: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).ravel()
    permutation = _convolutional_permutation(values.size, branches, increment)
    return values[np.argsort(permutation, kind="stable")]


def pseudo_random_interleave(bits: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).ravel()
    generator = np.random.default_rng(seed)
    return values[generator.permutation(values.size)]
