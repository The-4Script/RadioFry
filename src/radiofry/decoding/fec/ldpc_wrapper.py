"""Explicit classification-only LDPC adapter."""

import numpy as np

from .viterbi_wrapper import FECResult


def decode_ldpc(bits: np.ndarray) -> FECResult:
    values = np.asarray(bits, dtype=np.uint8).ravel() & 1
    return FECResult(values, "ldpc", False, "LDPC was classified, but decoding is disabled until code parameters are supplied.")
