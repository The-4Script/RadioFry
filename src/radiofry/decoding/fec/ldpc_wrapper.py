"""Best-effort LDPC adapter with honest degradation."""

import numpy as np

from .viterbi_wrapper import FECResult


def decode_ldpc(bits: np.ndarray) -> FECResult:
    values = np.asarray(bits, dtype=np.uint8).ravel() & 1
    try:
        import pyldpc  # noqa: F401
    except ImportError:
        return FECResult(values, "ldpc", False, "LDPC decoding is unavailable because pyldpc is not installed.")
    return FECResult(values, "ldpc", False, "LDPC code parameters are required for decoding.")
