"""Optional reedsolo adapter."""

import numpy as np

from .viterbi_wrapper import FECResult


def decode_reed_solomon(data: bytes, *, nsym: int = 32) -> FECResult:
    try:
        from reedsolo import RSCodec, ReedSolomonError
    except ImportError as error:
        fallback = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        return FECResult(fallback, "reed_solomon", False, f"Reed-Solomon decode failed: {error}")
    try:
        decoded = RSCodec(nsym).decode(data)[0]
        return FECResult(np.unpackbits(np.frombuffer(decoded, dtype=np.uint8)), "reed_solomon", True)
    except (ValueError, IndexError, TypeError, ReedSolomonError) as error:
        fallback = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        return FECResult(fallback, "reed_solomon", False, f"Reed-Solomon decode failed: {error}")
