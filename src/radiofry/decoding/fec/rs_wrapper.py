"""Optional reedsolo adapter."""

from .viterbi_wrapper import FECResult


def decode_reed_solomon(data: bytes, *, nsym: int = 32) -> FECResult:
    try:
        from reedsolo import RSCodec
        decoded = RSCodec(nsym).decode(data)[0]
        return FECResult(__import__("numpy").frombuffer(decoded, dtype="uint8"), "reed_solomon", True)
    except (ImportError, ValueError, IndexError, TypeError) as error:
        import numpy as np
        return FECResult(np.frombuffer(data, dtype=np.uint8), "reed_solomon", False, f"Reed-Solomon decode failed: {error}")
